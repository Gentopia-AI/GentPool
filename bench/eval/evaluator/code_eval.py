import multiprocessing
import random
import time

from gentopia.agent import BaseAgent
from gentopia.model import AgentOutput

from bench.eval import BaseEval
from bench.eval.base_eval import EvalResult
from bench.prompt.code_eval import APPSPrompt, HumanEvalPrompt, MBPPPrompt
from .utils import *


class CodeEval(BaseEval):
    """
    Evaluation class for coding tasks. 
    Such tasks should have the following keys in the json file:
    - problem: the problem description
    - test_case: the test cases to the problem
    - dataset: the dataset the task belongs to
    Now the dataset is temporarily hard-coded to support 3 types of datasets ["apps", "humaneval" and "mbpp"].
    """
    eval_class: str
    eval_subclass: str
    grader: Optional[BaseGrader] = None

    def _print_result(self, result: EvalResult):
        output = [
            "\n### FINISHING Agent EVAL ###",
            " (づ￣ ³￣)づ",
            f"Agent score: {result.score * 100}",
            f"Agent run exception rate: {result.fail_rate * 100}%",
            f"Avg runtime per task: {round(result.avg_runtime, 2)}s",
            f"Avg cost per 1000 runs: ${round(result.avg_cost * 1000, 3)}",
            f"Avg token usage per task: {round(result.avg_token_usage, 1)} tokens",
            f"... And the total cost for evaluation ${round(result.eval_cost, 5)}"
        ]

        for line in output:
            print(line, end=' ', flush=True)
            time.sleep(0.7)
            print()

        if result.score >= 0.8:
            print("Excellent Scoring! (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧")
        elif result.score >= 0.6:
            print("It passed, at least. (￣▽￣)ノ")
        else:
            print(f"Your agent needs some tuning for {self.eval_class}/{self.eval_subclass}. (╯°□°）╯︵ ┻━┻)")

    def _check_correctness(self, program: str, timeout: float) -> str:
        """
        Evaluates the functional correctness of a completion by running the test
        suite provided in the problem. 

        :param completion_id: an optional completion ID so we can match
            the results later even if execution finishes asynchronously.
        """

        def unsafe_execute():

            with create_tempdir():

                # These system calls are needed when cleaning up tempdir.
                import os
                import shutil
                rmtree = shutil.rmtree
                rmdir = os.rmdir
                chdir = os.chdir

                # Disable functionalities that can make destructive changes to the test.
                reliability_guard()

                try:
                    exec_globals = {}
                    with swallow_io():
                        with time_limit(timeout):
                            exec(program, exec_globals)
                    result.append("pass")
                except TimeoutException:
                    result.append("timed out")
                except BaseException as e:
                    result.append(f"failed: {e}")

                # Needed for cleaning up.
                shutil.rmtree = rmtree
                os.rmdir = rmdir
                os.chdir = chdir

        manager = multiprocessing.Manager()
        result = manager.list()

        p = multiprocessing.Process(target=unsafe_execute)
        p.start()
        p.join(timeout=timeout + 1)
        if p.is_alive():
            p.kill()

        if not result:
            result.append("timed out")

        return result[0]

    def evaluate(self, agent: BaseAgent, n_smaple: int, seed=0, private=False, verbose=True,
                 time_limit=5) -> EvalResult:
        ## Randomly sample 
        random.seed(seed)
        data = []

        if private:
            file_path = f"bench/benchmark/private/{self.eval_class}/{self.eval_subclass}/"
        else:
            file_path = f"bench/benchmark/public/{self.eval_class}/{self.eval_subclass}/"

        for file_name in os.listdir(file_path):
            if file_name.endswith(".json"):
                with open(file_path + file_name, "r") as f:
                    tmp = json.load(f)
                    data += [tmp]

        random.shuffle(data)
        data = data[:n_smaple]
        ## Run the agent and grader        
        total_score = 0
        total_cost = 0
        total_token = 0
        total_runtime = 0
        num_failed = 0
        eval_grader_cost = 0
        count = 0
        for task in data:
            count += 1
            print(f">>> Running Eval {count}/{n_smaple} ...")
            st = time.time()
            problem = task.get("problem", None)
            dataset = task.get("dataset", None)
            if dataset == "apps":
                agent_instruction = APPSPrompt.format(problem=problem)
            elif dataset == "humaneval":
                agent_instruction = HumanEvalPrompt.format(problem=problem)
            elif dataset == "mbpp":
                agent_instruction = MBPPPrompt.format(problem=problem)

            try:
                response = agent.run(agent_instruction)
                if verbose:
                    print("> Agent run successful.")
            except Exception as e:
                num_failed += 1
                response = AgentOutput(output="Agent failed", cost=0, token_usage=0)
                if verbose:
                    print("> Agent run failed.")
            et = time.time() - st
            total_cost += response.cost
            total_token += response.token_usage

            if response.output != "Agent failed":
                total_runtime += et
                if dataset == "apps":
                    test = convert_apps_code(response.output, task["test_case"])
                elif dataset == "humaneval":
                    test = response.output + "\n" + task["test_case"]
                elif dataset == "mbpp":
                    test = response.output + "\n" + task["test_case"]
                output = self._check_correctness(test, time_limit)
                if verbose:
                    print(f"> Grader: {output}")
                eval_grader_cost += 0
                if "pass" in output.lower():
                    total_score += 1

        valid_sample = n_smaple - num_failed

        result = EvalResult(score=total_score / valid_sample,
                            fail_rate=num_failed / n_smaple,
                            avg_runtime=total_runtime / valid_sample,
                            avg_cost=total_cost / valid_sample,
                            avg_token_usage=total_token / valid_sample,
                            eval_cost=eval_grader_cost)

        if verbose:
            self._print_result(result)
        return result

    def eval_async(self, agent: BaseAgent, n_smaple: int, seed: int = 0, *args, **kwargs) -> EvalResult:
        raise NotImplementedError("Async evaluation not supported yet.")
