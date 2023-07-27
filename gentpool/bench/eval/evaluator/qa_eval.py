import json
import os
import random
import time
from typing import Tuple

from gentopia.agent import BaseAgent
from gentopia.llm import OpenAIGPTClient
from gentopia.model import AgentOutput

from gentpool.bench.eval import BaseEval
from gentpool.bench.eval.base_eval import EvalResult
from gentpool.bench.grader import *


class QAEval(BaseEval):
    """
    Evaluation class for QA tasks. 
    Such tasks should have the following keys in the json file:
    - problem: the problem description
    - solution: the solution to the problem
    """
    eval_class: str
    eval_subclass: str
    grader: BaseGrader = GateGrader(llm=OpenAIGPTClient(model_name="gpt-4"))

    def _print_result(self, result: EvalResult):
        output = [
            "\n### FINISHING Agent EVAL ###",
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

    def evaluate(self, agent: BaseAgent, n_smaple: int, seed=0, private=False, verbose=True) -> EvalResult:
        ## Randomly sample 
        random.seed(seed)
        data = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if private:
            file_path = os.path.join(current_dir, f"../../../../benchmark/private/{self.eval_class}/{self.eval_subclass}/")
        else:
            file_path = os.path.join(current_dir, f"../../../../benchmark/public/{self.eval_class}/{self.eval_subclass}/")

        with open(file_path + "full.json", "r") as f:
            tmp = json.load(f)
            data = tmp

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
            #print(f">>> Running Eval {count}/{n_smaple} ...")
            st = time.time()
            agent_instruction = task.get("problem", None)
            try:
                response = agent.run(agent_instruction)
                assert response is not None
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
                # Here we assume math tasks are all problem-solution pairs
                try:
                    grader_output = self.grader.run(task=agent_instruction,
                                                    ground_truth=task["solution"],
                                                    prediciton=response.output)
                    if verbose:
                        print(f"> Grader: {grader_output.output}")
                except Exception as e:
                    grader_output = AgentOutput(output="Grader failed", cost=0, token_usage=0)
                eval_grader_cost += grader_output.cost
                if isinstance(self.grader, GateGrader):
                    total_score += 1 if "pass" in grader_output.output.lower() else 0
                elif isinstance(self.grader, ScoreGrader):
                    try:
                        total_score += int(grader_output.output) / 100
                    except Exception as e:
                        total_score += 0
                else:
                    raise NotImplementedError("Grader type not supported.")

        valid_sample = n_smaple - num_failed

        result = EvalResult(score=0 if not n_smaple else total_score / n_smaple,
                            fail_rate=0 if not n_smaple else num_failed / n_smaple,
                            avg_runtime=0 if not valid_sample else total_runtime / valid_sample,
                            avg_cost=0 if not valid_sample else total_cost / valid_sample,
                            avg_token_usage=0 if not valid_sample else total_token / valid_sample,
                            eval_cost=eval_grader_cost)

        if verbose:
            self._print_result(result)
        return result

    def eval_async(self, agent: BaseAgent, n_smaple: int, seed: int = 0, *args, **kwargs) -> EvalResult:
        raise NotImplementedError("Async evaluation not supported yet.")

    def evaluate_single(self, agent: BaseAgent, index: int, n_smaple: int, seed=0, private=False, verbose=True) \
            -> Tuple["QAEval", int, EvalResult, AgentOutput]:
        ## Randomly sample
        if self.data is None:
            self.data = self._get_data(seed, private, n_smaple)

        total_score, total_cost, total_token, total_runtime, num_failed, eval_grader_cost = [0] * 6
        task = self.data[index]

        st = time.time()
        agent_instruction = task.get("problem", None)
        try:
            response = agent.run(agent_instruction)
            assert response is not None
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


        valid_sample = n_smaple - num_failed

        result = EvalResult(score=0,
                            fail_rate=num_failed,
                            avg_runtime=total_runtime,
                            avg_cost=total_cost,
                            avg_token_usage=total_token,
                            eval_cost=0)

        if verbose:
            self._print_result(result)
        return self, index, result, response

    def grade_single(self, response: AgentOutput, index:int, verbose: bool=False) -> Tuple["QAEval", EvalResult]:
        task = self.data[index]
        eval_grader_cost, total_score = 0, 0
        agent_instruction = task.get("problem", None)
        if response.output != "Agent failed":
            try:
                grader_output = self.grader.run(task=agent_instruction,
                                                ground_truth=task["solution"],
                                                prediciton=response.output)
                if verbose:
                    print(f"> Grader: {grader_output.output}")
            except Exception as e:
                grader_output = AgentOutput(output="Grader failed", cost=0, token_usage=0)
            eval_grader_cost += grader_output.cost
            if isinstance(self.grader, GateGrader):
                total_score += 1 if "pass" in grader_output.output.lower() else 0
            elif isinstance(self.grader, ScoreGrader):
                try:
                    total_score += int(grader_output.output) / 100
                except Exception as e:
                    total_score += 0
            else:
                raise NotImplementedError("Grader type not supported.")

        return self, EvalResult(score=total_score,
                          fail_rate=0,
                          avg_runtime=0,
                          avg_cost=0,
                          avg_token_usage=0,
                          eval_cost=eval_grader_cost)
