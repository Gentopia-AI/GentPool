import multiprocessing
import random
import time
from typing import Tuple, Dict, List

from gentopia.agent import BaseAgent
from gentopia.model import AgentOutput
from gentopia.output.base_output import BaseOutput

from gentpool.bench.grader import BaseGrader
from gentpool.bench.eval import BaseEval
from gentpool.bench.eval.base_eval import EvalResult
from gentpool.bench.prompt.code_eval import APPSPrompt, HumanEvalPrompt, MBPPPrompt
from .utils import *
import os


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


    def _get_agent_instruction(self, dataset: str, problem: str) -> str:
        if dataset == "apps":
            return APPSPrompt.format(problem=problem)
        elif dataset == "humaneval":
            return HumanEvalPrompt.format(problem=problem)
        elif dataset == "mbpp":
            return MBPPPrompt.format(problem=problem)
        else:
            raise NotImplementedError(f"Dataset {dataset} not supported yet.")

    def _get_output(self, response: AgentOutput, dataset: str, task: Dict) -> str:
        test_case = task["test_case"]
        if dataset == "apps":
            return convert_apps_code(response.output, test_case)
        elif dataset == "humaneval":
            return response.output + "\n" + test_case
        elif dataset == "mbpp":
            return response.output + "\n" + test_case
        else:
            raise NotImplementedError(f"Dataset {dataset} not supported yet.")

    def evaluate(self, agent: BaseAgent, n_smaple: int, seed=0, private=False, verbose=True,
                 time_limit=5) -> Tuple[EvalResult, List[Tuple[List[str], Dict]]]:
        Log = []
        result = EvalResult()
        for index in range(n_smaple):
            _, index, eval_result, response, eval_log = self.evaluate_single(agent, 0, n_smaple, seed, private, time_limit)
            _, grade_result, grade_log = self.grade_single(response, index, time_limit)
            result += eval_result + grade_result
            Log.append((eval_log, grade_log))

        return result.avg(n_smaple), Log


    def eval_async(self, agent: BaseAgent, n_smaple: int, seed: int = 0, *args, **kwargs) -> EvalResult:
        raise NotImplementedError("Async evaluation not supported yet.")

    def evaluate_single(self, agent: BaseAgent, index: int, n_smaple: int, seed=0, private=False, time_limit=5) -> \
            Tuple["CodeEval", int, EvalResult, AgentOutput, List[str]]:

        if self.data is None:
            self.data = self._get_data(seed, private, n_smaple, True)

        result = EvalResult(avg_runtime=time.time())
        task = self.data[index]
        opt = BaseOutput()

        problem, dataset = task.get("problem", None), task.get("dataset", None)
        agent_instruction = self._get_agent_instruction(dataset, problem)

        agent_log = []
        try:
            response = agent.run(agent_instruction, opt)
            if hasattr(agent, "message_scratchpad"):
                agent_log = agent.message_scratchpad[-1]
            assert response is not None
        except Exception as e:
            result.fail_rate = 1
            response = AgentOutput(output="Agent failed", cost=0, token_usage=0)
        result.avg_runtime = time.time() - result.avg_runtime
        result.avg_cost = response.cost
        result.avg_token_usage = response.token_usage

        return self, index, result, response, agent_log + opt.log

    def grade_single(self, response: AgentOutput, index:int, time_limit: float = 5) -> Tuple["CodeEval", EvalResult, Dict]:
        task = self.data[index]
        result = EvalResult()
        dataset = task.get("dataset", None)
        code = "Agent failed"
        output = "Skip"
        if response.output != "Agent failed":
            code = self._get_output(response, dataset, task)
            output = check_correctness(code, time_limit)
            if "pass" in output.lower():
                result.score = 1
        return self, result, dict(prompt=code, output=output, solution='pass')
