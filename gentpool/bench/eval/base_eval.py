import json
import os
from abc import ABC, abstractmethod
import random
from typing import Dict, List, Optional, Tuple

from gentopia.agent import BaseAgent
from gentopia.model import AgentOutput
from pydantic import BaseModel

from gentpool.bench.grader import BaseGrader


class EvalResult(BaseModel):
    score: float  # score of the evaluation from 0 to 1
    fail_rate: float  # fail rate of the agent from 0 to 1
    avg_runtime: float  # avg runtime of the evaluation per task
    avg_cost: float  # avg cost of the evaluation per task
    avg_token_usage: float  # avg token usage of the evaluation per task 
    eval_cost: float  # total cost introducted by evaluation (such as grader cost)

    def __add__(self, other):
        return EvalResult(
            score=self.score + other.score,
            fail_rate=self.fail_rate + other.fail_rate,
            avg_runtime=self.avg_runtime + other.avg_runtime,
            avg_cost=self.avg_cost + other.avg_cost,
            avg_token_usage=self.avg_token_usage + other.avg_token_usage,
            eval_cost=self.eval_cost + other.eval_cost
        )

    def __iadd__(self, other):
        self.score += other.score
        self.fail_rate += other.fail_rate
        self.avg_runtime += other.avg_runtime
        self.avg_cost += other.avg_cost
        self.avg_token_usage += other.avg_token_usage
        self.eval_cost += other.eval_cost
        return self


class EvalPipelineResult(BaseModel):
    eval_results: Dict[str, EvalResult]  # eval results for each eval task
    avg_score: float  # weighted average (by count) of scores
    avg_fail_rate: float  # weighted average (by count) of fail rates
    avg_runtime: float  # weighted average (by count) of runtimes
    avg_cost: float  # weighted average (by count) of costs
    avg_token_usage: float  # weighted average (by count) of token usages
    total_eval_cost: float  # total cost introducted by evaluation (such as grader cost)


class BaseEval(ABC, BaseModel):
    eval_class: str
    eval_subclass: str
    grader: BaseGrader
    data: Optional[List[Dict]] = None

    @abstractmethod
    def evaluate(self, agent: BaseAgent, n_smaple: int, seed: int = 0, *args, **kwargs) -> EvalResult:
        pass

    @abstractmethod
    def eval_async(self, agent: BaseAgent, n_smaple: int, seed: int = 0, *args, **kwargs) -> EvalResult:
        pass

    @abstractmethod
    def evaluate_single(self, agent: BaseAgent, index: int, n_smaple: int, seed: int = 0, *args, **kwargs) -> Tuple["BaseEval", int, EvalResult, AgentOutput]:
        pass

    @abstractmethod
    def grade_single(self, response: AgentOutput, index: int, *args, **kwargs) -> Tuple["BaseEval", EvalResult]:
        pass


    def _get_data(self, seed: int, private: bool, n_smaple: int) -> List[Dict]:
        random.seed(seed)
        data = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if private:
            file_path = os.path.join(current_dir,
                                     f"../../../benchmark/private/{self.eval_class}/{self.eval_subclass}/")
        else:
            file_path = os.path.join(current_dir,
                                     f"../../../benchmark/public/{self.eval_class}/{self.eval_subclass}/")

        for file_name in os.listdir(file_path):
            if file_name.endswith(".json"):
                with open(file_path + file_name, "r") as f:
                    tmp = json.load(f)
                    data += [tmp]

        random.shuffle(data)
        return data[:n_smaple]


class BaseEvalPipeline(ABC, BaseModel):
    eval_config: Dict
    grader_llm: str

    @abstractmethod
    def run_eval(self, agent: BaseAgent, seed: int = 0, *args, **kwargs) -> EvalResult:
        pass

    @abstractmethod
    def run_eval_async(self, agent: BaseAgent, seed: int = 0, *args, **kwargs) -> EvalResult:
        pass
