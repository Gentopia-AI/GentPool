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
    score: float = 0 # score of the evaluation from 0 to 1
    fail_rate: float = 0 # fail rate of the agent from 0 to 1
    avg_runtime: float = 0 # avg runtime of the evaluation per task
    avg_cost: float = 0 # avg cost of the evaluation per task
    avg_token_usage: float = 0 # avg token usage of the evaluation per task
    eval_cost: float = 0 # total cost introducted by evaluation (such as grader cost)

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

    def avg(self, num: int):
        return EvalResult(
            score=self.score / num,
            fail_rate=self.fail_rate / num,
            avg_runtime=self.avg_runtime / num,
            avg_cost=self.avg_cost / num,
            avg_token_usage=self.avg_token_usage / num,
            eval_cost=self.eval_cost
        )


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


    def _get_data(self, seed: int, private: bool, n_smaple: int, code_eval: bool = False) -> List[Dict]:
        random.seed(seed)
        data = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if private:
            file_path = os.path.join(current_dir,
                                     f"../../../benchmark/private/{self.eval_class}/{self.eval_subclass}/")
        else:
            file_path = os.path.join(current_dir,
                                     f"../../../benchmark/public/{self.eval_class}/{self.eval_subclass}/")
            
        types = ['apps', 'humaneval', 'mbpp']
        if code_eval:
            for type in types:
                with open(file_path + f"full_{type}.json", "r") as f:
                    tmp = json.load(f)
                    data.extend(tmp)
        else:
            with open(file_path + "full.json", "r") as f:
                tmp = json.load(f)
                data = tmp

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
