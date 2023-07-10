from pydantic import BaseModel
from gentopia.agent import BaseAgent
from bench.grader import BaseGrader
from abc import ABC, abstractmethod
from typing import Dict
import yaml


class EvalResult(BaseModel):
    score: float  # score of the evaluation from 0 to 1
    fail_rate: float  # fail rate of the agent from 0 to 1
    avg_runtime: float  # avg runtime of the evaluation per task
    avg_cost: float  # avg cost of the evaluation per task
    avg_token_usage: float  # avg token usage of the evaluation per task 
    eval_cost: float # total cost introducted by evaluation (such as grader cost)
    

class EvalPipelineResult(BaseModel):
    eval_results: Dict[str, EvalResult]  # eval results for each eval task
    avg_score: float  # weighted average (by count) of scores
    avg_fail_rate: float  # weighted average (by count) of fail rates
    avg_runtime: float  # weighted average (by count) of runtimes
    avg_cost: float  # weighted average (by count) of costs
    avg_token_usage: float  # weighted average (by count) of token usages
    total_eval_cost: float # total cost introducted by evaluation (such as grader cost)





class BaseEval(ABC, BaseModel):
    eval_class: str
    eval_subclass: str
    grader: BaseGrader
    

    @abstractmethod
    def evaluate(self, agent: BaseAgent, n_smaple: int, seed: int = 0, *args, **kwargs) -> EvalResult:
        pass


    @abstractmethod
    def eval_async(self, agent: BaseAgent, n_smaple: int, seed: int = 0, *args, **kwargs) -> EvalResult:
        pass
    


class BaseEvalPipeline(ABC, BaseModel):
    eval_config: Dict 
    grader_llm: str

    @abstractmethod
    def run_eval(self, agent: BaseAgent, seed: int = 0, *args, **kwargs) -> EvalResult:
        pass

    @abstractmethod
    def run_eval_async(self, agent: BaseAgent, seed: int = 0, *args, **kwargs) -> EvalResult:
        pass
    

