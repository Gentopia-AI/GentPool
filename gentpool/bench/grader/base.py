from abc import ABC

from gentopia.agent.base_agent import BaseAgent


class BaseGrader(BaseAgent, ABC):
    """
    Base class for graders.
    Graders are special type of agents that are used to grade/eval the outputs of other agents.
    A "gate" Grader decide if a given task is "passed" or "failed".
    A "score" Grader instead gives continuous score from 0 to 100.
    A "dojo" Grader takes two agents in an arena and decide which one "win" or "lose".
    A “instructed” Grader takes an explicit instruction and correspondingly gives an output.
    """

    grader_type: str  # gate, score, dojo, instructed







