from typing import Dict


def get_instruction_by_task(task: Dict) -> str:
    """
    Given a task in json file, read the keys and determine the instruction for agent input.
    """
    if "problem" in task and "solution" in task:
        return task["problem"]
    if "prompt" in task and "eval_instruction" in task:
        return task["prompt"]
    else:
        raise NotImplementedError("Cannot determine the instruction for the task.")

# def get_grader_from_task(task: Dict) -> BaseGrader:
#     """
#     Given a task in json file, read the keys and determine the grader for the task.
#     """
#     if "problem" in task and "solution" in task:
#         eval_llm = OpenAIGPTClient(model_name="gpt-4")
#         return GateGrader(llm=eval_llm)
#     if "prompt" in task and "eval_instruction" in task:
#         raise NotImplementedError("TODO.")
#     else:
#         raise NotImplementedError("Cannot determine the grader for the task.")
