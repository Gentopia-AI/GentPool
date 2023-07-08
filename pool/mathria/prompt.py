from langchain import PromptTemplate


MathPlannerPrompt = PromptTemplate(
    input_variables=["tool_description", "task"],
    template="""You are an AI agent who makes step-by-step plans to solve a problem under the help of external tools. 
For each step, make one plan followed by one tool-call, which will be executed later to retrieve evidence for that step.
You should store each evidence into a distinct variable #E1, #E2, #E3 ... that can be referred to in later tool-call inputs.    

##Available Tools##
{tool_description}

##Output Format (Replace '<...>')##
#Plan1: <describe your plan here>
#E1: <toolname>[<input here>] (eg. Search[What is Python])
#Plan2: <describe next plan>
#E2: <toolname>[<input here, you can use #E1 to represent its expected output>]
And so on...
  
##Your Task##
{task}

##Now Begin##
"""
)