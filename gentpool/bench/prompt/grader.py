from gentopia import PromptTemplate


#### Gate Grader ####

TeacherStudentGatePrompt = PromptTemplate(
    input_variables=["task", "ground_truth", "prediction"],
    template="""You are a fair teacher. Given a task and a ground truth solution, you will grade your student's answer with "passed" or "failed".

## Rules
- Judge student answer based on given ground truth (not your own opinion).
- Only return "passed" or "failed" and no extra words.
- Ground truth and student answer might contain different explanations, but you should only judge the correctness of final answers.

## Task
{task}

## Ground Truth
{ground_truth}

## Student Answer
{prediction}

## Your Grade
""")

BatchTeacherStudentGatePrompt = PromptTemplate(
    input_variables=["task", "ground_truth", "prediction"],
    template="""You are a fair teacher. Given a list of tasks and corresponding ground truth solutions, you will grade your student's answers with a list of "passed" or "failed".

## Rules
- Judge student answer based on given ground truth (not your own opinion).
- For each question, only return "passed" or "failed" and no extra words.
- Ground truth and student answer might contain different explanations, but you should only judge the correctness of final answers.

## Tasks
(0). What is the capital of France?
{task}

## Ground Truth
(0). Paris
{ground_truth}

## Student Answers
(0). paris
{prediction}

## Your Grades
(0). passed
""")




#### Score Grader ####


TeacherStudentScorePrompt = PromptTemplate(
    input_variables=["task", "ground_truth", "prediction"],
    template="""You are a fair teacher. Given a task and a ground truth solution, you will grade your student's answer with a score from 0 to 100.

## Rules
- Be absolutely fair and only judge students' answers based on the ground truth solution (not your own opinion).
- Only return the score (from 0 to 100) and no extra words.
- A high score answer should be both correct and well explained (unless the ground truth doesn't contain explanation).

## Task
{task}

## Ground Truth
{ground_truth}

## Student Answer
{prediction}

## Your Grade
""")



#### Compare Grader ####


TeacherStudentDojoPrompt = PromptTemplate(
    input_variables=["task", "ground_truth", "left", "right"],
    template="""You are a fair teacher. Given a task and a ground truth solution, you will compare two answers (left and right) and decide which side is better, or a tie. 

## Rules
- Be absolutely fair and only judge students' answers based on the ground truth solution (not your own opinion).
- Only return "left" or "right" or "tie" with no extra words.
- Compare the correctness of their answers first. If both are correct, compare the quality of their explanations.

## Task
{task}

## Ground Truth
{ground_truth}

## Left Side
{left}

## Right Side
{right}

## Which side wins? (left or right or tie)
""")



#### Instruction Following Grader ####

InstructionFollowingPrompt = PromptTemplate(
    input_variables=["eval_instruction", "agent_message"],
    template="""You are a fair evaluator of AI agents. Your should follow the instruction and grade the agent.

## Your Instruction
{eval_instruction}    

## Agent Response
{agent_message}

## Your Grade
""")
