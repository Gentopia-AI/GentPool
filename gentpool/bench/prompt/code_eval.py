from gentopia import PromptTemplate

#### APPS ####

APPSPrompt = PromptTemplate(
    input_variables=["problem"],
    template="""# Language: Python 3
# Task: Synthesize program

\"\"\"
Contains programming exercises for single functions specified by their doc-strings and with solutions in simple code and with a lot of comments that explain what is done and why and how it is related to the specification. The solutions of all examples have a similar structure and are written in a similar style.
\"\"\"

# Example 1.

\"\"\"
Given an array of integers, find the sum of all the positive integers in the array.
Your function should return the sum of all the positive integers present in the array.


-----Input-----

The first line contains a list of integers.


-----Output-----

Output the sum of all positive integers.


-----Examples-----
Input
1 2 3 -1

Output
6

Input
-1 -2 -3 -4

Output
0

Input
1 1 1 3 3 4 3 2 4 2

Output
24

\"\"\"

# Solution:
ls = list(map(int, input().split()))
result = sum([i for i in ls if i > 0])
print(result)

# Example 2.

\"\"\"

{problem}

\"\"\"

# Solution:

""")


#### Human Eval ####

HumanEvalPrompt = PromptTemplate(
    input_variables=["problem"],
    template="""# Language: Python 3
# Task: Synthesize program

\"\"\"
Contains programming exercises for single functions specified by their doc-strings and with solutions in simple code and with a lot of comments that explain what is done and why and how it is related to the specification. The solutions of all examples have a similar structure and are written in a similar style.
\"\"\"

# Example 1.

def sum_positive(ls):

    \"\"\"
    Given an array of integers, find the sum of all the positive integers in the array.
    Your function should return the sum of all the positive integers present in the array.
    Examples:
    For ls = [1, 2, 3, -1], the output should be 6.
    For ls = [-1, -2, -3, -4], the output should be 0.
    For ls = [1, 1, 1, 3, 3, 4, 3, 2, 4, 2], the output should be 24.
    \"\"\"

# Solution:
def sum_positive(ls):
    result = sum([i for i in ls if i > 0])
    return result

# Example 2.

{problem}

# Solution:

""")


#### MBPP ####

MBPPPrompt = PromptTemplate(
    input_variables=["problem"],
    template="""# Language: Python 3
# Task: Synthesize program

\"\"\"
Contains programming exercises for single functions specified by their doc-strings and with solutions in simple code and with a lot of comments that explain what is done and why and how it is related to the specification. The solutions of all examples have a similar structure and are written in a similar style.
\"\"\"

# Example 1.

\"\"\"

Given an array of integers, find the sum of all the positive integers in the array.
Your function should return the sum of all the positive integers present in the array.

Your code should pass these tests:

assert sum_positive([1, 2, 3, -1]) == 6
assert sum_positive([-1, -2, -3, -4]) == 0
assert sum_positive([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]) == 24

\"\"\"

# Solution:
def sum_positive(ls):
    result = sum([i for i in ls if i > 0])
    return result

# Example 2.

\"\"\"

{problem}

\"\"\"

# Solution:

""")