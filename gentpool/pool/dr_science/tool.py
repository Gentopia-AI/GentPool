### Define your custom tools here. Check prebuilts in gentopia.tools (:###
from gentopia.tools import *


class CustomCodeInterpreter(PythonCodeInterpreter):
    description = ("A tool to execute Python code and retrieve the command line output. Input should be executable Python code."
                   "Note that for the first time to use this tool, run \"conda activate gentenv\" for prebuilt packages.")
