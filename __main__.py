import os
from pathlib import Path


def create_agent(agent_name: str):
    dir_path = Path(f"./gentpool/pool/{agent_name}")
    if dir_path.exists():
        print(f"Agent {agent_name} already exists.")
        return
    answer = input("Initializing agent ${agent_name} in folder ${dir_path}, continue? (y/n)")
    if answer.lower() != 'y':
        print("Exiting...")
        return
    dir_path.mkdir(parents=True, exist_ok=True)

    with open(dir_path.joinpath("__init__.py"), 'w') as f:
        f.writelines(
            [
                "from .prompt import *\n",
                "from .tools import *\n",
            ]
        )

    with open("./gentpool/pool/__init__.py", 'a') as f:
        f.write(f"from .{agent_name} import *\n")

    with open(dir_path.joinpath("agent.yaml"), 'w') as f:
        f.writelines(
            [
                "### Author: ###\n",
                f"name: {agent_name}\n",
                "type: \n",
                "version: \n",
                "description: \n",
                "target_tasks: \n",
                "prompt_template: \n",
                "llm: \n",
                "plugins: \n",
            ]
        )

    with open(dir_path.joinpath("prompt.py"), 'w') as f:
        f.writelines(
            [
                "### Define your custom prompt here. Check prebuilts in gentopia.prompt :)###\n",
                "from gentopia.prompt import *\n",
                "from gentopia import PromptTemplate\n",
            ]
        )

    with open(dir_path.joinpath("tool.py"), 'w') as f:
        f.writelines(
            [
                "### Define your custom tools here. Check prebuilts in gentopia.tools :)###\n",
                "from gentopia.tools import *\n",
            ]
        )

    print(f"Agent {agent_name} has been initialized.")


def delete_agent(agent_name: str):
    dir_path = Path(f"./gentpool/pool/{agent_name}")
    answer = input(f"Deleting agent ${agent_name} in folder ${dir_path}, this is irreversible, are you sure? (y/n)")
    if answer.lower() != 'y':
        print("Agent deletion cancelled.")
        return
    if dir_path.exists():
        print(f"Deleting agent {agent_name}...")
        dir_path.rmdir()
        with open("./gentpool/pool/__init__.py", 'r') as f:
            lines = f.readlines()
        with open("./gentpool/pool/__init__.py", 'w') as f:
            for line in lines:
                if f"from .{agent_name} import *" not in line:
                    f.write(line)
        print(f"Agent {agent_name} has been deleted.")
    else:
        print(f"Agent {agent_name} does not exist.")