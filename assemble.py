import argparse
import os
import signal

import dotenv
from gentopia.assembler.agent_assembler import AgentAssembler
from gentopia.output import enable_log
from gentopia.output.console_output import ConsoleOutput

from pool import *


def ask(agent):
    out = ConsoleOutput()

    def handler(signum, frame):
        out.print("\n[red]Bye!")
        exit(0)

    signal.signal(signal.SIGINT, handler)
    while True:
        out.print("[green]User: ", end="")
        text = input()
        if text:
            response = agent.stream(text, output=out)
        else:
            response = agent.stream(output=out)

        out.done(_all=True)
        print("\n")


def main():
    enable_log(log_level='info')
    dotenv.load_dotenv(".env")

    parser = argparse.ArgumentParser(description='Assemble an agent with given name.')
    parser.add_argument('name', type=str, help='Name of the agent to assemble.')

    args = parser.parse_args()
    agent_name = args.name

    # check if agent_name is under directory ./pool/
    if not os.path.exists(f'./pool/{agent_name}'):
        raise ValueError(f'Agent {agent_name} does not exist. Check GentPool/pool/ for available agents.')

    agent_config_path = f'./pool/{agent_name}/agent.yaml'

    assembler = AgentAssembler(file=agent_config_path)

    # # assembler.manager = LocalLLMManager()
    print(f">>> Assembling agent {agent_name}...")
    agent = assembler.get_agent()

    if agent.name != agent_name:
        raise ValueError(f"Agent name mismatch. Expected {agent_name}, got {agent.name}.")

    ask(agent)


if __name__ == '__main__':
    main()
