import argparse
import os

import dotenv
from gentopia import chat
from gentopia.assembler.agent_assembler import AgentAssembler
from gentopia.output import enable_log


def main():
    enable_log(log_level='info')
    dotenv.load_dotenv(".env")

    parser = argparse.ArgumentParser(description='Assemble an agent with given name.')
    parser.add_argument('name', type=str, help='Name of the agent to assemble.')

    args = parser.parse_args()
    agent_name = args.name

    # check if agent_name is under directory ./gentpool/pool/
    if not os.path.exists(f'./gentpool/pool/{agent_name}'):
        raise ValueError(f'Agent {agent_name} does not exist. Check ./gentpool/pool/ for available agents.')

    agent_config_path = f'./gentpool/pool/{agent_name}/agent.yaml'

    assembler = AgentAssembler(file=agent_config_path)

    # # assembler.manager = LocalLLMManager()
    print(f">>> Assembling agent {agent_name}...")
    agent = assembler.get_agent()

    if agent.name != agent_name:
        raise ValueError(f"Agent name mismatch. Expected {agent_name}, got {agent.name}.")

    chat(agent)


if __name__ == '__main__':
    main()
