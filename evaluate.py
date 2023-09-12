import argparse
import os
import time
import dotenv
from gentpool import EvalPipeline
from gentpool.bench.eval.multiprocess_eval_pipe import MultiProcessEvalPipeline

from gentopia.assembler.agent_assembler import AgentAssembler
from gentopia.output import enable_log


def main():
    enable_log(log_level='info')
    dotenv.load_dotenv(".env")

    parser = argparse.ArgumentParser(description='Assemble an agent with given name.')
    parser.add_argument('name', type=str, help='Name of the agent to evaluate.')
    parser.add_argument('--eval_config', type=str, default="./config/eval_config.yaml", help='Path to eval config file.')
    parser.add_argument('--save_dir', type=str, default="./", help='Path to save eval results.')
    parser.add_argument('--mode', type=str, default="parallel", help="EvalPipeline mode. Can be 'parallel' or 'sequential'.")

    args = parser.parse_args()
    agent_name = args.name
    eval_config = args.eval_config
    save_dir = args.save_dir

    # check if agent_name is under directory ./gentpool/pool/
    if not os.path.exists(f'./gentpool/pool/{agent_name}'):
        raise ValueError(f'Agent {agent_name} does not exist. Check ./gentpool/pool/ for available agents.')

    agent_config_path = f'./gentpool/pool/{agent_name}/agent.yaml'

    assembler = AgentAssembler(file=agent_config_path)

    # assembler.manager = LoncalLLMManager()
    # print(f">>> Assembling aget {agent_name}...")
    agent = assembler.get_agent()

    if agent.name != agent_name:
        raise ValueError(f"Agent name mismatch. Expected {agent_name}, got {agent.name}.")

    if args.mode == "parallel":
        evaluator = MultiProcessEvalPipeline(eval_config=eval_config)
        start = time.time()
        _, log = evaluator.run_eval(agent, save_dir=args.save_dir)
        end = time.time()
        print(f"MultiProcessEvalPipeline Complete in {end - start} seconds.")
        evaluator.vis(log, "openai-chat-markdown")
    elif args.mode == "sequential":
        evaluator = EvalPipeline(eval_config=eval_config)
        start = time.time()
        evaluator.run_eval(agent, save_dir=args.save_dir)
        end = time.time()
        print(f"EvalPipeline Complete in {end - start} seconds.")

if __name__ == '__main__':
    main()
