# GentPool
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Read the Docs](https://img.shields.io/readthedocs/gentopia)](https://gentopia.readthedocs.io/en/latest/gentpool.html)
[![Static Badge](https://img.shields.io/badge/Gentopia-873503)](https://github.com/Gentopia-AI/Gentopia)
[![Open Issues](https://img.shields.io/github/issues-raw/Gentopia-AI/GentPool)](https://github.com/Gentopia-AI/GentPool/issues)
[![Twitter Follow](https://img.shields.io/twitter/follow/GentopiaAI)](https://twitter.com/GentopiaAI)
[![](https://dcbadge.vercel.app/api/server/ASPP9MY9QK?compact=true&style=flat)](https://discord.gg/ASPP9MY9QK)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/views/UC9QCjcsHJVKjKZ2Zmrq83vA)](https://www.youtube.com/channel/UC9QCjcsHJVKjKZ2Zmrq83vA)
[![GitHub star chart](https://img.shields.io/github/stars/Gentopia-AI/GentPool?style=social)](https://star-history.com/Gentopia-AI/GentPool)

GentPool is the companion platform of [Gentopia](https://github.com/Gentopia-AI/Gentopia), where people share specialized agents, clone, customize or build upon each other, and run agent evaluation with [GentBench](https://gentopia.readthedocs.io/en/latest/gentpool.html#agent-evaluation).

## Installation 💻
Check the full installation guide [here](https://gentopia.readthedocs.io/en/latest/installation.html).
```
conda create --name gentenv python=3.10
conda activate gentenv
pip install gentopia
```
Clone and create a `.env` file under GentPool/ (ignored by git) and put your API Keys inside. They will be registered as environmental variables at run time.
```
git clone git@github.com:Gentopia-AI/GentPool.git
cd GentPool
touch .env
echo "OPENAI_API_KEY=<your_openai_api_key>" >> .env
echo "WOLFRAM_ALPHA_APPID=<your_wolfram_alpha_api_key>" >> .env
```
.. and so on if you plan to use other service keys. 

Now you are all set! Let's create your first Gentopia Agent.
## Quick Start ☘️
Find a cool name for your agent and create a template.
```
./create_agent <your_agent_name> 
```
You can start by cloning others' shared agents.
```
./clone_agent elon <your_agent_name> 
```
Both commands will initiate an agent template under `./gentpool/pool/<your_agent_name>`. Follow this [document](https://gentopia.readthedocs.io/en/latest/quick_start.html) to tune your agent, or check out our demo [tutorials](https://www.youtube.com/channel/UC9QCjcsHJVKjKZ2Zmrq83vA).  You can test and chat with your agent by 
```
python assemble.py <your_agent_name> --print_agent
```
`--print_agent` is optional and gives you an overview of your agent class. \
Sometimes an agent can upset you. To wipe it out completely,
```
./delete_agent <your_agent_name> 
```

## Agent Eval with GentBench 🥇
See [here](https://gentopia.readthedocs.io/en/latest/gentpool.html#agent-evaluation) to check more about Gentopia's unique agent evaluation benchmark. GentBench is released half *public* and half *private*. Check `GentPool/benchmark/` for samples. 
To download the full *public* benchmark, 
```sh
git lfs fetch --all
git lfs pull
```
This will populate all the pointer files under `benchmark/public`. We keep a *private* part of this benchmark to test the generalizability of agents on unseen tasks. This eval will be triggered when you share and publish your agent to GentPool.

Note that GentBench is hard as hell.👻 As of July 2023, OpenAI `gpt-3.5-turbo` LLM could pass less than 10% of the tasks. We mean to test agent ability **beyond** pure LLMs, which usually rely on powerful plugins, and how capable your agent is to tame the horse.

To run eval in parallel, config the number of tasks of each class in `GentPool/config/eval_config.yaml`, and run with 
```
python evaluate.py my_agent
```
Check [here](https://gentopia.readthedocs.io/en/latest/gentpool.html#running-eval) to see more details, including how to use `graders` (a special type of agent) to grade on your own tasks.


## Share your Agents 🌎

Ship your agent to the world! Every single step you've made towards agent specialization **exponentially** accelerates the growth of the community. Refer to the following checklist:

- Tune your agent towards some specific set of tasks or goals.  
- Eval with public GentBench for some reference (especially on your targeted abilities).
- Create a PR to merge your agent into `main` branch, 
- After merge, we will initiate a Wiki page for your agent, together with eval scores from both public and private benchmarks.
