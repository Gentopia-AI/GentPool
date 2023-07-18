# GentPool
The registry pool and benchmark of Agents in Gentopia.

## Installation
Create a new conda env called `gentenv` (choose another name if it sucks). Don't have [conda](https://www.anaconda.com/)? 
```
conda create --name gentenv python=3.9
conda activate gentenv
```
In your workspace, clone [Gentopia](https://github.com/Gentopia-AI/Gentopia) and GentPool.
```
git clone git@github.com:Gentopia-AI/Gentopia.git
git clone git@github.com:Gentopia-AI/GentPool.git
```
Install Gentopia (we will package gentopia after completing this open beta).
```
pip install -e Gentopia
```
Create a `.env` file under GentPool/ (ignored by git) and put your API Keys inside. They will be registered as environmental variables at run time.
```
cd GentPool
touch .env
echo "OPENAI_API_KEY=<your_openai_api_key>" >> .env
echo "WOLFRAM_ALPHA_APPID=<your_wolfram_alpha_api_key>" >> .env
```
.. and so on if you plan to use other service keys. 

Now you are all set! Let's create your first Gentopia Agent.
## Assembling Agents
Find a cool name for your agent and create a template.
```
./create_agent <your_agent_name> 
```
This will initiate an agent template under `./gentpool/pool/<your_agent_name>`. Follow [Agent Config Guide]() to tune your agent!  You can test and play with your agent anytime by 
```
python assemble.py <your_agent_name> --print_agent
```
Ignore the `--print_agent` if you don't want the overview of your agent class. \
Sometimes an agent can upset you. To wipe it out completely,
```
./delete_agent <your_agent_name> 
```

## Agent  Eval and GentBench
A core feature of GentPool is an ALM (Augmented Language Model) benchmark to evaluate agent capability **beyond** plain LLMs (since otherwise why do you pay the [tool tax]())? We carefully choose and curate NLP tasks tailored for ALM eval. We release *public* data for a quick overview and store *filtered* data using Git-LFS [Git-LFS](https://git-lfs.com/). Once you have downloaded and installed LFS following the official repo, you can fetch the benchmark data (from within your local copy of the GentPool repo) with:

```sh
git lfs fetch --all
git lfs pull
```

This will populate all the pointer files under `benchmark/filtered`.

Note that GentBench aims to be hard as hell. For example, a common approach to build GentBench is to run a plain `chatgpt` agent and only keep failed cases. See this [Notebook]() for details.

You can make single eval using `graders`, or you can use `EvalPipeline` to generate a holistic report for your agent. Check out this Quick Start [Notebook](https://github.com/Gentopia-AI/GentPool/blob/main/notebooks/gentpool_eval_quickstart.ipynb) for basic eval utilities.

Eval classes in GentBench by far:
```
Reasoning
 - Math 
 - Coding
 - Planning
 - Commonsense

Knowledge
 - World Knowledge 
 - Domain Specific Knowledge 
 - Web Retrieval (Online update)

Safety
 - Integrity (Jailbreaks, Ethics, ..)
 - Harmlessness (Toxicity, Bias, ..)

Multilingual 
 - Translation
 - Understanding
 
Robustness
 - Consistency 
 - Resilience (Self-correct when presented with false info.)

Memory 
 - Context Length
 - Retrieval Accuracy

Efficiency 
 - Token usage.
 - Run time.
```

## Share your Agents

Ship your agent to the world! Every single step you've made towards agent specialization **exponentially** accelerates the growth of the community. Refer to following checklist:

- Tune your agent towards some specific set of tasks or goals.  
- Eval with public GentBench for some reference (especially on your targeted abilities).
- Create a PR to merge your agent into main branch, 
- After merge, we will initiate a Wiki page for your agent, together with eval scores from both public and private benchmarks.
- Explore what people have been made -- You can easily `!include` others' agents as plugins, grabbing components and rewrite yourself!