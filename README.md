# GentPool
The registry pool and benchmark of Agents in Gentopia.

## Install Gentopia
Currently, we haven't registered Gentopia to pip. Fork and clone [Gentopia](https://github.com/Gentopia-AI/Gentopia) and install locally.
```
pip install Gentopia
```

## Register Keys in dotenv
Create a `.env` file in the project directory (this will be ignored by git), and write into the keys you might use:
```
OPENAI_API_KEY=xxxx
WOLFRAM_ALPHA_APPID=xxxx
```

## Register an empty new agent under ./gentpool/pool/
```
./create_agent <your_agent_name> 
```

## Assemble your agent and chat
To test agent by interaction while specializing your agent (prompt tuning, tool tuning, SFT on LMs, etc.), simply assemble your agent and chat by 
```
python assemble.py <your_agent_name>
```

## To Run Eval
Check out this Quick Start [Notebook](https://github.com/Gentopia-AI/GentPool/blob/main/notebooks/gentpool_eval_quickstart.ipynb) for basic eval utilities.

## Delete an agent under ./gentpool/pool/
```
./delete_agent <your_agent_name> 
```
