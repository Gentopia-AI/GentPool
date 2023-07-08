# GentPool
The pool of Agents for Gentopia

## Install Gentopia
```
pip install -e Gentopia
```

## Register an empty new agent under ./pool/
```
./create_agent <your_agent_name> 
```

## Assemble your agent
After specializing your agent (prompt tuning, tool tuning, SFT on LMs, etc.), simply assemble your agent by 
```
python assemble.py <your_agent_name>
```
