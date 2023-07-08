# GentPool
The pool of Agents for Gentopia

## Register an empty new agent under ./pool/
```
./create_agent <your_agent_name> 
```
if it fails for authentication, try `chmod +x create_agent`


## Assemble your agent
After specializing your agent (prompt tuning, tool tuning, SFT on LMs, etc.), simply assemble your agent by 
```
python assemble.py <your_agent_name>
```
