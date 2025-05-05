# Evaluating Long-Term Ethical Decision-Making Capabilities of Current Thinking Models in Simulated Scenarios
This repo contains simulations of complex moral and ethical decision making under uncertainty and ambiguity. 
It tests the ability of LLMs to make choices in these situations and compares reasoning models to non-reasoning counterparts.
Each scenario is constructed as a branching decision tree with up to three levels.

## Reproducing
First set up an API keys file (api_keys.json) in the root directory like follows:

```
{
    "OPENAI_API_KEY": "...",
    "ANTHROPIC_API_KEY": "...",
    "DEEPINFRA_API_KEY": "...",
    "GEMINI_API_KEY": "..."
}
```
In theory, you can run any models supported by (LiteLLM)[https://docs.litellm.ai/docs/providers] as long as you have the correct API key set up.
The script will set environment variables equal to the values set in that json file.

Run with `uv run main.py scenarios/${scenario}.yaml --model ${model} -n 30` or use `run_all.sh`.