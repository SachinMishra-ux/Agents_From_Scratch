# Pure Python ReAct Agent From Scratch

This directory contains a lightweight, framework-free implementation of a **ReAct (Reasoning and Acting)** pattern AI agent using pure Python and the official Google GenAI SDK.

## Design Architecture

The agent is designed entirely from scratch without using heavy packages like LangChain or LangGraph. It runs in a feedback loop (Thought -> Action -> Observation -> Thought -> Final Answer) and makes autonomous decisions on which tools to call depending on the user's query.

### Prompt Pattern
The agent uses a structured system prompt directing it to output thoughts and tools in the following format:
```text
Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, should be one of [get_github_repo_info, get_github_repo_prs, get_weather]
Action Input: the input to the action (do NOT quote the input, pass it as raw text)
Observation: the result of the action
...
Thought: I now know the final answer
Final Answer: the final answer to the original input question
```

### Flow execution
1. The question is formatted into the prompt template.
2. The agent calls Gemini (`gemini-2.5-flash`) with the prompt, utilizing `stop_sequences=["Observation:"]`. This halts LLM text generation the instant the model decides on an action.
3. The parser extracts the Action and Action Input using regex.
4. The local Python function matching the Action is run, returning results.
5. The result is appended to the prompt under `Observation:` and fed back into the agentic loop.
6. Once the model outputs `Final Answer:`, the loop terminates and outputs the answer to the user.

---

## Pre-requisites & Keys

- **Gemini API Key**: Requires a Google Gemini API Key. Ensure either `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set in your environment or in the `.env` file.
- **GitHub Token (Optional but Recommended)**: GitHub's API rate limits anonymous requests to 60 per hour. To avoid rate limits, set `GITHUB_TOKEN` in your environment or `.env` file. You can create a classic Personal Access Token (PAT) with read-only permissions on GitHub.

---

## How to Run

### Command-Line Direct Query
You can run the agent directly with a query from your terminal:
```bash
python agents/main.py "Compare the weather in Tokyo with the stars of google/jax"
```

### Interactive REPL Mode
You can enter interactive REPL mode by running the script without arguments:
```bash
python agents/main.py
```
You can type questions at the prompt and exit by typing `exit` or `quit`.
