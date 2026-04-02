# Claude Agent SDK

Agent using Claude's tool use capabilities, instrumented with [TraceRoot](https://traceroot.ai).

## Setup
```bash
cp .env.example .env  # fill in your API keys
```

With `uv` (recommended):
```bash
uv run --no-project --python 3.13 --with-requirements requirements.txt python main.py
```

## What it does
Runs demo queries with a Claude-powered agent using tool use.
Tools: `get_weather`, `get_stock_price`, `calculate`

## Notes

This example uses the `anthropic` SDK directly with a Claude Agent SDK style
abstraction. It will be updated when the official `claude-agent-sdk` package
is released. TraceRoot auto-instruments all Anthropic LLM calls via
`Integration.ANTHROPIC`; `@observe` decorators capture the agent structure.
