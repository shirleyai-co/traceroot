"""
Claude Agent SDK example with TraceRoot observability.

Uses Claude Code as a library — Claude autonomously reads files,
runs commands, and searches the web using built-in tools.

Usage:
    cp .env.example .env  # fill in your API keys
    uv run --no-project --python 3.13 --with-requirements requirements.txt python main.py
"""

import asyncio

from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    print("No .env file found. Using process environment variables.")

import traceroot
from traceroot import observe

traceroot.initialize()

from claude_agent_sdk import query, ClaudeAgentOptions


@observe(name="claude_agent_query", type="agent")
async def run_query(prompt: str) -> str:
    """Run a Claude Agent SDK query with tracing."""
    result_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Glob", "Grep", "Bash"],
            max_turns=5,
        ),
    ):
        if hasattr(message, "result"):
            result_text = message.result
        elif hasattr(message, "content"):
            # Stream assistant content
            if isinstance(message.content, str):
                print(message.content, end="", flush=True)
    return result_text


@observe(name="demo_session", type="agent")
async def run_demo():
    queries = [
        "What Python files are in the current directory? List them briefly.",
        "What is 2 + 2 * 3? Use the Bash tool to calculate it.",
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n{'=' * 60}")
        print(f"Query {i}: {q}")
        print("=" * 60)
        result = await run_query(q)
        if result:
            print(f"\nResult: {result}")
        print()


if __name__ == "__main__":
    asyncio.run(run_demo())
    traceroot.flush()
