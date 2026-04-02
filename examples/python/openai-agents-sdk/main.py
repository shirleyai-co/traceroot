"""
OpenAI Agents SDK with TraceRoot observability.

Usage:
    cp .env.example .env
    uv run --no-project --python 3.13 --with-requirements requirements.txt python main.py
"""

import asyncio

from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    print("No .env file found (find_dotenv returned None).\nUsing process environment variables.")

# Initialize TraceRoot first — sets the global OTel TracerProvider
import traceroot

traceroot.initialize()

# Then instrument — uses the global provider set by TraceRoot
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

OpenAIAgentsInstrumentor().instrument()

# =============================================================================
# AGENT
# =============================================================================
from agents import Agent, Runner, function_tool


@function_tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    weather_db = {
        "san francisco": "68°F, foggy",
        "tokyo": "72°F, sunny",
        "new york": "45°F, cloudy",
    }
    return weather_db.get(city.lower(), "Unknown city")


@function_tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))
    except Exception:
        return "Error"


agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. Use tools when needed.",
    tools=[get_weather, calculate],
)


async def main():
    result = await Runner.run(
        agent,
        "What's the weather in SF? If the temp goes up 10 degrees, what would it be?",
    )
    print(result.final_output)
    traceroot.flush()


asyncio.run(main())
