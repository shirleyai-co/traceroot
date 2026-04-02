"""
CrewAI Multi-Agent Crew, instrumented with TraceRoot observability.

A simple 2-agent crew that:
1. Uses a Researcher agent to gather weather data using tools
2. Uses a Writer agent to produce a comparison report

Usage:
    cp .env.example .env  # fill in your API keys
    uv run --no-project --python 3.13 --with-requirements requirements.txt python main.py
"""

from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    print("No .env file found (find_dotenv returned None).\nUsing process environment variables.")

# =============================================================================
# TRACEROOT SETUP
# Must be initialized BEFORE importing CrewAI so instrumentation hooks in
# =============================================================================
import traceroot
from traceroot import observe

traceroot.initialize()

from openinference.instrumentation.crewai import CrewAIInstrumentor

CrewAIInstrumentor().instrument()

# =============================================================================
# CREWAI AGENTS
# =============================================================================
from crewai import Agent, Crew, Task
from crewai.tools import tool


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    weather_db = {
        "san francisco": "68°F, foggy",
        "tokyo": "72°F, sunny",
        "new york": "45°F, cloudy",
    }
    return weather_db.get(city.lower(), "Unknown city")


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))
    except:
        return "Error"


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

researcher = Agent(
    role="Research Analyst",
    goal="Gather and analyze information using available tools",
    backstory="Expert researcher who uses tools to find accurate data.",
    tools=[get_weather, calculate],
    verbose=True,
)

writer = Agent(
    role="Report Writer",
    goal="Write clear, concise reports based on research findings",
    backstory="Skilled writer who creates readable summaries.",
    verbose=True,
)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

research_task = Task(
    description="What's the weather in San Francisco and Tokyo? Compare them and calculate the temperature difference.",
    expected_output="Weather data with comparison and calculations",
    agent=researcher,
)

report_task = Task(
    description="Write a brief weather comparison report based on the research.",
    expected_output="A short weather comparison report",
    agent=writer,
)


# ---------------------------------------------------------------------------
# Crew
# ---------------------------------------------------------------------------

crew = Crew(agents=[researcher, writer], tasks=[research_task, report_task], verbose=True)

result = crew.kickoff()
print(result)

traceroot.flush()
