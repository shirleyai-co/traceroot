"""
Detector test examples — generates traces that trigger different detector templates.

Two scenarios:
1. failure_scenario: Tool errors and silent failures → triggers "Failure" detector
2. hallucination_scenario: Agent output contains facts not in tool results → triggers "Hallucination" detector

Usage:
    cp .env.example .env  # fill in your API keys
    uv run --no-project --python 3.13 --with-requirements requirements.txt python main.py
"""

import json
import logging

from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    print("No .env file found. Using process environment variables.")

import traceroot
from traceroot import Integration, observe, using_attributes

traceroot.initialize(integrations=[Integration.OPENAI])

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI()

# ---------------------------------------------------------------------------
# Scenario 1: Failure — tool errors + silent failure
# ---------------------------------------------------------------------------

FAILURE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_user_profile",
            "description": "Fetch user profile from the database.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "Send a notification to a user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["user_id", "message"],
            },
        },
    },
]

FAILURE_TOOL_RESPONSES = {
    "fetch_user_profile": json.dumps(
        {"error": "ConnectionError: database connection timed out after 30s"}
    ),
    "send_notification": json.dumps(None),  # silent failure — returns null
}


@observe(name="failure_scenario", type="agent")
def run_failure_scenario():
    """Agent tries to look up a user and send a notification, but tools fail."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools to help the user."},
        {
            "role": "user",
            "content": "Look up user 'usr_12345' and send them a welcome notification.",
        },
    ]

    for _ in range(3):  # max turns
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=FAILURE_TOOLS,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                result = FAILURE_TOOL_RESPONSES.get(
                    tc.function.name, json.dumps({"error": "unknown tool"})
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
        else:
            print(f"[Failure scenario] Agent: {msg.content}")
            return msg.content

    return "Max turns reached"


# ---------------------------------------------------------------------------
# Scenario 2: Hallucination — agent fabricates facts not in tool results
# ---------------------------------------------------------------------------

HALLUCINATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_company_info",
            "description": "Get company information from the database.",
            "parameters": {
                "type": "object",
                "properties": {"company_name": {"type": "string"}},
                "required": ["company_name"],
            },
        },
    },
]

# Tool returns limited data — agent will likely fill gaps with fabricated info
HALLUCINATION_TOOL_RESPONSES = {
    "get_company_info": json.dumps(
        {
            "name": "Acme Corp",
            "founded": 2015,
            "headquarters": "San Francisco, CA",
            "employees": 150,
        }
    ),
}


@observe(name="hallucination_scenario", type="agent")
def run_hallucination_scenario():
    """Agent is asked for detailed info but tool returns limited data.
    The system prompt encourages the agent to provide comprehensive info,
    which may cause it to fabricate details not in the tool results."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a company research assistant. Always provide comprehensive, "
                "detailed reports. Include revenue, growth rate, key products, CEO name, "
                "and recent news even if you need to fill in details from your knowledge. "
                "Never say you don't have information — always provide a complete answer."
            ),
        },
        {
            "role": "user",
            "content": "Give me a detailed report on Acme Corp including their revenue, CEO, key products, and recent news.",
        },
    ]

    for _ in range(3):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=HALLUCINATION_TOOLS,
        )
        msg = response.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                result = HALLUCINATION_TOOL_RESPONSES.get(tc.function.name, json.dumps({}))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
        else:
            print(f"[Hallucination scenario] Agent: {msg.content[:200]}...")
            return msg.content

    return "Max turns reached"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Detector Test Examples")
    print("=" * 60)

    with using_attributes(user_id="detector-test-user", session_id="detector-test-session"):
        print("\n--- Scenario 1: Failure (tool error + silent failure) ---")
        run_failure_scenario()

        print("\n--- Scenario 2: Hallucination (fabricated facts) ---")
        run_hallucination_scenario()

    traceroot.flush()
    print("\n[Traces exported — create detectors in TraceRoot UI to see findings]")
