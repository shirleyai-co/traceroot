"""
LlamaIndex RAG Pipeline - TraceRoot Observability

A simple in-memory RAG pipeline that:
1. Ingests documents into a VectorStoreIndex
2. Queries the index using a query engine
3. Demonstrates RAG pipeline tracing with TraceRoot

Trace hierarchy (auto-captured by LlamaIndexInstrumentor):
- query
  └── retrieve (vector search)
  └── synthesize (LLM response generation)
      └── llm (OpenAI call)
"""

from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    print("No .env file found. Using process environment variables.")

# =============================================================================
# TRACEROOT SETUP
# Must be initialized BEFORE importing LlamaIndex so instrumentation hooks in
# =============================================================================
import traceroot
from traceroot import observe, using_attributes

traceroot.initialize()

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

LlamaIndexInstrumentor().instrument()

print("[Observability: TraceRoot]")

# =============================================================================
# LLAMAINDEX RAG PIPELINE
# =============================================================================
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.llms.openai import OpenAI

# Set up LLM
Settings.llm = OpenAI(model="gpt-4o-mini")

# Create documents
documents = [
    Document(
        text="TraceRoot is an open-source observability platform for AI agents. It captures traces, debugs with AI, and helps ship with confidence."
    ),
    Document(
        text="TraceRoot supports OpenTelemetry-compatible tracing. You can capture LLM calls, agent actions, and tool usage."
    ),
    Document(
        text="TraceRoot provides agentic debugging with AI-native root cause analysis and GitHub integration. It supports BYOK for any model provider."
    ),
]

# Build index
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()

queries = [
    "What is TraceRoot?",
    "What tracing capabilities does TraceRoot support?",
]


@observe(name="rag_demo", type="agent")
def run_queries():
    for query in queries:
        print(f"\nQuery: {query}")
        response = query_engine.query(query)
        print(f"Answer: {response}")


if __name__ == "__main__":
    with using_attributes(user_id="example-user", session_id="llamaindex-session"):
        run_queries()

    traceroot.flush()
