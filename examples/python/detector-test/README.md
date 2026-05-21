# Detector Test Examples

Generates traces designed to trigger TraceRoot's detector templates.

## Scenarios

### 1. Failure Detection
- `fetch_user_profile` tool returns a connection error
- `send_notification` tool returns `null` (silent failure)
- **Expected**: "Failure" detector identifies tool errors

### 2. Hallucination Detection
- `get_company_info` tool returns limited data (name, founded, HQ, employees)
- Agent is prompted to provide detailed report including revenue, CEO, products, news
- Agent will fabricate details not present in tool results
- **Expected**: "Hallucination" detector flags claims not grounded in tool output

## Usage

```bash
cp .env.example .env  # fill in API keys + set TRACEROOT_HOST_URL to staging/production
uv run --no-project --python 3.13 --with-requirements requirements.txt python main.py
```

Then in TraceRoot UI:
1. Create a "Failure" detector on the project
2. Create a "Hallucination" detector on the project
3. Run the script again — new traces will be evaluated by detectors
4. Check Detectors page for findings
