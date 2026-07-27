# Learning Walkthrough

This project is intentionally small enough to understand one concept at a time.

## Study order

1. **`WeddingContext` and `WeddingState`** — fixed context versus changing state.
2. **`authenticate`** — `ToolRuntime` reads context and returns a `Command` that updates state.
3. **`save_wedding_details`** — another `ToolRuntime` state update.
4. **`trim_old_tool_messages`** — node-style middleware receives `state + Runtime`.
5. **`wedding_prompt`, `dynamic_tools`, and `choose_model`** — wrap-style middleware receives `ModelRequest`.
6. **`load_kiwi_flight_tools`** — connect to the external Kiwi MCP server.
7. **Flight, venue, and music agents** — build three focused specialists.
8. **`ask_*_specialist` tools** — expose each specialist agent as a tool to the coordinator.
9. **`HumanInTheLoopMiddleware`** — pause the email action before execution.
10. **`analyze_inspiration_image`** — send a text + Base64 image message.
11. **`demo.py`** — follow the complete runtime flow.

## Why these three specialist agents?

The foundation course's destination-wedding example uses three clear responsibilities:

```text
Flight agent → external travel MCP tools
Venue agent  → web search
Music agent  → SQL music database
```

The coordinator does not perform those jobs itself. It decides which specialist to call and combines their answers.

## Runtime flow

```text
Planner message
   ↓
Node middleware removes older tool-message clutter
   ↓
Wrap middleware selects tools, prompt, and model
   ↓
Coordinator model decides what to do
   ↓
ToolRuntime tools read context or update state
   ↓
Coordinator calls flight, venue, and music specialists
   ↓
Specialists use Kiwi MCP, Tavily, or SQLite
   ↓
Coordinator combines the destination-wedding plan
   ↓
Coordinator drafts the client email
   ↓
Human-in-the-loop interrupt
   ↓
Planner approves/rejects/edits
   ↓
Dummy email tool executes
```

## The three runtime objects

### `ModelRequest`

Used in wrap-style middleware because the middleware is changing the model call itself.

```python
request.override(tools=tools)
request.override(model=model)
```

In this project it controls:

- tools before versus after login;
- login prompt versus wedding-coordinator prompt;
- standard model versus stronger model.

### `Runtime`

Used with state in node-style middleware.

```python
@before_agent
def trim_old_tool_messages(state: WeddingState, runtime: Runtime):
    ...
```

The example mainly modifies `state`, but the runtime object is available for run-level context and services.

### `ToolRuntime`

Injected inside tools. The model does not supply it.

```python
@tool
def authenticate(email: str, password: str, runtime: ToolRuntime):
    ...
```

It lets tools:

- read planner credentials from context;
- read or update wedding state;
- obtain the current tool-call ID;
- read the client's email safely.

## Kiwi MCP connection

```python
kiwi_client = MultiServerMCPClient(
    {
        "kiwi_flights": {
            "url": KIWI_MCP_URL,
            "transport": "http",
        }
    }
)

kiwi_flight_tools = await kiwi_client.get_tools()
```

The loaded MCP tools are passed only to the flight specialist. The coordinator sees one simple tool named `ask_flight_specialist`, so it does not need to choose among every low-level Kiwi tool itself.

That is the main benefit of the multi-agent structure:

```text
Coordinator chooses the specialist.
Specialist chooses its domain tools.
```

## Graceful flight fallback

Remote servers can fail. The project catches a Kiwi connection error and supplies `flight_search_unavailable` instead.

This fallback does **not** create fake flight options. It tells the coordinator that live flight data is unavailable and explains what to verify.

## Email approval flow

Only this tool is sensitive:

```python
send_client_email
```

Therefore only that tool has:

```python
"send_client_email": True
```

When the model calls it, LangGraph pauses and returns an interrupt. The application resumes with an approve, reject, or edit decision using the same thread ID.

## Suggested first debugging exercise

Temporarily add prints inside these functions:

```python
trim_old_tool_messages
dynamic_tools
choose_model
authenticate
ask_flight_specialist
```

Then run `demo.py` and watch which layer executes at each point.
