"""Simple multi-agent Destination Wedding Planner built with LangChain.

 Concepts demonstrated:
- Multimodal image messages
- External Kiwi flight-search MCP server
- Multi-agent supervisor pattern
- State, context, Runtime, and ToolRuntime
- Node-style middleware
- Wrap-style middleware and dynamic agents
- Human-in-the-loop approval before sending a client email
"""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRequest,
    ModelResponse,
    before_agent,
    dynamic_prompt,
    wrap_model_call,
)
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, RemoveMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_community.utilities import SQLDatabase
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command
from tavily import TavilyClient

load_dotenv()

ROOT = Path(__file__).resolve().parent
STANDARD_MODEL_NAME = os.getenv("STANDARD_MODEL", "gpt-5-nano")
LARGE_MODEL_NAME = os.getenv("LARGE_MODEL", "gpt-5-mini")
KIWI_MCP_URL = os.getenv("KIWI_MCP_URL", "https://mcp.kiwi.com")


# ---------------------------------------------------------------------------
# 1. CONTEXT AND STATE
# ---------------------------------------------------------------------------

@dataclass
class WeddingContext:
    """Fixed information supplied by the application for one agent run."""

    planner_email: str = "planner@example.com"
    planner_password: str = "wedding123"
    client_name: str = "Ava and Noah"
    client_email: str = "couple@example.com"
    preferred_language: str = "English"


class WeddingState(AgentState):
    """Mutable information learned while the conversation continues."""

    authenticated: bool
    departure_city: str
    wedding_city: str
    wedding_date: str
    travel_start_date: str
    travel_end_date: str
    budget_usd: int
    guest_count: int
    style_summary: str


# ---------------------------------------------------------------------------
# 2. MULTIMODAL MESSAGE
# ---------------------------------------------------------------------------

def analyze_inspiration_image(image_path: str) -> str:
    """Analyze a wedding inspiration image and return a short style summary."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/png"
    image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")

    image_agent = create_agent(
        model=STANDARD_MODEL_NAME,
        system_prompt=(
            "You are a wedding design assistant. Analyze the uploaded inspiration "
            "image. Return a concise summary of colors, style, decor, and venue mood."
        ),
    )

    question = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Describe this wedding inspiration image in 5 short bullets.",
            },
            {
                "type": "image",
                "base64": image_b64,
                "mime_type": mime_type,
            },
        ]
    )

    response = image_agent.invoke({"messages": [question]})
    return response["messages"][-1].content


# ---------------------------------------------------------------------------
# 3. EMAIL + STATE TOOLS USING ToolRuntime
# ---------------------------------------------------------------------------

@tool
def authenticate(email: str, password: str, runtime: ToolRuntime) -> Command:
    """Authenticate the wedding planner before private tools become available."""
    valid = (
        email == runtime.context.planner_email
        and password == runtime.context.planner_password
    )

    message = "Successfully authenticated." if valid else "Authentication failed."
    return Command(
        update={
            "authenticated": valid,
            "messages": [
                ToolMessage(content=message, tool_call_id=runtime.tool_call_id)
            ],
        }
    )


@tool
def save_wedding_details(
    departure_city: str,
    wedding_city: str,
    wedding_date: str,
    travel_start_date: str,
    travel_end_date: str,
    budget_usd: int,
    guest_count: int,
    style_summary: str,
    runtime: ToolRuntime,
) -> Command:
    """Save or update the couple's wedding and travel details in agent state."""
    return Command(
        update={
            "departure_city": departure_city,
            "wedding_city": wedding_city,
            "wedding_date": wedding_date,
            "travel_start_date": travel_start_date,
            "travel_end_date": travel_end_date,
            "budget_usd": budget_usd,
            "guest_count": guest_count,
            "style_summary": style_summary,
            "messages": [
                ToolMessage(
                    content="Wedding and travel details saved to state.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def get_wedding_details(runtime: ToolRuntime) -> str:
    """Read wedding details from state and client identity from context."""
    return str(
        {
            "client_name": runtime.context.client_name,
            "client_email": runtime.context.client_email,
            "departure_city": runtime.state.get("departure_city"),
            "wedding_city": runtime.state.get("wedding_city"),
            "wedding_date": runtime.state.get("wedding_date"),
            "travel_start_date": runtime.state.get("travel_start_date"),
            "travel_end_date": runtime.state.get("travel_end_date"),
            "budget_usd": runtime.state.get("budget_usd"),
            "guest_count": runtime.state.get("guest_count"),
            "style_summary": runtime.state.get("style_summary"),
        }
    )


@tool
def send_client_email(subject: str, body: str, runtime: ToolRuntime) -> str:
    """Send a wedding-plan email to the client stored in runtime context."""
    # This starter project intentionally uses a dummy sender.
    return (
        f"Email sent to {runtime.context.client_email} "
        f"with subject '{subject}'.\n\n{body}"
    )


@tool
def web_search(query: str) -> dict[str, Any]:
    """Search the web for current wedding venue information."""
    return TavilyClient().search(query)


@tool
def flight_search_unavailable(request: str) -> str:
    """Report that the remote Kiwi MCP server could not be reached in this run."""
    return (
        "The Kiwi MCP flight tools could not be loaded, so no live flight results "
        "are available. Do not invent airlines, prices, or schedules. Requested "
        f"itinerary: {request}. Verify KIWI_MCP_URL and internet access, then retry."
    )


# ---------------------------------------------------------------------------
# 4. NODE-STYLE MIDDLEWARE: State + Runtime
# ---------------------------------------------------------------------------

@before_agent
def trim_old_tool_messages(
    state: WeddingState, runtime: Runtime
) -> dict[str, Any] | None:
    """Keep only the four newest ToolMessages to reduce conversation clutter."""
    del runtime  # Runtime is available here even though this hook only needs state.

    tool_messages = [
        message
        for message in state["messages"]
        if isinstance(message, ToolMessage) and message.id is not None
    ]

    if len(tool_messages) <= 4:
        return None

    return {
        "messages": [
            RemoveMessage(id=message.id) for message in tool_messages[:-4]
        ]
    }


# ---------------------------------------------------------------------------
# 5. WRAP-STYLE MIDDLEWARE: ModelRequest
# ---------------------------------------------------------------------------

standard_model = init_chat_model(STANDARD_MODEL_NAME)
large_model = init_chat_model(LARGE_MODEL_NAME)


@wrap_model_call
def choose_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Use a stronger model only after the conversation becomes long."""
    model = large_model if len(request.messages) > 12 else standard_model
    return handler(request.override(model=model))


@dynamic_prompt
def wedding_prompt(request: ModelRequest) -> str:
    """Change the model's role after authentication."""
    authenticated = request.state.get("authenticated", False)

    if not authenticated:
        return (
            "You are the login assistant for a wedding-planning application. "
            "Ask for the planner email and password, then call authenticate."
        )

    language = request.runtime.context.preferred_language
    client_name = request.runtime.context.client_name
    return f"""
You are a friendly destination-wedding coordinator helping {client_name}.
Respond in {language}.

Your workflow:
1. Collect departure city, wedding destination, wedding date, travel start/end dates,
   total budget, guest count, and wedding style.
2. Save them with save_wedding_details.
3. Delegate flight, venue, and music tasks to their specialist agents.
4. Combine the results into a short, practical destination-wedding plan.
5. Only call send_client_email when the user explicitly asks you to email the plan.
6. Clearly label flight prices, venue prices, schedules, and availability as estimates
   unless a connected tool provides live confirmed information.
7. Never invent flight results when the flight MCP tool is unavailable.
"""


def create_dynamic_tool_middleware(planning_tools: list[Any]):
    """Factory because specialist tools are created inside build_wedding_agent."""

    @wrap_model_call
    def dynamic_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Expose only authenticate until login succeeds."""
        if request.state.get("authenticated", False):
            tools = planning_tools
        else:
            tools = [authenticate]

        return handler(request.override(tools=tools))

    return dynamic_tools


# ---------------------------------------------------------------------------
# 6. EXTERNAL MCP + MULTI-AGENT COORDINATION
# ---------------------------------------------------------------------------

async def load_kiwi_flight_tools() -> list[Any]:
    """Load live flight-search tools from Kiwi's remote MCP server.

    A small fallback tool keeps the rest of the learning project usable when the
    remote endpoint is unavailable or changes in the future.
    """
    kiwi_client = MultiServerMCPClient(
        {
            "kiwi_flights": {
                "url": KIWI_MCP_URL,
                "transport": "http",
            }
        }
    )

    try:
        tools = await kiwi_client.get_tools()
        return tools or [flight_search_unavailable]
    except Exception as exc:
        print(
            "Warning: Kiwi MCP tools could not be loaded. "
            f"Using the learning fallback tool instead. Details: {exc}"
        )
        return [flight_search_unavailable]


async def build_wedding_agent():
    """Build specialist agents and return the main wedding coordinator."""

    # Flight specialist: external remote MCP, matching the course architecture.
    kiwi_flight_tools = await load_kiwi_flight_tools()

    flight_agent = create_agent(
        model=STANDARD_MODEL_NAME,
        tools=kiwi_flight_tools,
        system_prompt=(
            "You are a destination-wedding flight specialist. Use the connected "
            "Kiwi MCP tools to search for suitable round-trip flight options. "
            "Return a concise comparison including route, dates, major trade-offs, "
            "and booking links when the tool provides them. Never invent live "
            "prices, schedules, airlines, or availability."
        ),
    )

    # Venue specialist: Tavily web search.
    venue_agent = create_agent(
        model=STANDARD_MODEL_NAME,
        tools=[web_search],
        system_prompt=(
            "You are a venue research specialist. Search the web for 2-3 suitable "
            "destination-wedding venues. Return venue name, location, why it fits, "
            "and a source URL. Do not claim availability or confirmed pricing."
        ),
    )

    # Music specialist: Chinook SQLite database supplied with the course.
    database = SQLDatabase.from_uri(f"sqlite:///{ROOT / 'data' / 'Chinook.db'}")

    @tool
    def sql_query(query: str) -> str:
        """Query the sample music database using SQL."""
        try:
            return database.run(query)
        except Exception as exc:  # beginner-friendly error returned to the agent
            return f"SQL error: {exc}"

    music_agent = create_agent(
        model=STANDARD_MODEL_NAME,
        tools=[sql_query],
        system_prompt=(
            "You are a wedding music specialist. Query the Chinook sample database "
            "and recommend artists or tracks that fit the requested mood. Return a "
            "short playlist idea; never provide copyrighted lyrics."
        ),
    )

    # Specialist agents become tools for the supervisor/coordinator.
    @tool
    async def ask_flight_specialist(request: str) -> str:
        """Delegate flight research to the Kiwi MCP flight specialist agent."""
        response = await flight_agent.ainvoke(
            {"messages": [HumanMessage(content=request)]}
        )
        return response["messages"][-1].content

    @tool
    async def ask_venue_specialist(request: str) -> str:
        """Delegate venue research to the venue specialist agent."""
        response = await venue_agent.ainvoke(
            {"messages": [HumanMessage(content=request)]}
        )
        return response["messages"][-1].content

    @tool
    async def ask_music_specialist(request: str) -> str:
        """Delegate playlist research to the music specialist agent."""
        response = await music_agent.ainvoke(
            {"messages": [HumanMessage(content=request)]}
        )
        return response["messages"][-1].content

    planning_tools = [
        save_wedding_details,
        get_wedding_details,
        ask_flight_specialist,
        ask_venue_specialist,
        ask_music_specialist,
        send_client_email,
    ]

    coordinator = create_agent(
        model=STANDARD_MODEL_NAME,
        tools=[authenticate, *planning_tools],
        state_schema=WeddingState,
        context_schema=WeddingContext,
        checkpointer=InMemorySaver(),
        middleware=[
            trim_old_tool_messages,  # node-style middleware
            create_dynamic_tool_middleware(planning_tools),  # wrap-style tools
            wedding_prompt,  # wrap-style dynamic prompt
            choose_model,  # wrap-style dynamic model
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "authenticate": False,
                    "save_wedding_details": False,
                    "get_wedding_details": False,
                    "ask_flight_specialist": False,
                    "ask_venue_specialist": False,
                    "ask_music_specialist": False,
                    "send_client_email": True,
                },
                description_prefix="Client email requires planner approval",
            ),
        ],
    )

    return coordinator
