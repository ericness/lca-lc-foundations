# %% [markdown]
# # Multi-Agent Wedding Planner
#
# A multi-agent system that coordinates three specialist sub-agents to plan a
# wedding, building on concepts from modules 2.1-2.3:
# - **2.1 MCP / Tools**: Tool definitions with `@tool`
# - **2.2 State**: Persistent conversation memory with `InMemorySaver`
# - **2.3 Multi-Agent**: Sub-agents wrapped as tools for a coordinator
#
# **Architecture:**
# - **Travel Agent** (flights) -- searches for flight options
# - **Venue Agent** -- searches for wedding venue options
# - **DJ Agent** (playlists) -- curates music and playlist suggestions
# - **Coordinator Agent** -- orchestrates the three specialists

# %% Environment setup
from dotenv import load_dotenv

load_dotenv()

# %% Imports
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient

# %% Web search tool (shared by all sub-agents)
tavily_client = TavilyClient()


@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information about flights, venues, music, or anything wedding-related."""
    results = tavily_client.search(query)
    return str(results)


# %% Travel Agent (flights sub-agent)
travel_agent = create_agent(
    model="gpt-5-mini",
    tools=[web_search],
    system_prompt="""You are a travel agent specializing in wedding travel logistics.

Your job is to find flight options for the wedding party. When given details about
a wedding (origin city, destination, dates, budget, group size), search the web for:
- Best flight routes and airlines for the trip
- Approximate pricing and deals
- Tips for group travel bookings

Keep your response concise and actionable. Present 2-3 flight options with pricing
estimates. If details are missing, work with what you have and note assumptions.""",
)

# %% Venue Agent (sub-agent)
venue_agent = create_agent(
    model="gpt-5-mini",
    tools=[web_search],
    system_prompt="""You are a wedding venue specialist.

Your job is to find and recommend wedding venues. When given details about a wedding
(location, date, guest count, budget, style preferences), search the web for:
- Specific venue recommendations with names and descriptions
- Capacity and pricing information
- What makes each venue special

Keep your response concise. Present 2-3 venue options with key details.
If details are missing, work with what you have and note assumptions.""",
)

# %% DJ Agent (playlist sub-agent)
dj_agent = create_agent(
    model="gpt-5-mini",
    tools=[web_search],
    system_prompt="""You are a professional wedding DJ and music curator.

Your job is to suggest music and playlists for a wedding. When given details about
a wedding (style, vibe, genre preferences, cultural background), search the web for:
- Curated playlist suggestions for different parts of the wedding (ceremony,
  cocktail hour, reception, first dance, party)
- Specific song recommendations
- Current trending wedding songs

Keep your response concise and organized by wedding moment. If details are missing,
suggest a well-rounded mix and note what you assumed.""",
)

# %% Wrapper tools that let the coordinator call each sub-agent


@tool
def consult_travel_agent(request: str) -> str:
    """Consult the travel agent to find flight options for the wedding party.
    Pass along all relevant details: origin city, destination, dates, budget, group size."""
    response = travel_agent.invoke(
        {"messages": [HumanMessage(content=request)]}
    )
    return response["messages"][-1].content


@tool
def consult_venue_agent(request: str) -> str:
    """Consult the venue specialist to find wedding venue options.
    Pass along all relevant details: location, date, guest count, budget, style."""
    response = venue_agent.invoke(
        {"messages": [HumanMessage(content=request)]}
    )
    return response["messages"][-1].content


@tool
def consult_dj_agent(request: str) -> str:
    """Consult the DJ to get music and playlist suggestions for the wedding.
    Pass along all relevant details: style, vibe, genre preferences, cultural notes."""
    response = dj_agent.invoke(
        {"messages": [HumanMessage(content=request)]}
    )
    return response["messages"][-1].content


# %% Coordinator Agent
coordinator = create_agent(
    model="gpt-5-mini",
    tools=[consult_travel_agent, consult_venue_agent, consult_dj_agent],
    system_prompt="""You are a wedding planning coordinator. You manage a team of three
specialists to create comprehensive wedding plans:

1. **Travel Agent** - finds flights and travel logistics
2. **Venue Specialist** - finds and recommends wedding venues
3. **DJ** - curates music and playlists

When a user describes their wedding, extract the key details and consult your
specialists. You should:
- Parse the user's free-form description for relevant details (location, date,
  budget, guest count, music preferences, departure city, style, etc.)
- Delegate to the right specialists with clear, detailed requests
- Compile their recommendations into a cohesive wedding plan
- Be ready to refine any part of the plan based on follow-up requests

If the user's description is missing important details, make reasonable assumptions
and note them. Always present the final plan in a clear, organized format with
sections for Travel, Venue, and Music.

You can also consult individual specialists if the user wants to refine just one
part of the plan (e.g., "show me different venue options" or "I want more upbeat music").""",
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}

# %% Interactive chat loop
print("=" * 60)
print("  Wedding Planner - Multi-Agent System")
print("  Describe your dream wedding and we'll plan it!")
print("  e.g. 'Plan a summer wedding in Napa Valley for 150")
print("        guests with a $50k budget'")
print("  Type 'quit' to exit.")
print("=" * 60)
print()

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit", "q"):
        print("\nCongratulations on your upcoming wedding! Goodbye.")
        break

    print("\nPlanner: ", end="", flush=True)
    for token, metadata in coordinator.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        stream_mode="messages",
    ):
        if token.content:
            print(token.content, end="", flush=True)
    print("\n")

# %%
