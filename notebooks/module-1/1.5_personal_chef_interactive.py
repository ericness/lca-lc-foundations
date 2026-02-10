# %% [markdown]
# # Personal Chef Agent
#
# An interactive recipe assistant that combines concepts from modules 1.1–1.4:
# - **1.1 Foundational Models**: Model initialization and streaming
# - **1.1 Prompting**: System prompts with few-shot examples
# - **1.2 Tools**: Custom tool definitions with `@tool`
# - **1.2 Web Search**: Real-time recipe lookup via Tavily
# - **1.3 Memory**: Persistent conversation with `InMemorySaver`

# %% Environment setup
from dotenv import load_dotenv

load_dotenv()

# %% Imports
from typing import Dict, Any
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient

# %% Tool definition (1.2)
tavily_client = TavilyClient()


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for recipes and cooking information."""
    return tavily_client.search(query)


# %% System prompt with few-shot examples (1.1)
system_prompt = """You are a friendly personal chef assistant. Your job is to help \
the user find recipes based on ingredients they have on hand.

When the user tells you what ingredients they have:
1. Use the web_search tool to find recipes that use those ingredients.
2. Suggest 2-3 recipe options with a brief description of each.
3. When the user picks a recipe, search for the full instructions and present them clearly.

Here are some example interactions:

User: I have chicken, garlic, and lemon.
Chef: Great ingredients! Let me search for some recipes for you.
[searches web]
Here are a few options:
1. **Lemon Garlic Chicken** – A classic roasted chicken with bright citrus and garlic flavors.
2. **Chicken Piccata** – Pan-seared chicken in a lemon-garlic butter sauce.
3. **Greek Lemon Chicken Soup** – A warm, comforting soup with egg-lemon broth.

Which one sounds good? I can get you the full recipe.

User: Tell me more about the piccata.
Chef: [searches web for full recipe]
Here's a simple Chicken Piccata recipe: ...

Guidelines:
- Be warm and encouraging.
- If the user has very few ingredients, suggest simple recipes or note what one extra item could unlock.
- Always search the web for up-to-date recipes rather than relying on memory.
- Keep responses concise but helpful.
"""

# %% Agent creation with memory (1.3) and tools (1.2)
agent = create_agent(
    model="gpt-5-nano",
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}

# %% Interactive chat loop with streaming (1.1)
print("=" * 60)
print("  Welcome to your Personal Chef!")
print("  Tell me what ingredients you have and I'll find recipes.")
print("  Type 'quit' to exit.")
print("=" * 60)
print()

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit", "q"):
        print("\nBon appétit! Goodbye.")
        break

    print("\nChef: ", end="", flush=True)
    for token, metadata in agent.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        stream_mode="messages",
    ):
        if token.content:
            print(token.content, end="", flush=True)
    print("\n")

# %%
