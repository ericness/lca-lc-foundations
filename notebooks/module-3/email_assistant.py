from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, ToolMessage
from langchain.agents.middleware import dynamic_prompt, HumanInTheLoopMiddleware, ModelRequest

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Sample Data
# ---------------------------------------------------------------------------

SAMPLE_INBOX = [
    {
        "id": "1",
        "from": "jane@example.com",
        "subject": "Coffee this weekend?",
        "body": "Hey! Are you free this Saturday for coffee? I found a great new café downtown. Let me know!",
    },
    {
        "id": "2",
        "from": "boss@company.com",
        "subject": "Q3 Report Review",
        "body": "Hi, could you review the Q3 report and send me your comments by end of day Thursday? Thanks.",
    },
    {
        "id": "3",
        "from": "deals@spammy-cruises.biz",
        "subject": "YOU WON A FREE CRUISE!!!",
        "body": "Congratulations! You have been selected for a FREE luxury cruise. Click here to claim your prize now!!!",
    },
    {
        "id": "4",
        "from": "mike@example.com",
        "subject": "Hike on Sunday?",
        "body": "Hey, a group of us are hiking the ridge trail this Sunday morning. Want to join? We're meeting at 8am at the trailhead.",
    },
    {
        "id": "5",
        "from": "hr@company.com",
        "subject": "Mandatory Compliance Training Reminder",
        "body": "This is a reminder that all employees must complete the annual compliance training module by Friday. Please log in to the training portal to complete it.",
    },
]

# ---------------------------------------------------------------------------
# 2. State Schema
# ---------------------------------------------------------------------------


class InboxState(AgentState):
    inbox: list[dict]
    processed_ids: list[str]


# ---------------------------------------------------------------------------
# 3. Tools
# ---------------------------------------------------------------------------


@tool
def check_inbox(runtime: ToolRuntime) -> str:
    """List all emails in the inbox with their status (NEW or DONE)."""
    inbox = runtime.state["inbox"]
    processed_ids = runtime.state.get("processed_ids", [])
    lines = []
    for email in inbox:
        status = "[DONE]" if email["id"] in processed_ids else "[NEW]"
        lines.append(f'{status} ID:{email["id"]} From:{email["from"]} Subject:{email["subject"]}')
    return "\n".join(lines)


@tool
def read_email(email_id: str, runtime: ToolRuntime) -> str:
    """Read the full content of an email by its ID."""
    inbox = runtime.state["inbox"]
    for email in inbox:
        if email["id"] == email_id:
            return (
                f'From: {email["from"]}\n'
                f'Subject: {email["subject"]}\n'
                f'Body: {email["body"]}'
            )
    return f"Email with ID {email_id} not found."


@tool
def reply_to_email(email_id: str, body: str, runtime: ToolRuntime) -> Command:
    """Reply to an email. Requires human approval."""
    inbox = runtime.state["inbox"]
    recipient = None
    for email in inbox:
        if email["id"] == email_id:
            recipient = email["from"]
            break

    if recipient is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Email with ID {email_id} not found.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    processed_ids = list(runtime.state.get("processed_ids", []))
    processed_ids.append(email_id)

    return Command(
        update={
            "processed_ids": processed_ids,
            "messages": [
                ToolMessage(
                    f"Reply sent to {recipient}: {body}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def delete_email(email_id: str, runtime: ToolRuntime) -> Command:
    """Delete an email from the inbox. Requires human approval."""
    inbox = runtime.state["inbox"]
    found = any(email["id"] == email_id for email in inbox)

    if not found:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Email with ID {email_id} not found.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    processed_ids = list(runtime.state.get("processed_ids", []))
    processed_ids.append(email_id)

    return Command(
        update={
            "processed_ids": processed_ids,
            "messages": [
                ToolMessage(
                    f"Email {email_id} deleted.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


# ---------------------------------------------------------------------------
# 4. Dynamic Prompt Middleware
# ---------------------------------------------------------------------------


@dynamic_prompt
def inbox_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on how many emails remain unprocessed."""
    inbox = request.state.get("inbox", [])
    processed_ids = request.state.get("processed_ids", [])
    remaining = len([e for e in inbox if e["id"] not in processed_ids])

    if remaining == 0:
        return (
            "All emails have been processed. Let the user know you are done "
            "and summarize the actions you took."
        )

    return (
        f"You are an email assistant. There are {remaining} unprocessed email(s) in the inbox. "
        "Work through them one at a time: check the inbox, read each unprocessed email, "
        "then either reply or delete it. For spam or junk mail, delete it. "
        "For legitimate emails, draft a polite reply. "
        "Always use the reply_to_email or delete_email tools to take action — never just describe what you would do."
    )


# ---------------------------------------------------------------------------
# 5. Agent Creation
# ---------------------------------------------------------------------------

agent = create_agent(
    "gpt-5-nano",
    tools=[check_inbox, read_email, reply_to_email, delete_email],
    state_schema=InboxState,
    checkpointer=InMemorySaver(),
    middleware=[
        inbox_prompt,
        HumanInTheLoopMiddleware(
            interrupt_on={
                "check_inbox": False,
                "read_email": False,
                "reply_to_email": True,
                "delete_email": True,
            },
        ),
    ],
)

# ---------------------------------------------------------------------------
# 6. Demo Block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-1"}}

    print("=" * 60)
    print("  EMAIL ASSISTANT — Human-in-the-Loop Demo")
    print("=" * 60)

    # Initial invocation
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Please process all emails in my inbox. "
                        "Check the inbox first, then read and handle each email one at a time."
                    )
                )
            ],
            "inbox": SAMPLE_INBOX,
            "processed_ids": [],
        },
        config=config,
    )

    # HITL loop
    while response.get("__interrupt__"):
        interrupt = response["__interrupt__"][0]
        action_requests = interrupt.value.get("action_requests", [])

        for i, action in enumerate(action_requests):
            tool_name = action["name"]
            tool_args = action["args"]

            print("\n" + "-" * 60)
            print(f"  PROPOSED ACTION: {tool_name}")
            print("-" * 60)
            for key, value in tool_args.items():
                print(f"  {key}: {value}")
            print("-" * 60)

            choice = input("\n  Decision — (a)pprove / (r)eject / (e)dit: ").strip().lower()

            if choice == "a":
                decision = {"type": "approve"}
            elif choice == "r":
                reason = input("  Reason for rejection: ").strip()
                decision = {"type": "reject", "message": reason}
            elif choice == "e":
                print(f"\n  Current args: {tool_args}")
                edited_args = dict(tool_args)
                for key in tool_args:
                    new_val = input(f"  New value for '{key}' (Enter to keep): ").strip()
                    if new_val:
                        edited_args[key] = new_val
                decision = {
                    "type": "edit",
                    "edited_action": {
                        "name": tool_name,
                        "args": edited_args,
                    },
                }
            else:
                print("  Unrecognized choice, approving by default.")
                decision = {"type": "approve"}

        # Resume with decisions
        response = agent.invoke(
            Command(resume={"decisions": [decision]}),
            config=config,
        )

    # Final summary
    print("\n" + "=" * 60)
    print("  ALL DONE")
    print("=" * 60)
    final_message = response["messages"][-1].content
    print(f"\n  Agent: {final_message}")
    print(f"\n  Processed IDs: {response.get('processed_ids', [])}")
    print("=" * 60)
