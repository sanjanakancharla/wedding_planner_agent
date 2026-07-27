"""Small command-line demonstration of the Destination Wedding Planner Agent."""

import asyncio
from pprint import pprint

from langchain.messages import HumanMessage
from langgraph.types import Command

from wedding_planner import WeddingContext, build_wedding_agent


async def main() -> None:
    agent = await build_wedding_agent()
    context = WeddingContext()
    config = {"configurable": {"thread_id": "wedding-demo-1"}}

    print("\n1) AUTHENTICATING\n")
    login = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="planner@example.com, wedding123")
            ]
        },
        context=context,
        config=config,
    )
    print(login["messages"][-1].content)

    print("\n2) CREATING THE DESTINATION-WEDDING PLAN\n")
    plan = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Plan a romantic beach wedding in Cancun on October 10, "
                        "2027 for 80 guests with a total budget of $45,000. Guests "
                        "will travel from Dallas. Search round-trip flights for "
                        "October 7 through October 12, 2027. The style is ivory, "
                        "blush, tropical greenery, and candlelight. Find flight "
                        "options, venue ideas, and a romantic music direction. "
                        "Do not email yet."
                    )
                )
            ]
        },
        context=context,
        config=config,
    )
    print(plan["messages"][-1].content)

    print("\n3) DRAFTING A CLIENT EMAIL\n")
    email_attempt = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Email the final destination-wedding plan to the client."
                )
            ]
        },
        context=context,
        config=config,
    )

    if "__interrupt__" not in email_attempt:
        print(email_attempt["messages"][-1].content)
        return

    request = email_attempt["__interrupt__"][0].value["action_requests"][0]
    print("\nEMAIL WAITING FOR HUMAN APPROVAL:\n")
    pprint(request["args"])

    decision = input("\nApprove email? Type yes or no: ").strip().lower()
    decision_type = "approve" if decision in {"yes", "y"} else "reject"

    resumed = await agent.ainvoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": decision_type,
                        "message": "The wedding planner reviewed the draft.",
                    }
                ]
            }
        ),
        context=context,
        config=config,
    )

    print("\nFINAL RESULT:\n")
    print(resumed["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
