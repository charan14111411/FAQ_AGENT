from app.agents.base_agent import run_specialist_agent

async def run(history: list, user_message: str, context: str) -> dict:
    return await run_specialist_agent(
        agent_key="investor",
        history=history,
        user_message=user_message,
        context=context,
        max_tokens=700,
        temperature=0.4,
    )
