from app.agents import grower_agent, corporate_agent, investor_agent, agritech_agent

AGENT_REGISTRY = {
    "grower": grower_agent,
    "corporate": corporate_agent,
    "investor": investor_agent,
    "agritech": agritech_agent
}

async def dispatch_agent(category: str, history: list, user_message: str, context: str) -> dict:
    cat = category.lower().strip()
    if cat not in AGENT_REGISTRY:
        raise ValueError(f"Unknown category: {category}")
    agent = AGENT_REGISTRY[cat]
    return await agent.run(history, user_message, context)

async def detect_switch_intent(user_message: str) -> bool:
    msg = user_message.lower()
    phrases = [
        "change category", 
        "switch category", 
        "change agent", 
        "switch agent", 
        "different category", 
        "change to", 
        "switch to", 
        "i want to talk to", 
        "talk to someone else", 
        "change my category", 
        "switch my category"
    ]
    for phrase in phrases:
        if phrase in msg:
            return True
    return False
