from app.agents import grower_agent, corporate_agent, investor_agent, agritech_agent
import json
import re
from app.agents.base_agent import _call_llm

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


def _extract_json_block(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


async def route_category(
    user_message: str,
    current_category: str | None,
    history: list | None = None,
    context: str = "",
) -> dict:
    safe_current = (current_category or "").strip().lower()
    history = history or []
    category_list = ", ".join(sorted(AGENT_REGISTRY.keys()))

    system_prompt = (
        "You are an agent router. Decide the best specialist category for the user's message.\n"
        f"Allowed categories: {category_list}\n"
        "Return only strict JSON with keys: category, switch_requested, reason.\n"
        "category must be one of allowed values.\n"
        "switch_requested must be true if current category is present and differs from chosen category."
    )
    user_prompt = (
        f"Current category: {safe_current or 'none'}\n"
        f"Message: {user_message}\n"
        f"History: {history[-4:]}\n"
        f"Context: {context[:1200]}"
    )
    llm_result = await _call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=120,
        temperature=0.0,
    )
    parsed = _extract_json_block(llm_result.get("reply", ""))
    if parsed and parsed.get("category") in AGENT_REGISTRY:
        chosen = parsed["category"]
        switch_requested = bool(parsed.get("switch_requested", False))
        if safe_current and chosen != safe_current:
            switch_requested = True
        return {
            "category": chosen,
            "switch_requested": switch_requested,
            "reason": str(parsed.get("reason", "auto_route")),
        }

    heuristics = {
        "investor": ["roi", "revenue", "valuation", "market", "tam", "funding", "investor"],
        "corporate": ["compliance", "eudr", "dashboard", "enterprise", "audit", "carbon"],
        "agritech": ["integration", "api", "device", "sensor", "warranty", "installation", "reseller"],
        "grower": ["farm", "crop", "plantation", "leaf", "grower", "yield"],
    }
    message = (user_message or "").lower()
    chosen = safe_current if safe_current in AGENT_REGISTRY else "grower"
    for cat, terms in heuristics.items():
        if any(term in message for term in terms):
            chosen = cat
            break
    return {
        "category": chosen,
        "switch_requested": bool(safe_current and chosen != safe_current),
        "reason": "heuristic_fallback",
    }
