import time
from app.agents.base_agent import _call_llm
from app.data.personas import get_persona

async def run(history: list, user_message: str, context: str) -> dict:
    start_time = time.time()
    system_prompt = get_persona("investor")
    
    if context and context.strip():
        system_prompt += f"\n\nMOST RELEVANT CONTEXT FOR THIS QUESTION:\n{context}"
        
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    
    result = await _call_llm(messages, max_tokens=700, temperature=0.4)
    
    latency_ms = int((time.time() - start_time) * 1000)
    result["latency_ms"] = latency_ms
    result["agent"] = "investor_agent"
    return result
