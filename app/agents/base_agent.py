from app.config import settings
from groq import AsyncGroq
from openai import AsyncOpenAI
import time
from typing import Any

def _result_from_response(response: Any, model: str) -> dict:
    reply = response.choices[0].message.content
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    return {
        "reply": reply,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model,
    }

async def _call_openai(messages: list, max_tokens: int, temperature: float) -> dict:
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    model = "gpt-4o-mini"
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return _result_from_response(response, model)


async def _call_groq(messages: list, max_tokens: int, temperature: float) -> dict:
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    model = "llama-3.3-70b-versatile"
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return _result_from_response(response, model)


async def _call_llm(messages: list, max_tokens: int, temperature: float) -> dict:
    provider = settings.LLM_PROVIDER.lower()
    openai_key = (settings.OPENAI_API_KEY or "").strip()

    if provider == "groq":
        try:
            return await _call_groq(messages, max_tokens, temperature)
        except Exception as e:
            msg = str(e).lower()
            if ("rate limit" in msg or "rate_limit_exceeded" in msg) and openai_key:
                return await _call_openai(messages, max_tokens, temperature)
            raise

    if provider == "openai":
        return await _call_openai(messages, max_tokens, temperature)

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}")


async def run_specialist_agent(
    agent_key: str,
    history: list,
    user_message: str,
    context: str,
    max_tokens: int = 600,
    temperature: float = 0.3,
) -> dict:
    from app.data.personas import get_persona

    start_time = time.time()
    system_prompt = get_persona(agent_key)

    if context and context.strip():
        system_prompt += f"\n\nMOST RELEVANT CONTEXT FOR THIS QUESTION:\n{context}"

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    result = await _call_llm(messages, max_tokens=max_tokens, temperature=temperature)
    result["latency_ms"] = int((time.time() - start_time) * 1000)
    result["agent"] = f"{agent_key}_agent"
    return result
