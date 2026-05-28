from app.config import settings
from groq import AsyncGroq


async def _call_llm(messages: list, max_tokens: int, temperature: float) -> dict:
    """
    Groq-only LLM call.
    Returns: {reply, input_tokens, output_tokens, model}
    On failure: returns error_fallback dict so callers can detect and handle gracefully.
    """
    try:
        client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        model = "llama-3.3-70b-versatile"
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {
            "reply":         resp.choices[0].message.content,
            "input_tokens":  resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "model":         model,
        }
    except Exception as e:
        from app.logger import get_logger
        get_logger().error(f"Groq LLM call failed: {e}")
        return {
            "reply":         "I am experiencing technical difficulties. Please try again in a moment.",
            "input_tokens":  0,
            "output_tokens": 0,
            "model":         "error_fallback",
        }
