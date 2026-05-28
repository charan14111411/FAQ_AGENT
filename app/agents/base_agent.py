from app.config import settings
from groq import AsyncGroq
from openai import AsyncOpenAI


async def _call_llm(messages: list, max_tokens: int, temperature: float) -> dict:
    """
    Unified LLM caller. Supports Groq and OpenAI providers.
    Configured via LLM_PROVIDER in .env.
    Returns: { reply, input_tokens, output_tokens, model }
    On error: returns a safe fallback dict instead of raising.
    """
    provider = settings.LLM_PROVIDER.lower()

    try:
        if provider == "groq":
            client = AsyncGroq(api_key=settings.GROQ_API_KEY)
            model = "llama-3.3-70b-versatile"
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        elif provider == "openai":
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            model = "gpt-4o-mini"
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: '{settings.LLM_PROVIDER}'. Use 'groq' or 'openai'.")

        return {
            "reply":         response.choices[0].message.content,
            "input_tokens":  response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "model":         model,
        }

    except Exception as e:
        from app.logger import get_logger
        get_logger().error(f"LLM call failed ({provider}): {e}")
        return {
            "reply":         "I'm experiencing a brief technical issue. Please try your message again in a moment.",
            "input_tokens":  0,
            "output_tokens": 0,
            "model":         "error_fallback",
        }
