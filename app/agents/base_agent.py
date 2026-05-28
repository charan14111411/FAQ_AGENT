from app.config import settings
from groq import AsyncGroq
from openai import AsyncOpenAI

async def _call_llm(messages: list, max_tokens: int, temperature: float) -> dict:
    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "groq":
        client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        model = "llama-3.3-70b-versatile"
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        reply = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
    elif provider == "openai":
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        model = "gpt-4o-mini"
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        reply = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}")
        
    return {
        "reply": reply,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model
    }
