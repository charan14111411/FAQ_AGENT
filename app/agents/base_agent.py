import os
from app.config import settings
from groq import AsyncGroq
from openai import AsyncOpenAI


async def _call_llm(messages: list, max_tokens: int, temperature: float) -> dict:
    """
    Unified LLM caller. Supports Groq, OpenAI, and Gemini providers.
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
                temperature=temperature,
            )
            reply = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        elif provider == "gemini":
            from google import genai
            from google.genai import types

            credentials_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "varsapradaya-credentials.json")
            if os.path.exists(credentials_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
                client = genai.Client(vertexai=True, project="varsapradaya", location="us-central1")
            else:
                api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "No Gemini API key or Google Application Credentials was provided. "
                        "Please pass a valid API key via GEMINI_API_KEY in your .env file or place "
                        "varsapradaya-credentials.json in the project root."
                    )
                client = genai.Client(api_key=api_key)

            model = "gemini-2.5-flash"

            # Map OpenAI/Groq messages format to Gemini Content objects
            gemini_contents = []
            system_instruction = None
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role == "system":
                    system_instruction = content
                elif role in ["user", "assistant"]:
                    gemini_role = "model" if role == "assistant" else "user"
                    gemini_contents.append(
                        types.Content(
                            role=gemini_role,
                            parts=[types.Part.from_text(text=content)]
                        )
                    )

            if not gemini_contents:
                gemini_contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text="Please respond.")]
                    )
                )

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=max_tokens,
                temperature=temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )

            import asyncio
            max_retries = 3
            retry_delay = 1.0
            response = None
            for attempt in range(max_retries):
                try:
                    response = await client.aio.models.generate_content(
                        model=model,
                        contents=gemini_contents,
                        config=config,
                    )
                    break
                except Exception as e:
                    is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "rate limit" in str(e).lower()
                    if is_rate_limit and attempt < max_retries - 1:
                        from app.logger import get_logger
                        get_logger().warning(f"Gemini API rate limited (429). Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2.0
                    else:
                        raise e

            reply = response.text or ""
            input_tokens = (response.usage_metadata.prompt_token_count or 0) if response.usage_metadata else 0
            output_tokens = (response.usage_metadata.candidates_token_count or 0) if response.usage_metadata else 0

        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: '{settings.LLM_PROVIDER}'. Use 'groq', 'openai', or 'gemini'.")

        # Beautiful console logging for LLM calls (full clarity on query, response, and tokens)
        print(f"\n{'='*60}")
        print(f"[*] [LLM CALL] Provider: {provider.upper()} | Model: {model}")
        print(f"    Prompt: '{messages[-1]['content'] if messages else 'N/A'}'")
        print(f"    Response: '{reply}'")
        print(f"    Tokens: Input={input_tokens} | Output={output_tokens} | Total={input_tokens + output_tokens}")
        print(f"{'='*60}\n")

        return {
            "reply":         reply,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "model":         model,
        }

    except Exception as e:
        import traceback
        from app.logger import get_logger
        get_logger().error(f"LLM call failed ({provider}): {e}\n{traceback.format_exc()}")
        return {
            "reply":         "I'm experiencing a brief technical issue. Please try your message again in a moment.",
            "input_tokens":  0,
            "output_tokens": 0,
            "model":         "error_fallback",
        }
