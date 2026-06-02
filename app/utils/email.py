import httpx
import asyncio
import time
from app.db import AsyncSessionLocal, get_all_session_messages, write_log
from app.logger import get_logger
from app.agents.base_agent import _call_llm

logger = get_logger()

EMAIL_API_URL = "https://api-mobile.farmfuture.io/api/email/send"

async def call_email_api_with_retry(payload: dict, max_retries: int = 3, initial_delay: float = 1.0) -> dict:
    """
    Asynchronously POSTs to the email endpoint with retry logic and exponential backoff.
    """
    delay = initial_delay
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(1, max_retries + 1):
            try:
                response = await client.post(EMAIL_API_URL, json=payload)
                if response.status_code in [200, 201, 202, 204]:
                    return {"status": "success", "attempts": attempt, "code": response.status_code}
                
                logger.warning(
                    f"Email API returned status {response.status_code} (attempt {attempt}/{max_retries})"
                )
            except httpx.RequestError as exc:
                logger.warning(
                    f"Network error calling email API: {exc} (attempt {attempt}/{max_retries})"
                )
            
            if attempt < max_retries:
                sleep_time = delay * (2 ** (attempt - 1)) + (time.time() % 0.5)
                await asyncio.sleep(sleep_time)
                
        raise RuntimeError(f"Failed to send email after {max_retries} attempts.")

def format_simple_transcript(messages: list, name: str, category: str) -> str:
    """
    Builds a simple, untemplated text summary of the conversation (used as a fallback).
    """
    lines = [
        "<b>Varsapradaya Chat Summary</b><br>",
        f"<b>Customer Name:</b> {name}<br>",
        f"<b>Audience Role:</b> {category.capitalize()}<br>",
        "---------------------------------------------<br><br>"
    ]
    for msg in messages:
        role = msg["role"]
        content = msg["content"].replace("\n", "<br>")
        if role == "user":
            lines.append(f"<b>{name}:</b> {content}<br>")
        else:
            lines.append(f"<b>Varsapradaya:</b> {content}<br>")
    
    return "\n".join(lines)

async def generate_conversation_summary(messages: list, name: str, category: str) -> str:
    """
    Uses the configured LLM to generate a clean, professional, and concise HTML summary of the conversation.
    """
    transcript_lines = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        speaker = name if role == "user" else "Varsapradaya AI"
        transcript_lines.append(f"{speaker}: {content}")
    transcript_str = "\n".join(transcript_lines)

    system_prompt = (
        "You are an expert agritech communications assistant for Varsapradaya.\n"
        "Your task is to analyze the conversation transcript provided by the user and write a highly professional, "
        "concise summary of the interaction.\n\n"
        "GUIDELINES:\n"
        "1. Write the summary directly. Do not include any introductory filler like 'Here is the summary' or 'Sure, I can help'.\n"
        "2. Keep the summary under 180 words total.\n"
        "3. Focus on: What the customer was inquiring about, key information or solutions given by the advisor, and next steps.\n"
        "4. Format the output using basic HTML tags (e.g. <b>, <ul>, <li>, <br>) so it is beautifully readable in an email client.\n"
        "5. Keep the tone warm, professional, and agritech-focused."
    )

    llm_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Customer Name: {name}\n"
                f"Audience Role: {category.capitalize()}\n\n"
                f"CONVERSATION TRANSCRIPT:\n{transcript_str}"
            )
        }
    ]

    try:
        # Call the LLM with low temperature for accurate factual summarization
        result = await _call_llm(llm_messages, max_tokens=400, temperature=0.2)
        summary_reply = result.get("reply", "").strip()
        
        # If the fallback model was used due to an error, we default to the simple transcript
        if result.get("model") == "error_fallback":
            return format_simple_transcript(messages, name, category)
            
        header = (
            f"<b>Varsapradaya Conversation Summary</b><br>"
            f"<b>Customer Name:</b> {name}<br>"
            f"<b>Audience Role:</b> {category.capitalize()}<br>"
            f"---------------------------------------------<br><br>"
        )
        return header + summary_reply
        
    except Exception as e:
        logger.error(f"Failed to generate LLM summary: {e}")
        # Fallback to simple transcript to ensure the email is still sent
        return format_simple_transcript(messages, name, category)

async def send_transcript_email(session_id: str, email: str, name: str, category: str):
    """
    Main background task. Fetches conversation history, builds the LLM summary,
    and calls the external email API.
    """
    start_time = time.time()
    logger.info(f"Triggering email summary task for session: {session_id} to: {email}")
    
    try:
        # 1. Fetch transcript messages
        async with AsyncSessionLocal() as db:
            messages = await get_all_session_messages(db, session_id)
            
        if not messages:
            logger.warning(f"No messages found for session: {session_id}. Aborting email summary.")
            return

        # 2. Generate a clean LLM summary of the conversation
        body_content = await generate_conversation_summary(messages, name, category)
        
        # 3. Formulate the external API payload
        payload = {
            "to": email,
            "subject": f"🌿 Varsapradaya Chat Summary — {name}",
            "body": body_content,
            "isBodyHtml": True
        }
        
        # 4. Post with retries
        api_result = await call_email_api_with_retry(payload)
        latency_ms = int((time.time() - start_time) * 1000)
        
        # 5. Log audit in database
        async with AsyncSessionLocal() as db:
            await write_log(
                db,
                level="INFO",
                event="email_sent",
                message=f"Summary email successfully sent to {email} in {latency_ms}ms",
                session_id=session_id,
                meta={
                    "recipient": email,
                    "attempts": api_result["attempts"],
                    "latency_ms": latency_ms,
                    "status": "delivered"
                }
            )
        logger.info(f"Summary email successfully sent to {email}")
        
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Failed to execute email task: {e}")
        
        # Log failure in database
        try:
            async with AsyncSessionLocal() as db:
                await write_log(
                    db,
                    level="ERROR",
                    event="email_failed",
                    message=f"Failed to send email to {email}: {str(e)}",
                    session_id=session_id,
                    meta={
                        "recipient": email,
                        "latency_ms": latency_ms,
                        "status": "failed",
                        "error": str(e)
                    }
                )
        except Exception as db_err:
            logger.error(f"Could not log email failure to DB: {db_err}")

FOLLOWUP_API_URL = "https://api-mobile.farmfuture.io/api/User/SendPostChatFollowup"

async def call_followup_api_with_retry(payload: dict, max_retries: int = 3, initial_delay: float = 1.0) -> dict:
    """
    Asynchronously POSTs to the followup endpoint with retry logic and exponential backoff.
    """
    delay = initial_delay
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(1, max_retries + 1):
            try:
                response = await client.post(FOLLOWUP_API_URL, json=payload)
                if response.status_code in [200, 201, 202, 204]:
                    return {"status": "success", "attempts": attempt, "code": response.status_code}
                
                logger.warning(
                    f"Followup API returned status {response.status_code} (attempt {attempt}/{max_retries})"
                )
            except httpx.RequestError as exc:
                logger.warning(
                    f"Network error calling Followup API: {exc} (attempt {attempt}/{max_retries})"
                )
            
            if attempt < max_retries:
                sleep_time = delay * (2 ** (attempt - 1)) + (time.time() % 0.5)
                await asyncio.sleep(sleep_time)
                
        raise RuntimeError(f"Failed to trigger followup after {max_retries} attempts.")

async def trigger_post_chat_followup(phone: str, name: str, session_id: str = None):
    """
    Sends a POST request to trigger the WhatsApp/SMS followup in the background.
    """
    if not phone:
        logger.warning("No phone number available for post-chat followup.")
        return

    logger.info(f"Triggering post-chat followup for: {name} ({phone})")
    try:
        payload = {
            "mobileNumber": phone,
            "name": name
        }
        
        api_result = await call_followup_api_with_retry(payload)
        
        # Log audit in database
        async with AsyncSessionLocal() as db:
            await write_log(
                db,
                level="INFO",
                event="followup_sent",
                message=f"Post-chat followup successfully triggered for {name} ({phone})",
                session_id=session_id,
                meta={
                    "phone": phone,
                    "name": name,
                    "attempts": api_result["attempts"],
                    "status": "triggered"
                }
            )
        logger.info(f"Post-chat followup successfully triggered for {name} ({phone})")
        
    except Exception as e:
        logger.error(f"Failed to execute followup task: {e}")
        # Log failure in database
        try:
            async with AsyncSessionLocal() as db:
                await write_log(
                    db,
                    level="ERROR",
                    event="followup_failed",
                    message=f"Failed to trigger followup for {name} ({phone}): {str(e)}",
                    session_id=session_id,
                    meta={
                        "phone": phone,
                        "name": name,
                        "status": "failed",
                        "error": str(e)
                    }
                )
        except Exception as db_err:
            logger.error(f"Could not log followup failure to DB: {db_err}")

