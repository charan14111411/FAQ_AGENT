import os
import httpx
import asyncio
import time
from sqlalchemy import text
from app.db import AsyncSessionLocal, write_log
from app.logger import get_logger

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

def get_html_thank_you_template(name: str) -> str:
    """
    Returns the user's custom HTML thank-you template with name interpolated.
    """
    return f"""<!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="x-apple-disable-message-reformatting">
  <!-- Safest dark-mode path: declare light so Apple Mail / iOS won't auto-invert and mangle the palette -->
  <meta name="color-scheme" content="light">
  <meta name="supported-color-schemes" content="light">
  <title>Thank you for your interest — Varsapradaya</title>
  <style>
    :root {{ color-scheme: light; supported-color-schemes: light; }}
 
    /* Reset */
    body {{ margin: 0; padding: 0; width: 100% !important; }}
    table {{ border-collapse: collapse; }}
    img {{ border: 0; line-height: 100%; outline: none; text-decoration: none; }}
 
    /* Link hover (progressive enhancement, modern clients) */
    a.link {{ transition: opacity .2s ease; }}
    a.link:hover {{ opacity: .7 !important; }}
 
    /* Mobile */
    @media only screen and (max-width: 600px) {{
      .card        {{ width: 100% !important; border-radius: 14px !important; }}
      .gutter      {{ padding-left: 24px !important; padding-right: 24px !important; }}
      .body-pad    {{ padding-top: 36px !important; padding-bottom: 36px !important; }}
      .header-pad  {{ padding-top: 32px !important; padding-bottom: 32px !important; }}
      .wordmark    {{ font-size: 20px !important; letter-spacing: 3px !important; }}
      .greeting    {{ font-size: 22px !important; }}
      .contact-pad {{ padding: 22px 20px !important; }}
    }}
 
    /* Gentle dark-mode courtesy for clients that ignore the opt-out */
    @media (prefers-color-scheme: dark) {{
      .bg-page  {{ background-color: #10140f !important; }}
      .card     {{ background-color: #ffffff !important; }}
    }}
  </style>
</head>
 
<body class="bg-page" style="margin:0; padding:0; background-color:#f4f7f2;">
 
  <!-- Preheader: shows in inbox preview, hidden in the body -->
  <div style="display:none; max-height:0; overflow:hidden; mso-hide:all; font-size:1px; line-height:1px; color:#f4f7f2; opacity:0;">
    Thank you for connecting with Varsapradaya. Here is our direct support contact info.
    &zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;
  </div>
 
  <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" class="bg-page" style="background-color:#f4f7f2;">
    <tr>
      <td align="center" style="padding:56px 16px;">
 
        <!-- Card -->
        <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="600" class="card" style="width:600px; max-width:600px; background-color:#ffffff; border-radius:18px; overflow:hidden; box-shadow:0 16px 40px rgba(33,46,28,0.06); border:1px solid rgba(78,109,61,0.12);">
 
          <!-- Header band with Logo on Left of Wordmark -->
          <tr>
            <td class="header-pad gutter" style="background-color:#ffffff; padding:40px 45px 30px; border-bottom:1px solid rgba(78,109,61,0.08); text-align:center;">
              <table role="presentation" border="0" cellpadding="0" cellspacing="0" align="center" style="margin:0 auto;">
                <tr>
                  <td style="vertical-align:middle; padding-right:14px; line-height:0;">
                    <img src="https://varsapradaya.com/assets/images/logo.png" width="40" height="37" alt="Varsapradaya Logo" style="display:block; border:0; outline:none; text-decoration:none;">
                  </td>
                  <td style="vertical-align:middle;">
                    <div class="wordmark" style="font-family:Georgia,'Times New Roman',Times,serif; font-size:25px; font-weight:700; color:#243420; letter-spacing:6px; text-transform:uppercase; line-height:1.2; margin:0;">
                      Varsapradaya
                    </div>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
 
          <!-- Body -->
          <tr>
            <td class="body-pad gutter" style="padding:48px 45px 40px;">
 
              <!-- Greeting -->
              <div class="greeting" style="font-family:Georgia,'Times New Roman',Times,serif; font-size:24px; font-weight:700; color:#243420; line-height:1.25; margin-bottom:20px;">
                Hi {name},
              </div>
 
              <!-- Intro -->
              <p style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; font-size:15px; line-height:1.75; color:#3a4a34; margin:0 0 32px 0;">
                Thank you for showing interest in Varsapradaya. We are committed to revolutionizing high-value plantation agriculture with state-of-the-art precision technology — helping growers, corporate partners, and investors make smarter, data-driven decisions.
              </p>
 
              <!-- Contact card -->
              <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="background-color:#f4f7f2; border-left:4px solid #4e6d3d; border-radius:4px 14px 14px 4px;">
                <tr>
                  <td class="contact-pad" style="padding:24px 28px;">
                    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; font-size:11px; font-weight:700; color:#4e6d3d; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:12px;">
                      Contact Channels
                    </div>
                    <p style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; font-size:14px; line-height:1.65; color:#3a4a34; margin:0 0 16px 0;">
                      If you have any further queries, are interested in collaboration, or would like to explore our solutions, please reach out to our dedicated team:
                    </p>
 
                    <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%">
                      <tr>
                        <td style="padding:4px 0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; font-size:14px; color:#243420;">
                          <span style="font-weight:700; color:#243420;">Email&nbsp;&nbsp;</span>
                          <a class="link" href="mailto:info@varsapradaya.com" style="color:#4e6d3d; text-decoration:none; font-weight:600; border-bottom:1px solid rgba(78,109,61,0.5);">info@varsapradaya.com</a>
                        </td>
                      </tr>
                      <tr>
                        <td style="padding:6px 0 2px; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; font-size:14px; color:#243420;">
                          <span style="font-weight:700; color:#243420;">Phone&nbsp;&nbsp;</span>
                          <a class="link" href="tel:+91 70323 25050" style="color:#4e6d3d; text-decoration:none; font-weight:600;">+91 70323 25050</a>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>
 
              <!-- Sign-off -->
              <p style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; font-size:15px; line-height:1.6; color:#3a4a34; margin:32px 0 0 0;">
                Warm regards,<br>
                <span style="font-family:Georgia,'Times New Roman',Times,serif; font-weight:700; color:#4e6d3d;">The Varsapradaya Team</span>
              </p>
 
            </td>
          </tr>
 
          <!-- Footer -->
          <tr>
            <td class="gutter" style="background-color:#243420; padding:32px 45px; text-align:center;">
              <div style="font-family:Georgia,'Times New Roman',Times,serif; font-size:14px; font-weight:700; color:#ffffff; letter-spacing:2px; text-transform:uppercase;">
                Varsapradaya
              </div>
              <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; font-size:11px; color:#a3bfa3; line-height:1.5; margin-top:8px;">
                &copy; 2026 Varsapradaya. All rights reserved.
              </div>
            </td>
          </tr>
 
        </table>
        <!-- /Card -->
 
      </td>
    </tr>
  </table>
 
</body>
</html>"""


async def send_transcript_email(session_id: str, email: str, name: str, category: str):
    """
    Main background task. Constructs the custom HTML thank-you email and calls
    the external email API.
    """
    start_time = time.time()
    user_id = None
    if session_id:
        try:
            async with AsyncSessionLocal() as db:
                res = await db.execute(
                    text("SELECT user_id FROM sessions WHERE id = :session_id"), 
                    {"session_id": session_id}
                )
                row = res.fetchone()
                if row:
                    user_id = row[0]
        except Exception as db_err:
            logger.error(f"Failed to fetch user_id for logging: {db_err}")

    logger.info(
        f"Triggering email summary task for session: {session_id} to: {email}",
        extra={"session_id": session_id, "user_id": user_id}
    )
    
    try:
        # Generate the premium HTML thank-you content
        body_content = get_html_thank_you_template(name)
        
        # Formulate the external API payload
        payload = {
            "to": email,
            "subject": f"Thank you for showing interest in Varsapradaya — {name}",
            "body": body_content,
            "isBodyHtml": True
        }
        
        # Post with retries
        api_result = await call_email_api_with_retry(payload)
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Log audit in database
        async with AsyncSessionLocal() as db:
            await write_log(
                db,
                level="INFO",
                event="email_sent",
                message=f"Summary email successfully sent to {email} in {latency_ms}ms",
                session_id=session_id,
                user_id=user_id,
                meta={
                    "recipient": email,
                    "attempts": api_result["attempts"],
                    "latency_ms": latency_ms,
                    "status": "delivered"
                }
            )
        logger.info(
            f"Summary email successfully sent to {email}",
            extra={"session_id": session_id, "user_id": user_id}
        )
        
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"Failed to execute email task: {e}",
            extra={"session_id": session_id, "user_id": user_id}
        )
        
        # Log failure in database
        try:
            async with AsyncSessionLocal() as db:
                await write_log(
                    db,
                    level="ERROR",
                    event="email_failed",
                    message=f"Failed to send email to {email}: {str(e)}",
                    session_id=session_id,
                    user_id=user_id,
                    meta={
                        "recipient": email,
                        "latency_ms": latency_ms,
                        "status": "failed",
                        "error": str(e)
                    }
                )
        except Exception as db_err:
            logger.error(
                f"Could not log email failure to DB: {db_err}",
                extra={"session_id": session_id, "user_id": user_id}
            )

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

    user_id = None
    if session_id:
        try:
            async with AsyncSessionLocal() as db:
                res = await db.execute(
                    text("SELECT user_id FROM sessions WHERE id = :session_id"), 
                    {"session_id": session_id}
                )
                row = res.fetchone()
                if row:
                    user_id = row[0]
        except Exception as db_err:
            logger.error(f"Failed to fetch user_id for logging: {db_err}")

    # Format phone number: strip any '+' and leading '91' country code since the external API
    # automatically prepends '91' to the payload input, preventing double country code errors.
    digits = "".join(c for c in phone if c.isdigit())
    if len(digits) == 12 and digits.startswith("91"):
        formatted_phone = digits[2:]
    elif len(digits) == 10:
        formatted_phone = digits
    else:
        formatted_phone = digits

    logger.info(
        f"Triggering post-chat followup for: {name} (original={phone}, sent={formatted_phone})",
        extra={"session_id": session_id, "user_id": user_id}
    )
    try:
        payload = {
            "mobileNumber": formatted_phone,
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
                user_id=user_id,
                meta={
                    "phone": phone,
                    "name": name,
                    "attempts": api_result["attempts"],
                    "status": "triggered"
                }
            )
        logger.info(
            f"Post-chat followup successfully triggered for {name} ({phone})",
            extra={"session_id": session_id, "user_id": user_id}
        )
        
    except Exception as e:
        logger.error(
            f"Failed to execute followup task: {e}",
            extra={"session_id": session_id, "user_id": user_id}
        )
        # Log failure in database
        try:
            async with AsyncSessionLocal() as db:
                await write_log(
                    db,
                    level="ERROR",
                    event="followup_failed",
                    message=f"Failed to trigger followup for {name} ({phone}): {str(e)}",
                    session_id=session_id,
                    user_id=user_id,
                    meta={
                        "phone": phone,
                        "name": name,
                        "status": "failed",
                        "error": str(e)
                    }
                )
        except Exception as db_err:
            logger.error(
                f"Could not log followup failure to DB: {db_err}",
                extra={"session_id": session_id, "user_id": user_id}
            )

