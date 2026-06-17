from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import asyncio
import smtplib
import ssl
import httpx
from app.config import settings
from app.db import get_db
from app.logger import get_logger

logger = get_logger()
router = APIRouter()

@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "db_connected": True
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "db_connected": False
        }


def _check_smtp_sync():
    """Connects to the SMTP server and authenticates, without sending mail."""
    host = settings.INFO_SMTP_HOST
    port = settings.INFO_SMTP_PORT
    user = settings.INFO_SMTP_USER
    password = settings.INFO_SMTP_PASSWORD
    if not (host and port and user and password):
        return {"configured": False, "error_message": "SMTP credentials not configured"}

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=5) as client:
        client.ehlo()
        client.starttls(context=context)
        client.ehlo()
        client.login(user, password)
    return {"configured": True, "connected": True, "authenticated": True, "host": host, "port": port}


@router.get("/diagnose-network")
async def diagnose_network():
    results = {}

    # 1. Verify SMTP connectivity + authentication (email is now sent directly via SMTP).
    try:
        results["email_smtp"] = await asyncio.to_thread(_check_smtp_sync)
    except Exception as e:
        results["email_smtp"] = {
            "configured": bool(settings.INFO_SMTP_USER),
            "connected": False,
            "error_type": type(e).__name__,
            "error_message": str(e),
        }

    # 2. Verify reachability of the WhatsApp Cloud (Graph) API (followup is now sent directly).
    if not (settings.WHATSAPP_ACCESS_TOKEN and settings.WHATSAPP_PHONE_NUMBER_ID):
        results["whatsapp_api"] = {"configured": False, "error_message": "WhatsApp credentials not configured"}
    else:
        # GET the phone number node; a 200 confirms token + reachability without sending a message.
        url = (
            f"https://graph.facebook.com/{settings.WHATSAPP_API_VERSION}"
            f"/{settings.WHATSAPP_PHONE_NUMBER_ID}"
        )
        try:
            async with httpx.AsyncClient(timeout=5.0, trust_env=True) as client:
                response = await client.get(
                    url, headers={"Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}"}
                )
                results["whatsapp_api"] = {
                    "configured": True,
                    "status_code": response.status_code,
                    "body": response.text[:200],
                }
        except Exception as e:
            results["whatsapp_api"] = {
                "configured": True,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }

    return results
 
 