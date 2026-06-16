from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import httpx
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


@router.get("/diagnose-network")
async def diagnose_network():
    results = {}
   
    # 1. Test DNS and HTTP to api-mobile.farmfuture.io (Email)
    email_url = "https://api-mobile.farmfuture.io/api/email/send"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(email_url, json={
                "to": "test@example.com",
                "subject": "Diagnostic test",
                "body": "Hello",
                "isBodyHtml": False
            })
            results["email_api"] = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text[:200]
            }
    except Exception as e:
        results["email_api"] = {
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
 
    # 2. Test DNS and HTTP to api-mobile.farmfuture.io (Followup)
    followup_url = "https://api-mobile.farmfuture.io/api/User/SendPostChatFollowup"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(followup_url, json={
                "mobileNumber": "0000000000",
                "name": "Test"
            })
            results["followup_api"] = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text[:200]
            }
    except Exception as e:
        results["followup_api"] = {
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
       
    return results
 
 