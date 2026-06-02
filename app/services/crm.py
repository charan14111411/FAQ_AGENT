"""
CRM integration service — BusinessCentral Prospect API.

Called once per new session (new AND returning users) to register
the user as a prospect in the CRM system.

Endpoint: POST https://dev.businesscentral.in/rest/telecaller/backoffice/createProspect
Payload:  { "name": str, "mobile": str, "source": "FAQchat" }
Response: { "success": true, "data": { "message": "...", "prospectID": "363" } }
"""

import httpx
from app.logger import get_logger

logger = get_logger()

# OLD CODE (Commented out):
# CRM_PROSPECT_URL = "https://dev.businesscentral.in/rest/telecaller/backoffice/createProspect"
# CRM_SOURCE = "FAQ_Test_chat"
# CRM_TIMEOUT = 10.0  # seconds
# 
# async def create_crm_prospect(name: str, mobile: str) -> str | None:
#     """
#     Calls the BusinessCentral createProspect API and returns the prospectID string.
#     Returns None if the call fails or the API reports an error — session creation
#     is NOT blocked by a CRM failure.
# 
#     Args:
#         name:   Full name of the user (collected during onboarding).
#         mobile: Phone number of the user (collected during onboarding).
# 
#     Returns:
#         prospectID as a string (e.g. "363"), or None on failure.
#     """
#     payload = {
#         "name": name,
#         "mobile": mobile,
#         "source": CRM_SOURCE,
#     }
# 
#     try:
#         async with httpx.AsyncClient(timeout=CRM_TIMEOUT) as client:
#             response = await client.post(CRM_PROSPECT_URL, json=payload)
#             response.raise_for_status()
#             data = response.json()
# 
#             if data.get("success") and data.get("data", {}).get("prospectID"):
#                 prospect_id = str(data["data"]["prospectID"])
#                 logger.info(
#                     f"CRM prospect created: prospectID={prospect_id}, name={name}",
#                     extra={"event": "crm_prospect_created"},
#                 )
#                 return prospect_id
# 
#             # CRM says mobile already registered — prospect exists in CRM already.
#             # Return sentinel so caller can look up the existing prospect_id from DB.
#             msg = data.get("data", {}).get("message", "")
#             if "already exists" in msg.lower() or "duplicate" in msg.lower():
#                 logger.info(
#                     f"CRM: mobile already registered for name={name}, mobile={mobile}. Will reuse existing prospect_id.",
#                     extra={"event": "crm_prospect_duplicate"},
#                 )
#                 return "DUPLICATE"
# 
#             logger.warning(
#                 f"CRM API returned unexpected response: {data}",
#                 extra={"event": "crm_prospect_unexpected"},
#             )
#             return None
# 
#     except httpx.TimeoutException:
#         logger.error(
#             f"CRM prospect API timed out after {CRM_TIMEOUT}s for name={name}",
#             extra={"event": "crm_prospect_timeout"},
#         )
#         return None
#     except httpx.HTTPStatusError as e:
#         logger.error(
#             f"CRM prospect API HTTP error {e.response.status_code}: {e.response.text}",
#             extra={"event": "crm_prospect_http_error"},
#         )
#         return None
#     except Exception as e:
#         logger.error(
#             f"CRM prospect API unexpected error: {e}",
#             extra={"event": "crm_prospect_error"},
#         )
#         return None

from app.config import settings
from app.db import AsyncSessionLocal
from sqlalchemy import text

CRM_TIMEOUT = 10.0  # seconds
CRM_SOURCE = "FAQ_Test_chat"

async def create_crm_prospect_local_direct(name: str, mobile: str) -> str | None:
    """
    Direct database fallback for prospect creation when HTTP server is unreachable
    (e.g., during startup, offline scripts, or test executions).
    """
    try:
        async with AsyncSessionLocal() as db:
            # Check if prospect already exists
            res = await db.execute(
                text("SELECT prospect_id FROM prospects WHERE phone_number = :phone"),
                {"phone": mobile}
            )
            row = res.fetchone()
            if row:
                logger.info(f"Local CRM (Direct DB Fallback): phone {mobile} already registered. Reusing prospect_id={row[0]}")
                return str(row[0])
            
            # Create a new prospect
            res = await db.execute(
                text("""
                    INSERT INTO prospects (name, phone_number, source)
                    VALUES (:name, :phone, :source)
                    RETURNING prospect_id
                """),
                {"name": name, "phone": mobile, "source": CRM_SOURCE}
            )
            await db.commit()
            p_id = str(res.scalar())
            logger.info(f"Local CRM (Direct DB Fallback): created prospect_id={p_id} for phone={mobile}")
            return p_id
    except Exception as db_err:
        logger.error(f"Local CRM Direct DB Fallback failed: {db_err}")
        return None


async def create_crm_prospect(name: str, mobile: str) -> str | None:
    """
    Calls the local createprospect API to register the user as a prospect.
    If already registered, calls fetch_prospect/{phone_number} to retrieve the existing prospect ID.
    If the API server is unreachable, falls back to direct database operations.
    """
    payload = {
        "name": name,
        "phone_number": mobile,
        "source": CRM_SOURCE,
    }

    backend_url = getattr(settings, "BACKEND_URL", "http://localhost:8000")
    create_url = f"{backend_url}/api/createprospect"
    fetch_url = f"{backend_url}/api/fetch_prospect/{mobile}"

    try:
        async with httpx.AsyncClient(timeout=CRM_TIMEOUT) as client:
            logger.info(f"Local CRM: Hitting api {create_url}...")
            response = await client.post(create_url, json=payload)
            
            # Use same logic: if phone number already there, call fetch_prospect/{phone_number}
            is_duplicate = False
            if response.status_code == 400:
                try:
                    err_data = response.json()
                    detail = err_data.get("detail", "")
                    if "already existed" in detail or "already exists" in detail:
                        is_duplicate = True
                except Exception:
                    pass

            if is_duplicate:
                logger.info(f"Local CRM: Phone {mobile} is already registered. Calling fetch_prospect API...")
                fetch_response = await client.get(fetch_url)
                fetch_response.raise_for_status()
                fetch_data = fetch_response.json()
                p_id = fetch_data.get("data", {}).get("prospect_id")
                if p_id:
                    logger.info(f"Local CRM: Successfully fetched existing prospect_id={p_id} for phone={mobile}")
                    return str(p_id)
                return None

            response.raise_for_status()
            data = response.json()
            if data.get("success") and data.get("data", {}).get("prospect_id"):
                p_id = str(data["data"]["prospect_id"])
                logger.info(f"Local CRM: Successfully created prospect_id={p_id} for phone={mobile}")
                return p_id
            
            logger.warning(f"Local CRM: Unexpected response from createprospect: {data}")
            return None

    except (httpx.ConnectError, httpx.TimeoutException) as conn_err:
        logger.warning(
            f"Local CRM: Could not connect to API server at {backend_url} ({conn_err}). "
            "Falling back to direct database operations."
        )
        return await create_crm_prospect_local_direct(name, mobile)
    except Exception as e:
        logger.error(f"Local CRM error in create_crm_prospect: {e}")
        # Final fallback to direct DB just in case of any other HTTP/request lifecycle errors
        try:
            return await create_crm_prospect_local_direct(name, mobile)
        except Exception:
            return None

