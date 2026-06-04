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

# --- ACTIVE BUSINESS CENTRAL CRM PROSPECT SERVICE ---
CRM_PROSPECT_URL = "https://dev.businesscentral.in/rest/telecaller/backoffice/createProspect"
CRM_SOURCE = "FAQ_Test_chat"
CRM_TIMEOUT = 10.0  # seconds

async def create_crm_prospect(name: str, mobile: str) -> str | None:
    """
    Calls the BusinessCentral createProspect API and returns the prospectID string.
    Returns None if the call fails or the API reports an error — session creation
    is NOT blocked by a CRM failure.

    Args:
        name:   Full name of the user (collected during onboarding).
        mobile: Phone number of the user (collected during onboarding).

    Returns:
        prospectID as a string (e.g. "363"), or None on failure.
    """
    payload = {
        "name": name,
        "mobile": mobile,
        "source": CRM_SOURCE,
    }

    try:
        async with httpx.AsyncClient(timeout=CRM_TIMEOUT) as client:
            response = await client.post(CRM_PROSPECT_URL, json=payload)
            response.raise_for_status()
            data = response.json()

            inner_data = data.get("data") or {}
            prospect_id = inner_data.get("prospectID")
            if prospect_id:
                prospect_id_str = str(prospect_id)
                logger.info(
                    f"CRM prospect registered/retrieved: prospectID={prospect_id_str}, name={name}",
                    extra={"event": "crm_prospect_retrieved"},
                )
                return prospect_id_str

            # CRM says mobile already registered — prospect exists in CRM already.
            # Return sentinel so caller can look up the existing prospect_id from DB.
            msg = inner_data.get("message", "")
            if "already exists" in msg.lower() or "duplicate" in msg.lower():
                logger.info(
                    f"CRM: mobile already registered for name={name}, mobile={mobile} but no prospectID returned. Will reuse existing prospect_id via fallback.",
                    extra={"event": "crm_prospect_duplicate"},
                )
                return "DUPLICATE"

            logger.warning(
                f"CRM API returned unexpected response: {data}",
                extra={"event": "crm_prospect_unexpected"},
            )
            return None

    except httpx.TimeoutException:
        logger.error(
            f"CRM prospect API timed out after {CRM_TIMEOUT}s for name={name}",
            extra={"event": "crm_prospect_timeout"},
        )
        return None
    except httpx.HTTPStatusError as e:
        logger.error(
            f"CRM prospect API HTTP error {e.response.status_code}: {e.response.text}",
            extra={"event": "crm_prospect_http_error"},
        )
        return None
    except Exception as e:
        logger.error(
            f"CRM prospect API unexpected error: {e}",
            extra={"event": "crm_prospect_error"},
        )
        return None
