"""
Check which sessions were created AFTER the CRM integration (migration 012),
and look at their prospect_id status with exact timestamps.
"""
import asyncio
import asyncpg
import httpx
from app.config import settings

# OLD CODE (Commented out):
# CRM_URL = "https://dev.businesscentral.in/rest/telecaller/backoffice/createProspect"
CRM_URL = "http://localhost:8000/api/createprospect"

async def run():
    url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        # Show sessions with timestamps to identify which were after CRM was added
        print("=" * 70)
        print("ALL SESSIONS (newest first) with prospect_id status")
        print("=" * 70)
        rows = await conn.fetch("""
            SELECT s.id, s.prospect_id, s.is_returning, s.started_at,
                   u.name, u.phone
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            ORDER BY s.started_at DESC
        """)
        for r in rows:
            status = "[OK]" if r["prospect_id"] else "[NULL]"
            ts = str(r["started_at"])[:19]
            print(f"  {status} | {ts} | prospect={r['prospect_id']} | name={r['name']} | phone={r['phone']}")

        # Test CRM with the exact phone number that's failing
        print("\n" + "=" * 70)
        print("TEST CRM WITH phone=7569444260 (Charan's number)")
        print("=" * 70)
        # OLD CODE (Commented out):
        # payload = {"name": "Charan", "mobile": "7569444260", "source": "FAQchat"}
        payload = {"name": "Charan", "phone_number": "7569444260", "source": "FAQchat"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(CRM_URL, json=payload)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")

    finally:
        await conn.close()

asyncio.run(run())
