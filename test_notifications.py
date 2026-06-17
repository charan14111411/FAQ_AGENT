"""
Manual test harness for the direct SMTP email + WhatsApp followup.

Usage (from the project root, venv active):
    python test_notifications.py --email you@example.com --phone 9876543210 --name "Vinod"

Omit --email to skip the email test; omit --phone to skip the WhatsApp test.
This sends REAL messages, so use your own address / number.
"""
import sys
import asyncio
import argparse

# Required for psycopg/asyncpg compatibility on Windows (same as main.py)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.utils.email import send_email_with_retry, send_whatsapp_followup_with_retry


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", help="Recipient email address")
    parser.add_argument("--phone", help="10-digit mobile number (no country code)")
    parser.add_argument("--name", default="Tester", help="Name used in the message")
    args = parser.parse_args()

    if not args.email and not args.phone:
        parser.error("Provide at least --email or --phone")

    if args.email:
        print(f"\n[EMAIL] sending to {args.email} ...")
        try:
            result = await send_email_with_retry(
                to=args.email,
                subject=f"Test email — {args.name}",
                body="<h2>Hello!</h2><p>This is a direct SMTP test from the FAQ Agent.</p>",
                is_body_html=True,
            )
            print(f"[EMAIL] OK -> {result}")
        except Exception as e:
            print(f"[EMAIL] FAILED -> {type(e).__name__}: {e}")

    if args.phone:
        print(f"\n[WHATSAPP] sending '{args.name}' template to {args.phone} ...")
        try:
            result = await send_whatsapp_followup_with_retry(
                mobile_number=args.phone,
                name=args.name,
            )
            print(f"[WHATSAPP] OK -> {result}")
        except Exception as e:
            print(f"[WHATSAPP] FAILED -> {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
