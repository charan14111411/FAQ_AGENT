import asyncio
from app.db import AsyncSessionLocal, end_session, fetch_inactive_sessions
from app.utils.email import send_transcript_email, trigger_post_chat_followup
from app.logger import get_logger
from app.config import settings

logger = get_logger()

# Maintain strong references to active background tasks to prevent GC reclamation
_background_tasks = set()

async def monitor_inactive_sessions_loop():
    """
    Continuous background loop checking for inactive sessions.
    Auto-closes idle sessions after 30 minutes and sends summary transcripts.
    Checks every 60 seconds.
    """
    logger.info("Varsapradaya idle session background monitor loop started.")
    while True:
        try:
            async with AsyncSessionLocal() as db:
                # 1. Fetch sessions idle for the configured timeout
                inactive_sessions = await fetch_inactive_sessions(db, threshold_minutes=settings.SESSION_TIMEOUT_MINUTES)
                
                for session in inactive_sessions:
                    session_id = str(session.session_id)
                    category = session.category
                    email = session.email
                    name = session.name
                    phone = session.phone
                    
                    logger.info(
                        f"Auto-Timeout Triggered: Session {session_id} for user '{name}' ({email}) has been idle for > {settings.SESSION_TIMEOUT_MINUTES} minutes."
                    )
                    
                    # 2. Immediately close the session in the database (acts as a processing lock)
                    await end_session(db, session_id)
                    
                    # 3. Trigger email transcript & followup asynchronously (non-blocking task) if contact details are present
                    if email:
                        email_task = asyncio.create_task(
                            send_transcript_email(
                                session_id=session_id,
                                email=email,
                                name=name,
                                category=category
                            )
                        )
                        _background_tasks.add(email_task)
                        email_task.add_done_callback(_background_tasks.discard)
                    else:
                        logger.info(f"Skipping transcript email for timed-out session {session_id} because email is missing.")
                    
                    if phone:
                        followup_task = asyncio.create_task(
                            trigger_post_chat_followup(
                                phone=phone,
                                name=name,
                                session_id=session_id
                            )
                        )
                        _background_tasks.add(followup_task)
                        followup_task.add_done_callback(_background_tasks.discard)
                    else:
                        logger.info(f"Skipping WhatsApp/SMS followup for timed-out session {session_id} because phone is missing.")
                    
        except asyncio.CancelledError:
            # Handle cancellation gracefully during app shutdown
            logger.info("Varsapradaya idle session background monitor loop cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in idle session background monitor loop: {e}")
            
        # 4. Wait 60 seconds before scanning the database again
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            logger.info("Varsapradaya idle session background monitor loop cancelled during sleep.")
            break
