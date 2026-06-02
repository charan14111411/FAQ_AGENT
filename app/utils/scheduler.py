import asyncio
from app.db import AsyncSessionLocal, end_session, fetch_inactive_sessions
from app.utils.email import send_transcript_email, trigger_post_chat_followup
from app.logger import get_logger

logger = get_logger()

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
                # 1. Fetch sessions idle for 30 minutes
                inactive_sessions = await fetch_inactive_sessions(db, threshold_minutes=5)
                
                for session in inactive_sessions:
                    session_id = str(session.session_id)
                    category = session.category
                    email = session.email
                    name = session.name
                    phone = session.phone
                    
                    logger.info(
                        f"Auto-Timeout Triggered: Session {session_id} for user '{name}' ({email}) has been idle for > 30 minutes."
                    )
                    
                    # 2. Immediately close the session in the database (acts as a processing lock)
                    await end_session(db, session_id)
                    
                    # 3. Trigger email transcript & followup asynchronously (non-blocking task)
                    asyncio.create_task(
                        send_transcript_email(
                            session_id=session_id,
                            email=email,
                            name=name,
                            category=category
                        )
                    )
                    
                    if phone:
                        asyncio.create_task(
                            trigger_post_chat_followup(
                                phone=phone,
                                name=name,
                                session_id=session_id
                            )
                        )
                    
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
