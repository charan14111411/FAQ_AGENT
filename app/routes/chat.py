from fastapi import APIRouter, HTTPException, status, Request, BackgroundTasks
from app.models import ChatRequest, ChatResponse
from app.logger import get_logger
from app.utils.email import send_transcript_email, trigger_post_chat_followup

logger = get_logger()
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def handle_chat(req: ChatRequest, request: Request, background_tasks: BackgroundTasks):
    """
    Single endpoint for the entire FAQ chatbot experience.

    The frontend sends:
      - thread_id: a UUID that identifies the conversation (generated once per session)
      - message:   the user's text input

    LangGraph manages the full state machine:
      onboarding → name → phone → email → category selection → FAQ answering
    """
    try:
        faq_graph = request.app.state.faq_graph
        config = {"configurable": {"thread_id": req.thread_id}}

        # Get current state
        state_snapshot = await faq_graph.aget_state(config)
        
        current_step = "start"
        session_id = None
        if state_snapshot and state_snapshot.values:
            current_step = state_snapshot.values.get("step", "start")
            session_id = state_snapshot.values.get("session_id")

        # Sync check: check if the session has been ended in the database (e.g. by auto-timeout)
        if session_id and current_step != "ended":
            try:
                from app.db import AsyncSessionLocal
                from sqlalchemy import text
                async with AsyncSessionLocal() as db:
                    query = "SELECT ended_at FROM sessions WHERE id = :session_id"
                    res = await db.execute(text(query), {"session_id": session_id})
                    row = res.fetchone()
                    if row and row[0] is not None:
                        current_step = "ended"
                        # Sync the checkpointer so it persists the ended status
                        await faq_graph.aupdate_state(config, {"step": "ended"})
            except Exception as e:
                logger.error(f"Failed to check session status in DB: {e}")

        if current_step == "ended":
            return ChatResponse(
                reply="This conversation has timed out. Please refresh the page to start a new chat.",
                step="ended",
                agent=state_snapshot.values.get("agent_name") if state_snapshot and state_snapshot.values else None
            )


        # ── Fast regex content moderation (replaces LLM call, saves 2-5 sec per turn) ──
        # Covers severe profanity, slurs, and hostile patterns. ~0ms.
        import re as _re
        _TOXIC_PATTERN = _re.compile(
            r"\b(fuck|shit|ass(?:hole)?|bitch|bastard|cunt|dick|cock|pussy|whore|"
            r"nigger|nigga|faggot|retard|idiot|moron|stupid|hate you|kill you|"
            r"die|go to hell|wtf|stfu|shut up)\b",
            _re.IGNORECASE,
        )
        is_toxic = bool(_TOXIC_PATTERN.search(req.message))

        if is_toxic:
            agent_name = state_snapshot.values.get("agent_name") if state_snapshot and state_snapshot.values else None
            from app.agents.nodes import _generate_dynamic_reply
            prompt = "The user just used severe profanity or hostile language. Politely but firmly tell them that professional language is required to continue our conversation. Keep it to one sentence."
            toxic_reply = await _generate_dynamic_reply(
                prompt,
                language_code=req.language_code,
                language_name=req.language_name,
                language_native_name=req.language_native_name
            )
            return ChatResponse(reply=toxic_reply, step=current_step, agent=agent_name)

        # Build the input — inject user_input and current step
        graph_input = {
            "user_input": req.message,
            "step": current_step,
            "thread_id": req.thread_id,
            "language_code": req.language_code,
            "language_name": req.language_name,
            "language_native_name": req.language_native_name,
        }

        # Run the graph
        result = await faq_graph.ainvoke(graph_input, config=config)

        # --- DEBUG LOGGING ---
        print(f"\n{'='*50}")
        print(f"[CART] LANGGRAPH MEMORY FOR THREAD: {req.thread_id}")
        for key, value in result.items():
            if key not in ["reply", "user_input"]:  # Skip long text fields to make it readable
                print(f"  -> {key}: {value}")
        print(f"{'='*50}\n")

        # Check if conversation just ended, trigger background email & followup tasks
        if result.get("step") == "ended":
            session_id = result.get("session_id")
            user_id = result.get("user_id")
            email = result.get("email")
            phone = result.get("phone")
            name = result.get("name", "Explorer")
            category = result.get("category", "exploring")
            
            if session_id and email:
                background_tasks.add_task(
                    send_transcript_email,
                    session_id=session_id,
                    email=email,
                    name=name,
                    category=category
                )
                logger.info(
                    f"Queued transcript email background task for session {session_id} to {email}",
                    extra={"session_id": session_id, "user_id": user_id}
                )

            if phone:
                background_tasks.add_task(
                    trigger_post_chat_followup,
                    phone=phone,
                    name=name,
                    session_id=session_id
                )
                logger.info(
                    f"Queued post-chat followup background task for session {session_id} to {phone}",
                    extra={"session_id": session_id, "user_id": user_id}
                )

        return ChatResponse(
            reply=result.get("reply", ""),
            step=result.get("step", "start"),
            agent=result.get("agent_name"),
        )

    except Exception as e:
        logger.error(f"Error in LangGraph chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your message.",
        )
