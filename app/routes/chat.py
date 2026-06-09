import re as _re
from fastapi import APIRouter, HTTPException, status, Request, BackgroundTasks
from app.models import ChatRequest, ChatResponse
from app.logger import get_logger
from app.utils.email import send_transcript_email, trigger_post_chat_followup

logger = get_logger()
router = APIRouter()


# ---------------------------------------------------------------------------
# Language code → canonical name used throughout the backend
# Supported ISO 639-1 codes sent by the production frontend
# ---------------------------------------------------------------------------

_LANG_CODE_TO_NAME: dict[str, str] = {
    "en": "english",
    "hi": "hindi",
    "ta": "tamil",
    "te": "telugu",
    "kn": "kannada",
    "ml": "malayalam",
    "ur": "urdu",
}


def _resolve_language(req: ChatRequest) -> tuple[str, str | None]:
    """
    Returns (canonical_language_name, language_native_name).

    Priority:
      1. language_code  (ISO 639-1 from production frontend, e.g. 'te')
      2. language       (legacy full-name from HTML mockup, e.g. 'telugu')
      3. Default        → ('english', None)

    canonical_language_name  : 'telugu', 'hindi', etc.
    language_native_name     : 'తెలుగు', 'हिन्दी', etc.  Used verbatim in LLM prompts.
    """
    if req.language_code:
        code = req.language_code.strip().lower()
        name = _LANG_CODE_TO_NAME.get(code, "english")
        return name, (req.language_native_name or None)

    if req.language and req.language.strip().lower() not in ("", "english", "en"):
        return req.language.strip().lower(), (req.language_native_name or None)

    return "english", None


# ---------------------------------------------------------------------------
# Strip language prefix injected by the production frontend
# e.g. "Please reply in Telugu (తెలుగు).\nUser message: my name is ratan tata"
#   →  "my name is ratan tata"
# ---------------------------------------------------------------------------

_PREFIX_PATTERN = _re.compile(
    r"^Please reply in .+?\.\s*\nUser message:\s*",
    _re.IGNORECASE | _re.DOTALL,
)


def _strip_language_prefix(message: str) -> str:
    match = _PREFIX_PATTERN.match(message)
    return message[match.end():].strip() if match else message.strip()


# ---------------------------------------------------------------------------
# Toxicity pattern (compiled once)
# ---------------------------------------------------------------------------

_TOXIC_PATTERN = _re.compile(
    r"\b(fuck|shit|ass(?:hole)?|bitch|bastard|cunt|dick|cock|pussy|whore|"
    r"nigger|nigga|faggot|retard|idiot|moron|stupid|hate you|kill you|"
    r"die|go to hell|wtf|stfu|shut up)\b",
    _re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Main chat endpoint
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def handle_chat(req: ChatRequest, request: Request, background_tasks: BackgroundTasks):
    """
    Single endpoint for the entire FAQ chatbot experience.

    Supports two frontend formats:
      Legacy (HTML mockup):   { thread_id, message, language: "telugu" }
      Production frontend:    { thread_id, message, language_code: "te",
                                language_name: "Telugu", language_native_name: "తెలుగు" }

    The production frontend may also prepend a language instruction to the message:
      "Please reply in Telugu (తెలుగు).\\nUser message: <actual text>"
    This prefix is automatically stripped before processing.

    LangGraph manages the full state machine:
      onboarding → name → phone → email → category selection → FAQ answering
    """
    try:
        faq_graph = request.app.state.faq_graph
        config = {"configurable": {"thread_id": req.thread_id}}

        # ── 1. Resolve language from request ─────────────────────────────────
        language, language_native_name = _resolve_language(req)

        # ── 2. Strip language prefix injected by production frontend ─────────
        clean_message = _strip_language_prefix(req.message)

        # ── 3. Get current graph state ───────────────────────────────────────
        state_snapshot = await faq_graph.aget_state(config)

        current_step = "start"
        session_id = None
        if state_snapshot and state_snapshot.values:
            current_step = state_snapshot.values.get("step", "start")
            session_id = state_snapshot.values.get("session_id")

        # ── 4. Sync check: session already ended in DB? ───────────────────────
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
                        await faq_graph.aupdate_state(config, {"step": "ended"})
            except Exception as e:
                logger.error(f"Failed to check session status in DB: {e}")

        if current_step == "ended":
            return ChatResponse(
                reply="This conversation has timed out. Please refresh the page to start a new chat.",
                step="ended",
                agent=state_snapshot.values.get("agent_name") if state_snapshot and state_snapshot.values else None
            )

        # ── 5. Content moderation (runs on clean message, not prefixed raw) ──
        if _TOXIC_PATTERN.search(clean_message):
            agent_name = state_snapshot.values.get("agent_name") if state_snapshot and state_snapshot.values else None
            from app.agents.nodes import _generate_dynamic_reply
            prompt = (
                "The user just used severe profanity or hostile language. "
                "Politely but firmly tell them that professional language is required "
                "to continue our conversation. Keep it to one sentence."
            )
            toxic_reply = await _generate_dynamic_reply(prompt, language=language)
            return ChatResponse(reply=toxic_reply, step=current_step, agent=agent_name)

        # ── 6. Build graph input ──────────────────────────────────────────────
        graph_input = {
            "user_input": clean_message,
            "step": current_step,
            "thread_id": req.thread_id,
            "language": language,
            "language_native_name": language_native_name,  # e.g. "తెలుగు" — used in LLM prompt
        }

        # ── 7. Run the graph ──────────────────────────────────────────────────
        result = await faq_graph.ainvoke(graph_input, config=config)

        # ── 8. Debug logging ──────────────────────────────────────────────────
        print(f"\n{'='*50}")
        print(f"[CART] LANGGRAPH MEMORY FOR THREAD: {req.thread_id}")
        for key, value in result.items():
            if key not in ["reply", "user_input"]:
                print(f"  -> {key}: {value}")
        print(f"{'='*50}\n")

        # ── 9. Post-session background tasks ──────────────────────────────────
        if result.get("step") == "ended":
            session_id = result.get("session_id")
            user_id    = result.get("user_id")
            email      = result.get("email")
            phone      = result.get("phone")
            name       = result.get("name", "Explorer")
            category   = result.get("category", "exploring")

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
