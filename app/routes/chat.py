from fastapi import APIRouter, HTTPException, status, Request
from app.models import ChatRequest, ChatResponse
from app.logger import get_logger

logger = get_logger()
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def handle_chat(req: ChatRequest, request: Request):
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
        if state_snapshot and state_snapshot.values:
            current_step = state_snapshot.values.get("step", "start")

        if current_step == "ended":
            return ChatResponse(
                reply="This conversation has ended. Please refresh the page to start a new chat.",
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
            toxic_reply = await _generate_dynamic_reply(prompt)
            return ChatResponse(reply=toxic_reply, step=current_step, agent=agent_name)

        # Build the input — inject user_input and current step
        graph_input = {
            "user_input": req.message,
            "step": current_step,
            "thread_id": req.thread_id,
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
