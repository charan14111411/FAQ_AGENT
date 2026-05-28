from fastapi import APIRouter, HTTPException, status, Request
from app.models import ChatRequest, ChatResponse
from app.logger import get_logger

logger = get_logger()
router = APIRouter()

VALID_AGENT_HINTS = {"grower_agent", "investor_agent", "corporate_agent", "general_agent"}


@router.post("/chat", response_model=ChatResponse)
async def handle_chat(req: ChatRequest, request: Request):
    """
    Unified chat endpoint for v3 master/slave architecture.

    source="button" + agent_hint → fast-path directly to a slave agent (no master call).
    source="text"               → master agent classifies and routes or answers directly.
    """
    try:
        faq_graph = request.app.state.faq_graph
        config = {"configurable": {"thread_id": req.thread_id}}

        # ── Content moderation (pre-filter) ──────────────────────────────────
        from app.agents.base_agent import _call_llm
        mod_result = await _call_llm(
            messages=[
                {"role": "system", "content": "You are a content moderation bot. If the message contains severe profanity, insults, or offensive language reply EXACTLY 'TOXIC'. Otherwise reply 'SAFE'."},
                {"role": "user",   "content": req.message},
            ],
            max_tokens=10,
            temperature=0.0,
        )

        # error_fallback model means Groq itself rejected it — treat as toxic
        is_toxic = (
            mod_result.get("model") == "error_fallback"
            or "TOXIC" in mod_result.get("reply", "").upper()
        )

        if is_toxic:
            # Dynamically generated refusal — no hardcoded strings
            from app.agents.nodes import _slave_welcome  # reuse LLM helper
            refusal = await _call_llm(
                messages=[
                    {"role": "system", "content": "You are Varsapradaya. The user just used severe profanity. Politely but firmly tell them professional language is required. One sentence only."},
                    {"role": "user",   "content": req.message},
                ],
                max_tokens=60,
                temperature=0.3,
            )
            # Read current step from snapshot (don't advance state)
            snap = await faq_graph.aget_state(config)
            current_step = "start"
            current_agent = None
            if snap and snap.values:
                current_step  = snap.values.get("step", "start")
                current_agent = snap.values.get("agent_name")

            return ChatResponse(
                reply=refusal["reply"],
                step=current_step,
                agent=current_agent,
            )

        # ── Validate agent_hint ───────────────────────────────────────────────
        agent_hint = req.agent_hint
        if req.source == "button" and agent_hint not in VALID_AGENT_HINTS:
            agent_hint = None  # Fall through to master if hint is invalid

        # ── Build graph input ─────────────────────────────────────────────────
        graph_input = {
            "user_input": req.message,
            "source":     req.source,
            "agent_hint": agent_hint,
            "thread_id":  req.thread_id,
        }

        # ── Run the graph ─────────────────────────────────────────────────────
        result = await faq_graph.ainvoke(graph_input, config=config)

        # ── Debug dump ────────────────────────────────────────────────────────
        print(f"\n{'='*55}")
        print(f"🤖 THREAD: {req.thread_id} | SOURCE: {req.source} | AGENT_HINT: {agent_hint}")
        for k, v in result.items():
            if k not in ("reply", "user_input"):
                print(f"   {k}: {v}")
        print(f"{'='*55}\n")

        return ChatResponse(
            reply=result.get("reply", ""),
            step=result.get("step", "start"),
            agent=result.get("agent_name"),
            master_handled=result.get("master_handled", False),
            routed_to=result.get("routed_to"),
        )

    except Exception as e:
        logger.error(f"Chat handler error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred processing your message.",
        )


@router.get("/agents")
async def list_agents():
    """Returns available agent categories for dynamic frontend button rendering."""
    return {
        "agents": [
            {"id": "grower_agent",    "label": "I'm a grower",          "description": "Farmers, plantation owners, estate managers"},
            {"id": "investor_agent",  "label": "I'm an investor",        "description": "VCs, analysts, financial professionals"},
            {"id": "corporate_agent", "label": "Corporate / Partnership","description": "Executives, compliance, agritech resellers"},
            {"id": "general_agent",   "label": "Just exploring",         "description": "General visitors and explorers"},
        ]
    }
