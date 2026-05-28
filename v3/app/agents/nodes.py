"""
nodes.py — All LangGraph node handlers for v3 master/slave architecture.

Entry flow:
  router_node (conditional entry)
    ├─ Button click  → slave node directly (fast path)
    ├─ Active agent  → continue in that slave
    └─ No agent      → master_node → dispatch → slave node
                                   └─ general/ambiguous → master answers, END

Slave flow (each slave):
  slave_node → _answer_faq_slave() → check_soft_collect → soft_collect_node? → END
"""

import re
import time
from app.agents.state import ChatState
from app.agents.base_agent import _call_llm
from app.data.farmfuture import get_farmfuture, get_master_farmfuture
from app.rag.retriever import retrieve
from app.db import (
    create_session, end_session, save_message,
    get_last_10_messages, write_log, AsyncSessionLocal,
)
from app.logger import get_logger

logger = get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────

VALID_AGENTS = {"grower_agent", "investor_agent", "corporate_agent", "general_agent"}

SWITCH_PHRASES = [
    "change category", "switch category", "change agent", "switch agent",
    "switch to", "change to", "i want to talk to", "different category",
    "talk to someone else", "change my category",
]

# Classification prompt used by master_node
MASTER_CLASSIFY_PROMPT = """You are a routing classifier for Varsapradaya, a plantation intelligence company.
Classify the user's message into EXACTLY one of these four words:
- grower   (farmers, plantation, crops, sensors, soil, fertilizer, pest, monsoon, canopy, estate, harvest)
- investor (TAM, funding, revenue, roadmap, burn rate, LTV, market, competitors, R&D, valuation, capital)
- corporate (EUDR, compliance, supply chain, dashboard, ESG, reseller, partnership, agritech, hardware, distributor)
- general  (greeting, product overview, who are you, unclear, exploratory, off-topic, help, what is this)
Reply with ONLY the single category word. No punctuation. No explanation."""


# ── ROUTER NODE ───────────────────────────────────────────────────────────────

def router_node(state: ChatState) -> str:
    """
    Conditional entry point. Returns the name of the node to run.
    Priority: button click > switch intent > active agent > master
    """
    source = state.get("source", "text")
    agent_hint = state.get("agent_hint") or ""
    active_agent = state.get("active_agent") or ""
    user_input = state.get("user_input", "").lower()

    # 1. Button click with a valid agent → fast-path to slave
    if source == "button" and agent_hint in VALID_AGENTS:
        node = agent_hint.replace("_agent", "")
        logger.info(f"Router: button click → {node}_node", extra={"event": "route_button"})
        return node

    # 2. Switch intent while in an active agent → switch handler
    if active_agent in VALID_AGENTS:
        if any(p in user_input for p in SWITCH_PHRASES):
            return "handle_switch"
        # Continue in the current slave
        node = active_agent.replace("_agent", "")
        logger.info(f"Router: continue in {node}_node", extra={"event": "route_continue"})
        return node

    # 3. No active agent → master decides
    logger.info("Router: no active agent → master_node", extra={"event": "route_master"})
    return "master"


# ── MASTER NODE ───────────────────────────────────────────────────────────────

async def master_node(state: ChatState) -> dict:
    """
    LLM-powered orchestrator.
    - Classifies the user's message into a domain.
    - If domain-specific → sets routed_to so master_dispatch sends them to a slave.
    - If general/ambiguous → answers directly with surface knowledge.
    """
    user_msg = state["user_input"]

    classify_msgs = [
        {"role": "system", "content": MASTER_CLASSIFY_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    result = await _call_llm(classify_msgs, max_tokens=10, temperature=0.0)

    # Take only the first word in case the LLM adds punctuation
    category = result["reply"].strip().lower().split()[0] if result["reply"].strip() else "general"

    if category in ("grower", "investor", "corporate"):
        logger.info(f"Master: routing to {category}_agent", extra={"event": "master_route"})
        return {
            "routed_to":     category,
            "active_agent":  f"{category}_agent",
            "master_handled": False,
        }

    # General or ambiguous → master answers directly
    surface_msgs = [
        {"role": "system", "content": get_master_farmfuture()},
        {"role": "user",   "content": user_msg},
    ]
    answer = await _call_llm(surface_msgs, max_tokens=300, temperature=0.4)
    logger.info("Master: handled directly (general)", extra={"event": "master_direct"})
    return {
        "routed_to":     None,
        "active_agent":  "general_agent",
        "master_handled": True,
        "reply":         answer["reply"],
        "agent_name":    "master_agent",
        "step":          "chatting",
    }


def master_dispatch(state: ChatState) -> str:
    """
    Conditional edge function: called after master_node.
    Returns which node to run next — a slave or END.
    """
    if state.get("master_handled"):
        return "end"
    routed = state.get("routed_to", "general")
    return routed if routed in ("grower", "investor", "corporate", "general") else "general"


# ── SHARED SLAVE HELPERS ──────────────────────────────────────────────────────

async def _slave_welcome(category: str, name: str = None) -> str:
    """Warm welcome message when user clicks a button (no question to answer yet)."""
    name_part = f" {name}" if name else ""
    welcome_prompts = {
        "grower": (
            f"Welcome{name_part}! Greet the user as a plantation grower visiting Varsapradaya. "
            "Tell them you are their dedicated Grower Assistant and ask what aspect of their "
            "plantation you can help with today — soil, sensors, pests, or pricing. 2 sentences max."
        ),
        "investor": (
            f"Welcome{name_part}! Greet the user as an investor visiting Varsapradaya. "
            "Tell them you are their Investor Relations Assistant and invite them to ask about "
            "market opportunity, business model, roadmap, or competitive moat. 2 sentences max."
        ),
        "corporate": (
            f"Welcome{name_part}! Greet the user as a corporate or agritech partner visiting Varsapradaya. "
            "Tell them you are their Corporate & Partner Specialist and ask how you can help — "
            "EUDR compliance, multi-estate dashboards, or partnership terms. 2 sentences max."
        ),
        "general": (
            f"Welcome{name_part} to Varsapradaya — precision plantation intelligence. "
            "Tell them you are here to help them explore, and briefly mention that you serve growers, "
            "investors, and corporate/agritech partners. Ask what brings them here today. 3 sentences max."
        ),
    }
    msgs = [
        {"role": "system", "content": "You are Varsapradaya, a professional plantation intelligence AI. Be warm and concise."},
        {"role": "user",   "content": welcome_prompts.get(category, welcome_prompts["general"])},
    ]
    result = await _call_llm(msgs, max_tokens=150, temperature=0.4)
    return result["reply"]


async def _ensure_session(state: ChatState, category: str, db) -> str:
    """Creates a DB session if one does not exist yet. Returns session_id."""
    if state.get("session_id"):
        return state["session_id"]

    if not state.get("user_id"):
        return None  # No user yet — session will be created after onboarding

    session = await create_session(
        db, state["user_id"], category,
        is_returning=state.get("is_returning", False),
    )
    sid = str(session.id)
    await write_log(
        db, "INFO", "session_started",
        f"Category: {category} (auto-created)",
        user_id=state.get("user_id"), session_id=session.id,
        meta={"category": category, "source": state.get("source", "text")},
    )
    return sid


async def _answer_faq_slave(state: ChatState, user_msg: str, category: str) -> dict:
    """
    Core RAG pipeline — partitioned by category.
    Used by all 4 slave nodes when handling a real question.
    """
    start_time = time.time()
    turn_count = state.get("turn_count", 0) + 1

    async with AsyncSessionLocal() as db:
        session_id = await _ensure_session(state, category, db)

        # Save user message if we have a session
        if session_id:
            await save_message(db, session_id, "user", user_msg)
            history = await get_last_10_messages(db, session_id)
        else:
            history = []

        # Partitioned RAG retrieval
        if category == "corporate":
            context = await retrieve(db, user_msg, category="corporate",
                                     top_k=4, include_agritech=True)
        elif category == "general":
            context = await retrieve(db, user_msg, category=None, top_k=2)
        else:
            context = await retrieve(db, user_msg, category=category, top_k=3)

        rag_used = bool(context and context.strip())

        # Build system prompt: persona + guardrails + retrieved context
        system_prompt = get_farmfuture(category)
        system_prompt += (
            "\n\nGUARDRAILS:\n"
            "1. DOMAIN RESTRICTION: Only answer Varsapradaya-related questions.\n"
            "2. STRICT ANTI-HALLUCINATION: If the answer cannot be found in the context below, "
            "say clearly that you don't have that specific information. Never guess.\n"
            "3. MAINTAIN PERSONA: Never say 'As an AI' or break character.\n"
            "4. NO SYSTEM ACTIONS: You are read-only. You cannot delete, book, or modify anything.\n"
        )
        if context and context.strip():
            system_prompt += f"\n\nMOST RELEVANT CONTEXT FOR THIS QUESTION:\n{context}"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[:-1] if history else [])
        messages.append({"role": "user", "content": user_msg})

        # Per-slave token and temperature config
        TOKEN_LIMITS = {"grower": 500, "investor": 700, "corporate": 600, "general": 400}
        TEMPS       = {"grower": 0.3, "investor": 0.4, "corporate": 0.3, "general": 0.5}

        result = await _call_llm(
            messages,
            max_tokens=TOKEN_LIMITS.get(category, 500),
            temperature=TEMPS.get(category, 0.3),
        )
        reply = result["reply"]
        latency_ms = int((time.time() - start_time) * 1000)

        # Persist assistant reply
        if session_id:
            await save_message(db, session_id, "assistant", reply)

        await write_log(
            db, "INFO", "agent_call", f"{category}_agent replied",
            user_id=state.get("user_id"),
            session_id=session_id,
            meta={
                "agent":         f"{category}_agent",
                "category":      category,
                "latency_ms":    latency_ms,
                "input_tokens":  result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "rag_used":      rag_used,
                "source":        state.get("source", "text"),
                "turn_count":    turn_count,
            },
        )

    return {
        "reply":        reply,
        "agent_name":   f"{category}_agent",
        "active_agent": f"{category}_agent",
        "step":         "chatting",
        "turn_count":   turn_count,
        "session_id":   session_id or state.get("session_id"),
    }


# ── SLAVE NODES ───────────────────────────────────────────────────────────────

async def grower_node(state: ChatState) -> dict:
    source = state.get("source", "text")
    turn_count = state.get("turn_count", 0)

    if source == "button" and turn_count == 0:
        reply = await _slave_welcome("grower", state.get("name"))
        return {
            "reply": reply, "agent_name": "grower_agent",
            "active_agent": "grower_agent", "step": "chatting", "turn_count": 1,
        }
    return await _answer_faq_slave(state, state["user_input"], "grower")


async def investor_node(state: ChatState) -> dict:
    source = state.get("source", "text")
    turn_count = state.get("turn_count", 0)

    if source == "button" and turn_count == 0:
        reply = await _slave_welcome("investor", state.get("name"))
        return {
            "reply": reply, "agent_name": "investor_agent",
            "active_agent": "investor_agent", "step": "chatting", "turn_count": 1,
        }
    return await _answer_faq_slave(state, state["user_input"], "investor")


async def corporate_node(state: ChatState) -> dict:
    source = state.get("source", "text")
    turn_count = state.get("turn_count", 0)

    if source == "button" and turn_count == 0:
        reply = await _slave_welcome("corporate", state.get("name"))
        return {
            "reply": reply, "agent_name": "corporate_agent",
            "active_agent": "corporate_agent", "step": "chatting", "turn_count": 1,
        }
    return await _answer_faq_slave(state, state["user_input"], "corporate")


async def general_node(state: ChatState) -> dict:
    source = state.get("source", "text")
    turn_count = state.get("turn_count", 0)

    if source == "button" and turn_count == 0:
        reply = await _slave_welcome("general", state.get("name"))
        return {
            "reply": reply, "agent_name": "general_agent",
            "active_agent": "general_agent", "step": "chatting", "turn_count": 1,
        }
    return await _answer_faq_slave(state, state["user_input"], "general")


# ── SWITCH HANDLER ────────────────────────────────────────────────────────────

async def handle_switch_node(state: ChatState) -> dict:
    """Ends current session and resets to allow category re-selection."""
    async with AsyncSessionLocal() as db:
        if state.get("session_id"):
            await end_session(db, state["session_id"])
            await write_log(
                db, "INFO", "switch_requested", "User requested agent switch",
                user_id=state.get("user_id"), session_id=state.get("session_id"),
            )

    reply = (
        "Of course! Let me connect you with the right specialist. "
        "Please click one of the buttons — **I'm a grower**, **I'm an investor**, "
        "**Corporate / Partnership**, or **Just exploring** — and I'll get you there right away."
    )
    return {
        "session_id":   None,
        "active_agent": None,
        "routed_to":    None,
        "reply":        reply,
        "agent_name":   None,
        "step":         "start",
    }


# ── SOFT ONBOARDING NODE ──────────────────────────────────────────────────────

async def soft_collect_node(state: ChatState) -> dict:
    """
    Non-blocking identity collection appended after a slave reply.
    Fires after turn 3 (ask name) or turn 6 (ask email).
    """
    turn_count = state.get("turn_count", 0)
    current_reply = state.get("reply", "")

    if not state.get("name") and turn_count == 3:
        append = "\n\n*(By the way, may I have your name so I can personalise our conversation?)*"
        return {"reply": current_reply + append, "onboarding_asked": True}

    if state.get("name") and not state.get("email") and turn_count == 6:
        append = "\n\n*(If you'd like tailored follow-up material, feel free to share your email — completely optional!)*"
        return {"reply": current_reply + append, "onboarding_asked": True}

    return {}  # Nothing to append


def check_soft_collect(state: ChatState) -> str:
    """Edge function: after each slave, decide whether to pass through soft_collect."""
    turn_count = state.get("turn_count", 0)
    onboarding_asked = state.get("onboarding_asked", False)

    if not onboarding_asked and turn_count in (3, 6):
        return "soft_collect"
    return "end"
