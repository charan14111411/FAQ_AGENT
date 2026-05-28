import re
import time
from app.agents.state import ChatState
from app.agents.base_agent import _call_llm
from app.data.personas import get_persona
from app.rag.retriever import retrieve
from app.db import (
    find_user_by_email,
    create_user,
    get_last_session_category,
    create_session,
    end_session,
    save_message,
    get_last_10_messages,
    write_log,
    AsyncSessionLocal,
)
from app.logger import get_logger

logger = get_logger()

SWITCH_PHRASES = [
    "change category", "switch category", "change agent", "switch agent",
    "different category", "change to", "switch to", "i want to talk to",
    "talk to someone else", "change my category", "switch my category",
]

EMAIL_REGEX = r"^[^@]+@[^@]+\.[^@]+$"
PHONE_REGEX = r"^\+?[\d\s\-]{7,15}$"

CATEGORY_MENU_PROMPT = """
You MUST present these 4 role choices clearly to the user (preferably as a bulleted list or menu):
- 🌱 Grower (Farmer, planter, estate owner)
- 🏢 Corporate (Executive, compliance, supply-chain)
- 💰 Investor (VC, analyst, financial professional)
- 🔧 Agritech (Reseller, field technician, installer)
Tell them they can type their role or describe themselves and you will figure it out.
"""

async def _generate_dynamic_reply(system_prompt: str, user_input: str = "") -> str:
    """Helper to generate conversational filler dynamically without RAG."""
    
    # Base guardrails applied to every onboarding LLM call
    base_rules = (
        "You are Varsapradaya, a professional corporate assistant. "
        "CRITICAL RULE 1: Never break character. Never say 'I am a large language model' or 'I don't have feelings'. You are Varsapradaya.\n"
        "CRITICAL RULE 2: If the user's input contains profanity, insults, or offensive language, "
        "you MUST explicitly address it by politely but firmly stating that professional language is required, "
        "and then ask them to complete the current step.\n\n"
    )
    
    messages = [{"role": "system", "content": base_rules + system_prompt}]
    if user_input:
        messages.append({"role": "user", "content": user_input})
    
    # Fast, low-temp generation for onboarding
    result = await _call_llm(messages, max_tokens=250, temperature=0.3)
    return result["reply"].strip()

# ---------------------------------------------------------------------------
# ROUTER — determines which node runs next based on current step
# ---------------------------------------------------------------------------

def router_node(state: ChatState) -> str:
    """Conditional edge: returns the name of the next node to run."""
    step = state.get("step", "start")
    return step  # node names match step names exactly


# ---------------------------------------------------------------------------
# NODE: greet — first ever message, ask for name
# ---------------------------------------------------------------------------

async def greet_node(state: ChatState) -> dict:
    prompt = "You are Varsapradaya, a precision plantation intelligence AI assistant. Warmly welcome the user and politely ask for their full name."
    reply = await _generate_dynamic_reply(prompt)
    
    return {
        "reply": reply,
        "step": "await_name",
        "agent_name": None,
    }


# ---------------------------------------------------------------------------
# NODE: await_name — parse name, ask email
# ---------------------------------------------------------------------------

async def collect_name_node(state: ChatState) -> dict:
    raw = state["user_input"].strip()
    
    if not raw:
        return {"reply": "Please type something so I can help you.", "step": "await_name"}

    # Must have at least two letters and no numbers/symbols
    if len(re.sub(r'[^a-zA-Z]', '', raw)) < 2 or re.search(r'[\d\+\-\*\/=\(\)\!@#\$%\^&\*]', raw):
        prompt = f"The user entered '{raw}'. This contains numbers/symbols or is too short. Politely ask for their real full name using letters."
        reply = await _generate_dynamic_reply(prompt)
        return {"reply": reply, "step": "await_name"}

    # LLM Gibberish Detection
    check_prompt = "You are a validation bot. Is the following text a reasonable human name, or is it obvious keyboard mashing/gibberish (like 'asdfg' or 'gvufygufykvbfku')? Reply with EXACTLY 'VALID' or 'INVALID'."
    messages = [{"role": "system", "content": check_prompt}, {"role": "user", "content": raw}]
    validation_result = await _call_llm(messages, max_tokens=10, temperature=0.0)
    
    if "INVALID" in validation_result["reply"].upper():
        prompt = f"The user entered '{raw}' which looks like random keyboard mashing. Politely but firmly ask them to provide their real name to proceed."
        reply = await _generate_dynamic_reply(prompt)
        return {"reply": reply, "step": "await_name"}

    name = re.sub(r"(?i)^(i('m| am)|my name is|this is|hi[,!]?|hello[,!]?)\s*", "", raw).strip()
    name = name.title() if name else raw.title()

    prompt = f"The user just provided their name: {name}. Greet them warmly by name and ask for their phone number."
    reply = await _generate_dynamic_reply(prompt)

    return {
        "name": name,
        "reply": reply,
        "step": "await_phone",
        "phone_attempts": 0,
        "email_attempts": 0,
    }


# ---------------------------------------------------------------------------
# NODE: await_phone — validate phone, ask email
# ---------------------------------------------------------------------------

async def collect_phone_node(state: ChatState) -> dict:
    raw = state["user_input"].strip()
    attempts = state.get("phone_attempts", 0) + 1

    if not raw:
        return {"reply": "Please type something or ask to skip.", "step": "await_phone"}

    if not re.match(PHONE_REGEX, raw):
        if attempts == 1:
            prompt = f"The user replied '{raw}' instead of a valid phone number. Acknowledge it, and ask them to try entering their phone number again."
        else:
            prompt = f"The user replied '{raw}'. They have failed to provide a valid phone number. Politely but firmly state that a valid phone number is strictly required to proceed with Varsapradaya."
        
        reply = await _generate_dynamic_reply(prompt)
        return {
            "phone_attempts": attempts,
            "reply": reply,
            "step": "await_phone",
        }

    prompt = "The user successfully provided their phone number. Acknowledge it briefly and ask for their email address."
    reply = await _generate_dynamic_reply(prompt)

    return {
        "phone": raw,
        "reply": reply,
        "step": "await_email",
        "email_attempts": 0,
    }


# ---------------------------------------------------------------------------
# NODE: await_email — validate email, save user to DB
# ---------------------------------------------------------------------------

async def collect_email_node(state: ChatState) -> dict:
    raw = state["user_input"].strip().lower()
    attempts = state.get("email_attempts", 0) + 1
    has_phone = bool(state.get("phone"))

    if not raw:
        return {"reply": "Please type something to proceed.", "step": "await_email"}

    if not re.match(EMAIL_REGEX, raw):
        if attempts == 1:
            prompt = f"The user replied '{raw}' instead of a valid email address. Ask them to try providing a valid email address again."
        else:
            prompt = f"The user replied '{raw}'. They have failed to provide a valid email address. Firmly explain that a valid email address is strictly required to proceed, as it acts as their unique identifier for future sessions."
        
        reply = await _generate_dynamic_reply(prompt)
        return {
            "email_attempts": attempts,
            "reply": reply,
            "step": "await_email",
        }

    return await _proceed_to_db_save(state, raw)

async def _proceed_to_db_save(state: ChatState, email: str) -> dict:
    async with AsyncSessionLocal() as db:
        user = await find_user_by_email(db, email)

        if user:
            # Returning user — load last category
            last_category = await get_last_session_category(db, user.id)
            user_id = str(user.id)
            name = user.name

            await write_log(db, "INFO", "user_returning", f"Returning user: {email}", user_id=user.id)

            if last_category:
                prompt = f"Welcome back the user ({name}) who is returning. Mention that last time they were chatting as a '{last_category.title()}'. Ask if they want to continue with that or choose a different role.\n\n{CATEGORY_MENU_PROMPT}"
                reply = await _generate_dynamic_reply(prompt)
            else:
                prompt = f"Welcome back the user ({name}) who is returning. \n\n{CATEGORY_MENU_PROMPT}"
                reply = await _generate_dynamic_reply(prompt)

            return {
                "email": email,
                "user_id": user_id,
                "name": name,
                "is_returning": True,
                "reply": reply,
                "step": "await_category",
            }
        else:
            # NEW USER
            new_user = await create_user(db, state["name"], email, state["phone"])
            user_id = str(new_user.id)

            await write_log(db, "INFO", "user_created", f"User data gathered", user_id=new_user.id)

            prompt = f"Tell the user ({state.get('name', 'there')}) that they are perfectly set up and ready to go. \n\n{CATEGORY_MENU_PROMPT}"
            reply = await _generate_dynamic_reply(prompt)

            return {
                "email": email,
                "user_id": user_id,
                "is_returning": False,
                "reply": reply,
                "step": "await_category",
            }


# ---------------------------------------------------------------------------
# NODE: await_category — LLM classifies user's intent → category
# ---------------------------------------------------------------------------

async def collect_category_node(state: ChatState) -> dict:
    raw = state["user_input"].strip()

    if not raw:
        return {"reply": "Please describe yourself or choose a role.", "step": "await_category"}

    # Fast keyword match first
    KEYWORD_MAP = {
        "grower":    ["grower", "farmer", "planter", "farm", "estate owner", "coffee", "tea", "spice", "crop"],
        "corporate": ["corporate", "executive", "compliance", "supply chain", "manager", "officer", "eudr", "sustainability"],
        "investor":  ["investor", "invest", "vc", "venture", "analyst", "financial", "capital", "fund"],
        "agritech":  ["agritech", "reseller", "technician", "installer", "distributor", "hardware", "equipment", "sell"],
    }
    for cat, keywords in KEYWORD_MAP.items():
        if any(kw in raw.lower() for kw in keywords):
            return await _finalize_category(state, cat)

    # Fall back to LLM classification
    classify_messages = [
        {
            "role": "system",
            "content": (
                "You are a classifier. Based on the user's message, classify them into EXACTLY one of these categories:\n"
                "- grower (farmers, planters, estate owners, crop growers)\n"
                "- corporate (executives, compliance officers, supply-chain managers)\n"
                "- investor (VCs, analysts, financial professionals)\n"
                "- agritech (hardware resellers, technicians, installers)\n\n"
                "Reply with ONLY the category word, nothing else. If unclear, reply 'unclear'."
            ),
        },
        {"role": "user", "content": raw},
    ]

    result = await _call_llm(classify_messages, max_tokens=10, temperature=0.0)
    category = result["reply"].strip().lower()

    if category not in ("grower", "corporate", "investor", "agritech"):
        prompt = "The user provided an invalid role. Apologize that you didn't quite catch that, and ask them to clearly choose one of: Grower, Corporate, Investor, or Agritech."
        reply = await _generate_dynamic_reply(prompt)
        return {
            "reply": reply,
            "step": "await_category",
        }

    return await _finalize_category(state, category)


async def _finalize_category(state: ChatState, category: str) -> dict:
    """Creates DB session and confirms the selected agent to the user."""
    async with AsyncSessionLocal() as db:
        session = await create_session(db, state["user_id"], category, state.get("is_returning", False))
        session_id = str(session.id)

        await write_log(
            db, "INFO", "session_started",
            f"Category selected: {category}",
            user_id=state["user_id"], session_id=session.id,
            meta={"category": category}
        )

    prompt = (
        f"Acknowledge that the user is focusing on the {category.title()} perspective. "
        "CRITICAL: Keep your response extremely brief (maximum 2 sentences). "
        "Do not use the word 'Agent'. Simply tell them you are ready to answer their questions "
        "about Varsapradaya, and mention they can type 'switch role' anytime if they want to change topics."
    )
    reply = await _generate_dynamic_reply(prompt)

    return {
        "category": category,
        "session_id": session_id,
        "reply": reply,
        "step": "chatting",
        "agent_name": f"{category}_agent",
    }


# ---------------------------------------------------------------------------
# NODE: chatting — detect switch OR answer FAQ with RAG
# ---------------------------------------------------------------------------

async def chat_node(state: ChatState) -> dict:
    user_msg = state["user_input"]

    # Check for switch intent
    msg_lower = user_msg.lower()
    if any(phrase in msg_lower for phrase in SWITCH_PHRASES):
        return await _handle_switch(state, user_msg)

    # Normal FAQ answer flow
    return await _answer_faq(state, user_msg)


async def _handle_switch(state: ChatState, user_msg: str) -> dict:
    """End current session and ask which category they want."""
    async with AsyncSessionLocal() as db:
        if state.get("session_id"):
            await end_session(db, state["session_id"])
            await write_log(
                db, "INFO", "switch_requested", "User requested category switch",
                user_id=state.get("user_id"), session_id=state.get("session_id")
            )

    prompt = f"The user requested to switch their category/agent. Enthusiastically agree to switch their agent.\n\n{CATEGORY_MENU_PROMPT}"
    reply = await _generate_dynamic_reply(prompt)

    return {
        "session_id": None,
        "category": None,
        "reply": reply,
        "step": "await_category",
        "agent_name": None,
    }


async def _answer_faq(state: ChatState, user_msg: str) -> dict:
    """RAG retrieval + persona agent LLM call."""
    start_time = time.time()
    category = state["category"]

    async with AsyncSessionLocal() as db:
        # Save user message
        await save_message(db, state["session_id"], "user", user_msg)

        # Load conversation history (last 10 msgs)
        history = await get_last_10_messages(db, state["session_id"])

        # RAG: find top 3 relevant FAQs
        context = await retrieve(db, user_msg, top_k=3)
        rag_used = bool(context and context.strip())

        # Build prompt: persona + RAG context
        system_prompt = get_persona(category)
        system_prompt += (
            "\n\nGUARDRAILS & LIMITATIONS:\n"
            "1. DOMAIN RESTRICTION: You are strictly an informational assistant for Varsapradaya. If the user asks about entirely unrelated topics (e.g., 'what is 4+3', politics, recipes, coding), politely decline and state that you only answer questions related to Varsapradaya.\n"
            "2. NO SYSTEM ACTIONS: You are a read-only informational bot. You CANNOT perform system actions. You cannot delete conversations, clear history, delete accounts, reset passwords, or book meetings.\n"
            "3. STRICT ANTI-HALLUCINATION: You are strictly limited to the provided context. If the answer cannot be explicitly found in the context provided below, you MUST say 'I apologize, but I do not have that specific information in my current knowledge base.' Do NOT attempt to answer using outside knowledge. Do NOT guess.\n"
            "4. MAINTAIN PERSONA: Do not break character. Do not say 'As an AI...'."
        )
        
        if context and context.strip():
            system_prompt += f"\n\nMOST RELEVANT CONTEXT FOR THIS QUESTION:\n{context}"

        # Build messages list for LLM
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[:-1])  # exclude the message we just saved
        messages.append({"role": "user", "content": user_msg})

        # LLM call
        TOKEN_LIMITS = {"grower": 500, "corporate": 600, "investor": 700, "agritech": 600}
        TEMPS = {"grower": 0.3, "corporate": 0.3, "investor": 0.4, "agritech": 0.3}

        result = await _call_llm(
            messages,
            max_tokens=TOKEN_LIMITS.get(category, 500),
            temperature=TEMPS.get(category, 0.3),
        )

        reply = result["reply"]
        latency_ms = int((time.time() - start_time) * 1000)

        # Save assistant reply
        await save_message(db, state["session_id"], "assistant", reply)

        # Write structured log
        await write_log(
            db,
            level="INFO",
            event="agent_call",
            message=f"{category}_agent replied",
            user_id=state.get("user_id"),
            session_id=state.get("session_id"),
            meta={
                "agent": f"{category}_agent",
                "category": category,
                "latency_ms": latency_ms,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "rag_used": rag_used,
            },
        )

        logger.info(
            f"Agent call: {category}_agent",
            extra={"event": "agent_call", "user_id": state.get("user_id"), "session_id": state.get("session_id")}
        )

    return {
        "reply": reply,
        "agent_name": f"{category}_agent",
        "step": "chatting",
    }
