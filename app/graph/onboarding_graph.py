import json
import re
from uuid import uuid4
from typing import Any, Optional, TypedDict
from langgraph.graph import END, StateGraph
from app.agents.base_agent import _call_llm
from app.db import (
    create_session,
    create_user,
    find_user_by_email,
    get_last_session_category,
    get_onboarding_state,
    upsert_onboarding_state,
)


CATEGORY_VALUES = {"grower", "corporate", "investor", "agritech"}
EMAIL_REGEX = r"^[^@]+@[^@]+\.[^@]+$"


class OnboardingState(TypedDict, total=False):
    db: Any
    conversation_id: str
    message: str
    step: str
    profile: dict
    reply: str
    onboarding_complete: bool
    user_id: Optional[str]
    session_id: Optional[str]
    category: Optional[str]


STEP_PROMPTS = {
    "name": "Ask for the user's full name.",
    "phone": "Ask for the user's phone number (10-15 digits).",
    "email": "Ask for the user's email address.",
    "category": "Ask user to choose one category: grower, corporate, investor, or agritech.",
}


async def _load_state_node(state: OnboardingState) -> dict[str, Any]:
    row = await get_onboarding_state(state["db"], state["conversation_id"])
    if not row:
        return {"step": "name", "profile": {}}
    return {
        "step": row["step"],
        "profile": row["profile"] or {},
        "user_id": row.get("user_id"),
        "session_id": row.get("session_id"),
    }


async def _detect_conversational_intent(text: str) -> tuple[str, str]:
    """
    Detect if input is conversational (greeting, small talk, etc.) BEFORE form validation.
    Returns: (intent_type, response)
    intent_type: "greeting" | "acknowledgement" | "formality" | "none"
    """
    system = (
        "Classify user input during onboarding as conversational or not.\n"
        "Return strict JSON only: {\"intent\": \"greeting|acknowledgement|formality|none\", \"response\": \"string\"}.\n"
        "intent=greeting for greetings (hi, hey, hello, how are you, what's up, etc).\n"
        "intent=acknowledgement for confirmations (ok, yes, sure, got it, etc).\n"
        "intent=formality for politeness (please, thanks, thank you, sorry, etc).\n"
        "intent=none for all other inputs (actual form data).\n"
        "If intent is not 'none', provide a warm response that acknowledges and guides to task. Keep under 30 words."
    )
    user = f"User message: {text}"
    try:
        result = await _call_llm(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=100,
            temperature=0.4,
        )
        parsed = json.loads(
            re.search(r"\{.*\}", result.get("reply", ""), re.DOTALL).group(0)
        )
        intent = (parsed.get("intent") or "none").strip().lower()
        response = (parsed.get("response") or "").strip()
        if intent in {"greeting", "acknowledgement", "formality"}:
            return intent, response
    except Exception:
        pass
    return "none", ""


async def _llm_validate_name(text: str) -> tuple[bool, str]:
    system = (
        "You validate onboarding name input.\n"
        "Return strict JSON only: {\"valid\": boolean, \"normalized_name\": string, \"reason\": string}.\n"
        "Mark invalid for random words or non-name text.\n"
        "Mark valid for realistic person names.\n"
        "Note: Greetings are handled separately, so only validate actual name attempts here."
    )
    user = f"Candidate input: {text}"
    try:
        result = await _call_llm(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=90,
            temperature=0.0,
        )
        parsed = json.loads(re.search(r"\{.*\}", result.get("reply", ""), re.DOTALL).group(0))
        if bool(parsed.get("valid")):
            normalized = (parsed.get("normalized_name") or text).strip()
            return True, normalized
    except Exception:
        pass
    return False, text


def _valid_phone(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    return 10 <= len(digits) <= 15


def _valid_email(text: str) -> bool:
    return bool(re.match(EMAIL_REGEX, text.strip()))


async def _llm_onboarding_reply(step: str, user_message: str, instruction: str, invalid_reason: str | None = None) -> str:
    system = (
        "You are a friendly onboarding assistant for Varsapradaya.\n"
        "Sound natural, warm, and concise like a production chat assistant.\n"
        "Do not use bullet points. Keep reply under 45 words.\n"
        "If input is invalid, acknowledge politely and ask again."
    )
    user = (
        f"Current onboarding step: {step}\n"
        f"User message: {user_message}\n"
        f"Task: {instruction}\n"
        f"Invalid reason: {invalid_reason or 'none'}"
    )
    try:
        result = await _call_llm(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=80,
            temperature=0.4,
        )
        text = (result.get("reply") or "").strip()
        if text:
            return text
    except Exception:
        pass
    if invalid_reason:
        return f"I got that, but {invalid_reason}. {STEP_PROMPTS.get(step, 'Please continue.')}"
    return STEP_PROMPTS.get(step, "Please continue.")


async def _llm_parse_category(text: str) -> str | None:
    allowed = ", ".join(sorted(CATEGORY_VALUES))
    system = (
        "You map user input to a single allowed onboarding category.\n"
        f"Allowed: {allowed}\n"
        "Return strict JSON only: {\"category\": \"allowed_value_or_none\"}.\n"
        "If no confident match, return {\"category\":\"none\"}."
    )
    try:
        result = await _call_llm(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
            max_tokens=40,
            temperature=0.0,
        )
        parsed = json.loads(re.search(r"\{.*\}", result.get("reply", ""), re.DOTALL).group(0))
        category = (parsed.get("category") or "").strip().lower()
        if category in CATEGORY_VALUES:
            return category
    except Exception:
        pass
    return None


async def _llm_contact_intent(step: str, text: str) -> str:
    system = (
        "Classify user intent during onboarding contact collection.\n"
        "Return strict JSON only: {\"intent\": \"valid|invalid|refuse|abusive\"}.\n"
        "intent=refuse when user declines to share requested detail.\n"
        "intent=abusive when user uses insults/profanity/hostile tone.\n"
        "intent=valid when user provides requested detail for the current step."
    )
    user = f"Step: {step}\nMessage: {text}"
    try:
        result = await _call_llm(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=30,
            temperature=0.0,
        )
        parsed = json.loads(re.search(r"\{.*\}", result.get("reply", ""), re.DOTALL).group(0))
        intent = (parsed.get("intent") or "").strip().lower()
        if intent in {"valid", "invalid", "refuse", "abusive"}:
            return intent
    except Exception:
        pass
    return "invalid"


async def _process_node(state: OnboardingState) -> dict[str, Any]:
    message = (state.get("message") or "").strip()
    step = state.get("step", "name")
    profile = dict(state.get("profile") or {})

    if not message:
        reply = await _llm_onboarding_reply(step, message, STEP_PROMPTS.get(step, "Ask the user to continue onboarding."))
        return {"step": step, "profile": profile, "reply": reply, "onboarding_complete": False}

    # Detect conversational intent BEFORE form validation
    intent_type, conversational_response = await _detect_conversational_intent(message)
    
    if intent_type in {"greeting", "acknowledgement", "formality"}:
        # User said something conversational - respond warmly and continue with current step
        step_instruction = STEP_PROMPTS.get(step, "Please continue.")
        combined_reply = f"{conversational_response} {step_instruction}"
        return {
            "step": step,
            "profile": profile,
            "reply": combined_reply,
            "onboarding_complete": False,
        }

    if step == "name":
        is_valid_name, normalized_name = await _llm_validate_name(message)
        if not is_valid_name:
            reply = await _llm_onboarding_reply(
                "name",
                message,
                STEP_PROMPTS["name"],
                invalid_reason="that does not look like a full real name",
            )
            return {
                "step": "name",
                "reply": reply,
                "onboarding_complete": False,
            }
        profile["name"] = normalized_name
        reply = await _llm_onboarding_reply("phone", message, STEP_PROMPTS["phone"])
        return {
            "step": "phone",
            "profile": profile,
            "reply": reply,
            "onboarding_complete": False,
        }

    if step == "phone":
        if not _valid_phone(message):
            reply = await _llm_onboarding_reply(
                "phone",
                message,
                STEP_PROMPTS["phone"],
                invalid_reason="the number format is not valid",
            )
            return {
                "step": "phone",
                "profile": profile,
                "reply": reply,
                "onboarding_complete": False,
            }
        profile["phone"] = re.sub(r"\D", "", message)
        reply = await _llm_onboarding_reply("email", message, STEP_PROMPTS["email"])
        return {
            "step": "email",
            "profile": profile,
            "reply": reply,
            "onboarding_complete": False,
        }

    if step == "email":
        intent = await _llm_contact_intent("email", message)
        email_attempts = int(profile.get("email_attempts", 0))

        if intent in {"refuse", "abusive"}:
            profile["email_opt_out"] = True
            profile["email"] = f"optout_{uuid4().hex[:12]}@varsapradaya.local"
            reply = await _llm_onboarding_reply(
                "category",
                message,
                "Acknowledge and continue without forcing email. Ask category: grower, corporate, investor, or agritech.",
            )
            return {
                "step": "category",
                "profile": profile,
                "reply": reply,
                "onboarding_complete": False,
            }

        if not _valid_email(message):
            profile["email_attempts"] = email_attempts + 1
            if profile["email_attempts"] >= 3:
                profile["email_opt_out"] = True
                profile["email"] = f"optout_{uuid4().hex[:12]}@varsapradaya.local"
                reply = await _llm_onboarding_reply(
                    "category",
                    message,
                    "Do not continue asking email. Politely move on and ask category: grower, corporate, investor, or agritech.",
                )
                return {
                    "step": "category",
                    "profile": profile,
                    "reply": reply,
                    "onboarding_complete": False,
                }
            reply = await _llm_onboarding_reply(
                "email",
                message,
                STEP_PROMPTS["email"],
                invalid_reason="the email format seems incorrect",
            )
            return {
                "step": "email",
                "profile": profile,
                "reply": reply,
                "onboarding_complete": False,
            }
        profile["email"] = message.strip().lower()
        profile.pop("email_attempts", None)
        reply = await _llm_onboarding_reply("category", message, STEP_PROMPTS["category"])
        return {
            "step": "category",
            "profile": profile,
            "reply": reply,
            "onboarding_complete": False,
        }

    if step == "category":
        category = await _llm_parse_category(message) or message.strip().lower()
        if category not in CATEGORY_VALUES:
            reply = await _llm_onboarding_reply(
                "category",
                message,
                STEP_PROMPTS["category"],
                invalid_reason="I could not map that to a valid category",
            )
            return {
                "step": "category",
                "profile": profile,
                "reply": reply,
                "onboarding_complete": False,
            }

        user = await find_user_by_email(state["db"], profile["email"])
        if user:
            user_id = str(user.id)
            is_returning = (await get_last_session_category(state["db"], user.id)) is not None
        else:
            created = await create_user(state["db"], profile["name"], profile["phone"], profile["email"])
            user_id = str(created.id)
            is_returning = False

        session = await create_session(state["db"], user_id, category, is_returning)
        session_id = str(session.id)
        reply = await _llm_onboarding_reply(
            "done",
            message,
            f"Confirm setup completion and mention user is connected to {category} assistant.",
        )
        return {
            "step": "done",
            "profile": profile,
            "category": category,
            "user_id": user_id,
            "session_id": session_id,
            "reply": reply,
            "onboarding_complete": True,
        }

    return {
        "step": "done",
        "profile": profile,
        "reply": "You are already onboarded. Please continue with your question.",
        "onboarding_complete": True,
    }


async def _save_state_node(state: OnboardingState) -> dict[str, Any]:
    await upsert_onboarding_state(
        state["db"],
        conversation_id=state["conversation_id"],
        step=state["step"],
        profile=state.get("profile") or {},
        user_id=state.get("user_id"),
        session_id=state.get("session_id"),
    )
    return {}


def build_onboarding_graph():
    graph = StateGraph(OnboardingState)
    graph.add_node("load_state", _load_state_node)
    graph.add_node("process", _process_node)
    graph.add_node("save_state", _save_state_node)
    graph.set_entry_point("load_state")
    graph.add_edge("load_state", "process")
    graph.add_edge("process", "save_state")
    graph.add_edge("save_state", END)
    return graph.compile()


onboarding_graph = build_onboarding_graph()
