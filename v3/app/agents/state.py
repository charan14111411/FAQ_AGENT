from typing import Optional
from typing_extensions import TypedDict


class ChatState(TypedDict):
    """
    LangGraph state for v3 master/slave architecture.
    Persisted across every message in a thread via the checkpointer.
    """
    # ── Request context ──────────────────────────────────────────────────────
    thread_id:        str            # Frontend UUID — identifies the conversation
    user_input:       str            # Current raw message from user
    source:           str            # "button" | "text"
    agent_hint:       Optional[str]  # Populated on button click: "grower_agent" etc.

    # ── Routing ──────────────────────────────────────────────────────────────
    active_agent:     Optional[str]  # "grower_agent" | "investor_agent" | "corporate_agent" | "general_agent"
    routed_to:        Optional[str]  # Set by master_node when routing to a slave: "grower" etc.
    master_handled:   bool           # True if master answered directly (no slave invoked)

    # ── Conversation ─────────────────────────────────────────────────────────
    step:             str            # "start" | "chatting" | "await_onboarding"
    reply:            str            # Bot response text
    agent_name:       Optional[str]  # Which agent produced the reply

    # ── Deferred onboarding ──────────────────────────────────────────────────
    name:             Optional[str]
    email:            Optional[str]
    phone:            Optional[str]
    user_id:          Optional[str]
    session_id:       Optional[str]
    is_returning:     bool
    turn_count:       int            # Number of substantive exchanges in this session
    onboarding_asked: bool           # Whether we've already asked for name/email
