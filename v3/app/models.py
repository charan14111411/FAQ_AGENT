from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    """
    Unified request model for v3 master/slave architecture.

    source:     "button" | "text"
                  button = user clicked a category button (fast-path to slave)
                  text   = user typed a free-form message (goes through master)
    agent_hint: populated only when source="button"
                  one of: grower_agent | investor_agent | corporate_agent | general_agent
    """
    thread_id:  str
    message:    str
    source:     str = "text"          # "button" | "text"
    agent_hint: Optional[str] = None  # "grower_agent" | "investor_agent" | "corporate_agent" | "general_agent"


class ChatResponse(BaseModel):
    """
    Response from the v3 master/slave FAQ agent.

    reply:          Bot's text response.
    step:           Conversation step (chatting | await_onboarding).
    agent:          Which slave produced the reply (grower_agent, etc.) or "master_agent".
    master_handled: True if the master answered directly without routing to a slave.
    routed_to:      Which slave the master routed to (None if master_handled=True or button click).
    """
    reply:          str
    step:           str
    agent:          Optional[str] = None
    master_handled: bool = False
    routed_to:      Optional[str] = None
