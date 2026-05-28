from pydantic import BaseModel, Field
from typing import Any, Optional

class UserRequest(BaseModel):
    name: str
    phone: str
    email: str

class SessionRequest(BaseModel):
    user_id: str
    category: str

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    category: Optional[str] = None
    message: str

class EndSessionRequest(BaseModel):
    session_id: str

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    is_returning: bool
    last_category: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    conversation_id: Optional[str] = None
    step: Optional[str] = None
    onboarding_complete: bool = False
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    category: Optional[str] = None
    agent: str
    switch_requested: bool

class CategorySwitchRequest(BaseModel):
    session_id: str
    user_id: str
    new_category: str


class CheckpointItem(BaseModel):
    id: int
    turn_id: str
    checkpoint_type: str
    status: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    category: Optional[str] = None
    agent: Optional[str] = None
    user_message_id: Optional[str] = None
    assistant_message_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class CheckpointListResponse(BaseModel):
    session_id: str
    count: int
    items: list[CheckpointItem]


class OnboardingChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class OnboardingChatResponse(BaseModel):
    conversation_id: str
    reply: str
    step: str
    onboarding_complete: bool
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    category: Optional[str] = None
