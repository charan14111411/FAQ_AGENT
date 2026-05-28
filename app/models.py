from pydantic import BaseModel
from typing import Optional

class UserRequest(BaseModel):
    name: str
    phone: str
    email: str

class SessionRequest(BaseModel):
    user_id: str
    category: str

class ChatRequest(BaseModel):
    session_id: str
    user_id: str
    category: str
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
    session_id: str
    agent: str
    switch_requested: bool

class CategorySwitchRequest(BaseModel):
    session_id: str
    user_id: str
    new_category: str
