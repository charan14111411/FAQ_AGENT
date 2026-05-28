from typing import Optional
from typing_extensions import TypedDict


class ChatState(TypedDict):
    """
    LangGraph state object — persisted across every message in a conversation thread.
    """
    thread_id: str                    # UUID from frontend — identifies the conversation
    user_input: str                   # Current raw message from the user
    step: str                         # Conversation step:
                                      #   "start" | "await_name" | "await_email"
                                      #   "await_phone" | "await_category" | "chatting"
    name: Optional[str]               # Collected user name
    email: Optional[str]              # Collected user email
    phone: Optional[str]              # Collected user phone
    user_id: Optional[str]            # UUID from users table (after DB lookup/create)
    session_id: Optional[str]         # UUID from sessions table (after session created)
    category: Optional[str]           # grower | corporate | investor | agritech
    reply: str                        # The bot's response to send back to the frontend
    agent_name: Optional[str]         # Name of the agent that produced the reply
    is_returning: bool                # True if user already existed in DB
    phone_attempts: int               # Count of invalid phone attempts
    email_attempts: int               # Count of invalid email attempts

