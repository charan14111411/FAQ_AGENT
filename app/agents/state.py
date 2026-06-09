from typing import Optional
from typing_extensions import TypedDict


class ChatState(TypedDict):
    """
    LangGraph state object — persisted across every message via PostgreSQL checkpointer.

    Lifecycle:
      step=start      → category resolved here (from button click or free text)
      step=await_name → name collected
      step=await_phone → phone collected
      step=await_email → email collected, user created/found in DB, session created
      step=chatting   → FAQ answering loop (runs indefinitely)
    """
    thread_id:      str             # UUID from frontend — identifies the conversation
    user_input:     str             # Current raw message from the user
    step:           str             # Current node: start | await_name | await_phone | await_email | chatting
    category:       Optional[str]  # grower | investor | corporate | exploring — set at step=start
    name:           Optional[str]  # Collected at await_name
    phone:          Optional[str]  # Collected at await_phone
    email:          Optional[str]  # Collected at await_email
    user_id:        Optional[str]  # UUID from users table (after DB lookup/create)
    session_id:     Optional[str]  # UUID from sessions table (after session created)
    reply:          str             # The bot's response text
    agent_name:     Optional[str]  # e.g. "grower_agent", "investor_agent"
    is_returning:   bool            # True if user already existed in DB
    phone_attempts: int             # Invalid phone entry counter
    email_attempts: int             # Invalid email entry counter
    farewell_attempts: int          # Counter for polite farewell/end attempts
    classify_attempts: int          # Counter for category classification retries
    language:              Optional[str]  # Canonical language name: 'telugu', 'hindi', 'english' etc.
    language_code:         Optional[str]  # ISO 639-1 code: 'te', 'hi', 'en' etc. Stored for programmatic use
    language_native_name:  Optional[str]  # Native script: 'తెలుగు', 'हिन्दी' etc. Used in LLM prompts

