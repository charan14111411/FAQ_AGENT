# 📊 FAQ Agent - Chat Data Storage Guide

## 🗄️ Database Overview

All chat data is stored in **PostgreSQL** with **pgvector** extension.

**Database URL** (from `.env`):
```
DATABASE_URL = postgresql+asyncpg://user:password@localhost:5432/faq_db
```

---

## 📍 WHERE CHAT DATA IS STORED

### 1️⃣ **USERS Table** - User Profile Information
**Location**: PostgreSQL `users` table

| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID (Primary Key) | Unique user identifier |
| `name` | VARCHAR(120) | User's full name |
| `phone` | VARCHAR(15) | User's phone number |
| `email` | VARCHAR(255) | User's email (UNIQUE) |
| `created_at` | TIMESTAMPTZ | Registration timestamp |
| `updated_at` | TIMESTAMPTZ | Last update timestamp |

**Query**: Who is this user?
```sql
SELECT * FROM users WHERE email = 'farmer@example.com';
```

---

### 2️⃣ **SESSIONS Table** - Chat Sessions
**Location**: PostgreSQL `sessions` table

Each conversation = one session

| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID (Primary Key) | Session identifier |
| `user_id` | UUID (FK) | Which user owns this session |
| `category` | VARCHAR(30) | Agent type: `grower`, `corporate`, `investor`, `agritech` |
| `is_returning` | BOOLEAN | Is this a returning user? |
| `started_at` | TIMESTAMPTZ | When session began |
| `ended_at` | TIMESTAMPTZ | When session ended (null if active) |

**Query**: Get all chat sessions for a user
```sql
SELECT * FROM sessions 
WHERE user_id = 'user-uuid' 
ORDER BY started_at DESC;
```

**Query**: Get all active sessions
```sql
SELECT * FROM sessions 
WHERE ended_at IS NULL;
```

---

### 3️⃣ **MESSAGES Table** - Individual Chat Messages ⭐ **MAIN CHAT DATA**
**Location**: PostgreSQL `messages` table

**This is where ALL chat messages (user input + agent response) are stored**

| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID (Primary Key) | Message identifier |
| `session_id` | UUID (FK) | Which session this message belongs to |
| `role` | VARCHAR(15) | `'user'` or `'assistant'` |
| `content` | TEXT | The actual message text |
| `created_at` | TIMESTAMPTZ | When message was created |

**Indices**:
- `idx_messages_session_id` - Fast lookup by session
- `idx_messages_created_at` - Time-based queries

**Query**: Get all chat messages for a session
```sql
SELECT role, content, created_at FROM messages 
WHERE session_id = 'session-uuid' 
ORDER BY created_at ASC;
```

**Query**: Get conversation history (last 10 messages)
```sql
SELECT role, content FROM (
    SELECT role, content, created_at FROM messages
    WHERE session_id = 'session-uuid'
    ORDER BY created_at DESC
    LIMIT 10
) sub
ORDER BY created_at ASC;
```

---

### 4️⃣ **CHECKPOINTS Table** - Agent Decision Logs
**Location**: PostgreSQL `checkpoints` table

Stores every decision/state the agent makes (for debugging & analysis)

| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID | Checkpoint identifier |
| `turn_id` | UUID | Which chat turn (request-response) |
| `checkpoint_type` | VARCHAR | Type of checkpoint (e.g., `routing_decision`, `response_generated`) |
| `status` | VARCHAR | `success`, `error`, `pending` |
| `user_id` | UUID | Which user |
| `session_id` | UUID | Which session |
| `category` | VARCHAR(30) | Agent used: `grower`, `corporate`, etc. |
| `agent` | VARCHAR | Specific agent name |
| `user_message_id` | UUID | Reference to user message |
| `assistant_message_id` | UUID | Reference to assistant response |
| `metadata` | JSONB | Additional data (routing scores, retrieved FAQ, etc.) |
| `created_at` | TIMESTAMPTZ | Timestamp |

**Query**: Debug what happened in a session
```sql
SELECT turn_id, checkpoint_type, agent, metadata 
FROM checkpoints 
WHERE session_id = 'session-uuid' 
ORDER BY created_at DESC;
```

---

### 5️⃣ **LOGS Table** - System Logs
**Location**: PostgreSQL `logs` table

General application logging

| Column | Type | Purpose |
|--------|------|---------|
| `id` | BIGSERIAL | Auto-incrementing ID |
| `level` | VARCHAR(10) | Log level: `INFO`, `ERROR`, `WARNING` |
| `event` | VARCHAR(80) | Event type |
| `user_id` | UUID | Associated user |
| `session_id` | UUID | Associated session |
| `meta` | JSONB | Metadata |
| `message` | TEXT | Log message |
| `created_at` | TIMESTAMPTZ | Timestamp |

**Query**: Get errors for a session
```sql
SELECT level, event, message, created_at 
FROM logs 
WHERE session_id = 'session-uuid' AND level = 'ERROR';
```

---

### 6️⃣ **ONBOARDING_STATES Table** - Onboarding Progress
**Location**: PostgreSQL `onboarding_states` table

Stores partial onboarding data (useful for resuming interrupted flows)

| Column | Type | Purpose |
|--------|------|---------|
| `conversation_id` | VARCHAR | Chat conversation ID |
| `step` | VARCHAR | Current step: `name`, `phone`, `email`, `category` |
| `profile` | JSONB | Collected data: `{name, phone, email, email_opt_out}` |
| `user_id` | UUID | Linked user (after creation) |
| `session_id` | UUID | Linked session (after creation) |

**Query**: Get onboarding progress
```sql
SELECT * FROM onboarding_states 
WHERE conversation_id = 'conv-id';
```

---

### 7️⃣ **FAQ_EMBEDDINGS Table** - Vector Search
**Location**: PostgreSQL `faq_embeddings` table (with pgvector)

Stores FAQ documents + vector embeddings for semantic search

| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID | Document ID |
| `question` | TEXT | FAQ question |
| `answer` | TEXT | FAQ answer |
| `category` | VARCHAR(30) | Relevant agent category |
| `embedding` | `vector(1536)` | OpenAI embedding (for semantic search) |
| `created_at` | TIMESTAMPTZ | Timestamp |

**Query**: Find similar FAQs (semantic search)
```sql
SELECT question, answer, category 
FROM faq_embeddings 
WHERE category = 'grower'
ORDER BY embedding <-> $1 LIMIT 5;  -- $1 is query embedding
```

---

## 🔄 Data Flow - How Chat Gets Stored

```
User sends message
    ↓
POST /chat/onboarding  (if not onboarded)
    ↓ (once onboarded)
POST /chat
    ↓
chat_graph processes request
    ↓
Message SAVED to messages table:
  • Saved user message (role='user')
  • Saved agent response (role='assistant')
    ↓
Checkpoint SAVED to checkpoints table (debugging info)
    ↓
Response returned to user
```

---

## 💾 Database Functions Used

From [app/db.py](app/db.py):

### Save Message
```python
await save_message(db, session_id, role='user', content=message_text)
await save_message(db, session_id, role='assistant', content=response_text)
```

### Get Chat History
```python
await get_last_10_messages(db, session_id)
# Returns: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
```

### Get Session Checkpoints
```python
await get_checkpoints_by_session(db, session_id, limit=100)
# Returns detailed decision log
```

---

## 🔍 How To Query Chat Data

### 1️⃣ Get all conversations for a user
```sql
SELECT s.id, s.category, s.started_at, COUNT(m.id) as message_count
FROM sessions s
LEFT JOIN messages m ON s.id = m.session_id
WHERE s.user_id = 'user-uuid'
GROUP BY s.id
ORDER BY s.started_at DESC;
```

### 2️⃣ Get a complete conversation
```sql
SELECT m.role, m.content, m.created_at
FROM messages m
JOIN sessions s ON m.session_id = s.id
WHERE s.id = 'session-uuid'
ORDER BY m.created_at ASC;
```

### 3️⃣ Get agents used by a user
```sql
SELECT DISTINCT category, COUNT(*) as count
FROM sessions
WHERE user_id = 'user-uuid'
GROUP BY category;
```

### 4️⃣ Get user engagement stats
```sql
SELECT 
  COUNT(DISTINCT s.id) as total_sessions,
  COUNT(m.id) as total_messages,
  COUNT(CASE WHEN s.is_returning THEN 1 END) as returning_sessions,
  AVG(EXTRACT(EPOCH FROM (COALESCE(s.ended_at, NOW()) - s.started_at))) as avg_session_duration_seconds
FROM sessions s
LEFT JOIN messages m ON s.id = m.session_id
WHERE s.user_id = 'user-uuid';
```

---

## 🗂️ Complete Data Model Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         USERS                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ id (UUID)│ name │ phone │ email │ created_at      │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ (1-to-many)
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                      SESSIONS                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ id (UUID)│ user_id │ category │ started_at        │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ (1-to-many)
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                      MESSAGES  ⭐                             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ id (UUID)│ session_id │ role │ content│ created_at  │    │
│  │          │ (user/asst)│                              │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘

                         │
                         │ (1-to-many)
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                     CHECKPOINTS                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ turn_id │ session_id │ agent │ metadata │ created_at│    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Running Database Queries

### Using psql (command line):
```bash
psql postgresql://user:password@localhost:5432/faq_db

# Then run queries like:
SELECT * FROM messages WHERE session_id = 'session-uuid';
```

### Using Python (from app):
```python
from app.db import get_last_10_messages
messages = await get_last_10_messages(db, session_id)
for msg in messages:
    print(f"{msg['role']}: {msg['content']}")
```

---

## 📌 Summary

| **What** | **Where** | **Table** |
|---------|----------|----------|
| User profiles | PostgreSQL | `users` |
| Chat sessions | PostgreSQL | `sessions` |
| **All chat messages** ⭐ | PostgreSQL | `messages` |
| Agent decisions | PostgreSQL | `checkpoints` |
| Logs | PostgreSQL | `logs` |
| Onboarding progress | PostgreSQL | `onboarding_states` |
| FAQ embeddings | PostgreSQL (pgvector) | `faq_embeddings` |

**All data is in PostgreSQL → NOT stored in memory → Persists across restarts** ✅
