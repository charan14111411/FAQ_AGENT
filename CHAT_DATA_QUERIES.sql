-- 📊 Quick Chat Data Queries
-- Use these to explore your chat data

-- ============================================================================
-- 1️⃣  GET ALL USERS WHO HAVE CHATTED
-- ============================================================================
SELECT 
    u.id,
    u.name,
    u.email,
    u.phone,
    COUNT(DISTINCT s.id) as total_sessions,
    COUNT(m.id) as total_messages,
    MAX(m.created_at) as last_chat_time
FROM users u
LEFT JOIN sessions s ON u.id = s.user_id
LEFT JOIN messages m ON s.id = m.session_id
GROUP BY u.id, u.name, u.email, u.phone
ORDER BY MAX(m.created_at) DESC;


-- ============================================================================
-- 2️⃣  GET COMPLETE CHAT HISTORY FOR A SPECIFIC USER
-- ============================================================================
-- Replace 'user-email@example.com' with actual email
SELECT 
    s.id as session_id,
    s.category as agent_type,
    s.started_at,
    s.ended_at,
    m.role,
    m.content,
    m.created_at
FROM sessions s
LEFT JOIN messages m ON s.id = m.session_id
JOIN users u ON s.user_id = u.id
WHERE u.email = 'user-email@example.com'
ORDER BY s.started_at DESC, m.created_at ASC;


-- ============================================================================
-- 3️⃣  GET A SPECIFIC CONVERSATION (BY SESSION ID)
-- ============================================================================
-- Replace 'session-uuid' with actual session ID
SELECT 
    m.created_at,
    m.role as speaker,
    m.content,
    LENGTH(m.content) as message_length
FROM messages m
WHERE m.session_id = 'session-uuid'
ORDER BY m.created_at ASC;


-- ============================================================================
-- 4️⃣  GET AGENT USAGE STATISTICS
-- ============================================================================
SELECT 
    category as agent,
    COUNT(*) as total_sessions,
    COUNT(CASE WHEN is_returning = TRUE THEN 1 END) as returning_users,
    COUNT(CASE WHEN is_returning = FALSE THEN 1 END) as new_users,
    SUM(
        SELECT COUNT(*) FROM messages m 
        WHERE m.session_id = s.id
    ) as total_messages
FROM sessions s
GROUP BY category
ORDER BY total_sessions DESC;


-- ============================================================================
-- 5️⃣  GET TOP FAQ QUESTIONS USED BY EACH AGENT
-- ============================================================================
SELECT 
    s.category,
    fe.question,
    COUNT(cp.id) as times_retrieved,
    COUNT(DISTINCT s.user_id) as unique_users
FROM checkpoints cp
JOIN sessions s ON cp.session_id = s.id
JOIN faq_embeddings fe ON cp.metadata->>'faq_id' = fe.id::text
WHERE cp.checkpoint_type = 'faq_retrieved'
GROUP BY s.category, fe.question
ORDER BY s.category, times_retrieved DESC;


-- ============================================================================
-- 6️⃣  GET CONVERSATION LENGTH STATISTICS
-- ============================================================================
SELECT 
    category,
    COUNT(*) as total_sessions,
    ROUND(AVG(message_count)::numeric, 2) as avg_messages_per_session,
    MAX(message_count) as longest_conversation,
    MIN(message_count) as shortest_conversation,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY message_count) as median_messages
FROM (
    SELECT 
        s.category,
        COUNT(m.id) as message_count
    FROM sessions s
    LEFT JOIN messages m ON s.id = m.session_id
    GROUP BY s.id, s.category
) subquery
GROUP BY category;


-- ============================================================================
-- 7️⃣  GET ACTIVE USERS (LAST 7 DAYS)
-- ============================================================================
SELECT 
    u.id,
    u.name,
    u.email,
    COUNT(DISTINCT s.id) as sessions_last_7_days,
    COUNT(m.id) as messages_last_7_days,
    MAX(m.created_at) as last_active
FROM users u
JOIN sessions s ON u.id = s.user_id
LEFT JOIN messages m ON s.id = m.session_id AND m.created_at > NOW() - INTERVAL '7 days'
WHERE s.started_at > NOW() - INTERVAL '7 days'
GROUP BY u.id, u.name, u.email
ORDER BY MAX(m.created_at) DESC;


-- ============================================================================
-- 8️⃣  GET SESSION DURATION ANALYTICS
-- ============================================================================
SELECT 
    category,
    COUNT(*) as total_sessions,
    ROUND(
        AVG(EXTRACT(EPOCH FROM (COALESCE(ended_at, NOW()) - started_at)))::numeric / 60, 
        2
    ) as avg_duration_minutes,
    ROUND(
        MAX(EXTRACT(EPOCH FROM (COALESCE(ended_at, NOW()) - started_at)))::numeric / 60,
        2
    ) as max_duration_minutes,
    ROUND(
        MIN(EXTRACT(EPOCH FROM (COALESCE(ended_at, NOW()) - started_at)))::numeric / 60,
        2
    ) as min_duration_minutes
FROM sessions
WHERE ended_at IS NOT NULL
GROUP BY category;


-- ============================================================================
-- 9️⃣  GET ERRORS/FAILED INTERACTIONS
-- ============================================================================
SELECT 
    cp.turn_id,
    cp.checkpoint_type,
    cp.agent,
    cp.status,
    cp.metadata->>'error' as error_message,
    cp.created_at
FROM checkpoints cp
WHERE cp.status = 'error'
ORDER BY cp.created_at DESC
LIMIT 50;


-- ============================================================================
-- 🔟 GET ONBOARDING DROPOUT ANALYSIS
-- ============================================================================
SELECT 
    step,
    COUNT(*) as users_at_step,
    COUNT(CASE WHEN user_id IS NOT NULL THEN 1 END) as completed_onboarding,
    ROUND(
        100.0 * COUNT(CASE WHEN user_id IS NOT NULL THEN 1 END) / COUNT(*),
        2
    ) as completion_percentage
FROM onboarding_states
GROUP BY step
ORDER BY 
    CASE WHEN step = 'name' THEN 1
         WHEN step = 'phone' THEN 2
         WHEN step = 'email' THEN 3
         WHEN step = 'category' THEN 4
    END;


-- ============================================================================
-- 🔗 EXPORT: Get all data for a user (for backup/analysis)
-- ============================================================================
-- Step 1: Get user info
SELECT * FROM users WHERE email = 'user-email@example.com';

-- Step 2: Get sessions
SELECT * FROM sessions WHERE user_id = 'user-uuid';

-- Step 3: Get messages (pipe to file)
\copy (SELECT * FROM messages WHERE session_id = 'session-uuid') TO '/tmp/chat_export.csv' CSV HEADER

-- Step 4: Get checkpoints
SELECT * FROM checkpoints WHERE session_id = 'session-uuid' ORDER BY created_at;
