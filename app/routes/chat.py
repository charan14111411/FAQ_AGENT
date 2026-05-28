from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_db, save_message, get_last_10_messages, write_log
from app.models import ChatRequest, ChatResponse
from app.agents.dispatcher import detect_switch_intent, dispatch_agent
from app.rag.retriever import retrieve
from app.config import settings
from app.logger import get_logger

logger = get_logger()
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def handle_chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    try:
        switch_requested = await detect_switch_intent(req.message)
        
        if switch_requested:
            return ChatResponse(
                reply="",
                session_id=req.session_id,
                agent="none",
                switch_requested=True
            )
            
        await save_message(db, req.session_id, "user", req.message)
        
        history = await get_last_10_messages(db, req.session_id)
        
        context = await retrieve(db, req.message, top_k=3)
        rag_used = bool(context and context.strip())
        
        agent_result = await dispatch_agent(req.category, history[:-1], req.message, context)
        
        reply = agent_result["reply"]
        agent_name = agent_result["agent"]
        latency_ms = agent_result["latency_ms"]
        input_tokens = agent_result["input_tokens"]
        output_tokens = agent_result["output_tokens"]
        
        await save_message(db, req.session_id, "assistant", reply)
        
        meta = {
            "agent": agent_name,
            "category": req.category,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "rag_used": rag_used,
            "llm_provider": settings.LLM_PROVIDER
        }
        await write_log(
            db,
            level="INFO",
            event="agent_call",
            message=f"Agent {agent_name} replied to message",
            user_id=req.user_id,
            session_id=req.session_id,
            meta=meta
        )
        
        logger.info(
            f"Agent call completed: {agent_name}",
            extra={"event": "agent_call", "user_id": req.user_id, "session_id": req.session_id, "meta": meta}
        )
        
        return ChatResponse(
            reply=reply,
            session_id=req.session_id,
            agent=agent_name,
            switch_requested=False
        )
        
    except Exception as e:
        logger.error(f"Error in chat route: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your message."
        )
