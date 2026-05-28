from typing import Any, Optional, TypedDict
from sqlalchemy.ext.asyncio import AsyncSession
from langgraph.graph import END, StateGraph
from app.agents.dispatcher import detect_switch_intent, dispatch_agent, route_category
from app.checkpointer import write_checkpoint
from app.config import settings
from app.db import get_last_10_messages, save_message, write_log
from app.rag.retriever import retrieve


class ChatGraphState(TypedDict, total=False):
    db: AsyncSession
    turn_id: str
    session_id: str
    user_id: str
    category: str
    message: str
    switch_requested: bool
    reply: str
    agent: str
    history: list[dict[str, str]]
    context: str
    rag_used: bool
    user_message_id: Optional[str]
    assistant_message_id: Optional[str]
    latency_ms: int
    input_tokens: int
    output_tokens: int
    route_reason: str
    resolved_category: str


async def _request_received_node(state: ChatGraphState) -> dict[str, Any]:
    await write_checkpoint(
        state["db"],
        turn_id=state["turn_id"],
        checkpoint_type="request_received",
        user_id=state["user_id"],
        session_id=state["session_id"],
        category=state["category"],
        metadata={
            "message_length": len(state.get("message", "") or ""),
            "llm_provider": settings.LLM_PROVIDER,
        },
    )
    return {}


async def _detect_switch_node(state: ChatGraphState) -> dict[str, Any]:
    current_category = (state.get("category") or "").strip().lower()
    switch_by_phrase = await detect_switch_intent(state["message"])
    if switch_by_phrase:
        await write_checkpoint(
            state["db"],
            turn_id=state["turn_id"],
            checkpoint_type="switch_requested",
            user_id=state["user_id"],
            session_id=state["session_id"],
            category=current_category or None,
            metadata={"reason": "user_phrase"},
        )
        return {
            "switch_requested": True,
            "reply": "Sure, I can switch that for you. Please choose the category you want to continue with: Grower, Corporate, Investor, or Agritech.",
            "agent": "none",
        }
    auto_route = await route_category(
        user_message=state["message"],
        current_category=current_category or None,
        history=state.get("history", []),
        context=state.get("context", ""),
    )
    await write_checkpoint(
        state["db"],
        turn_id=state["turn_id"],
        checkpoint_type="route_resolved",
        user_id=state["user_id"],
        session_id=state["session_id"],
        category=auto_route["category"],
        metadata={
            "reason": auto_route["reason"],
            "switch_requested": auto_route["switch_requested"],
            "current_category": current_category or None,
        },
    )
    return {
        "switch_requested": auto_route["switch_requested"],
        "resolved_category": auto_route["category"],
        "route_reason": auto_route["reason"],
    }


def _route_after_switch(state: ChatGraphState) -> str:
    if state.get("switch_requested") and state.get("reply"):
        return "finish_after_switch"
    return "save_user_message"


async def _save_user_message_node(state: ChatGraphState) -> dict[str, Any]:
    user_message_row = await save_message(state["db"], state["session_id"], "user", state["message"])
    history = await get_last_10_messages(state["db"], state["session_id"])
    return {"user_message_id": str(user_message_row.id), "history": history}


async def _retrieve_context_node(state: ChatGraphState) -> dict[str, Any]:
    context = await retrieve(state["db"], state["message"], top_k=3)
    rag_used = bool(context and context.strip())
    await write_checkpoint(
        state["db"],
        turn_id=state["turn_id"],
        checkpoint_type="retrieval_completed",
        user_id=state["user_id"],
        session_id=state["session_id"],
        category=state["category"],
        user_message_id=state.get("user_message_id"),
        metadata={"rag_used": rag_used, "context_length": len(context or ""), "top_k": 3},
    )
    return {"context": context, "rag_used": rag_used}


async def _call_agent_node(state: ChatGraphState) -> dict[str, Any]:
    history_without_latest_user = (state.get("history") or [])[:-1]
    effective_category = state.get("resolved_category") or state.get("category") or "grower"
    agent_result = await dispatch_agent(
        effective_category,
        history_without_latest_user,
        state["message"],
        state.get("context", ""),
    )
    return {
        "reply": agent_result["reply"],
        "agent": agent_result["agent"],
        "latency_ms": agent_result["latency_ms"],
        "input_tokens": agent_result["input_tokens"],
        "output_tokens": agent_result["output_tokens"],
    }


async def _finalize_node(state: ChatGraphState) -> dict[str, Any]:
    assistant_message_row = await save_message(state["db"], state["session_id"], "assistant", state["reply"])
    meta = {
        "agent": state["agent"],
        "category": state.get("resolved_category") or state.get("category"),
        "latency_ms": state["latency_ms"],
        "input_tokens": state["input_tokens"],
        "output_tokens": state["output_tokens"],
        "rag_used": state.get("rag_used", False),
        "route_reason": state.get("route_reason"),
        "llm_provider": settings.LLM_PROVIDER,
    }
    await write_log(
        state["db"],
        level="INFO",
        event="agent_call",
        message=f"Agent {state['agent']} replied to message",
        user_id=state["user_id"],
        session_id=state["session_id"],
        meta=meta,
    )
    await write_checkpoint(
        state["db"],
        turn_id=state["turn_id"],
        checkpoint_type="response_completed",
        user_id=state["user_id"],
        session_id=state["session_id"],
        category=state.get("resolved_category") or state.get("category"),
        agent=state["agent"],
        user_message_id=state.get("user_message_id"),
        assistant_message_id=str(assistant_message_row.id),
        metadata=meta,
    )
    return {"assistant_message_id": str(assistant_message_row.id)}


def build_chat_graph():
    graph = StateGraph(ChatGraphState)
    graph.add_node("request_received", _request_received_node)
    graph.add_node("detect_switch", _detect_switch_node)
    graph.add_node("save_user_message", _save_user_message_node)
    graph.add_node("retrieve_context", _retrieve_context_node)
    graph.add_node("call_agent", _call_agent_node)
    graph.add_node("finalize", _finalize_node)

    graph.set_entry_point("request_received")
    graph.add_edge("request_received", "detect_switch")
    graph.add_conditional_edges(
        "detect_switch",
        _route_after_switch,
        {"finish_after_switch": END, "save_user_message": "save_user_message"},
    )
    graph.add_edge("save_user_message", "retrieve_context")
    graph.add_edge("retrieve_context", "call_agent")
    graph.add_edge("call_agent", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


chat_graph = build_chat_graph()
