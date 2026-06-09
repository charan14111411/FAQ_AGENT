from langgraph.graph import StateGraph, END
from app.agents.state import ChatState
from app.agents.nodes import (
    router_node,
    classify_entry_node,
    collect_name_node,
    collect_phone_on_exit_node,
    collect_email_on_exit_node,
    chat_node,
)
from app.logger import get_logger

logger = get_logger()


def build_graph_builder():
    """
    Build the LangGraph StateGraph.

    Flow:
      start (classify_entry_node)
        → await_name (collect_name_node)
          → chatting (chat_node) [loops indefinitely]
            ↳ [on exit] await_phone_on_exit (collect_phone_on_exit_node)
              → await_email_on_exit (collect_email_on_exit_node)
                → ended
    """
    builder = StateGraph(ChatState)

    # Register nodes for each step in the conversation
    builder.add_node("start",               classify_entry_node)
    builder.add_node("await_name",          collect_name_node)
    builder.add_node("await_phone_on_exit", collect_phone_on_exit_node)
    builder.add_node("await_email_on_exit", collect_email_on_exit_node)
    builder.add_node("chatting",            chat_node)
    builder.add_node("ended",               lambda state: state)

    # Conditional entry point: router reads state.step → picks node
    builder.set_conditional_entry_point(
        router_node,
        path_map={
            "start":               "start",
            "await_name":          "await_name",
            "await_phone_on_exit": "await_phone_on_exit",
            "await_email_on_exit": "await_email_on_exit",
            "chatting":            "chatting",
            "ended":               "ended",
        },
    )

    # All nodes produce a reply and hand control back to the frontend
    builder.add_edge("start",               END)
    builder.add_edge("await_name",          END)
    builder.add_edge("await_phone_on_exit", END)
    builder.add_edge("await_email_on_exit", END)
    builder.add_edge("chatting",            END)
    builder.add_edge("ended",               END)


    return builder


# Graph builder (compiled with checkpointer in main.py lifespan)
faq_graph_builder = build_graph_builder()
