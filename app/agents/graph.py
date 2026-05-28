from langgraph.graph import StateGraph, END
from app.agents.state import ChatState
from app.agents.nodes import (
    router_node,
    classify_entry_node,
    collect_name_node,
    collect_phone_node,
    collect_email_node,
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
          → await_phone (collect_phone_node)
            → await_email (collect_email_node)
              → chatting (chat_node) [loops indefinitely]

    The conditional entry point reads state.step and dispatches
    to the correct node. Each node runs, updates state, and ends —
    the response is returned to the frontend. On the next message,
    the checkpointer restores state and the router picks the right node.
    """
    builder = StateGraph(ChatState)

    # Register nodes
    builder.add_node("start",       classify_entry_node)
    builder.add_node("await_name",  collect_name_node)
    builder.add_node("await_phone", collect_phone_node)
    builder.add_node("await_email", collect_email_node)
    builder.add_node("chatting",    chat_node)
    builder.add_node("ended",       lambda state: state)

    # Conditional entry point: router reads state.step → picks node
    builder.set_conditional_entry_point(
        router_node,
        path_map={
            "start":       "start",
            "await_name":  "await_name",
            "await_phone": "await_phone",
            "await_email": "await_email",
            "chatting":    "chatting",
            "ended":       "ended",
        },
    )

    # All nodes produce a reply and hand control back to the frontend
    builder.add_edge("start",       END)
    builder.add_edge("await_name",  END)
    builder.add_edge("await_phone", END)
    builder.add_edge("await_email", END)
    builder.add_edge("chatting",    END)
    builder.add_edge("ended",       END)


    return builder


# Graph builder (compiled with checkpointer in main.py lifespan)
faq_graph_builder = build_graph_builder()
