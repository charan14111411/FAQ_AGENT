from langgraph.graph import StateGraph, END
from app.agents.state import ChatState
from app.agents.nodes import (
    router_node,
    greet_node,
    collect_name_node,
    collect_email_node,
    collect_phone_node,
    collect_category_node,
    chat_node,
)
from app.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Build the LangGraph StateGraph (Graph builder)
# ---------------------------------------------------------------------------

def build_graph_builder():
    builder = StateGraph(ChatState)

    # Register all nodes
    builder.add_node("start",           greet_node)
    builder.add_node("await_name",      collect_name_node)
    builder.add_node("await_email",     collect_email_node)
    builder.add_node("await_phone",     collect_phone_node)
    builder.add_node("await_category",  collect_category_node)
    builder.add_node("chatting",        chat_node)

    # Entry point: conditional router reads state.step and picks node
    builder.set_conditional_entry_point(
        router_node,
        path_map={
            "start":           "start",
            "await_name":      "await_name",
            "await_email":     "await_email",
            "await_phone":     "await_phone",
            "await_category":  "await_category",
            "chatting":        "chatting",
        },
    )

    # All nodes end after producing their reply
    builder.add_edge("start",          END)
    builder.add_edge("await_name",     END)
    builder.add_edge("await_email",    END)
    builder.add_edge("await_phone",    END)
    builder.add_edge("await_category", END)
    builder.add_edge("chatting",       END)

    return builder

# Compile without checkpointer for testing/initialization if needed
faq_graph_builder = build_graph_builder()
