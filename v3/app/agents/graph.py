from langgraph.graph import StateGraph, END
from app.agents.state import ChatState
from app.agents.nodes import (
    router_node,
    master_node,
    master_dispatch,
    grower_node,
    investor_node,
    corporate_node,
    general_node,
    handle_switch_node,
    soft_collect_node,
    check_soft_collect,
)
from app.logger import get_logger

logger = get_logger()


def build_graph():
    """
    v3 Master/Slave LangGraph.

    Entry → router_node (conditional)
      ├─ "master"        → master_node → master_dispatch (conditional)
      │                       ├─ "grower" / "investor" / "corporate" / "general" → slave
      │                       └─ "end"                                             → END
      ├─ "grower"        → grower_node   → check_soft_collect → soft_collect? → END
      ├─ "investor"      → investor_node → check_soft_collect → soft_collect? → END
      ├─ "corporate"     → corporate_node→ check_soft_collect → soft_collect? → END
      ├─ "general"       → general_node  → check_soft_collect → soft_collect? → END
      └─ "handle_switch" → handle_switch_node → END
    """
    builder = StateGraph(ChatState)

    # ── Register nodes ─────────────────────────────────────────────────────
    builder.add_node("master",        master_node)
    builder.add_node("grower",        grower_node)
    builder.add_node("investor",      investor_node)
    builder.add_node("corporate",     corporate_node)
    builder.add_node("general",       general_node)
    builder.add_node("handle_switch", handle_switch_node)
    builder.add_node("soft_collect",  soft_collect_node)

    # ── Conditional entry point (the global router) ─────────────────────────
    builder.set_conditional_entry_point(
        router_node,
        path_map={
            "master":        "master",
            "grower":        "grower",
            "investor":      "investor",
            "corporate":     "corporate",
            "general":       "general",
            "handle_switch": "handle_switch",
        },
    )

    # ── Master dispatches to a slave OR ends ────────────────────────────────
    builder.add_conditional_edges(
        "master",
        master_dispatch,
        path_map={
            "grower":    "grower",
            "investor":  "investor",
            "corporate": "corporate",
            "general":   "general",
            "end":       END,
        },
    )

    # ── Each slave checks if soft_collect should fire ───────────────────────
    for slave in ("grower", "investor", "corporate", "general"):
        builder.add_conditional_edges(
            slave,
            check_soft_collect,
            path_map={"soft_collect": "soft_collect", "end": END},
        )

    # ── Switch handler and soft_collect always end ──────────────────────────
    builder.add_edge("handle_switch", END)
    builder.add_edge("soft_collect",  END)

    return builder


# Module-level builder — compiled in main.py with a checkpointer
faq_graph_builder = build_graph()
