from langgraph.graph import StateGraph, START, END
from modules.states import AgentState
from modules.edges import condition_a1, condition_a2, condition_d, condition_v
from modules.agents import (
    agent_a1_node,
    agent_a2_node,
    agent_analyze_node,
    agent_verify_node,
    agent_aggregate_node
)

def build_graph():
    # Xây dựng StateGraph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent_a1", agent_a1_node)
    workflow.add_node("agent_a2", agent_a2_node)
    workflow.add_node("agent_analyze", agent_analyze_node)
    workflow.add_node("agent_verify", agent_verify_node)
    workflow.add_node("agent_aggregate", agent_aggregate_node)
    workflow.add_node("error_handler", lambda state: state)
    workflow.add_node("error_final_handler", lambda state: state)

    # Direct edges
    workflow.add_edge(START, "agent_a1")
    workflow.add_edge("error_handler", "agent_aggregate")
    workflow.add_edge("error_final_handler", "agent_aggregate")

    # Conditional edges
    workflow.add_conditional_edges("agent_a1", condition_a1, {
        "agent_a2": "agent_a2",
        "agent_a1": "agent_a1",
        "error": "error_handler"
    })
    workflow.add_conditional_edges("agent_a2", condition_a2, {
        "agent_analyze": "agent_analyze",
        "agent_a2": "agent_a2",
        "error": "error_handler"
    })
    workflow.add_conditional_edges("agent_verify", condition_v, {
        "agent_aggregate": "agent_aggregate",
        "agent_analyze": "agent_analyze",
        "error_final": "error_final_handler"
    })
    workflow.add_conditional_edges("agent_aggregate", condition_d, {
        "agent_verify": "agent_verify",
        END: END
    })

    # Biên dịch và chạy
    graph = workflow.compile()
    return graph
