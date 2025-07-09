from langgraph.graph import StateGraph, START, END
from modules.states import AgentState
from modules.routers import condition_a1, condition_a2, condition_analyze, condition_verify, condition_aggregate
from modules.agents import (
    extracted_agent,
    chunked_and_embedded_agent,
    analyzed_agent,
    verified_agent,
    aggregated_agent
)

def build_graph():
    # Xây dựng StateGraph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent_a1", extracted_agent)
    workflow.add_node("agent_a2", chunked_and_embedded_agent)
    workflow.add_node("agent_analyze", analyzed_agent)
    workflow.add_node("agent_verify", verified_agent)
    workflow.add_node("agent_aggregate", aggregated_agent)
    
    # Fix error_final node - không dùng dict unpacking với dataclass
    def error_final_handler(state: AgentState) -> AgentState:
        """Handler cho error final state"""
        return {
            "error": state.error or "Workflow failed after max retries",
            "report": f"ERROR: {state.error or 'Workflow failed'}",
            "messages": []
        }
    
    workflow.add_node("error_final", error_final_handler)

    # Direct edges
    workflow.add_edge(START, "agent_a1")
    workflow.add_edge("error_final", END)

    # Conditional edges với exit conditions rõ ràng
    workflow.add_conditional_edges("agent_a1", condition_a1, {
        "agent_a2": "agent_a2",
        "agent_a1": "agent_a1", 
        "error_final": "error_final"
    })
    
    workflow.add_conditional_edges("agent_a2", condition_a2, {
        "agent_analyze": "agent_analyze",
        "agent_a2": "agent_a2",
        "error_final": "error_final"
    })
    
    workflow.add_conditional_edges("agent_analyze", condition_analyze, {
        "agent_verify": "agent_verify",
        "agent_analyze": "agent_analyze",
        "agent_a2": "agent_a2",
        "error_final": "error_final"
    })
    
    workflow.add_conditional_edges("agent_verify", condition_verify, {
        "agent_aggregate": "agent_aggregate",
        "agent_verify": "agent_verify",
        "agent_analyze": "agent_analyze",
        "error_final": "error_final"
    })
    
    workflow.add_conditional_edges("agent_aggregate", condition_aggregate, {
        "agent_aggregate": "agent_aggregate",
        "error_final": "error_final",
        END: END
    })

    # Biên dịch và chạy
    graph = workflow.compile()
    return graph
