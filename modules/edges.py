from langgraph.graph import END
from modules.tools import check_chunks
from modules.states import AgentState
# EDGE
def condition_a1(state: AgentState) -> str:
    if state.get("error") or not state.get("cleaned_text"):
        return "error" if state.get("retry_count_a1", 0) >= 3 else "agent_a1"
    return "agent_a2"

def condition_a2(state: AgentState) -> str:
    if state.get("error") or not check_chunks(state.get("chunks", [])):
        return "error" if state.get("retry_count_a2", 0) >= 3 else "agent_a2"
    return "agent_analyze"

def condition_d(state: AgentState) -> str:
    if state.get("error") or not state.get("report"):
        return "agent_verify"
    return END

def condition_v(state: AgentState) -> str:
    if state.get("error"):
        return "error_final" if state.get("retry_count_analyze", 0) >= 3 else "agent_analyze"
    return "agent_aggregate"