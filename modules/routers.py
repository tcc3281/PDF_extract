from langgraph.graph import END
from modules.tools import check_chunks
from modules.states import AgentState

# ROUTER
def condition_a1(state: AgentState) -> str:
    """Router cho extracted_agent"""
    retry_count = state.get("retry_count_a1", 0)
    if state.get("error"):
        if retry_count >= 3:
            return "error_final"
        return "agent_a1"
    if not state.get("cleaned_text"):
        if retry_count >= 3:
            return "error_final"
        return "agent_a1"
    return "agent_a2"

def condition_a2(state: AgentState) -> str:
    """Router cho chunked_and_embedded_agent"""
    retry_count = state.get("retry_count_a2", 0)
    if state.get("error"):
        if retry_count >= 3:
            return "error_final"
        return "agent_a2"
    if not check_chunks(state.get("chunks", [])):
        if retry_count >= 3:
            return "error_final"
        return "agent_a2"
    return "agent_analyze"

def condition_analyze(state: AgentState) -> str:
    """Router cho analyzed_agent"""
    retry_count = state.get("retry_count_analyze", 0)
    
    if state.get("error"):
        # Kiểm tra tổng số retry trước
        if retry_count >= 3:
            return "error_final"
            
        # Nếu lỗi là do chunk size và chưa retry quá 2 lần, quay lại A2
        if (retry_count <= 1 and 
            any(msg.get("to") == "agent_a2" and msg.get("action") == "adjust_chunk" 
                for msg in state.get("messages", []))):
            return "agent_a2"
        
        # Các trường hợp khác: retry analyze hoặc kết thúc
        if retry_count < 3:
            return "agent_analyze"
        else:
            return "error_final"
    
    # Kiểm tra kết quả - nếu không có summary hoặc entities
    if not state.get("summary"):
        if retry_count >= 3:
            return "error_final"
        return "agent_analyze"
        
    # Summary-only mode hoặc có entities đầy đủ
    if state.get("summary_only_mode") or state.get("entities"):
        return "agent_verify"
    
    # Default: retry nếu chưa đủ retry count
    if retry_count < 3:
        return "agent_analyze"
    else:
        return "error_final"

def condition_verify(state: AgentState) -> str:
    """Router cho verified_agent"""
    if state.get("error"):
        # Kiểm tra retry count để tránh vòng lặp
        total_retries = (state.get("retry_count_analyze", 0) + 
                        state.get("retry_count_verify", 0))
        
        if total_retries >= 5:  # Tổng retry limit
            return "error_final"
            
        # Nếu message yêu cầu reanalyze và chưa retry quá nhiều
        if (total_retries <= 3 and
            any(msg.get("to") == "agent_analyze" and msg.get("action") == "reanalyze" 
                for msg in state.get("messages", []))):
            return "agent_analyze"
        
        # Các trường hợp lỗi khác: kết thúc
        return "error_final"
        
    # Kiểm tra verified_data
    if not state.get("verified_data"):
        verify_retries = state.get("retry_count_verify", 0)
        if verify_retries >= 2:
            return "error_final"
        return "agent_verify"
        
    return "agent_aggregate"

def condition_aggregate(state: AgentState) -> str:
    """Router cho aggregated_agent"""
    if state.get("error"):
        return "error_final"  # Aggregate errors always end
        
    if not state.get("report"):
        agg_retries = state.get("retry_count_aggregate", 0)
        if agg_retries >= 2:
            return "error_final"
        return "agent_aggregate"
        
    return END

def get_total_retries(state: AgentState) -> int:
    """Helper function để tính tổng số retries"""
    return (state.get("retry_count_a1", 0) + 
            state.get("retry_count_a2", 0) + 
            state.get("retry_count_analyze", 0) + 
            state.get("retry_count_verify", 0) + 
            state.get("retry_count_aggregate", 0))