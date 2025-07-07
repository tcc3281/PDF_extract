from typing import TypedDict, Optional, Dict, Any, List

class AgentState(TypedDict):
    file_path: str                      # Đường dẫn file PDF
    cleaned_text: Optional[str]         # Văn bản đã làm sạch
    chunks: List[str]                   # Danh sách các đoạn văn bản
    embeddings: List[List[float]]       # Danh sách vector embedding
    db: str                            # Đường dẫn FAISS index
    question: str                       # Câu hỏi người dùng
    summary: Optional[str]              # Tóm tắt nội dung
    entities: Optional[Dict[str, Any]]  # Entities trích xuất
    verified_data: Optional[Dict[str, Any]]  # Dữ liệu đã xác minh
    report: Optional[str]               # Báo cáo JSON cuối cùng
    error: Optional[str]                # Lỗi nếu có
    messages: List[Dict[str, str]]      # Thông điệp giữa các tác nhân
    retry_count_a1: int                 # Số lần retry của A1
    retry_count_a2: int                 # Số lần retry của A2
    retry_count_analyze: int            # Số lần retry của Analyze