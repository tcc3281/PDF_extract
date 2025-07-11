from typing import TypedDict, Optional, Dict, Any, List
from dataclasses import dataclass, field

@dataclass
class AgentState:
    # Required fields
    file_path: str                      # Đường dẫn file PDF
    question: str                       # Câu hỏi người dùng
    
    # Optional fields with default values
    cleaned_text: Optional[str] = None
    chunks: List[str] = field(default_factory=list)
    embeddings: List[Any] = field(default_factory=list)
    db: str = ""
    summary: Optional[str] = None
    entities: Optional[Dict] = None
    verified_data: Optional[Dict] = None
    report: Optional[str] = None
    error: Optional[str] = None
    messages: List[Dict] = field(default_factory=list)
    
    # Retry counters
    retry_count_a1: int = 0
    retry_count_a2: int = 0
    retry_count_analyze: int = 0
    retry_count_verify: int = 0
    retry_count_aggregate: int = 0
    
    # Model configurations
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    embedding_model: Optional[str] = None
    
    # Special modes
    summary_only_mode: Optional[bool] = None

    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []
        if self.embeddings is None:
            self.embeddings = []
        if self.messages is None:
            self.messages = []

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)