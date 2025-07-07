# Hệ Thống Multi-Agent Trích Xuất PDF

Hệ thống sử dụng kiến trúc Multi-Agent để trích xuất và phân tích thông tin từ tài liệu PDF, với trọng tâm là xử lý song song và tối ưu hiệu suất thông qua việc phân chia công việc cho các agent chuyên biệt.

## Thư Viện Sử Dụng

Dựa trên imports trong code:
- `langgraph`: Xây dựng StateGraph cho luồng xử lý multi-agent
- `langchain_openai`: Tương tác với OpenAI API (ChatOpenAI)
- `pydantic`: Định nghĩa model cho output
- `python-dotenv`: Quản lý biến môi trường (OPENAI_API_KEY, MODEL_NAME)

## Luồng Xử Lý Multi-Agent

```mermaid
graph TD
    START(("Start")) --> A1["Agent A1<br/>extract_pdf.invoke"]
    
    A1 --> A1_C{"cleaned_text?<br/><small>retry_count_a1 < 3</small>"}
    A1_C -->|"Không"| A1
    A1_C -->|"Lỗi ≥ 3"| ERR["Error Handler"]
    A1_C -->|"OK"| A2["Agent A2<br/>chunk_and_embed.invoke"]
    
    A2 --> A2_C{"check_chunks?<br/><small>retry_count_a2 < 3</small>"}
    A2_C -->|"Không"| A2
    A2_C -->|"Lỗi ≥ 3"| ERR
    A2_C -->|"OK"| AN["Agent Analyze<br/>Xử lý song song<br/><small>batch_size=20, workers=15</small>"]
    
    AN --> AN_SAVE["analyze_intermediate.json<br/><small>summaries, entities, stats</small>"]
    AN_SAVE --> V["Agent Verify<br/>search_tool.invoke"]
    
    V --> V_C{"verified?<br/><small>retry_count_analyze < 3</small>"}
    V_C -->|"Không"| AN
    V_C -->|"Lỗi ≥ 3"| ERR_F["Error Final"]
    V_C -->|"OK"| AG["Agent Aggregate<br/>FinalOutput"]
    
    ERR --> AG
    ERR_F --> AG
    
    AG --> AG_C{"report?"}
    AG_C -->|"Không"| V
    AG_C -->|"Có"| END(("End"))

    style START fill:#9cf
    style END fill:#9cf
    style ERR fill:#ffcccc
    style ERR_F fill:#ffcccc
    style AN_SAVE fill:#f9f,stroke:#333,stroke-width:2px
```

## Các Module Trong Hệ Thống

### 1. `modules/agents.py`
Định nghĩa các agents và luồng xử lý:

**Các Agent Chính:**
- `agent_a1_node`: Trích xuất nội dung PDF
- `agent_a2_node`: Phân đoạn và tạo embeddings
- `agent_analyze_node`: 
  - Xử lý song song với batch_size=20
  - ThreadPoolExecutor(max_workers=15)
  - Delay 0.2s giữa các batches
  - Lưu kết quả trung gian vào analyze_intermediate.json
- `agent_verify_node`: Xác minh kết quả với search_tool
- `agent_aggregate_node`: Tạo output theo FinalOutput model

**Xử lý Lỗi:**
- Mỗi agent có 3 lần retry
- Các error handlers: error_handler và error_final_handler
- Logging với timestamp và emoji

### 2. `modules/states.py`
Định nghĩa trạng thái của hệ thống:
```python
class AgentState(TypedDict):
    file_path: str                      # Đường dẫn PDF
    cleaned_text: Optional[str]         # Text sau xử lý
    chunks: List[str]                   # Các đoạn văn bản
    embeddings: List[List[float]]       # Vector embeddings
    db: str                            # FAISS index
    question: str                       # Câu hỏi
    summary: Optional[str]              # Tóm tắt
    entities: Optional[Dict[str, Any]]  # Entities
    verified_data: Optional[Dict[str, Any]]  # Data đã verify
    report: Optional[str]               # JSON output
    error: Optional[str]                # Lỗi
    messages: List[Dict[str, str]]      # Messages giữa agents
    retry_count_a1: int                 # Số lần retry A1
    retry_count_a2: int                 # Số lần retry A2
    retry_count_analyze: int            # Số lần retry Analyze
```

### 3. Output Format
```python
class FinalOutput(BaseModel):
    answer: str = Field(description="Câu trả lời cho câu hỏi")
    summary: str = Field(description="Tóm tắt nội dung")
    entities: Dict[str, Any] = Field(description="Entities trích xuất")
    verified_data: Dict[str, Any] = Field(description="Dữ liệu đã xác minh")
```

## Cấu Trúc Project
```
PDF_extract/
  ├── data/              # Thư mục PDF
  ├── main.py           # Entry point
  ├── modules/
  │   ├── __init__.py
  │   ├── agents.py     # Các agents
  │   ├── states.py     # AgentState
  │   └── tools.py      # Công cụ hỗ trợ
  └── requirements.txt  # Dependencies
```

## Hiệu Suất Xử Lý

### Tối ưu song song:
- Batch size: 20 chunks/lần
- Workers: 15 threads đồng thời
- Delay: 0.2s giữa các batch
- Token limit: 150k/chunk

### Rate Limits:
- 200k tokens/phút
- 500 requests/phút
- Xử lý chunk thông minh

### Xử lý lỗi:
- Retry tự động (3 lần/agent)
- Logging chi tiết
- Lưu trạng thái trung gian
- Fallback strategies

## Thời Gian Xử Lý
- Trích xuất PDF: ~13s
- Phân đoạn & embedding: ~13s
- Phân tích nội dung: ~3-4s
- Tổng thời gian: 30-35s

## Cài Đặt

1. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

2. Thiết lập môi trường:
```bash
OPENAI_API_KEY=your_api_key
MODEL_NAME=gpt-3.5-turbo
```

3. Chuẩn bị file PDF trong thư mục data

## Performance Metrics

- Total processing time: 30-35s (improved from 15-20s)
- PDF extraction: ~13s
- Chunking & embeddings: ~13s
- Content analysis: ~3-4s
- Verification & aggregation: ~1-2s

## State Management

The system uses a TypedDict-based state management system (`AgentState`) to track:
- File processing status
- Extracted text and chunks
- Embeddings and search indices
- Analysis results and entities
- Error states and retry counts
- Inter-agent messages

## Error Handling

- Automatic retries (3 attempts per agent)
- Detailed logging with timestamps
- Error classification and recovery
- State preservation during retries
- Graceful degradation options

## Output Format

The final output is a JSON structure containing:
- Answer to the specific question
- Document summary
- Extracted entities (names, dates, locations, numbers)
- Verified data points
- Processing metadata

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 