# 🚀 Demo Streamlit - PDF Extract Multi-Agent System

## Mô tả

Demo Streamlit cho hệ thống Multi-Agent trích xuất và phân tích PDF với giao diện web thân thiện. Người dùng chỉ cần upload file PDF, nhập OpenAI API key, chọn model và nhấn Extract.

## Cài đặt

### 1. Cài đặt dependencies

#### Cách 1: Automatic Setup (Khuyến nghị)
```bash
python setup.py
```

#### Cách 2: Manual Setup
```bash
# Cài đặt dependencies cơ bản
pip install -r requirements.txt

# Hoặc nếu bạn muốn development environment
pip install -r requirements-dev.txt

# Tạo thư mục cần thiết
mkdir uploads outputs temp_files indices
```

**Lưu ý**: 
- File requirements.txt đã được cập nhật với version chính xác từ môi trường hiện tại
- Script `setup.py` sẽ tự động kiểm tra Python version, cài đặt dependencies, tạo thư mục và tạo .env template

### 2. Chạy ứng dụng Streamlit

```bash
streamlit run streamlit_app.py
```

Ứng dụng sẽ mở trên `http://localhost:8501`

## Cách sử dụng

### 1. Giao diện chính

- **Sidebar**: Cấu hình API key và model
- **Cột trái**: Upload file PDF và extract
- **Cột phải**: Hiển thị kết quả

### 2. Quy trình sử dụng

1. **Nhập OpenAI API Key** trong sidebar
2. **Chọn Model** (mặc định: gpt-4o-mini)
3. **Upload file PDF** (hỗ trợ .pdf)
4. **Nhấn "Extract Information"**
5. **Xem kết quả** trong các tab:
   - 📋 Tóm tắt
   - 🏷️ Entities
   - 📄 Raw Data

### 3. Tính năng

- ✅ Upload multiple PDF files
- ✅ Automatic file naming với timestamp
- ✅ Real-time processing status
- ✅ Download results as JSON
- ✅ View previous results
- ✅ Error handling và retry
- ✅ Clean file organization

## Cấu trúc thư mục

```
PDF_extract/
├── streamlit_app.py      # Ứng dụng Streamlit chính
├── main.py              # CLI interface
├── uploads/             # File PDF được upload
├── outputs/             # Kết quả JSON
├── temp_files/          # File tạm (chunks, intermediate)
├── indices/             # FAISS index files
└── modules/             # Core modules
    ├── agents.py        # Multi-agent logic
    ├── graphs.py        # LangGraph workflow
    ├── states.py        # State management
    ├── tools.py         # PDF extraction tools
    └── routers.py       # Edge conditions
```

## API Keys

### OpenAI API Key

1. Đăng ký tại [OpenAI Platform](https://platform.openai.com/)
2. Tạo API key trong mục "API Keys"
3. Nhập vào sidebar của ứng dụng

**Lưu ý**: API key chỉ được lưu trong session, không được lưu trữ.

## Models hỗ trợ

- **gpt-4o-mini** (recommended) - Nhanh, tiết kiệm chi phí
- **gpt-4o** - Chất lượng cao nhất
- **gpt-3.5-turbo** - Cân bằng tốc độ/chất lượng
- **gpt-4-turbo** - Phiên bản turbo của GPT-4

## Hiệu suất

### Thời gian xử lý (ước tính)
- **PDF nhỏ (< 10 trang)**: 15-25 giây
- **PDF trung bình (10-50 trang)**: 30-60 giây  
- **PDF lớn (> 50 trang)**: 1-3 phút

### Tối ưu hóa
- ✅ Parallel processing (batch_size=20, workers=15)
- ✅ Smart retry mechanism (3 attempts per agent)
- ✅ Rate limit handling
- ✅ Memory management

## CLI Interface

Ngoài Streamlit, bạn cũng có thể sử dụng command line:

```bash
# Cơ bản
python main.py --file path/to/document.pdf

# Với câu hỏi tùy chỉnh
python main.py --file document.pdf --question "Tóm tắt nội dung chính"

# Lưu kết quả ra file
python main.py --file document.pdf --output results.json
```

## File outputs

### JSON Structure

```json
{
  "timestamp": "20250108_112345",
  "input_file": "document.pdf",
  "result": {
    "report": "{...}",
    "summary": "...",
    "entities": {
      "names": ["..."],
      "dates": ["..."],
      "locations": ["..."],
      "numbers": ["..."]
    },
    "verified_data": {...}
  }
}
```

### Download Options

- 📥 Download individual results
- 📁 Browse previous results
- 🗂️ Organized by timestamp

## Troubleshooting

### Lỗi thường gặp

1. **"OpenAI API Key not found"**
   - Đảm bảo đã nhập API key trong sidebar
   - Kiểm tra API key hợp lệ

2. **"Rate limit exceeded"**
   - Đợi vài phút và thử lại
   - Hệ thống có retry tự động

3. **"PDF extraction failed"**
   - Kiểm tra file PDF không bị corrupt
   - Đảm bảo file có text (không phải scan)

4. **Streamlit connection error**
   - Restart ứng dụng: `Ctrl+C` và chạy lại
   - Kiểm tra port 8501 không bị chiếm

### Performance Issues

- **Slow processing**: Kiểm tra internet connection
- **Memory errors**: Xử lý file PDF nhỏ hơn
- **API errors**: Kiểm tra API key và credits

## Environment Variables (tùy chọn)

Tạo file `.env` để set mặc định:

```bash
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-4o-mini
```

## Security

- ⚠️ **Không chia sẻ API key**
- ⚠️ **Không commit API key vào git**
- ✅ File uploads được lưu local
- ✅ API key chỉ tồn tại trong session

## Support

Nếu gặp vấn đề:

1. Kiểm tra logs trong terminal
2. Restart ứng dụng
3. Kiểm tra file trong thư mục `temp_files/` cho debug info

---

**🏗️ Built with**: Streamlit, LangChain, OpenAI, FAISS, LangGraph 