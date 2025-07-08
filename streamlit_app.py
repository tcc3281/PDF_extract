import streamlit as st
import json
import os
import logging
import queue
from datetime import datetime
from pathlib import Path
import threading

from modules.graphs import build_graph
from modules.states import AgentState

# Thiết lập page
st.set_page_config(
    page_title="PDF Extract",
    page_icon="📄",
    layout="wide"
)

# Khởi tạo session state
if 'logs' not in st.session_state:
    st.session_state.logs = []

# Tạo global log queue để thread-safe
log_queue = queue.Queue()

# Thiết lập logging
class ThreadSafeLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
    
    def emit(self, record):
        log_entry = self.format(record)
        # Sử dụng queue thay vì truy cập trực tiếp session_state
        log_queue.put(log_entry)

# Khởi tạo logger
logger = logging.getLogger()
handler = ThreadSafeLogHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Tạo thư mục cần thiết
for folder in ["uploads", "outputs", "temp_files", "indices"]:
    Path(folder).mkdir(exist_ok=True)

def save_uploaded_file(uploaded_file):
    """Lưu file PDF được upload"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"uploads/{timestamp}_{uploaded_file.name}"
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def get_optimized_prompt():
    """Tạo prompt tối ưu cho việc trích xuất thông tin"""
    return """Trích xuất thông tin quan trọng, chính xác và ngắn gọn từ tài liệu này. 
Tập trung vào:
1. Các dữ kiện chính (facts) và số liệu quan trọng
2. Tên người, tổ chức, địa điểm và thời gian chính xác
3. Các mốc thời gian và sự kiện quan trọng
4. Các thông tin định lượng (số liệu, thống kê)

Kết quả cần:
- Ngắn gọn, súc tích, không dài dòng
- Chính xác, trung thực với nội dung gốc
- Có cấu trúc rõ ràng
- Ưu tiên thông tin có giá trị cao

Bỏ qua các thông tin:
- Mang tính chủ quan, đánh giá
- Thông tin trùng lặp
- Chi tiết không quan trọng
- Nội dung mang tính quảng cáo"""

def run_extraction(file_path, api_key, model_name, embedding_model):
    """Chạy quá trình trích xuất PDF"""
    try:
        # Khởi tạo state với prompt tối ưu
        initial_state = AgentState(
            file_path=file_path,
            question=get_optimized_prompt(),
            cleaned_text=None,
            chunks=[],
            embeddings=[],
            db="",
            summary=None,
            entities=None,
            verified_data=None,
            report=None,
            error=None,
            messages=[],
            retry_count_a1=0,
            retry_count_a2=0,
            retry_count_analyze=0,
            api_key=api_key,
            model_name=model_name,
            embedding_model=embedding_model
        )
        
        # Xây dựng và chạy graph
        graph = build_graph()
        result = graph.invoke(initial_state)
        
        return result, None
    except Exception as e:
        return None, str(e)

def save_result_json(result, filename):
    """Lưu kết quả vào file JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{timestamp}_{Path(filename).stem}_result.json"
    
    # Chuyển đổi AgentState thành dict
    if hasattr(result, '__dict__'):
        result_dict = {}
        for key in ['file_path', 'cleaned_text', 'chunks', 'embeddings', 'db', 
                   'question', 'summary', 'entities', 'error']:
            if hasattr(result, key):
                result_dict[key] = getattr(result, key)
    else:
        result_dict = result
    
    # Tạo output data
    output_data = {
        "timestamp": timestamp,
        "input_file": filename,
        "result": result_dict
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return output_path

def main():
    st.title("📄 PDF Extract - Trích xuất thông tin quan trọng")
    
    # Sidebar cho cấu hình
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        api_key = st.text_input("OpenAI API Key", type="password")
        
        model_options = ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-4.1-mini"]
        model = st.selectbox("Chọn Model", model_options, index=0)
        
        embedding_options = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
        embedding_model = st.selectbox("Chọn Embedding Model", embedding_options, index=0)
        
        with st.expander("ℹ️ Về công cụ này"):
            st.write("""
            **PDF Extract** là công cụ multi-agent được tối ưu hóa để:
            - Trích xuất thông tin quan trọng và chính xác từ PDF
            - Tóm tắt nội dung một cách ngắn gọn, súc tích
            - Nhận diện entities: tên, địa điểm, thời gian, số liệu
            - Loại bỏ thông tin thừa, không quan trọng
            """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload PDF")
        
        uploaded_file = st.file_uploader("Chọn file PDF", type=['pdf'])
        
        if uploaded_file:
            st.success(f"✅ Đã chọn: {uploaded_file.name}")
            
            if st.button("🚀 Extract Thông Tin Quan Trọng", type="primary", disabled=not api_key):
                if not api_key:
                    st.error("⚠️ Vui lòng nhập API key")
                else:
                    # Reset logs
                    st.session_state.logs = []
                    
                    with st.spinner("Đang trích xuất thông tin quan trọng..."):
                        # Lưu file
                        file_path = save_uploaded_file(uploaded_file)
                        
                        # Chạy extraction với các model được chọn
                        result, error = run_extraction(
                            file_path=file_path,
                            api_key=api_key,
                            model_name=model,
                            embedding_model=embedding_model
                        )
                        
                        # Lấy logs từ queue
                        while not log_queue.empty():
                            try:
                                log_entry = log_queue.get_nowait()
                                st.session_state.logs.append(log_entry)
                            except:
                                break
                        
                        if error:
                            st.error(f"❌ Lỗi: {error}")
                        else:
                            # Lưu kết quả
                            output_path = save_result_json(result, uploaded_file.name)
                            
                            # Lưu vào session state
                            st.session_state.result = result
                            st.session_state.output_path = output_path
                            
                            st.success("✅ Trích xuất thành công!")
                            st.balloons()
        else:
            st.info("👆 Vui lòng upload file PDF để bắt đầu trích xuất thông tin quan trọng")
    
    with col2:
        st.header("Kết quả trích xuất")
        
        # Cập nhật logs từ queue
        while not log_queue.empty():
            try:
                log_entry = log_queue.get_nowait()
                st.session_state.logs.append(log_entry)
            except:
                break
        
        # Hiển thị logs
        if st.session_state.logs:
            with st.expander("📋 Logs", expanded=False):
                for log in st.session_state.logs[-20:]:
                    st.text(log)
        
        # Hiển thị kết quả
        if 'result' in st.session_state and st.session_state.result:
            result = st.session_state.result
            
            # Lấy data từ result
            if hasattr(result, 'summary'):
                summary = result.summary
                entities = result.entities
            else:
                summary = result.get('summary')
                entities = result.get('entities', {})
            
            # Hiển thị kết quả
            tab1, tab2, tab3 = st.tabs(["📝 Thông tin quan trọng", "🏷️ Entities", "📄 Raw Data"])
            
            with tab1:
                if summary:
                    st.markdown("### Thông tin quan trọng")
                    st.write(summary)
                else:
                    st.info("Chưa có thông tin được trích xuất")
            
            with tab2:
                if entities:
                    st.markdown("### Entities đã trích xuất")
                    
                    # Hiển thị các loại entities
                    for entity_type, icon in [
                        ('names', '👤'), 
                        ('dates', '📅'), 
                        ('locations', '📍'),
                        ('numbers', '🔢')
                    ]:
                        if entity_type in entities and entities[entity_type]:
                            st.write(f"**{icon} {entity_type.capitalize()}:**")
                            for item in entities[entity_type]:
                                st.write(f"- {item}")
                else:
                    st.info("Không có entities")
            
            with tab3:
                st.markdown("### Raw Data")
                
                # Convert to dict if needed
                if hasattr(result, '__dict__'):
                    result_dict = {}
                    for key in ['summary', 'entities', 'cleaned_text']:
                        if hasattr(result, key):
                            result_dict[key] = getattr(result, key)
                    st.json(result_dict)
                else:
                    st.json(result)
            
            # Download button
            if 'output_path' in st.session_state:
                try:
                    with open(st.session_state.output_path, 'r', encoding='utf-8') as f:
                        json_data = f.read()
                    
                    st.download_button(
                        "📥 Download Kết Quả (JSON)",
                        json_data,
                        file_name=f"extracted_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Lỗi tải file: {str(e)}")
        else:
            st.info("👈 Upload file và nhấn Extract để xem thông tin quan trọng được trích xuất")

if __name__ == "__main__":
    main() 