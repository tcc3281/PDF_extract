import os
import argparse
import json
from pathlib import Path
from modules.graphs import build_graph
from modules.states import AgentState

def run_extraction(file_path: str, question: str = "Trích xuất các nội dung quan trọng của văn bản mà người dùng có thể quan tâm, cần chú ý đến"):
    """
    Chạy quá trình trích xuất PDF
    
    Args:
        file_path (str): Đường dẫn đến file PDF
        question (str): Câu hỏi cho hệ thống
    
    Returns:
        dict: Kết quả xử lý
    """
    # Khởi tạo state
    initial_state = AgentState(
        file_path=file_path,
        question=question,
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
        retry_count_analyze=0
    )
    
    # Xây dựng và chạy graph
    graph = build_graph()
    result = graph.invoke(initial_state)
    
    return result

def main():
    """Main function cho command line interface"""
    parser = argparse.ArgumentParser(description="PDF Extract Multi-Agent System")
    parser.add_argument("--file", "-f", required=True, help="Đường dẫn đến file PDF")
    parser.add_argument("--question", "-q", default="Trích xuất các nội dung quan trọng của văn bản mà người dùng có thể quan tâm, cần chú ý đến", help="Câu hỏi cho hệ thống")
    parser.add_argument("--output", "-o", help="Đường dẫn file output (tùy chọn)")
    
    args = parser.parse_args()
    
    # Kiểm tra file tồn tại
    if not Path(args.file).exists():
        print(f"❌ File không tồn tại: {args.file}")
        return
    
    print(f"🚀 Bắt đầu xử lý file: {args.file}")
    print(f"❓ Câu hỏi: {args.question}")
    print("=" * 50)
    
    try:
        # Chạy extraction
        result = run_extraction(args.file, args.question)
        
        if result.get("error"):
            print(f"❌ Lỗi: {result['error']}")
        else:
            print("✅ Xử lý thành công!")
            
            # Parse và hiển thị kết quả
            if result.get("report"):
                try:
                    report_data = json.loads(result["report"])
                    
                    print("\n📋 TÓM TẮT:")
                    print("-" * 30)
                    print(report_data.get("summary", "Không có tóm tắt"))
                    
                    print("\n💬 CÂU TRẢ LỜI:")
                    print("-" * 30)
                    print(report_data.get("answer", "Không có câu trả lời"))
                    
                    print("\n🏷️ ENTITIES:")
                    print("-" * 30)
                    entities = report_data.get("entities", {})
                    
                    if entities.get("names"):
                        print(f"👥 Tên: {', '.join(entities['names'])}")
                    if entities.get("dates"):
                        print(f"📅 Ngày: {', '.join(entities['dates'])}")
                    if entities.get("locations"):
                        print(f"📍 Địa điểm: {', '.join(entities['locations'])}")
                    if entities.get("numbers"):
                        print(f"🔢 Số liệu: {', '.join(entities['numbers'])}")
                    
                    # Lưu file output nếu được chỉ định
                    if args.output:
                        output_path = Path(args.output)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(report_data, f, ensure_ascii=False, indent=2)
                        
                        print(f"\n💾 Kết quả đã lưu: {output_path}")
                
                except json.JSONDecodeError:
                    print("❌ Không thể parse JSON result")
                    print("Raw result:", result.get("report", ""))
            
    except Exception as e:
        print(f"❌ Lỗi không mong muốn: {str(e)}")

if __name__ == '__main__':
    main()