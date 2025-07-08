from modules.tools import search_tool, extract_pdf, chunk_and_embed, check_chunks
from modules.states import AgentState
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import json
import logging
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")  # default if not set
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    api_key=API_KEY
)


def extracted_agent(state: AgentState) -> AgentState:
    logging.info("🚀 Agent A1: Bắt đầu trích xuất PDF...")
    retry_count = state.get("retry_count_a1", 0)
    if retry_count >= 3:
        logging.error("❌ Agent A1: Đã thử 3 lần nhưng không thành công")
        return {"error": "Invalid PDF after 3 retries", "messages": state.get("messages", [])}
    
    messages = state.get("messages", [])
    if any(msg.get("to") == "agent_a1" and msg.get("action") == "retry_clean" for msg in messages):
        logging.info("🔄 Agent A1: Thử lại việc làm sạch và OCR...")
    
    result = extract_pdf.invoke({"pdf_path": state["file_path"]})
    
    if result["error"]:
        logging.error(f"❌ Agent A1: Lỗi khi trích xuất PDF - {result['error']}")
    else:
        logging.info(f"✅ Agent A1: Đã trích xuất thành công {len(result['cleaned_text'])} ký tự")
    
    return {
        "cleaned_text": result["cleaned_text"],
        "error": result["error"],
        "retry_count_a1": retry_count + 1,
        "messages": messages
    }


def chunked_and_embedded_agent(state: AgentState) -> AgentState:
    logging.info("🚀 Agent A2: Bắt đầu chia nhỏ và tạo embeddings...")
    retry_count = state.get("retry_count_a2", 0)
    if retry_count >= 3:
        logging.error("❌ Agent A2: Đã thử 3 lần nhưng không thành công")
        return {"error": "Invalid chunks after 3 retries", "messages": state.get("messages", [])}
    
    if state["error"] or not state["cleaned_text"]:
        logging.error(f"❌ Agent A2: Không có text để xử lý - {state['error'] or 'No cleaned text'}")
        return {"error": state["error"] or "No cleaned text", "messages": state.get("messages", [])}
    
    messages = state.get("messages", [])
    chunk_size, chunk_overlap = 2000, 200
    for msg in messages:
        if msg.get("to") == "agent_a2" and msg.get("action") == "adjust_chunk":
            chunk_size = int(msg.get("chunk_size", 2000))
            chunk_overlap = int(msg.get("chunk_overlap", 200))
            logging.info(f"🔄 Agent A2: Điều chỉnh kích thước chunk={chunk_size}, overlap={chunk_overlap}")
    
    # Tạo file_id từ file_path
    file_id = Path(state["file_path"]).stem if state.get("file_path") else "unknown"
    
    result = chunk_and_embed.invoke({
        "text": state["cleaned_text"],
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "file_id": file_id
    })

    if result.get("error"):
        logging.error(f"❌ Agent A2: Lỗi khi tạo embeddings - {result['error']}")
        return {
            "error": result["error"],
            "retry_count_a2": retry_count + 1,
            "messages": messages
        }

    if not check_chunks(result["chunks"]):
        logging.warning("⚠️ Agent A2: Chunks không hợp lệ, thử điều chỉnh kích thước...")
        return {
            "error": "Invalid chunks",
            "retry_count_a2": retry_count + 1,
            "messages": messages + [{"to": "agent_a2", "action": "adjust_chunk", "chunk_size": chunk_size - 500, "chunk_overlap": chunk_overlap - 50}]
        }

    logging.info(f"✅ Agent A2: Đã tạo {len(result['chunks'])} chunks và embeddings thành công")
    return {
        "chunks": result["chunks"],
        "embeddings": result["embeddings"],
        "db": result["db"],
        "retry_count_a2": retry_count + 1,
        "error": None,
        "messages": messages
    }


summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Nhiệm vụ của bạn là trích xuất thông tin quan trọng, chính xác và ngắn gọn từ văn bản. 
Tập trung vào:
1. Dữ kiện chính (facts) và thông tin cốt lõi
2. Tên người, tổ chức quan trọng
3. Địa điểm và thời gian cụ thể
4. Số liệu định lượng và thống kê
5. Mối quan hệ giữa các thực thể

Bỏ qua:
- Thông tin trùng lặp
- Chi tiết không quan trọng
- Nội dung mang tính quảng cáo
- Đánh giá chủ quan

Tóm tắt ngắn gọn, súc tích (tối đa 100 từ), chỉ giữ lại thông tin quan trọng nhất."""),
    ("user", "{text}")
])

extract_prompt = ChatPromptTemplate.from_messages([
    ("system", """Trích xuất các thực thể (entities) quan trọng từ văn bản sau một cách chính xác và đầy đủ.
Phân loại thành 4 nhóm:
1. names: Tên người, tổ chức, công ty, thương hiệu quan trọng
2. dates: Ngày tháng, mốc thời gian, khoảng thời gian
3. locations: Địa điểm, quốc gia, thành phố, khu vực địa lý
4. numbers: Số liệu thống kê, tiền tệ, phần trăm, số đo lường

Chỉ trích xuất các entities thực sự quan trọng và có giá trị thông tin cao.
Bỏ qua các entities không rõ ràng hoặc không quan trọng.
Trả về đúng định dạng JSON: {{"entities": {{"names": [], "dates": [], "locations": [], "numbers": []}}}}"""),
    ("user", "{text}")
])

final_summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Tổng hợp các phần thông tin đã được trích xuất thành một bản tóm tắt hoàn chỉnh, ngắn gọn và có cấu trúc.
Yêu cầu:
1. Ngắn gọn, súc tích, không quá 500 từ
2. Chỉ giữ lại thông tin quan trọng và có giá trị cao
3. Sắp xếp thông tin theo thứ tự logic và dễ hiểu
4. Liên kết các thông tin có liên quan với nhau
5. Đảm bảo chính xác và khách quan
6. Tập trung vào dữ kiện, số liệu và mối quan hệ giữa các thực thể
7. Loại bỏ thông tin trùng lặp, không quan trọng hoặc mang tính chủ quan

Mục đích: Giúp người đọc nắm bắt nhanh chóng những thông tin quan trọng nhất từ văn bản gốc."""),
    ("user", "{summaries}")
])

def analyze_chunk_batch(chunk: str) -> Dict:
    """Xử lý một chunk đơn lẻ - để dùng trong parallel processing"""
    try:
        # Xử lý summary
        summary_result = llm.invoke(summarize_prompt.format(text=chunk))
        summary = summary_result.content if hasattr(summary_result, "content") else str(summary_result)
        
        # Xử lý entities với error handling tốt hơn
        entities_result = llm.invoke(extract_prompt.format(text=chunk))
        
        # Fix: Xử lý đúng ChatPromptValue và các định dạng khác
        if hasattr(entities_result, "content"):
            entities_content = entities_result.content
        elif hasattr(entities_result, "text"):
            entities_content = entities_result.text
        else:
            entities_content = str(entities_result)
        
        # Parse JSON entities
        try:
            if isinstance(entities_content, str):
                entities_data = json.loads(entities_content)
                if "entities" in entities_data:
                    entities = entities_data["entities"]
                else:
                    entities = entities_data
            else:
                entities = {"names": [], "dates": [], "locations": [], "numbers": []}
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"⚠️ Không thể parse entities JSON: {str(e)}")
            entities = {"names": [], "dates": [], "locations": [], "numbers": []}
        
        # Đảm bảo entities có đúng structure
        if not isinstance(entities, dict):
            entities = {"names": [], "dates": [], "locations": [], "numbers": []}
        
        for key in ["names", "dates", "locations", "numbers"]:
            if key not in entities:
                entities[key] = []
            elif not isinstance(entities[key], list):
                entities[key] = []
        
        result = {
            "summary": summary,
            "entities": entities
        }
        
        logging.info(f"✅ Chunk processed: {len(chunk)} chars, entities: {sum(len(v) for v in entities.values())} items")
        return result
        
    except Exception as e:
        logging.error(f"❌ Error processing chunk: {str(e)}")
        return {
            "summary": "Error processing this chunk",
            "entities": {"names": [], "dates": [], "locations": [], "numbers": []}
        }

def analyze_batch_parallel(batch: List[str], max_workers: int = 10) -> List[Dict]:
    """Xử lý batch với parallel processing để tăng tốc"""
    try:
        results = []
        
        # Sử dụng ThreadPoolExecutor để xử lý parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tất cả chunks trong batch
            future_to_chunk = {
                executor.submit(analyze_chunk_batch, chunk): chunk 
                for chunk in batch
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                try:
                    result = future.result(timeout=30)  # 30s timeout per chunk
                    if result:
                        results.append(result)
                except Exception as e:
                    logging.error(f"❌ Timeout or error in parallel processing: {str(e)}")
                    # Add fallback result
                    results.append({
                        "summary": "Error processing chunk",
                        "entities": {"names": [], "dates": [], "locations": [], "numbers": []}
                    })
        
        return results
    except Exception as e:
        logging.error(f"❌ Error in parallel batch processing: {str(e)}")
        return []

def count_tokens_estimate(text: str) -> int:
    """Ước tính số tokens (khoảng 3.5 ký tự = 1 token cho tiếng Việt)"""
    return len(text) // 3

def chunk_summaries_for_final(summaries: List[str], max_tokens: int = 150000) -> List[List[str]]:
    """Chia summaries thành các chunks lớn hơn để tận dụng token limit cao"""
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for summary in summaries:
        summary_tokens = count_tokens_estimate(summary)
        
        if current_tokens + summary_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [summary]
            current_tokens = summary_tokens
        else:
            current_chunk.append(summary)
            current_tokens += summary_tokens
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def analyzed_agent(state: AgentState) -> AgentState:
    logging.info("🚀 Agent Analyze: Bắt đầu phân tích nội dung với hiệu suất cao...")
    retry_count = state.get("retry_count_analyze", 0)
    if retry_count >= 3:
        logging.error("❌ Agent Analyze: Đã thử 3 lần nhưng không thành công")
        return {"error": "Analysis failed after 3 retries", "messages": state.get("messages", [])}
    
    if state["error"] or not state["chunks"]:
        logging.error(f"❌ Agent Analyze: Không có chunks để phân tích - {state['error'] or 'No chunks available'}")
        return {"error": state["error"] or "No chunks available", "messages": state.get("messages", [])}

    try:
        # Tăng batch size để tận dụng rate limit cao
        batch_size = 20  # Tăng từ 5 lên 20 
        chunks = state["chunks"]
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        logging.info(f"📊 Agent Analyze: Xử lý {len(batches)} batches với batch_size={batch_size} (parallel mode)...")

        summaries = []
        entities = {"names": [], "dates": [], "locations": [], "numbers": []}
        
        # Xử lý các batches với parallel processing
        for i, batch in enumerate(batches):
            # Giảm delay xuống chỉ 0.2s giữa batches để tận dụng 500 RPM
            if i > 0:
                time.sleep(0.2)
            
            logging.info(f"🔄 Processing batch {i+1}/{len(batches)} với {len(batch)} chunks...")
            
            # Sử dụng parallel processing cho từng batch
            batch_results = analyze_batch_parallel(batch, max_workers=min(15, len(batch)))
            
            if not batch_results:
                logging.warning(f"⚠️ Batch {i+1} không có kết quả")
                continue
                
            for result in batch_results:
                if not result:
                    continue
                summaries.append(result["summary"])
                for key in entities:
                    if key in result["entities"] and isinstance(result["entities"][key], list):
                        entities[key].extend(result["entities"][key])
        
        if not summaries:
            raise ValueError("Không có summary nào được tạo thành công")
            
        # Loại bỏ duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        # Optimized final summary processing với token limit cao hơn
        try:
            # Tận dụng token limit 200k - chia thành chunks lớn hơn
            summary_chunks = chunk_summaries_for_final(summaries, max_tokens=150000)
            logging.info(f"📝 Chia {len(summaries)} summaries thành {len(summary_chunks)} chunks để xử lý")
            
            if len(summary_chunks) == 1:
                # Xử lý 1 chunk lớn
                final_summary_result = llm.invoke(final_summarize_prompt.format(summaries="\n".join(summary_chunks[0])))
                final_summary = final_summary_result.content if hasattr(final_summary_result, "content") else str(final_summary_result)
            else:
                # Xử lý parallel các summary chunks
                chunk_summaries = []
                
                def process_summary_chunk(chunk_data):
                    chunk_idx, chunk = chunk_data
                    try:
                        chunk_result = llm.invoke(final_summarize_prompt.format(summaries="\n".join(chunk)))
                        return chunk_result.content if hasattr(chunk_result, "content") else str(chunk_result)
                    except Exception as e:
                        logging.warning(f"⚠️ Lỗi khi xử lý summary chunk {chunk_idx}: {str(e)}")
                        return " ".join(chunk[:5])  # Fallback với nhiều summaries hơn
                
                # Parallel processing cho summary chunks
                with ThreadPoolExecutor(max_workers=min(5, len(summary_chunks))) as executor:
                    chunk_futures = {
                        executor.submit(process_summary_chunk, (i, chunk)): i 
                        for i, chunk in enumerate(summary_chunks)
                    }
                    
                    for future in as_completed(chunk_futures):
                        try:
                            result = future.result(timeout=60)  # Longer timeout for summary
                            chunk_summaries.append(result)
                        except Exception as e:
                            logging.warning(f"⚠️ Timeout in summary processing: {str(e)}")
                            chunk_summaries.append("Summary processing failed")
                
                # Final combination
                if len(chunk_summaries) > 1:
                    try:
                        time.sleep(0.5)  # Brief delay
                        final_result = llm.invoke(final_summarize_prompt.format(summaries="\n".join(chunk_summaries)))
                        final_summary = final_result.content if hasattr(final_result, "content") else str(final_result)
                    except Exception as e:
                        logging.warning(f"⚠️ Lỗi khi combine final summary: {str(e)}")
                        final_summary = "\n\n".join(chunk_summaries)
                else:
                    final_summary = chunk_summaries[0] if chunk_summaries else "Không thể tạo summary"
                    
        except Exception as e:
            logging.error(f"❌ Lỗi khi tạo final summary: {str(e)}")
            # Fallback với nhiều summaries hơn
            final_summary = "\n".join(summaries[:10]) + ("..." if len(summaries) > 10 else "")
        
        # Lưu intermediate results với tên file unique
        file_id = Path(state["file_path"]).stem if state.get("file_path") else "unknown"
        intermediate_filename = f"analyze_intermediate_{file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        intermediate_path = Path("temp_files") / intermediate_filename
        
        with open(intermediate_path, "w", encoding='utf-8') as f:
            json.dump({
                "summaries": summaries, 
                "entities": entities,
                "final_summary": final_summary,
                "performance_stats": {
                    "total_chunks": len(chunks),
                    "total_batches": len(batches),
                    "batch_size": batch_size,
                    "total_summaries": len(summaries),
                    "total_entities": sum(len(v) for v in entities.values())
                }
            }, f, ensure_ascii=False, indent=2)
        
        logging.info(f"""✅ Agent Analyze: Phân tích thành công (High Performance Mode):
        - {len(summaries)} summaries
        - {len(entities['names'])} tên
        - {len(entities['dates'])} ngày tháng  
        - {len(entities['locations'])} địa điểm
        - {len(entities['numbers'])} số liệu
        - Final summary: {len(final_summary)} ký tự
        - Processed {len(chunks)} chunks in {len(batches)} batches""")
        
        return {
            "summary": final_summary,
            "entities": entities,
            "retry_count_analyze": retry_count + 1,
            "error": None,
            "messages": state.get("messages", [])
        }
    except Exception as e:
        logging.error(f"❌ Agent Analyze: Lỗi khi phân tích - {str(e)}")
        return {
            "error": str(e),
            "retry_count_analyze": retry_count + 1,
            "messages": state.get("messages", [])
        }

verify_prompt = ChatPromptTemplate.from_messages([
    ("system", """Xác minh tính chính xác và đầy đủ của các thực thể (entities) đã được trích xuất.
Nhiệm vụ của bạn:
1. Kiểm tra xem các entities có liên quan đến chủ đề chính không
2. Xác minh tính chính xác của các entities (tên, ngày tháng, địa điểm, số liệu)
3. Đánh giá mức độ đầy đủ của thông tin đã trích xuất
4. Phát hiện các thông tin quan trọng bị bỏ sót

Nếu phát hiện vấn đề, gửi message yêu cầu xử lý lại (ví dụ: {{"to": "agent_a1", "action": "retry_clean"}}).
Trả về JSON: {{"verified": bool, "verified_data": dict, "message": dict}}"""),
    ("user", "Entities: {entities}")
])

def verified_agent(state: AgentState) -> AgentState:
    logging.info("🚀 Agent Verify: Bắt đầu xác minh kết quả...")
    if state["error"] or not state["entities"] or not state["db"]:
        logging.error(f"❌ Agent Verify: Thiếu dữ liệu để xác minh - {state['error'] or 'Missing data'}")
        return {"error": state["error"] or "Missing data", "messages": state.get("messages", [])}
    
    try:
        result = search_tool.invoke({
            "faiss_index": state["db"],
            "query": state["question"],
            "chunks": state["chunks"]
        })
        
        response = llm.invoke(verify_prompt.format(
            question=state["question"],
            summary=state["summary"],
            entities=str(result["entities"])
        ))
        
        parsed = json.loads(response.content)
        messages = state.get("messages", []) + [parsed["message"]] if parsed["message"] else state.get("messages", [])
        
        if parsed["verified"]:
            logging.info("✅ Agent Verify: Xác minh thành công")
        else:
            logging.warning("⚠️ Agent Verify: Xác minh thất bại, cần phân tích lại")
            
        return {
            "verified_data": parsed["verified_data"],
            "messages": messages,
            "error": None if parsed["verified"] else "Verification failed"
        }
    except Exception as e:
        logging.error(f"❌ Agent Verify: Lỗi khi xác minh - {str(e)}")
        return {"error": str(e), "messages": state.get("messages", [])}


from pydantic import BaseModel, Field
class FinalOutput(BaseModel):
    answer: str = Field(description="Câu trả lời cho câu hỏi")
    summary: str = Field(description="Tóm tắt nội dung")
    entities: Dict[str, Any] = Field(description="Entities trích xuất")
    verified_data: Dict[str, Any] = Field(description="Dữ liệu đã xác minh")

def aggregated_agent(state: AgentState) -> AgentState:
    logging.info("🚀 Agent Aggregate: Bắt đầu tổng hợp kết quả...")
    if state["error"]:
        logging.error(f"❌ Agent Aggregate: Không thể tổng hợp do lỗi - {state['error']}")
        return {"report": f"Error: {state['error']}", "messages": state.get("messages", [])}
    
    try:
        final_result = FinalOutput(
            answer=f"Response to '{state['question']}': {state['verified_data']}",
            summary=state["summary"],
            entities=state["entities"],
            verified_data=state["verified_data"]
        )
        
        logging.info("✅ Agent Aggregate: Đã tổng hợp thành công")
        return {"report": final_result.model_dump_json(), "error": None, "messages": state.get("messages", [])}
    except Exception as e:
        logging.error(f"❌ Agent Aggregate: Lỗi khi tổng hợp - {str(e)}")
        return {
            "error": str(e),
            "messages": state.get("messages", []) + [{"to": "agent_verify", "action": "reverify"}]
        }


