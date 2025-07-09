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
from pydantic import BaseModel, Field

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

load_dotenv()

def get_llm(api_key: str, model_name: str) -> ChatOpenAI:
    """Tạo ChatOpenAI instance"""
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key
    )

# Global LLM instance (will be set by the first call)
_global_llm = None

def set_global_llm(api_key: str, model_name: str):
    """Set global LLM instance"""
    global _global_llm
    _global_llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key
    )

def get_global_llm():
    """Get global LLM instance"""
    return _global_llm

def extracted_agent(state: AgentState) -> AgentState:
    retry_count = state.get("retry_count_a1", 0)
    logging.info(f"🚀 Agent A1: Bắt đầu trích xuất PDF (retry: {retry_count})...")
    
    if retry_count >= 3:
        logging.error("❌ Agent A1: Đã thử 3 lần nhưng không thành công")
        return {
            "error": "Invalid PDF after 3 retries", 
            "retry_count_a1": retry_count + 1,
            "messages": []
        }
    
    result = extract_pdf.invoke({"pdf_path": state["file_path"]})
    
    if result["error"]:
        logging.error(f"❌ Agent A1: Lỗi khi trích xuất PDF - {result['error']}")
        return {
            "error": result["error"],
            "retry_count_a1": retry_count + 1,
            "messages": []
        }
    else:
        logging.info(f"✅ Agent A1: Đã trích xuất thành công {len(result['cleaned_text'])} ký tự")
        return {
            "cleaned_text": result["cleaned_text"],
            "error": None,
            "retry_count_a1": retry_count + 1,
            "messages": []
        }

def chunked_and_embedded_agent(state: AgentState) -> AgentState:
    """Agent phân đoạn và tạo embeddings"""
    retry_count = state.get("retry_count_a2", 0)
    logging.info(f"🚀 Agent A2: Bắt đầu chia nhỏ và tạo embeddings (retry: {retry_count})...")
    
    if retry_count >= 3:
        logging.error("❌ Agent A2: Đã thử 3 lần nhưng không thành công")
        return {
            "error": "Invalid chunks after 3 retries", 
            "retry_count_a2": retry_count + 1,
            "messages": []
        }
    
    if state["error"] or not state["cleaned_text"]:
        logging.error(f"❌ Agent A2: Không có text để xử lý - {state['error'] or 'No cleaned text'}")
        return {
            "error": state["error"] or "No cleaned text", 
            "retry_count_a2": retry_count + 1,
            "messages": []
        }
    
    # Lấy chunk settings từ messages nếu có
    chunk_size, chunk_overlap = 2000, 200
    for msg in state.get("messages", []):
        if msg.get("to") == "agent_a2" and msg.get("action") == "adjust_chunk":
            chunk_size = int(msg.get("chunk_size", 2000))
            chunk_overlap = int(msg.get("chunk_overlap", 200))
            logging.info(f"🔄 Agent A2: Điều chỉnh kích thước chunk={chunk_size}, overlap={chunk_overlap}")
            break
    
    # Tạo file_id từ file_path
    file_id = Path(state["file_path"]).stem if state.get("file_path") else "unknown"
    
    result = chunk_and_embed.invoke({
        "text": state["cleaned_text"],
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "file_id": file_id,
        "api_key": state.api_key,
        "embedding_model": state.embedding_model
    })

    if result.get("error"):
        logging.error(f"❌ Agent A2: Lỗi khi tạo embeddings - {result['error']}")
        return {
            "error": result["error"],
            "retry_count_a2": retry_count + 1,
            "messages": []
        }

    if not check_chunks(result["chunks"]):
        logging.warning("⚠️ Agent A2: Chunks không hợp lệ, thử điều chỉnh kích thước...")
        return {
            "error": "Invalid chunks",
            "retry_count_a2": retry_count + 1,
            "messages": [{
                "to": "agent_a2", 
                "action": "adjust_chunk", 
                "chunk_size": max(chunk_size - 500, 800), 
                "chunk_overlap": max(chunk_overlap - 50, 50)
            }]
        }

    logging.info(f"✅ Agent A2: Đã tạo {len(result['chunks'])} chunks và embeddings thành công")
    return {
        "chunks": result["chunks"],
        "embeddings": result["embeddings"],
        "db": result["db"],
        "retry_count_a2": retry_count + 1,
        "error": None,
        "messages": []  # Clear messages
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
    ("system", """Trích xuất các thực thể (entities) quan trọng từ văn bản.

QUAN TRỌNG: Luôn trả về JSON hợp lệ theo định dạng:
{
  "entities": {
    "names": ["tên người", "tên công ty", "tên tổ chức"],
    "dates": ["ngày tháng", "thời gian"],
    "locations": ["địa điểm", "thành phố", "quốc gia"],
    "numbers": ["số liệu", "phần trăm", "tiền tệ"]
  }
}

Ví dụ:
Input: "Công ty ABC có 1000 nhân viên tại Hà Nội từ năm 2020"
Output:
{
  "entities": {
    "names": ["Công ty ABC"],
    "dates": ["năm 2020"],
    "locations": ["Hà Nội"],
    "numbers": ["1000 nhân viên"]
  }
}

Nếu không tìm thấy entities nào, trả về:
{
  "entities": {
    "names": [],
    "dates": [],
    "locations": [],
    "numbers": []
  }
}"""),
    ("user", "Văn bản: {text}")
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

fallback_extract_prompt = ChatPromptTemplate.from_messages([
    ("system", """Hãy liệt kê từng dòng những thông tin quan trọng bạn tìm thấy:

Tên người/công ty:
- [liệt kê nếu có]

Ngày tháng:
- [liệt kê nếu có]

Địa điểm:
- [liệt kê nếu có]

Số liệu:
- [liệt kê nếu có]

Nếu không có thì ghi "Không có"."""),
    ("user", "Văn bản: {text}")
])

def extract_entities_from_text(text_response: str) -> Dict:
    """Extract entities từ text response thay vì JSON"""
    entities = {"names": [], "dates": [], "locations": [], "numbers": []}
    
    try:
        lines = text_response.split('\n')
        current_category = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Xác định category
            if "tên người" in line.lower() or "công ty" in line.lower():
                current_category = "names"
            elif "ngày" in line.lower() or "tháng" in line.lower():
                current_category = "dates"  
            elif "địa điểm" in line.lower() or "vị trí" in line.lower():
                current_category = "locations"
            elif "số liệu" in line.lower() or "số" in line.lower():
                current_category = "numbers"
            elif line.startswith('- ') and current_category:
                # Extract item
                item = line[2:].strip()
                if item and item.lower() != "không có" and len(item) > 2:
                    entities[current_category].append(item)
    
    except Exception as e:
        logging.warning(f"⚠️ Lỗi khi extract entities từ text: {str(e)}")
    
    return entities

def analyze_chunk_batch_with_mode(chunk: str, use_fallback: bool = False) -> Dict:
    """Xử lý một chunk đơn lẻ với tùy chọn sử dụng fallback prompt"""
    try:
        llm = get_global_llm()
        if not llm:
            raise ValueError("Global LLM not initialized")
            
        # Xử lý summary
        try:
            summary_result = llm.invoke(summarize_prompt.format(text=chunk))
            summary = summary_result.content if hasattr(summary_result, "content") else str(summary_result)
        except Exception as e:
            logging.warning(f"⚠️ Summary generation failed: {str(e)}")
            summary = f"Summary processing error for chunk: {chunk[:100]}..."
        
        # Xử lý entities với 2 approaches khác nhau
        if use_fallback:
            # Approach 1: Non-JSON fallback
            try:
                entities_result = llm.invoke(fallback_extract_prompt.format(text=chunk))
                entities_content = entities_result.content if hasattr(entities_result, "content") else str(entities_result)
                
                logging.info(f"🔄 Fallback: Sử dụng text extraction thay vì JSON")
                entities = extract_entities_from_text(entities_content)
            except Exception as e:
                logging.warning(f"⚠️ Fallback extraction failed: {str(e)}")
                entities = {"names": [], "dates": [], "locations": [], "numbers": []}
            
        else:
            # Approach 2: Standard JSON với backup
            try:
                entities_result = llm.invoke(extract_prompt.format(text=chunk))
                entities_content = entities_result.content if hasattr(entities_result, "content") else str(entities_result)
                
                # Clean và validate JSON content trước khi parse
                entities_content = entities_content.strip()
                
                # Remove markdown code blocks
                if entities_content.startswith("```json"):
                    entities_content = entities_content.replace("```json", "").replace("```", "").strip()
                elif entities_content.startswith("```"):
                    entities_content = entities_content.replace("```", "").strip()
                
                # Kiểm tra nếu content quá ngắn hoặc không hợp lệ
                if len(entities_content) < 10 or not entities_content.startswith("{"):
                    raise ValueError(f"Invalid JSON response: '{entities_content[:50]}...'")
                
                entities_data = json.loads(entities_content)
                
                # Extract entities từ response
                if "entities" in entities_data:
                    entities = entities_data["entities"]
                else:
                    entities = entities_data
                    
                # Validate structure
                if not isinstance(entities, dict):
                    raise ValueError("Invalid entities structure - not a dict")
                    
                # Ensure all required keys exist
                for key in ["names", "dates", "locations", "numbers"]:
                    if key not in entities:
                        entities[key] = []
                    elif not isinstance(entities[key], list):
                        entities[key] = []
                        
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logging.warning(f"⚠️ JSON parsing failed: {str(e)[:100]}. Content: '{entities_content[:100] if 'entities_content' in locals() else 'N/A'}'")
                # Backup: Sử dụng text extraction
                try:
                    entities = extract_entities_from_text(entities_content if 'entities_content' in locals() else "")
                except Exception as e2:
                    logging.warning(f"⚠️ Backup text extraction also failed: {str(e2)}")
                    entities = {"names": [], "dates": [], "locations": [], "numbers": []}
            except Exception as e:
                logging.error(f"❌ Unexpected error in JSON processing: {str(e)}")
                entities = {"names": [], "dates": [], "locations": [], "numbers": []}
        
        # Chuẩn hóa entities
        for key in entities:
            if isinstance(entities[key], list):
                entities[key] = [e for e in entities[key] if isinstance(e, str) and len(e.strip()) > 2]
            else:
                entities[key] = []
        
        result = {
            "summary": summary,
            "entities": entities,
            "text": chunk
        }
        
        entities_count = sum(len(v) for v in entities.values())
        prompt_type = "fallback" if use_fallback else "standard"
        logging.info(f"✅ ({prompt_type}) Chunk processed: {len(chunk)} chars, entities: {entities_count} items")
        
        if entities_count > 0:
            logging.info(f"   📋 Entities found: {dict((k, len(v)) for k, v in entities.items())}")
        
        return result
        
    except Exception as e:
        logging.error(f"❌ Critical error processing chunk: {str(e)}")
        return {
            "summary": "Error processing this chunk",
            "entities": {"names": [], "dates": [], "locations": [], "numbers": []},
            "text": chunk
        }

def analyze_batch_parallel_with_mode(batch: List[str], max_workers: int = 10, use_fallback: bool = False) -> List[Dict]:
    """Xử lý batch với parallel processing và tùy chọn fallback prompt"""
    try:
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(analyze_chunk_batch_with_mode, chunk, use_fallback): chunk 
                for chunk in batch
            }
            
            for future in as_completed(future_to_chunk):
                try:
                    result = future.result(timeout=30)
                    if result:
                        results.append(result)
                except Exception as e:
                    logging.error(f"❌ Timeout or error in parallel processing: {str(e)}")
                    results.append({
                        "summary": "Error processing chunk",
                        "entities": {"names": [], "dates": [], "locations": [], "numbers": []},
                        "text": ""
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
    """Agent phân tích nội dung"""
    if not state.api_key or not state.model_name:
        return {"error": "API key and model name are required", "messages": state.get("messages", [])}
        
    # Set global LLM
    set_global_llm(state.api_key, state.model_name)
    
    retry_count = int(state.get("retry_count_analyze", 0))
    use_fallback = retry_count > 0  # Sử dụng fallback prompt từ retry thứ 2
    
    logging.info(f"🚀 Agent Analyze: Bắt đầu phân tích nội dung ({'fallback mode' if use_fallback else 'standard mode'})...")
    
    if retry_count >= 3:
        logging.error("❌ Agent Analyze: Đã thử 3 lần nhưng không thành công")
        return {"error": "Analysis failed after 3 retries", "messages": []}
    
    # Kiểm tra chunks availability - nhưng KHÔNG check error cho retry case
    if not state.get("chunks"):
        logging.error(f"❌ Agent Analyze: Không có chunks để phân tích")
        return {"error": "No chunks available", "messages": []}
    
    # Check cho error CHỈ KHI không phải retry case
    if state.get("error") and retry_count == 0:
        error_msg = state.get("error")
        logging.error(f"❌ Agent Analyze: Lỗi từ agent trước - {error_msg}")
        return {"error": error_msg, "messages": []}
    
    # Clear error state nếu đang retry
    if retry_count > 0:
        logging.info(f"🔄 Agent Analyze: Clearing previous error for retry {retry_count}")
    
    try:
        batch_size = 20
        chunks = state["chunks"]
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        logging.info(f"📊 Agent Analyze: Xử lý {len(batches)} batches với batch_size={batch_size} ({'fallback' if use_fallback else 'standard'} mode)...")

        summaries = []
        entities = {"names": [], "dates": [], "locations": [], "numbers": []}
        failed_chunks = []
        
        # Xử lý các batches với prompt phù hợp
        for i, batch in enumerate(batches):
            if i > 0:
                time.sleep(0.2)
            
            logging.info(f"🔄 Processing batch {i+1}/{len(batches)} với {len(batch)} chunks...")
            
            batch_results = analyze_batch_parallel_with_mode(batch, max_workers=min(15, len(batch)), use_fallback=use_fallback)
            
            if not batch_results:
                logging.warning(f"⚠️ Batch {i+1} không có kết quả")
                failed_chunks.extend(batch)
                continue
                
            for result in batch_results:
                if not result:
                    continue
                    
                if isinstance(result.get("summary"), str) and result["summary"].strip():
                    summaries.append(result["summary"])
                    
                if result.get("entities"):
                    for key in entities:
                        if key in result["entities"] and isinstance(result["entities"][key], list):
                            entities[key].extend(result["entities"][key])
                else:
                    failed_chunks.append(result.get("text", ""))
        
        # Kiểm tra kết quả phân tích
        total_entities = sum(len(entities[key]) for key in entities)
        
        if total_entities == 0:
            logging.warning(f"⚠️ Agent Analyze: Không trích xuất được entities nào (retry: {retry_count})")
            
            # Strategy 1: Thử fallback prompt (retry 0 -> 1)
            if retry_count == 0:
                logging.info("🔄 Strategy 1: Thử lại với fallback text extraction...")
                return {
                    "error": "No entities extracted - retry with fallback",
                    "retry_count_analyze": retry_count + 1,
                    "messages": []  # Clear messages
                }
            
            # Strategy 2: Giảm chunk size (retry 1 -> 2)  
            elif retry_count == 1:
                logging.info("🔄 Strategy 2: Thử lại với chunk size nhỏ hơn...")
                return {
                    "error": "No entities after fallback - retry with smaller chunks", 
                    "retry_count_analyze": retry_count + 1,
                    "messages": [{
                        "to": "agent_a2",
                        "action": "adjust_chunk",
                        "chunk_size": 1200,  # Nhỏ hơn nữa
                        "chunk_overlap": 100
                    }]
                }
            
            # Strategy 3: Summary-only mode với smart fallback (retry 2+)
            else:
                logging.warning("⚠️ Strategy 3: Chuyển sang summary-only mode")
                
                # Tạo final summary trước
                llm = get_global_llm()
                try:
                    if summaries:
                        # Kết hợp summaries thành final summary
                        combined_summary = "\n".join(summaries[:5])  # Lấy tối đa 5 summaries
                        final_summary_result = llm.invoke(final_summarize_prompt.format(summaries=combined_summary))
                        final_summary = final_summary_result.content if hasattr(final_summary_result, "content") else str(final_summary_result)
                    else:
                        final_summary = "Tài liệu chứa thông tin nhưng không thể trích xuất chi tiết cụ thể."
                except Exception as e:
                    logging.warning(f"⚠️ Lỗi tạo final summary: {str(e)}")
                    final_summary = "\n".join(summaries[:3]) if summaries else "Không thể tạo tóm tắt"
                
                # Tạo entities từ summary (last attempt)
                try:
                    summary_entities_result = llm.invoke(fallback_extract_prompt.format(text=final_summary))
                    summary_entities_content = summary_entities_result.content if hasattr(summary_entities_result, "content") else str(summary_entities_result)
                    summary_entities = extract_entities_from_text(summary_entities_content)
                    
                    # Kiểm tra nếu có entities từ summary
                    summary_entities_count = sum(len(v) for v in summary_entities.values())
                    if summary_entities_count > 0:
                        logging.info(f"✓ Tìm thấy {summary_entities_count} entities từ final summary")
                        final_entities = summary_entities
                    else:
                        # Tạo entities placeholder  
                        final_entities = {
                            "names": ["Tài liệu"],
                            "dates": ["Không xác định"],
                            "locations": ["Không xác định"], 
                            "numbers": ["Không xác định"]
                        }
                        
                except Exception as e:
                    logging.warning(f"⚠️ Không thể extract từ summary: {str(e)}")
                    # Final fallback entities
                    final_entities = {
                        "names": ["Tài liệu"],
                        "dates": ["Không xác định"],
                        "locations": ["Không xác định"],
                        "numbers": ["Không xác định"]
                    }
                
                logging.info(f"✅ Summary-only mode: summary={len(final_summary)} chars, entities={sum(len(v) for v in final_entities.values())} items")
                
                return {
                    "summary": final_summary,
                    "entities": final_entities,
                    "retry_count_analyze": retry_count + 1,
                    "error": None,  # Không lỗi để tiếp tục workflow
                    "summary_only_mode": True,
                    "messages": []  # Clear messages
                }
        
        if not summaries:
            logging.error("❌ Agent Analyze: Không có summary nào được tạo thành công")
            return {
                "error": "No summaries generated",
                "retry_count_analyze": retry_count + 1,
                "messages": []  # Clear messages
            }
            
        # Loại bỏ duplicates và chuẩn hóa
        for key in entities:
            entities[key] = list(set(entities[key]))
            # Loại bỏ các giá trị rỗng hoặc quá ngắn
            entities[key] = [e for e in entities[key] if isinstance(e, str) and len(e.strip()) > 2]
        
        # Tạo final summary
        llm = get_global_llm()
        try:
            summary_chunks = chunk_summaries_for_final(summaries, max_tokens=150000)
            logging.info(f"📝 Chia {len(summaries)} summaries thành {len(summary_chunks)} chunks để xử lý")
            
            if len(summary_chunks) == 1:
                final_summary_result = llm.invoke(final_summarize_prompt.format(summaries="\n".join(summary_chunks[0])))
                final_summary = final_summary_result.content if hasattr(final_summary_result, "content") else str(final_summary_result)
            else:
                chunk_summaries = []
                with ThreadPoolExecutor(max_workers=min(5, len(summary_chunks))) as executor:
                    chunk_futures = {
                        executor.submit(process_summary_chunk, (i, chunk)): i 
                        for i, chunk in enumerate(summary_chunks)
                    }
                    
                    for future in as_completed(chunk_futures):
                        try:
                            result = future.result(timeout=60)
                            if result and isinstance(result, str):
                                chunk_summaries.append(result)
                        except Exception as e:
                            logging.warning(f"⚠️ Timeout in summary processing: {str(e)}")
                
                if len(chunk_summaries) > 1:
                    try:
                        time.sleep(0.5)
                        final_result = llm.invoke(final_summarize_prompt.format(summaries="\n".join(chunk_summaries)))
                        final_summary = final_result.content if hasattr(final_result, "content") else str(final_result)
                    except Exception as e:
                        logging.warning(f"⚠️ Lỗi khi combine final summary: {str(e)}")
                        final_summary = "\n\n".join(chunk_summaries[:3])  # Lấy 3 summary đầu làm fallback
                else:
                    final_summary = chunk_summaries[0] if chunk_summaries else summaries[0]
                    
        except Exception as e:
            logging.error(f"❌ Lỗi khi tạo final summary: {str(e)}")
            final_summary = "\n".join(summaries[:3])  # Fallback với 3 summary đầu
        
        # Log kết quả chi tiết
        logging.info(f"""✅ Agent Analyze: Phân tích thành công ({'fallback' if use_fallback else 'standard'} mode):
        - {len(summaries)}/{len(chunks)} chunks xử lý thành công
        - {len(failed_chunks)} chunks thất bại
        - {len(entities['names'])} tên
        - {len(entities['dates'])} ngày tháng  
        - {len(entities['locations'])} địa điểm
        - {len(entities['numbers'])} số liệu
        - Final summary: {len(final_summary)} ký tự""")
        
        if failed_chunks:
            logging.warning(f"⚠️ {len(failed_chunks)} chunks không xử lý được, có thể thiếu thông tin")
        
        return {
            "summary": final_summary,
            "entities": entities,
            "retry_count_analyze": retry_count + 1,
            "error": None,
            "messages": []  # Clear messages
        }
    except Exception as e:
        logging.error(f"❌ Agent Analyze: Lỗi khi phân tích - {str(e)}")
        return {
            "error": str(e),
            "retry_count_analyze": retry_count + 1,
            "messages": []  # Clear messages
        }

verify_prompt = ChatPromptTemplate.from_messages([
    ("system", """Xác minh tính chính xác và đầy đủ của các thực thể (entities) đã được trích xuất dựa trên câu hỏi và tóm tắt nội dung.

Câu hỏi gốc: {question}
Tóm tắt: {summary}

Nhiệm vụ của bạn:
1. Kiểm tra xem các entities có liên quan đến chủ đề chính không
2. Xác minh tính chính xác của các entities (tên, ngày tháng, địa điểm, số liệu)
3. Đánh giá mức độ đầy đủ của thông tin đã trích xuất
4. Phát hiện các thông tin quan trọng bị bỏ sót
5. So sánh entities từ search với entities từ analysis

Nếu phát hiện vấn đề, gửi message yêu cầu xử lý lại (ví dụ: {{"to": "agent_a1", "action": "retry_clean"}}).
Trả về JSON: {{"verified": bool, "verified_data": dict, "message": dict}}"""),
    ("user", "Entities từ search: {search_entities}\nEntities từ analysis: {analysis_entities}")
])

def verified_agent(state: AgentState) -> AgentState:
    """Agent xác minh thông tin"""
    retry_count = state.get("retry_count_verify", 0)
    logging.info(f"🔍 Agent Verify: Bắt đầu quá trình xác minh (retry: {retry_count})...")
    
    if not state.api_key or not state.model_name:
        logging.error("❌ Agent Verify: Thiếu API key hoặc model name")
        return {"error": "API key and model name are required", "messages": state.get("messages", [])}
        
    if state.get("error") or not state.get("entities") or not state.get("db"):
        error_msg = state.get("error") or "Missing data"
        logging.error(f"❌ Agent Verify: Thiếu dữ liệu để xác minh - {error_msg}")
        return {
            "error": error_msg, 
            "retry_count_verify": retry_count + 1,
            "messages": []  # Clear messages
        }
    
    # Kiểm tra summary-only mode
    summary_only_mode = state.get("summary_only_mode", False)
    if summary_only_mode:
        logging.info("🔍 Agent Verify: Xử lý summary-only mode")
    
    try:
        # Sử dụng summary từ analysis_agent làm query thay vì câu hỏi gốc
        search_query = state.get("summary", state["question"])
        logging.info(f"🔍 Agent Verify: Tìm kiếm với query dài {len(search_query)} ký tự")
        
        # Tăng k lên để có nhiều kết quả hơn
        result = search_tool.invoke({
            "faiss_index": state["db"],
            "query": search_query,
            "chunks": state["chunks"],
            "api_key": state.api_key,
            "embedding_model": state.embedding_model,
            "k": 5
        })
        
        if not result["entities"]["results"]:
            logging.warning("⚠️ Agent Verify: Không tìm thấy kết quả phù hợp, thử lại với câu hỏi gốc")
            # Fallback về câu hỏi gốc nếu tìm bằng summary không có kết quả
            result = search_tool.invoke({
                "faiss_index": state["db"],
                "query": state["question"],
                "chunks": state["chunks"],
                "api_key": state.api_key,
                "embedding_model": state.embedding_model,
                "k": 5
            })
        
        logging.info("✓ Agent Verify: Đã tìm kiếm xong với search tool")
        
        # Xử lý kết quả tìm kiếm với ngưỡng thấp hơn cho summary-only mode
        if not result["entities"]["results"]:
            if summary_only_mode:
                logging.warning("⚠️ Summary-only mode: Không có kết quả search, tiếp tục với verified_data từ entities")
                # Trong summary-only mode, chấp nhận entities hiện có
                verified_data = {
                    "verified_entities": state["entities"],
                    "confidence": "low", 
                    "mode": "summary_only",
                    "note": "Verified based on summary analysis only"
                }
                
                return {
                    "verified_data": verified_data,
                    "retry_count_verify": retry_count + 1,
                    "messages": [],  # Clear messages
                    "error": None
                }
            else:
                logging.error("❌ Agent Verify: Không tìm thấy kết quả nào phù hợp")
                return {
                    "error": "No matching results found",
                    "retry_count_verify": retry_count + 1,
                    "messages": [{"to": "agent_analyze", "action": "reanalyze"}]
                }
            
        # Lấy score trung bình của kết quả
        avg_score = sum(result["entities"]["scores"]) / len(result["entities"]["scores"])
        
        # Điều chỉnh ngưỡng score dựa trên mode
        min_score = 0.2 if summary_only_mode else 0.3
        
        if avg_score < min_score:
            if summary_only_mode:
                logging.warning(f"⚠️ Summary-only mode: Score thấp ({avg_score:.3f}) nhưng tiếp tục")
            else:
                logging.warning(f"⚠️ Agent Verify: Score trung bình ({avg_score:.3f}) quá thấp")
                return {
                    "error": "Low confidence in search results",
                    "retry_count_verify": retry_count + 1,
                    "messages": [{"to": "agent_analyze", "action": "reanalyze"}]
                }
        
        llm = get_llm(state.api_key, state.model_name)
        logging.info("✓ Agent Verify: Đã khởi tạo LLM")
            
        response = llm.invoke(verify_prompt.format(
            question=state["question"],
            summary=state["summary"],
            search_entities=str({
                "results": result["entities"]["results"],
                "scores": result["entities"]["scores"]
            }),
            analysis_entities=str(state["entities"])
        ))
        logging.info("✓ Agent Verify: Đã gọi LLM để xác minh")
        
        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            logging.warning("❌ Agent Verify: Không thể parse kết quả từ LLM, tạo fallback response")
            # Fallback verification cho summary-only mode
            if summary_only_mode:
                parsed = {
                    "verified": True,
                    "verified_data": {
                        "entities": state["entities"],
                        "confidence": "medium",
                        "mode": "summary_only_fallback"
                    },
                    "message": None
                }
            else:
                return {
                    "error": "Invalid LLM response format",
                    "retry_count_verify": retry_count + 1,
                    "messages": []
                }
        
        if parsed["verified"]:
            # Tính số lượng entities từ mỗi nguồn
            search_entities = result.get("entities", {})
            analysis_entities = state.get("entities", {})
            
            search_count = len(search_entities.get("results", []))
            analysis_count = sum(len(entities) for entities in analysis_entities.values())
            
            mode_info = f" (summary-only mode)" if summary_only_mode else ""
            
            logging.info(f"""✅ Agent Verify: Xác minh thành công{mode_info}
            - Entities từ search: {search_count} items (avg score: {avg_score:.3f})
            - Entities từ analysis: {analysis_count} items
            - Verified data: {len(parsed.get('verified_data', {}))} items
            - Categories: {', '.join(parsed.get('verified_data', {}).keys())}""")
            
            return {
                "verified_data": parsed["verified_data"],
                "retry_count_verify": retry_count + 1,
                "messages": [],  # Clear messages
                "error": None
            }
        else:
            logging.warning("⚠️ Agent Verify: Xác minh thất bại, cần phân tích lại")
            return {
                "error": "Verification failed",
                "retry_count_verify": retry_count + 1,
                "messages": [{"to": "agent_analyze", "action": "reanalyze"}]
            }
            
    except Exception as e:
        logging.error(f"❌ Agent Verify: Lỗi khi xác minh - {str(e)}")
        return {
            "error": str(e),
            "retry_count_verify": retry_count + 1,
            "messages": [{"to": "agent_analyze", "action": "reanalyze"}]
        }


class FinalOutput(BaseModel):
    answer: str = Field(description="Câu trả lời cho câu hỏi")
    summary: str = Field(description="Tóm tắt nội dung")
    entities: Dict[str, Any] = Field(description="Entities trích xuất")
    verified_data: Dict[str, Any] = Field(description="Dữ liệu đã xác minh")

def aggregated_agent(state: AgentState) -> AgentState:
    """Agent tổng hợp kết quả"""
    retry_count = state.get("retry_count_aggregate", 0)
    logging.info(f"🚀 Agent Aggregate: Bắt đầu tổng hợp kết quả (retry: {retry_count})...")
    
    if state.get("error"):
        error_msg = state.get("error")
        logging.error(f"❌ Agent Aggregate: Không thể tổng hợp do lỗi - {error_msg}")
        return {
            "report": f"Error: {error_msg}", 
            "retry_count_aggregate": retry_count + 1,
            "messages": []
        }
    
    try:
        final_result = FinalOutput(
            answer=f"Response to '{state['question']}': {state.get('verified_data', 'No verified data')}",
            summary=state.get("summary", "No summary available"),
            entities=state.get("entities", {}),
            verified_data=state.get("verified_data", {})
        )
        
        logging.info("✅ Agent Aggregate: Đã tổng hợp thành công")
        return {
            "report": final_result.model_dump_json(), 
            "retry_count_aggregate": retry_count + 1,
            "error": None, 
            "messages": []
        }
    except Exception as e:
        logging.error(f"❌ Agent Aggregate: Lỗi khi tổng hợp - {str(e)}")
        return {
            "error": str(e),
            "retry_count_aggregate": retry_count + 1,
            "messages": []
        }

def process_summary_chunk(chunk_data):
    """Process summary chunk for parallel execution"""
    chunk_idx, chunk = chunk_data
    try:
        llm = get_global_llm()
        chunk_result = llm.invoke(final_summarize_prompt.format(summaries="\n".join(chunk)))
        return chunk_result.content if hasattr(chunk_result, "content") else str(chunk_result)
    except Exception as e:
        logging.warning(f"⚠️ Lỗi khi xử lý summary chunk {chunk_idx}: {str(e)}")
        return " ".join(chunk[:3])  # Fallback với 3 summaries đầu

# Backward compatibility
def analyze_chunk_batch(chunk: str) -> Dict:
    """Backward compatibility wrapper"""
    return analyze_chunk_batch_with_mode(chunk, use_fallback=False)

def analyze_batch_parallel(batch: List[str], max_workers: int = 10) -> List[Dict]:
    """Backward compatibility wrapper"""
    return analyze_batch_parallel_with_mode(batch, max_workers, use_fallback=False)


