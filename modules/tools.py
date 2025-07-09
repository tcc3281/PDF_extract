import logging
import faiss
import numpy as np
import pdfplumber
import os
import json
from typing import Optional, Dict, Any, List
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time
from pathlib import Path
from datetime import datetime

load_dotenv()

def check_chunks(chunks: List[str]) -> bool:
    return all(500 <= len(chunk) <= 3000 for chunk in chunks) and len(chunks) < 1000 and all(chunk.strip() for chunk in chunks)

def get_timestamp_filename(base_name: str, extension: str) -> str:
    """Tạo tên file với timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{base_name}.{extension}"

# Tool: Trích xuất văn bản từ PDF
@tool
def extract_pdf(pdf_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        Dict with keys:
            - cleaned_text (str): Extracted and cleaned text
            - error (str or None): Error message if any
    """
    text = ''
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Kiểm tra page_text không None
                    text += page_text
        return {"cleaned_text": text.strip(), "error": None}
    except Exception as e:
        logging.error(f"Error extracting PDF: {e}")
        return {"cleaned_text": None, "error": str(e)}

@tool
def chunk_and_embed(text: str, chunk_size: int = 2000, chunk_overlap: int = 200, file_id: str = None, api_key: str = None, embedding_model: str = None) -> Dict[str, Any]:
    """
    Split text into chunks and create embeddings using OpenAI's API.
    
    Args:
        text (str): Input text to chunk and embed
        chunk_size (int): Size of each chunk (default: 2000)
        chunk_overlap (int): Overlap between chunks (default: 200)
        file_id (str): Optional file identifier for unique naming
        api_key (str): OpenAI API key
        embedding_model (str): Embedding model name
        
    Returns:
        Dict with keys:
            - chunks (List[str]): List of text chunks
            - embeddings (List[List[float]]): List of embeddings
            - db (str): Path to saved FAISS index
    """
    # Tạo tên file unique
    if file_id is None:
        file_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_text(text)
    
    # Initialize embeddings model
    embeddings_model = OpenAIEmbeddings(
        model=embedding_model or "text-embedding-ada-002",
        openai_api_key=api_key
    )
    
    # Process chunks in smaller batches with delay
    embeddings = []
    batch_size = 20  # Reduced batch size
    for i in range(0, len(chunks), batch_size):
        try:
            batch = chunks[i:i + batch_size]
            batch_embeddings = embeddings_model.embed_documents(batch)
            embeddings.extend(batch_embeddings)
            
            # Add delay between batches
            if i + batch_size < len(chunks):
                time.sleep(1)  # 1 second delay between batches
                
        except Exception as e:
            logging.error(f"Error in batch {i}-{i+batch_size}: {str(e)}")
            # If rate limit error, wait longer and retry
            if "rate_limit" in str(e).lower():
                time.sleep(30)  # Wait 30 seconds before retrying
                try:
                    batch_embeddings = embeddings_model.embed_documents(batch)
                    embeddings.extend(batch_embeddings)
                except Exception as retry_e:
                    logging.error(f"Retry failed: {str(retry_e)}")
                    return {"error": f"Embedding failed: {str(retry_e)}"}
    
    # Create and save FAISS index
    try:
        dimension = 1536  # Dimension of text-embedding-ada-002
        nlist = min(100, len(chunks))  # Adjust nlist based on number of vectors
        
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, nlist)
        embeddings_array = np.array(embeddings).astype("float32")
        
        if len(embeddings_array) < nlist:
            # If we have fewer vectors than clusters, adjust nlist
            nlist = len(embeddings_array)
            index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, nlist)
        
        index.train(embeddings_array)
        index.add(embeddings_array)
        
        # Tạo đường dẫn unique cho FAISS index
        faiss_index_filename = get_timestamp_filename(f"faiss_index_{file_id}", "index")
        faiss_index_path = Path("indices") / faiss_index_filename
        faiss.write_index(index, str(faiss_index_path))
        
        # Save chunks for later use với tên unique
        chunks_filename = get_timestamp_filename(f"chunks_{file_id}", "json")
        chunks_path = Path("temp_files") / chunks_filename
        with open(chunks_path, "w", encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        return {
            "chunks": chunks,
            "embeddings": embeddings,
            "db": str(faiss_index_path),
            "chunks_file": str(chunks_path),
            "error": None
        }
        
    except Exception as e:
        logging.error(f"Error creating FAISS index: {str(e)}")
        return {"error": f"FAISS index creation failed: {str(e)}"}

# Tool: Tìm kiếm FAISS
@tool
def search_tool(faiss_index: str, query: str, chunks: List[str], api_key: str = None, embedding_model: str = None, k: int = 5) -> Dict[str, Any]:
    """
    Search for relevant text chunks using FAISS index with advanced multi-stage search strategy.
    
    Args:
        faiss_index (str): Path to FAISS index file
        query (str): Search query
        chunks (List[str]): List of text chunks
        api_key (str): OpenAI API key
        embedding_model (str): Embedding model name
        k (int): Number of results to return (default: 5)
        
    Returns:
        Dict with keys:
            - entities (Dict): Contains search results and metadata
    """
    try:
        # 1. Tối ưu query
        optimized_query = _optimize_query(query)
        logging.info(f"Query gốc ({len(query)} ký tự) -> Query tối ưu ({len(optimized_query)} ký tự)")
        
        # 2. Multi-Stage Search Strategy
        index = faiss.read_index(faiss_index)
        embeddings_model = OpenAIEmbeddings(
            model=embedding_model or "text-embedding-ada-002",
            openai_api_key=api_key
        )
        
        # === STAGE 1: Broad Search ===
        # Tìm kiếm với k lớn để không miss thông tin quan trọng
        broad_k = min(max(k * 4, 15), len(chunks))  # Tăng từ 2x lên 4x, tối thiểu 15
        query_embedding = embeddings_model.embed_query(optimized_query)
        distances, indices = index.search(np.array([query_embedding]).astype("float32"), broad_k)
        
        # === STAGE 2: Multi-Query Expansion ===
        # Tạo các query variants để bắt được thông tin từ nhiều góc độ
        query_variants = _create_query_variants(optimized_query)
        all_results = []
        
        # Search với main query
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(chunks):
                score = 1 / (1 + distance)
                all_results.append({
                    "text": chunks[idx],
                    "score": score,
                    "distance": float(distance),
                    "chunk_idx": idx,
                    "query_type": "main"
                })
        
        # Search với các query variants (nếu có)
        for variant_type, variant_query in query_variants.items():
            if variant_query != optimized_query:  # Tránh duplicate
                try:
                    variant_embedding = embeddings_model.embed_query(variant_query)
                    v_distances, v_indices = index.search(np.array([variant_embedding]).astype("float32"), k)
                    
                    for distance, idx in zip(v_distances[0], v_indices[0]):
                        if idx < len(chunks):
                            # Check if chunk already exists
                            existing = next((r for r in all_results if r["chunk_idx"] == idx), None)
                            if existing:
                                # Boost score if found by multiple queries
                                existing["score"] = min(existing["score"] * 1.2, 1.0)
                                existing["query_type"] += f"+{variant_type}"
                            else:
                                all_results.append({
                                    "text": chunks[idx],
                                    "score": 1 / (1 + distance),
                                    "distance": float(distance),
                                    "chunk_idx": idx,
                                    "query_type": variant_type
                                })
                except Exception as e:
                    logging.warning(f"⚠️ Variant query '{variant_type}' failed: {str(e)}")
        
        # === STAGE 3: Context Expansion ===
        # Thêm chunks adjacent để có context đầy đủ
        expanded_results = _expand_with_context(all_results, chunks, max_expand=2)
        
        # === STAGE 4: Intelligent Ranking ===
        # Sắp xếp với multiple criteria
        ranked_results = _intelligent_ranking(expanded_results, optimized_query)
        
        # === STAGE 5: Adaptive K Selection ===
        # Điều chỉnh số lượng kết quả dựa trên quality distribution
        final_k = _adaptive_k_selection(ranked_results, target_k=k)
        final_results = ranked_results[:final_k]
        
        # === STAGE 6: Fallback nếu quality thấp ===
        if not final_results or final_results[0]["score"] < 0.3:
            logging.warning("⚠️ Low quality results, applying fallback strategy...")
            fallback_results = _fallback_search(index, query_embedding, chunks, k*2)
            if fallback_results:
                final_results = fallback_results[:k]
        
        # Log detailed results
        logging.info(f"""✓ Multi-stage search completed:
        - Broad search: {broad_k} candidates
        - Query variants: {len(query_variants)} types
        - Context expansion: {len(expanded_results)} total chunks
        - Final results: {len(final_results)} chunks
        - Score range: {final_results[0]["score"]:.3f} - {final_results[-1]["score"]:.3f}""")
        
        return {
            "entities": {
                "results": [r["text"] for r in final_results],
                "scores": [r["score"] for r in final_results],
                "metadata": {
                    "original_query_length": len(query),
                    "optimized_query_length": len(optimized_query),
                    "total_candidates": len(expanded_results),
                    "final_results": len(final_results),
                    "search_strategy": "multi_stage_enhanced",
                    "query_variants": list(query_variants.keys())
                }
            }
        }
    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        return {"entities": {"results": [], "scores": [], "error": str(e)}}

def _create_query_variants(query: str) -> Dict[str, str]:
    """Tạo các variants của query để search từ nhiều góc độ"""
    variants = {}
    
    # Truncated versions để bắt broader concepts
    if len(query) > 200:
        variants["short"] = query[:150]
    if len(query) > 100:
        variants["keywords"] = " ".join(query.split()[:15])  # Lấy 15 từ đầu
    
    # Entity-focused variants
    words = query.split()
    if len(words) > 10:
        # Focus on nouns và proper nouns (simplified)
        important_words = [w for w in words if len(w) > 3 and w[0].isupper()]
        if important_words:
            variants["entities"] = " ".join(important_words[:10])
    
    return variants

def _expand_with_context(results: List[Dict], chunks: List[str], max_expand: int = 2) -> List[Dict]:
    """Mở rộng kết quả bằng cách thêm chunks adjacent để có context đầy đủ"""
    expanded = {}  # Use dict để tránh duplicate chunks
    
    for result in results:
        chunk_idx = result["chunk_idx"]
        
        # Add original chunk
        if chunk_idx not in expanded:
            expanded[chunk_idx] = result.copy()
        else:
            # Merge scores nếu chunk đã tồn tại
            expanded[chunk_idx]["score"] = max(expanded[chunk_idx]["score"], result["score"])
        
        # Add context chunks
        for offset in range(-max_expand, max_expand + 1):
            context_idx = chunk_idx + offset
            if 0 <= context_idx < len(chunks) and context_idx not in expanded:
                # Context chunks có score thấp hơn original
                context_score = result["score"] * (0.7 ** abs(offset)) if offset != 0 else result["score"]
                expanded[context_idx] = {
                    "text": chunks[context_idx],
                    "score": context_score,
                    "distance": result["distance"] + abs(offset) * 0.1,
                    "chunk_idx": context_idx,
                    "query_type": f"context_{offset}" if offset != 0 else result["query_type"]
                }
    
    return list(expanded.values())

def _intelligent_ranking(results: List[Dict], query: str) -> List[Dict]:
    """Ranking thông minh dựa trên multiple factors"""
    
    def calculate_composite_score(result):
        base_score = result["score"]
        
        # Length bonus: chunks có length hợp lý
        text_len = len(result["text"])
        if 800 <= text_len <= 2500:  # Sweet spot
            length_bonus = 1.1
        elif 500 <= text_len <= 3000:
            length_bonus = 1.05
        else:
            length_bonus = 1.0
        
        # Query type bonus
        query_type = result.get("query_type", "")
        if "main" in query_type:
            type_bonus = 1.0
        elif "+" in query_type:  # Found by multiple queries
            type_bonus = 1.15
        else:
            type_bonus = 0.95
        
        # Context bonus: context chunks gần original có score cao hơn
        if "context" in query_type:
            offset = abs(int(query_type.split("_")[1]) if "_" in query_type else 0)
            context_bonus = 0.8 ** offset
        else:
            context_bonus = 1.0
        
        return base_score * length_bonus * type_bonus * context_bonus
    
    # Calculate composite scores
    for result in results:
        result["composite_score"] = calculate_composite_score(result)
    
    # Sort by composite score
    return sorted(results, key=lambda x: x["composite_score"], reverse=True)

def _adaptive_k_selection(results: List[Dict], target_k: int) -> int:
    """Điều chỉnh số lượng kết quả dựa trên quality distribution"""
    if not results:
        return target_k
    
    # Phân tích score distribution
    scores = [r["composite_score"] for r in results]
    top_score = scores[0]
    
    # Find natural quality cutoff
    quality_threshold = top_score * 0.6  # 60% of top score
    quality_results = [r for r in results if r["composite_score"] >= quality_threshold]
    
    # Adaptive selection
    min_results = max(target_k // 2, 2)  # Tối thiểu 2-3 kết quả
    max_results = target_k * 2  # Tối đa gấp đôi target
    
    if len(quality_results) < min_results:
        return min(target_k, len(results))
    elif len(quality_results) > max_results:
        return max_results
    else:
        return len(quality_results)

def _fallback_search(index, query_embedding, chunks: List[str], fallback_k: int) -> List[Dict]:
    """Fallback search với strategy khác khi main search fail"""
    try:
        # Tìm kiếm với số lượng lớn hơn và threshold thấp hơn
        large_k = min(fallback_k * 2, len(chunks))
        distances, indices = index.search(np.array([query_embedding]).astype("float32"), large_k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(chunks):
                results.append({
                    "text": chunks[idx],
                    "score": 1 / (1 + distance),
                    "distance": float(distance),
                    "chunk_idx": idx,
                    "query_type": "fallback"
                })
        
        return sorted(results, key=lambda x: x["score"], reverse=True)
    except Exception:
        return []

def _optimize_query(query: str, max_length: int = 500) -> str:
    """Tối ưu query để tìm kiếm hiệu quả hơn"""
    # 1. Lấy n ký tự đầu tiên nếu query quá dài
    if len(query) > max_length:
        query = query[:max_length]
    
    # 2. Loại bỏ các ký tự đặc biệt và khoảng trắng thừa
    query = " ".join(query.split())
    
    # 3. Thêm các tối ưu khác nếu cần
    
    return query