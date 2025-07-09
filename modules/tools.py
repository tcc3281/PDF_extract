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
    Search for relevant text chunks using FAISS index with optimized search and fallback.
    
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
        
        # 2. Tìm kiếm với k lớn hơn để có nhiều kết quả tiềm năng
        index = faiss.read_index(faiss_index)
        embeddings_model = OpenAIEmbeddings(
            model=embedding_model or "text-embedding-ada-002",
            openai_api_key=api_key
        )
        query_embedding = embeddings_model.embed_query(optimized_query)
        
        # Tăng k lên để có nhiều kết quả hơn cho reranking
        search_k = min(max(k * 2, 10), len(chunks))  # Ít nhất 10 kết quả hoặc gấp đôi k
        distances, indices = index.search(np.array([query_embedding]).astype("float32"), search_k)
        
        # 3. Rerank kết quả
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(chunks):  # Đảm bảo index hợp lệ
                score = 1 / (1 + distance)  # Chuyển khoảng cách thành điểm (0-1)
                results.append({
                    "text": chunks[idx],
                    "score": score,
                    "distance": float(distance)
                })
        
        # Sắp xếp theo score và lấy k kết quả tốt nhất
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
        
        # 4. Fallback nếu không có kết quả tốt
        if not results or results[0]["score"] < 0.5:  # Ngưỡng điểm tối thiểu
            logging.warning("⚠️ Không tìm thấy kết quả phù hợp, thử tìm kiếm với ngưỡng thấp hơn")
            # Thử tìm lại với k lớn hơn
            fallback_k = min(k * 3, len(chunks))
            distances, indices = index.search(np.array([query_embedding]).astype("float32"), fallback_k)
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(chunks):
                    results.append({
                        "text": chunks[idx],
                        "score": 1 / (1 + distance),
                        "distance": float(distance)
                    })
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
        
        # Log kết quả tìm kiếm
        logging.info(f"""✓ Kết quả tìm kiếm:
        - Query gốc: {len(query)} ký tự
        - Query tối ưu: {len(optimized_query)} ký tự
        - Số kết quả: {len(results)}
        - Score cao nhất: {results[0]["score"]:.3f} nếu có kết quả""")
        
        return {
            "entities": {
                "results": [r["text"] for r in results],
                "scores": [r["score"] for r in results],
                "metadata": {
                    "original_query_length": len(query),
                    "optimized_query_length": len(optimized_query),
                    "total_results": len(results)
                }
            }
        }
    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        return {"entities": {"results": [], "scores": [], "error": str(e)}}

def _optimize_query(query: str, max_length: int = 500) -> str:
    """Tối ưu query để tìm kiếm hiệu quả hơn"""
    # 1. Lấy n ký tự đầu tiên nếu query quá dài
    if len(query) > max_length:
        query = query[:max_length]
    
    # 2. Loại bỏ các ký tự đặc biệt và khoảng trắng thừa
    query = " ".join(query.split())
    
    # 3. Thêm các tối ưu khác nếu cần
    
    return query