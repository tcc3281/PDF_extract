from modules.graphs import build_graph

if __name__ == '__main__':
    graph = build_graph()
    result = graph.invoke({
        "file_path": "data/JD_Intern_AI_Engineer_technica_v1.1.pdf",
        "question": "Trích xuất các nội dung quan trọng của văn bản",
        "cleaned_text": None,
        "chunks": [],
        "embeddings": [],
        "db": "",
        "summary": None,
        "entities": None,
        "verified_data": None,
        "report": None,
        "error": None,
        "messages": [],
        "retry_count_a1": 0,
        "retry_count_a2": 0,
        "retry_count_analyze": 0
    })
    print(result["report"])