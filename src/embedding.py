
import os
import json
import glob
from config import CONFIG
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def safe_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and (x != x):  # NaN 확인
        return ""
    return str(x)

def create_vectorstore_from_all_json(processed_dir: str = CONFIG["paths"]["processed_json_path"],
                                     model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                                     persist_dir: str = CONFIG["paths"]["chroma_db_path"]):
    record_file = os.path.join(persist_dir, "processed_files.txt")
    if os.path.exists(record_file):
        with open(record_file, "r", encoding="utf-8") as f:
            processed_files = set(line.strip() for line in f if line.strip())
    else:
        processed_files = set()
    
    json_files = glob.glob(os.path.join(processed_dir, "*.json"))
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    new_files = []
    total_new_texts = 0
    
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        if file_name in processed_files:
            print(f"이미 처리됨: {file_name}, 스킵합니다.")
            continue
        
        print(f"처리 시작: {file_name}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

                if isinstance(json_data, dict) and "documents" in json_data:
                    documents = json_data["documents"]
                elif isinstance(json_data, list):
                    documents = json_data
                else:
                    print(f"지원하지 않는 JSON 구조입니다: {file_name}")
                    continue
        except Exception as e:
            print(f"파일 {file_name} 로드 실패: {e}")
            continue
        
        texts, metadatas, ids = [], [], []
        for doc in documents:
            content = safe_str(doc.get("content", ""))
            if not content.strip():
                continue

            metadata = {k: safe_str(v) for k, v in doc.get("metadata", {}).items()}
            doc_id = safe_str(doc.get("id", ""))

            texts.append(content)
            metadatas.append(metadata)
            ids.append(doc_id)
        
        if not texts:
            print(f"파일 {file_name}에 유효한 'content'가 없습니다. 스킵합니다.")
            continue
        
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        
        new_files.append(file_name)
        total_new_texts += len(texts)
        print(f"완료: {file_name} ({len(texts)}개의 문서)")
    
    print(f"ChromaDB에 총 {total_new_texts}개의 신규 문서를 추가했습니다. (경로: {persist_dir})")
    
    if new_files:
        with open(record_file, "a", encoding="utf-8") as f:
            for file_name in new_files:
                f.write(file_name + "\n")
        print(f"처리된 파일 목록 업데이트 완료: {new_files}")
    else:
        print("새로 처리할 파일이 없습니다.")
    
    return vectorstore

def search_vectorstore(query: str,
                       persist_dir: str = CONFIG["paths"]["chroma_db_path"],
                       model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                       k: int = 3):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    results = vectorstore.similarity_search(query, k=k)

    print("검색 결과:")
    for i, result in enumerate(results):
        print(f"Rank {i+1}: {result.page_content}")
        print("-" * 40)

if __name__ == "__main__":
    processed_dir = CONFIG["paths"]["processed_json_path"]
    persist_dir = CONFIG["paths"]["chroma_db_path"]
    vectorstore = create_vectorstore_from_all_json(processed_dir, persist_dir=persist_dir)
