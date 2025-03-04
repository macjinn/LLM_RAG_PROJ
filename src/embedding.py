import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def create_vectorstore_from_json(processed_json_path: str,
                                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                                 persist_dir: str = "data/chroma_db"):
    """
    전처리된 JSON 파일에 저장된 문서들을 로드하여, 내러티브 텍스트를 기반으로 임베딩을 생성하고,
    Chroma DB 벡터스토어에 저장합니다.
    
    Args:
        processed_json_path (str): 전처리된 JSON 파일 경로.
        model_name (str): 임베딩 모델 이름.
        persist_dir (str): Chroma DB 저장 디렉토리.
    
    Returns:
        vectorstore: 생성된 Chroma 벡터스토어 객체.
    """
    with open(processed_json_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
        
    # 각 문서에서 내러티브 텍스트 추출
    texts = [doc["narrative"] for doc in documents if "narrative" in doc]

    if not texts:
        raise ValueError("Error: JSON 파일 내 'narrative' 필드가 존재하지 않거나 데이터가 없습니다.")

    # 임베딩 모델 초기화 (LangChain의 HuggingFaceEmbeddings 사용)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Chroma DB 인스턴스 생성 (persist_directory: 저장 경로)
    vectorstore = Chroma.from_texts(texts, embeddings, persist_directory=persist_dir)
    print(f"ChromaDB에 {len(texts)}개의 문서를 저장했습니다. (경로: {persist_dir})")
    
    return vectorstore

def search_vectorstore(query: str,
                       persist_dir: str = "data/chroma_db",
                       model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                       k: int = 3):
    """
    저장된 Chroma DB 벡터스토어에서 쿼리와 유사한 문서를 검색합니다.
    
    Args:
        query (str): 검색할 쿼리 텍스트.
        persist_dir (str): Chroma DB 저장 디렉토리.
        model_name (str): 임베딩 모델 이름.
        k (int): 검색 결과 개수.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    results = vectorstore.similarity_search(query, k=k)
    
    print("검색 결과:")
    for i, result in enumerate(results):
        print(f"Rank {i+1}: {result.page_content}")
        print("-" * 40)

if __name__ == "__main__":
    # 전처리된 JSON 파일 경로 (예: data/processed/documents.json)
    processed_json_path = '/home/inseong/LLM_RAG_PROJ/data/processed/예금금리_입출금자유예금_20250213.json'
    persist_dir = '/home/inseong/LLM_RAG_PROJ/data/chroma_db'  # Chroma DB 저장 경로
    
    # 벡터스토어 생성: 전처리된 문서들을 임베딩하여 Chroma DB에 저장
    vectorstore = create_vectorstore_from_json(processed_json_path, persist_dir=persist_dir)
    
    # 검색 테스트: 사용자 쿼리 기반 유사 문서 검색
    query = "추천 예금 상품을 소개해줘"
    search_vectorstore(query, persist_dir=persist_dir)
