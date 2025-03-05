from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
from src.rag_pipeline import build_rag_pipeline

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Recommendation API")

# RAG 파이프라인 및 벡터스토어 초기화
rag_pipeline, vectorstore = build_rag_pipeline() 
retriever = rag_pipeline.retriever

if vectorstore is None:
    logger.error("Vectorstore is NOT initialized! Check ChromaDB path and ensure database exists.")
else:
    logger.info("Vectorstore successfully loaded and initialized.")

# 요청/응답 모델 정의
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatRequest(BaseModel):
    query: str

class AddDocumentRequest(BaseModel):
    id: str
    text: str
    metadata: dict = {}

# 엔드포인트: 벡터스토어 검색 (/search)
@app.post("/search")
def search_vectorstore(request: QueryRequest):
    """벡터스토어에서 유사한 문서를 검색합니다."""
    try:
        # 문서와 유사도 함께 검색
        docs_and_scores = vectorstore.similarity_search_with_score(request.query, k=request.top_k)

        if not docs_and_scores:
            return {"results": []}

        output = []
        for doc, score in docs_and_scores:  
            output.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })

        return {"results": output}

    except Exception as e:
        logger.error("Error during search: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Search error: " + str(e))


# 엔드포인트: 챗 응답 생성 (/chat)
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """RAG 파이프라인을 통해 질문에 대한 답변을 생성합니다."""
    try:
        answer = rag_pipeline.run(request.query)
        return {"answer": answer}
    except Exception as e:
        logger.error("Error during chat generation: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Chat error: " + str(e))

# 엔드포인트: 문서 추가 (/add)
@app.post("/add")
def add_document(request: AddDocumentRequest):
    """벡터스토어에 문서를 추가합니다."""
    if vectorstore is None:
        logger.error("Vectorstore is not initialized.")
        raise HTTPException(status_code=500, detail="Vectorstore not available.")
    try:
        # 사용 중인 라이브러리의 API에 따라 아래 메서드를 조정하세요.
        # 예시: vectorstore.add_texts(texts, metadatas, ids)
        vectorstore.add_texts([request.text], [request.metadata], [request.id])
        return JSONResponse(status_code=200, content={"detail": f"Document {request.id} added successfully."})
    except Exception as e:
        logger.error("Error adding document: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Add document error: " + str(e))

# 엔드포인트: 문서 삭제 (/delete/{doc_id})
@app.delete("/delete/{doc_id}")
def delete_document(doc_id: str):
    """벡터스토어에서 문서를 삭제합니다."""
    if vectorstore is None:
        logger.error("Vectorstore is not initialized.")
        raise HTTPException(status_code=500, detail="Vectorstore not available.")
    try:
        # 사용 중인 라이브러리의 API에 따라 아래 메서드를 조정하세요.
        # 예시: vectorstore.delete(ids=[doc_id])
        vectorstore.delete(ids=[doc_id])
        return JSONResponse(status_code=200, content={"detail": f"Document {doc_id} deleted successfully."})
    except Exception as e:
        logger.error("Error deleting document: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Delete document error: " + str(e))

# 헬스 체크 엔드포인트 (/health)
@app.get("/health")
def health_check():
    """서버 상태를 체크합니다."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
