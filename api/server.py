# api/server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
from src.config import CONFIG
from src.rag_pipeline import build_rag_pipeline

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Recommendation API")

# RAG 파이프라인 및 벡터스토어 초기화
rag_pipeline, vectorstore = build_rag_pipeline() 
hybrid_retriever = rag_pipeline.retriever.retriever_func.__self__ 


if vectorstore is None:
    logger.error("Vectorstore is NOT initialized! Check ChromaDB path and ensure database exists.")
else:
    logger.info("Vectorstore successfully loaded and initialized.")

# 요청/응답 모델 정의``
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatRequest(BaseModel):
    query: str

class AddDocumentRequest(BaseModel):
    id: str
    text: str
    metadata: dict = {}

def extract_user_answer(raw_text: str) -> str:
    """모델의 답변을 유저용으로 가공 후 반환합니다."""
    # raw_text가 dict인지 확인. langchain RetrievalQA의 답변은 딕셔너리로 반환됨. 따라서 str로 변환하여 가공. 
    if isinstance(raw_text, dict):
        result = raw_text.get("result", "")
    else:
        result = str(raw_text)

    # # 혹시 result도 dict일 경우 방어
    # if isinstance(result, dict):
    #     result = str(result)

    # 유저용 태그 추출
    marker = "##[USER_ANSWER]:"
    if marker in result:
        return result.split(marker, 1)[1].strip()

    # 태그가 없을 경우 전체 텍스트 반환
    return result.strip()


# 엔드포인트: 벡터스토어 검색 (/search)
@app.post("/search")
def search_vectorstore(request: QueryRequest):
    """
    벡터스토어에서 유사한 문서를 검색합니다.
    RetrievalQA 내부 retriever → CustomRetriever → hybrid_retriever 호출 
    """
    try:
        # hybrid_retriever를 통해 상세 점수(문서, combined, vector, bm25)를 받음
        docs_and_scores = hybrid_retriever.retrieve_with_scores(request.query)

        if not docs_and_scores:
            return {"results": []}

        output = []
        for doc, combined_score, vector_score, bm25_score in docs_and_scores:
            output.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "combined_score": round(float(combined_score), 4),
                "vector_score": round(float(vector_score), 4),
                "bm25_score": round(float(bm25_score), 4)
            })

        return {"results": output}

    except Exception as e:
        logger.error("Error during hybrid search: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Hybrid search error: " + str(e))



# 엔드포인트: 챗 응답 생성 (/chat)
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """RAG 파이프라인을 통해 질문에 대한 답변을 생성합니다."""
    try:
        raw_answer = rag_pipeline.invoke(request.query)  # 원본 답변 생성
        print("DEBUG: raw_answer =", raw_answer)

        # 사용자용 답변 추출
        user_answer = extract_user_answer(raw_answer)

        # 관리자용 전체 응답
        if isinstance(raw_answer, dict):
            admin_answer = raw_answer.get("result", str(raw_answer))
        else:
            admin_answer = str(raw_answer)

        return {
            "admin_answer": admin_answer,
            "user_answer": user_answer
        }

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
    uvicorn.run(app, host=CONFIG["api"]["host"], port=CONFIG["api"]["port"], reload=True)

