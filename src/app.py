# /home/inseong/LLM_RAG_PROJ/src/app.py
from src.config import CONFIG
import streamlit as st
import requests
import json

# FastAPI 서버 주소
API_URL = f"http://{CONFIG['api']['host']}:{CONFIG['api']['port']}"

# -------------------------------------------------------------
# -------- Chat Interface 함수 (기존 Gradio app.py 기능) --------

# 챗 질의응답 함수
def call_chat_endpoint(user_query: str):
    payload = {"query": user_query}
    try:
        response = requests.post(f"{API_URL}/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        user_answer = data.get("user_answer", "응답을 생성할 수 없습니다.")
        admin_answer = data.get("admin_answer", "관리자 응답 없음")
        return user_answer, admin_answer
    except Exception as e:
        return f"오류 발생: {e}", "관리자 응답 없음"
        
# -------------------------------------------------------------------------------- 
# -------- VectorStore Management 함수들 (기존 Streamlit dashboard.py 기능) --------

# 문서 검색 함수
def search_documents(query: str, top_k: int = 5):
    try:
        payload = {"query": query, "top_k": top_k}
        response = requests.post(f"{API_URL}/search", json=payload)
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        st.error(f"검색 에러: {e}")
        return []

# 문서 추가 함수
def add_document(doc_id: str, doc_text: str, doc_metadata: dict):
    try:
        payload = {"id": doc_id, "text": doc_text, "metadata": doc_metadata}
        response = requests.post(f"{API_URL}/add", json=payload)
        response.raise_for_status()
        return response.json().get("detail", "추가 성공")
    except Exception as e:
        return f"문서 추가 실패: {e}"

# 문서 삭제 함수
def delete_document(doc_id: str):
    try:
        response = requests.delete(f"{API_URL}/delete/{doc_id}")
        response.raise_for_status()
        return response.json().get("detail", "삭제 성공")
    except Exception as e:
        return f"문서 삭제 실패: {e}"

# ------------------------------------------------------------------------------------------
# Streamlit UI 구성
def main():
    st.set_page_config(page_title="금융 RAG 시스템", layout="wide")
    st.title("금융상품 RAG 시스템")

    tab_chat, tab_manage = st.tabs(["🔹 금융상품 추천 (Chat)", "🔸 벡터스토어 관리"])

    # =========================== [Tab 1: 금융상품 추천 (Chat)] ====================
    with tab_chat:
        st.subheader("금융상품 추천 챗봇")

        user_query = st.text_area("질문 입력", "", height=150)
        examples = [
            "현재 가장 높은 금리를 제공하는 예금 상품은?",
            "가입 제한이 없는 예금 상품을 추천해줘.",
            "청년을 위한 우대금리가 적용된 예금이 있을까?"
        ]
        example_question = st.selectbox("예시 질문 선택", ["직접 입력"] + examples, index=0)

        if example_question != "직접 입력":
            user_query = example_question
            st.info(f"예시 질문: {user_query}")

        if st.button("질문하기"):
            if not user_query.strip():
                st.warning("질문을 입력해주세요.")
            else:
                with st.spinner("AI가 답변을 생성 중..."):
                    user_answer, admin_answer = call_chat_endpoint(user_query)
                    st.markdown("### 사용자용 답변")
                    st.text_area("User", user_answer, height=300)
                    with st.expander("관리자용 전체 답변"):
                        st.text_area("Admin", admin_answer, height=600)

    # ============================== [Tab 2: 벡터스토어 관리] ====================
    with tab_manage:
        st.subheader("벡터스토어 관리")
        
        # 문서 검색
        st.markdown("#### 1) 문서 검색")
        search_query = st.text_input("검색어 입력", placeholder="예) 추천 예금 상품", key="search_query")
        top_k = st.slider("검색 결과 개수", min_value=1, max_value=10, value=3, key="search_top_k")

        if st.button("검색 실행", key="search_button"):
            try:
                results = search_documents(search_query, top_k)
                if results:
                    st.success(f"{len(results)}개의 결과를 찾았습니다.")
                    for idx, res in enumerate(results):
                        st.write(f"**Rank {idx+1}**")
                        st.text(res.get("text", "본문 없음"))

                        st.markdown("**유사도 세부 점수**")
                        st.write(f"- 결합 점수 (Hybrid): `{res.get('combined_score', 0.0):.4f}`")
                        st.write(f"- 벡터 유사도 점수: `{res.get('vector_score', 0.0):.4f}`")
                        st.write(f"- BM25 키워드 점수: `{res.get('bm25_score', 0.0):.4f}`")

                        st.markdown("**문서 메타데이터**")
                        st.json(res.get("metadata", {}))
                        st.markdown("---")
                else:
                    st.warning("검색 결과가 없습니다. 쿼리를 다시 확인해주세요.")

            except Exception as e:
                st.error(f"검색 실패: {e}")

        st.markdown("---") 
 
        # 문서 추가
        st.markdown("#### 2) 문서 추가")
        doc_id = st.text_input("문서 ID", placeholder="문서의 고유 ID", key="add_id")
        doc_text = st.text_area("문서 내용", placeholder="문서 텍스트 입력", key="add_text")
        doc_metadata_str = st.text_area("문서 메타데이터 (JSON)", placeholder='예: {"key": "value"}', key="add_meta")
        
        if st.button("문서 추가"):
            try:
                if doc_metadata_str.strip():
                    metadata_dict = json.loads(doc_metadata_str)
                else:
                    metadata_dict = {}
                detail = add_document(doc_id, doc_text, metadata_dict)
                st.success(detail)
            except Exception as e:
                st.error(f"문서 추가 실패: {e}")

        st.markdown("---")
        
        # 문서 삭제
        st.markdown("#### 3) 문서 삭제")
        delete_id = st.text_input("삭제할 문서 ID", placeholder="삭제할 문서의 ID 입력", key="delete_id")
        if st.button("문서 삭제", key="delete_button"):
            try:
                detail = delete_document(delete_id)
                st.success(detail)
            except Exception as e:
                st.error(f"문서 삭제 실패: {e}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
