# dashboard/dashboard.py
import streamlit as st
import requests
import json

# FastAPI 서버 주소 (내부 또는 공인 IP로 변경 필요)
API_URL = "http://localhost:8000"

st.title("ChromaDB 벡터스토어 관리 대시보드")
st.markdown("이 대시보드를 통해 벡터 데이터를 검색, 추가, 삭제할 수 있습니다.")

st.header("벡터 검색")
query = st.text_input("검색어 입력", placeholder="예) 추천 예금 상품")
top_k = st.slider("검색 결과 개수", min_value=1, max_value=10, value=5)

if st.button("검색 실행"):
    payload = {"query": query, "top_k": top_k}
    response = requests.post(f"{API_URL}/search", json=payload)
    if response.status_code == 200:
        try:
            response_data = response.json()
            results = response_data.get("results", [])  # "results" 키가 없으면 빈 리스트 반환
            
            if not isinstance(results, list):  # results가 리스트인지 확인 (예외 방지)
                st.error("Unexpected response format from API")
            else:
                st.success(f"{len(results)}개의 결과가 검색되었습니다.")
                for idx, res in enumerate(results):
                    text = res.get("text", "텍스트 없음")  # text 키가 없을 경우 대비
                    score = res.get("score", 0.0)  # score 키가 없을 경우 대비
                    metadata = res.get("metadata", {})  # metadata가 없을 경우 대비

                    st.markdown(f"**Rank {idx+1}:** {text} (유사도 (단순 가까운거리) : {score:.2f})")
                    st.json(metadata)
        except Exception as e:
            st.error(f"응답 처리 중 오류 발생: {e}")
    else:
        st.error(f"검색 실패: {response.status_code} - {response.text}")

st.header("문서 추가")
doc_id = st.text_input("문서 ID", placeholder="문서의 고유 ID")
doc_text = st.text_area("문서 내용", placeholder="문서 내러티브 텍스트 입력")
doc_metadata = st.text_area("문서 메타데이터 (JSON 형식)", placeholder='예: {"key": "value"}')
if st.button("문서 추가"):
    try:
        metadata_dict = json.loads(doc_metadata) if doc_metadata.strip() else {}
    except Exception as e:
        st.error("메타데이터 파싱 오류: " + str(e))
        metadata_dict = {}
    payload = {"id": doc_id, "text": doc_text, "metadata": metadata_dict}
    response = requests.post(f"{API_URL}/add", json=payload)
    if response.status_code == 200:
        st.success(response.json()["detail"])
    else:
        st.error("문서 추가 실패: " + response.text)

st.header("문서 삭제")
delete_id = st.text_input("삭제할 문서 ID", placeholder="삭제할 문서의 ID 입력")
if st.button("문서 삭제"):
    response = requests.delete(f"{API_URL}/delete/{delete_id}")
    if response.status_code == 200:
        st.success(response.json()["detail"])
    else:
        st.error("문서 삭제 실패: " + response.text)
