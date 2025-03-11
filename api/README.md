서비스 실행 순서 ( /LLM_RAG_PROJ 경로에서)

1. 가상환경 활성화
source venv/bin/activate # 리눅스 환경기준
.\venv\Scripts\activate # 윈도우 환경기준

2. 서버 활성화
uvicorn api.server:app --host 0.0.0.0 --port 9000

3. streamlit (통합 인터페이스) 활성화
streamlit run ./src/app.py