서비스 실행 순서 ( /LLM_RAG_PROJ 경로에서)

1. 가상환경 활성화
source venv/bin/activate 

2. 서버 활성화
uvicorn api.server:app --host 0.0.0.0 --port 9000

3. gradio 활성화
python -m src.app

4. chroma_db dashboard 활성화
streamlit run dashboard/dashboard.py