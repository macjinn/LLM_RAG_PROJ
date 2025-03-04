import gradio as gr
from src.rag_pipeline import build_rag_pipeline

# RAG 파이프라인 구축 (FinShibainu_4bit 기반 RetrievalQA 체인)
rag_pipeline = build_rag_pipeline()

def chat_interface(user_input):
    """
    사용자 입력을 받아 RAG 파이프라인을 통해 답변을 생성합니다.
    """
    response = rag_pipeline.run(user_input)
    return response

# Gradio 인터페이스 생성
iface = gr.Interface(fn=chat_interface,
                     inputs="text",
                     outputs="text",
                     title="금융상품 추천 시스템 (FinShibainu_4bit)")
iface.launch()
