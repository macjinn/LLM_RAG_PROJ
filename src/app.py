import os
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

def stop_instance():
    """ Gradio UI 종료 후 인스턴스를 종료합니다. """
    os._exit(0)

# 예제 질문 리스트
examples = [
    "현재 가장 높은 금리를 제공하는 예금 상품은?",
    "가입 제한이 없는 예금 상품을 추천해줘.",
    "청년을 위한 우대금리가 적용된 예금이 있을까?"
]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # **금융상품 추천 시스템 (FinShibainu_4bit)**
        **AI 기반 금융상품 추천 시스템**  
        원하는 금융상품을 입력하면 적절한 정보를 제공합니다.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 질문을 입력하세요")
            user_input = gr.Textbox(lines=5, placeholder="예) 나에게 맞는 정기예금 상품을 추천해줘.")
            example_buttons = gr.Radio(examples, label="예제 질문을 선택하세요")

        with gr.Column():
            gr.Markdown("### AI 응답")
            output_box = gr.Textbox(lines=6, interactive=False)

    example_buttons.change(fn=lambda x: x, inputs=[example_buttons], outputs=[user_input])
    submit_button = gr.Button("질문하기")
    submit_button.click(fn=chat_interface, inputs=[user_input], outputs=[output_box])

    # 종료 버튼 추가
    stop_button = gr.Button("인스턴스 종료")
    stop_button.click(fn=stop_instance)

# 실행
if __name__ == "__main__":
    demo.launch()

