import os
import time
import subprocess
import gradio as gr
import requests

# FastAPI 서버 실행
def run_fastapi():
    """FastAPI 서버를 별도의 프로세스로 실행"""
    return subprocess.Popen(
        ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "9000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

# FastAPI 실행
fastapi_process = run_fastapi()

# FastAPI 주소 설정
API_URL = "http://127.0.0.1:9000"

def chat_interface(user_input):
    """FastAPI의 /chat 엔드포인트를 호출하여 RAG 응답을 반환"""
    payload = {"query": user_input}
    try:
        response = requests.post(f"{API_URL}/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("user_answer", "응답을 생성할 수 없습니다."), data.get("admin_answer", "관리자 응답 없음")
    except Exception as e:
        return f"오류 발생: {e}", "관리자 응답 없음"

def stop_instance():
    """Gradio 종료 시 FastAPI 프로세스도 종료"""
    fastapi_process.terminate()  # FastAPI 종료
    os._exit(0)

# 예제 질문 리스트
examples = [
    "현재 가장 높은 금리를 제공하는 예금 상품은?",
    "가입 제한이 없는 예금 상품을 추천해줘.",
    "청년을 위한 우대금리가 적용된 예금이 있을까?"
]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 금융상품 추천 시스템 (FinShibainu_4bit)")
    gr.Markdown("AI 기반 금융상품 추천 시스템입니다. 원하는 금융상품에 대한 질문을 입력하세요.")

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 질문을 입력하세요")
            user_input = gr.Textbox(
                label="질문 입력",
                lines=5,
                placeholder="예) 나에게 맞는 정기예금 상품을 추천해줘.",
            )
            example_buttons = gr.Radio(
                examples, label="예제 질문을 선택하세요"
            )

        with gr.Column(scale=5):
            gr.Markdown("### AI 응답")
            with gr.Tabs():
                with gr.TabItem("사용자 답변"):
                    user_output = gr.Textbox(
                        label="유저용 응답",
                        lines=8,
                        interactive=False
                    )
                with gr.TabItem("관리자용 전체 응답"):
                    admin_output = gr.Textbox(
                        label="관리자 응답",
                        lines=12,
                        interactive=False
                    )

    example_buttons.change(fn=lambda x: x, inputs=[example_buttons], outputs=[user_input])

    submit_button = gr.Button("질문하기")
    submit_button.click(fn=chat_interface, inputs=[user_input], outputs=[user_output, admin_output])

    stop_button = gr.Button("인스턴스 종료")
    stop_button.click(fn=stop_instance)

# 실행
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
