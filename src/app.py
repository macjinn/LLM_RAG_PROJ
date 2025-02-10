import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 로드
model_name = "aiqwe/FinShibainu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(user_input):
    """ 사용자 입력을 받아 모델이 응답을 생성합니다. """
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

iface = gr.Interface(fn=generate_response, inputs="text", outputs="text", title="금융 추천 시스템")
iface.launch()

