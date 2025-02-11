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
##
iface = gr.Interface(fn=generate_response, inputs="text", outputs="text", title="금융 추천 시스템")
iface.launch()


import gradio as gr
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# 예시: vectorstore, retriever, 그리고 RAG 체인 구성
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("path_to_faiss_index", embeddings)
retriever = vectorstore.as_retriever()

# LLM 로드 (이미 재학습한 4-bit 모델)
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("path_to_finetuned_model", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("path_to_finetuned_model")

# Langchain LLM Wrapper (자체 구현 또는 community wrapper 활용)
def llm_generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Langchain RetrievalQA 구성 (Modular RAG)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_generate,
    chain_type="stuff",  # 혹은 Modular RAG에 맞게 chain 유형 선택
    retriever=retriever
)

def chat_interface(user_input):
    response = qa_chain.run(user_input)
    return response

gr.Interface(fn=chat_interface, inputs="text", outputs="text", title="금융상품 추천 시스템").launch()



