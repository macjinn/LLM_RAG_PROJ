import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

class FinShibainuLLM:
    """LangChain과 호환 가능한 LLM 래퍼 클래스"""
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 4-bit 양자화된 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto"
        )

        # HuggingFace Pipeline 생성 (LangChain과 호환)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=800
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    def get_llm(self):
        return self.llm

def build_rag_pipeline():
    """
    FinShibainu_4bit 모델과 Chroma 벡터스토어를 사용하여
    커스텀 프롬프트가 적용된 RAG 파이프라인을 구축합니다.
    """
    # 금융 특화 LLM 로드
    model_path = "/home/inseong/LLM_RAG_PROJ/models/FinShibainu_4bit"
    finshibainu_llm = FinShibainuLLM(model_path).get_llm()

    # 벡터 데이터베이스 로드 (Chroma 사용)
    chroma_db_path = "/home/inseong/LLM_RAG_PROJ/data/chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 프롬프트 템플릿 정의
    custom_template = """
    You are a financial product expert and consultant who always responds in Korean. Your task is to analyze the given financial product data and recommend exactly one product that has the highest "최고금리(우대금리포함)" (highest interest rate including preferential rates).

    Please follow these instructions carefully:
    1. Use the provided data only. Do not add any information that is not present in the data.
    2. If you do not know the answer or if the data does not contain sufficient information, simply respond with "모르겠습니다" (I don't know). Do not fabricate an answer.
    3. Clearly extract and present the key details: Bank Name, Product Name, Basic Interest Rate, Highest Interest Rate (including preferential rate), and any relevant conditions or restrictions.
    4. Provide a detailed recommendation reason based solely on the data, explaining why this product is the best choice.
    5. Format your answer exactly as shown in the output format below.

    Financial Product Data:
    {context}

    Output Format Example:
    [은행명]: [상품명]
    기본금리: 
    최고금리(우대금리포함): 
    가입조건/제한: 
    추천 사유: 

    Answer in Korean.
    """

    prompt_template = PromptTemplate(
        input_variables=["context"],
        template=custom_template,
    )

    # RetrievalQA 체인 생성
    rag_chain = RetrievalQA.from_chain_type(
        llm=finshibainu_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt_template
        }
    )


    return rag_chain

if __name__ == "__main__":
    rag_pipeline = build_rag_pipeline()
