import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

class FinShibainuLLM:
    """LangChain과 호환 가능한 LLM 래퍼"""
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 4bit 양자화된 모델 로드
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

        # HuggingFacePipeline 생성 (LangChain과 호환)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=500
        )

        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    def get_llm(self):
        """LangChain에서 사용할 LLM 반환"""
        return self.llm


def build_rag_pipeline():
    """RAG 파이프라인을 구축하는 함수"""
    
    # 모델 경로 설정
    model_path = "/home/inseong/LLM_RAG_PROJ/models/FinShibainu_4bit"

    # 금융 특화 LLM 로드
    finshibainu_llm = FinShibainuLLM(model_path).get_llm()

    # 벡터 데이터베이스 로드
    chroma_db_path = "/home/inseong/LLM_RAG_PROJ/data/chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    # LangChain 기반 RetrievalQA 구축
    rag_chain = RetrievalQA.from_chain_type(
        llm=finshibainu_llm,  # LangChain 호환 LLM
        chain_type="stuff",
        retriever=retriever
    )

    return rag_chain

# 테스트 실행
if __name__ == "__main__":
    rag_pipeline = build_rag_pipeline()
    print("RAG 파이프라인이 성공적으로 구축되었습니다.")
