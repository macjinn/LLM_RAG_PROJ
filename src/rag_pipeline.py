import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

from rank_bm25 import BM25Okapi
import numpy as np

from langchain_core.retrievers import BaseRetriever
from typing import Callable, List
from langchain_core.documents import Document


class FinShibainuLLM:
    """LangChain과 호환 가능한 LLM 래퍼 클래스"""
    def __init__(self, model_path):
        self.model_path = model_path

        # quantization_config 정의
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True  # CPU 오프로드 활성화
        )

        # 4-bit 양자화된 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=self.quantization_config,
            device_map="auto",
            torch_dtype=torch.float16  # 메모리 절약
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
        """LangChain과 호환 가능한 LLM 반환"""
        return self.llm

# 하이브리드 리트리버 정의
class HybridRetriever:
    """ HNSW(백터 유사도) + bm25(키워드 매칭) 선형결합 한 탐색 알고리즘 클래스
        search score = alpha * 백터 유사도(정규화) + (1-alpha) * bm25(정규화)
        정렬 후 top_k 정보 반환
    """
    def __init__(self, vectorstore, bm25_docs, alpha=0.5, top_k=5):
        self.vectorstore = vectorstore
        self.alpha = alpha
        self.top_k = top_k
        self.bm25_docs = bm25_docs

        tokenized_corpus = [doc.page_content.split(" ") for doc in bm25_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query):
        # 벡터 유사도를 통해 검색
        vector_results = self.vectorstore.similarity_search_with_score(query, k=self.top_k * 2)

        # BM25를 통해 키워드 중점 검색
        bm25_scores = self.bm25.get_scores(query.split(" "))
        bm25_top_n_idx = np.argsort(bm25_scores)[::-1][:self.top_k * 2]

        bm25_results = [(self.bm25_docs[idx], bm25_scores[idx]) for idx in bm25_top_n_idx]

        # 정규화
        def normalize(scores):
            arr = np.array(scores)
            return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

        vector_docs = [doc for doc, score in vector_results]
        vector_scores = normalize([score for doc, score in vector_results])

        bm25_docs = [doc for doc, score in bm25_results]
        bm25_scores_norm = normalize([score for doc, score in bm25_results])

        # 하이브리드 점수 계산
        all_docs = {}
        for doc, score in zip(vector_docs, vector_scores):
            doc_id = doc.metadata.get("id", doc.page_content[:20])
            all_docs[doc_id] = {"doc": doc, "score": (1 - self.alpha) * score}

        for doc, score in zip(bm25_docs, bm25_scores_norm):
            doc_id = doc.metadata.get("id", doc.page_content[:20])
            if doc_id in all_docs:
                all_docs[doc_id]["score"] += self.alpha * score
            else:
                all_docs[doc_id] = {"doc": doc, "score": self.alpha * score}

        # 정렬 후 top_k 선택
        sorted_docs = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:self.top_k]]


# LangChain 호환을 위해 lambda 함수로 래핑
class CustomRetriever(BaseRetriever):
    retriever_func: Callable[[str], List[Document]]
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.retriever_func(query)


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

    # 기존에 사용하던 단일 HNSW
    #retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 모든 문서 로딩 → BM25용으로
    raw_docs = vectorstore.get()
    docs_texts = raw_docs.get("documents", [])  # str 리스트
    metadatas = raw_docs.get("metadatas", [])

    from langchain.schema import Document
    all_docs = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(docs_texts, metadatas)
    ]
    print(f"총 문서 수: {len(docs_texts)}")
    print(f"None metadata 개수: {sum(1 for m in metadatas if m is None)}")

    # HybridRetriever 호출
    hybrid_retriever = HybridRetriever(vectorstore=vectorstore, bm25_docs=all_docs, alpha=0.5, top_k=3)

    # 최종 CustomRetriever 생성 (키워드 인자로 전달)
    retriever = CustomRetriever(retriever_func=hybrid_retriever.retrieve)

    # 프롬프트 템플릿 정의
    custom_template = """
    You are a financial product expert and consultant who always responds in Korean.
    Your task is to analyze the user's query and the given financial product data to recommend the most suitable financial product for the user.

    Please follow these instructions carefully:
    1. Use the provided data only. Do not add any information that is not present in the data.
    2. If you do not know the answer or if the data does not contain sufficient information, simply respond with "모르겠습니다" (I don't know). Do not fabricate an answer.
    3. Clearly extract and present the key details: Bank Name, Product Name, Basic Interest Rate, Highest Interest Rate (including preferential rate), and any relevant conditions or restrictions.
    4. Provide a detailed recommendation reason based solely on the data, explaining why this product is the best choice.
    5. Format your answer exactly as shown in the output format below.

    Financial Product Data:
    {context}

    Format Example:
    [은행명]: [상품명]
    기본금리: 
    최고금리(우대금리포함): 
    가입조건/제한: 
    추천 사유: 

    Please provide your final recommendation answer starting after the tag [USER_ANSWER]:
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

    return rag_chain, vectorstore

if __name__ == "__main__":
    rag_pipeline = build_rag_pipeline()
