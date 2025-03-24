import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

from rank_bm25 import BM25Okapi
import numpy as np
import hashlib

from langchain_core.retrievers import BaseRetriever
from typing import Callable, List
from langchain_core.documents import Document

# class FinShibainuLLM:
#     """LangChain과 호환 가능한 LLM 래퍼 클래스 (RTX 3080용)"""
#     def __init__(self, model_path):
#         self.model_path = model_path

#         # quantization_config 정의
#         self.quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             llm_int8_enable_fp32_cpu_offload=True  # CPU 오프로드 활성화
#         )

#         # 4-bit 양자화된 모델 로드
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_path,
#             quantization_config=self.quantization_config,
#             device_map="auto",
#             torch_dtype=torch.float16  # 메모리 절약
#         )

#         # HuggingFace Pipeline 생성 (LangChain과 호환)
#         self.pipe = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             max_new_tokens=800
#         )
#         self.llm = HuggingFacePipeline(pipeline=self.pipe)

#     def get_llm(self):
#         """LangChain과 호환 가능한 LLM 반환"""
#         return self.llm


class FinShibainuLLM:
    """LangChain과 호환 가능한 LLM 래퍼 클래스 (RTX 3090용)"""
    def __init__(self, model_path):
        self.model_path = model_path

        # 토크나이저 및 모델 로드 (Full Precision)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,     
            device_map="auto"              # GPU 자동 할당
        )

        # 텍스트 생성 파이프라인 (LangChain 호환)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=4000
        )

        # LangChain 통합용 LLM 객체
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    def get_llm(self):
        """LangChain과 호환 가능한 LLM 반환"""
        return self.llm


class HybridRetriever:
    """ HNSW(백터 유사도) + BM25(키워드 매칭) 선형 결합 탐색 알고리즘 클래스
        search score = alpha * (벡터 유사도 정규화) + (1 - alpha) * (BM25 정규화)
        최종 정렬 후 top_k 정보 반환
    """
    def __init__(self, vectorstore, bm25_docs, alpha=0.5, top_k=5):
        self.vectorstore = vectorstore
        self.alpha = alpha
        self.top_k = top_k
        self.bm25_docs = bm25_docs

        # BM25용 토큰화: key_summary가 있으면 해당 필드를, 없으면 전체 내용을 사용
        tokenized_corpus = []
        for doc in bm25_docs:
            text = doc.metadata.get("key_summary", doc.page_content)
            tokenized_corpus.append(text.split(" "))
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _get_doc_id(self, doc):
        # 고유 id가 없으면 page_content의 해시값을 fallback으로 사용
        if "id" in doc.metadata:
            return doc.metadata["id"]
        else:
            return hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:20]

    def retrieve_with_scores(self, query):
        try:
            vector_results = self.vectorstore.similarity_search_with_score(query, k=self.top_k * 2)
        except Exception as e:
            print(f"Vector search error: {e}")
            vector_results = []

        bm25_scores = self.bm25.get_scores(query.split(" "))
        bm25_top_n_idx = np.argsort(bm25_scores)[::-1][:self.top_k * 2]
        bm25_results = [(self.bm25_docs[idx], bm25_scores[idx]) for idx in bm25_top_n_idx]

        def normalize(scores):
            arr = np.array(scores)
            return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

        # 벡터 서치 결과 정규화
        vector_docs = [doc for doc, score in vector_results]
        vector_scores = normalize([score for doc, score in vector_results]) if vector_results else []
        print("DEBUG: vector_scores =", vector_scores)

        
        # BM25 서치 결과 정규화
        bm25_docs = [doc for doc, score in bm25_results]
        bm25_scores_norm = normalize([score for doc, score in bm25_results]) if bm25_results else []
        print("DEBUG: bm25_scores_norm =", bm25_scores)


        all_docs = {}
        # 벡터 서치 결과 반영 (가중치: (1 - alpha))
        for doc, score in zip(vector_docs, vector_scores):
            doc_id = self._get_doc_id(doc)
            all_docs[doc_id] = {
                "doc": doc,
                "vector_score": (1 - self.alpha) * score,
                "bm25_score": 0  # 아직 BM25 점수 없음
            }

        # BM25 서치 결과 반영 (가중치: alpha)
        for doc, score in zip(bm25_docs, bm25_scores_norm):
            doc_id = self._get_doc_id(doc)
            if doc_id in all_docs:
                all_docs[doc_id]["bm25_score"] = self.alpha * score
            else:
                all_docs[doc_id] = {
                    "doc": doc,
                    "vector_score": 0,
                    "bm25_score": self.alpha * score
                }
        
        # 최종 결합 점수 계산
        for doc_id, item in all_docs.items():
            item["score"] = item["vector_score"] + item["bm25_score"]

        sorted_docs = sorted(all_docs.values(), key=lambda x: x["score"], reverse=True)
        return [(item["doc"], item["score"], item["vector_score"], item["bm25_score"]) for item in sorted_docs[:self.top_k]]



class CustomRetriever(BaseRetriever):
    retriever_func: Callable[[str], List]
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.retriever_func(query)
        # RetrievalQA은 튜플을 받지 못 해서 따로 후처리
        # retrieve_with_scores 함수는 백터 스토어 점수 보고자 할 때 그대로 사용 가능
        if results and isinstance(results[0], tuple):
            return [doc for doc, score, vector_score, bm25_score in results]
        return results



def build_rag_pipeline():
    """
    FinShibainu 모델과 Chroma 벡터스토어를 사용하여
    커스텀 프롬프트가 적용된 RAG 파이프라인을 구축합니다.
    """
    # 금융 특화 LLM 로드
    model_path = "/home/inseong/LLM_RAG_PROJ/models/FinShibainu_full"
    finshibainu_llm = FinShibainuLLM(model_path).get_llm()

    # 벡터 데이터베이스 로드 (Chroma 사용)
    chroma_db_path = "/home/inseong/LLM_RAG_PROJ/data/chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)

    # 기존에 사용하던 단일 HNSW 대신 모든 문서를 로딩 (BM25용)
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

    # HybridRetriever 호출 (alpha, top_k 등 파라미터 조정 가능)
    hybrid_retriever = HybridRetriever(vectorstore=vectorstore, bm25_docs=all_docs, alpha=0.5, top_k=3)

    # 최종 CustomRetriever 생성
    retriever = CustomRetriever(retriever_func=hybrid_retriever.retrieve_with_scores)

    # 프롬프트 템플릿 정의 (JSON 형식 출력 요구 포함)
    custom_template = """
    You are a financial product expert and consultant who always responds in Korean.
    Your task is to analyze the user query and the given financial product data to recommend the most suitable financial product.

    ## User Query:
    {question}

    ## Financial Product Data:
    {context}

    ## Instructions:
    1. Use only the provided data. Do not add any information that is not present in the data.
    2. If you do not know the answer or the data is insufficient, respond with "모르겠습니다".
    3. Clearly extract and present the following format:
       - Bank Name
       - Product Name
       - Basic Interest Rate
       - Highest Interest Rate (including preferential rate)
       - Conditions/Restrictions
       - Recommendation Reason  
    4. Please provide your final answer after the tag [USER_ANSWER].

    ##[USER_ANSWER]:
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
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
    rag_pipeline, vectorstore = build_rag_pipeline()