from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

def create_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """ 텍스트 데이터를 벡터화하여 FAISS 인덱스 저장 """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local("data/embeddings/faiss_index")

if __name__ == "__main__":
    texts = ["금융 상품 추천 시스템 구축", "적금과 펀드의 차이", "ETF 투자 전략"]
    create_embeddings(texts)

