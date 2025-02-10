from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def build_rag_pipeline():
    """ Langchain을 사용하여 RAG 파이프라인을 설정합니다. """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("data/embeddings/faiss_index", embeddings)
    retriever = vectorstore.as_retriever()

    return RetrievalQA.from_chain_type(
        llm="aiqwe/FinShibainu",
        chain_type="stuff",
        retriever=retriever
    )

if __name__ == "__main__":
    rag_pipeline = build_rag_pipeline()
    print("RAG 파이프라인이 성공적으로 구축되었습니다.")

