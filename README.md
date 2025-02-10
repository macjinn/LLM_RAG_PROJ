금융 상품 추천 시스템 (FinShibainu 기반)

1. 프로젝트 개요

이 프로젝트는 금융 상품 추천을 위한 AI 기반 시스템으로, 모듈러 RAG (Modular RAG) 기법을 활용하여 사용자 입력을 처리하고 적절한 금융 상품을 추천하는 시스템입니다. 금융 투자 관련 데이터베이스를 구축하고, 대규모 언어 모델(LLM)을 통해 사용자 맞춤형 금융 상품 정보를 제공합니다.

2. 사용 모델 및 이유

FinShibainu 모델

이 시스템은 Hugging Face에서 제공하는 "FinShibainu" 모델을 사용합니다.

출처: Hugging Face - FinShibainu

사용 이유:

금융 특화 모델: 금융 및 회계 관련 데이터를 학습하여 금융 도메인에 최적화됨

자연어 처리 성능: 범용 LLM보다 금융 데이터를 정확하게 이해하고 응답할 수 있음

오픈소스 라이선스: 자유롭게 사용 및 확장 가능

3. 시스템 개요 및 구조

이 시스템은 모듈러 RAG를 활용하여 구축되며, LangChain을 기반으로 RAG 파이프라인을 구현합니다. 모듈화된 구조를 통해 각 요소를 독립적으로 최적화할 수 있습니다.

[데이터 수집 및 전처리]
        │
        ▼
[텍스트 임베딩 생성 및 벡터 저장]
        │
        ▼
[벡터 데이터베이스 구축 (FAISS)]
        │
        ▼
[사용자 쿼리 입력 (웹 인터페이스)]
        │
        ▼
[벡터 검색 (유사 금융 상품 탐색)]
        │
        ▼
[LLM 프롬프트 구성 및 추천 결과 생성]
        │
        ▼
[사용자에게 추천 금융 상품 표시]

4. 사용 알고리즘 및 기술

(1) 모듈러 RAG (Modular RAG) 기법

이유: 단순히 사전 학습된 모델로 응답하는 것이 아니라, 최신 금융 데이터를 검색하여 더욱 정확한 정보를 제공하기 위해 사용됩니다.

특징:

검색(retrieval)과 생성(generation) 모듈 분리

다양한 데이터 소스를 활용할 수 있도록 유연하게 설계

검색 및 생성 모듈을 개별적으로 최적화 가능

구성 요소:

Retriever: FAISS 벡터 검색을 통해 적절한 금융 상품을 탐색

Generator: FinShibainu 모델을 활용하여 사용자 맞춤형 추천 응답 생성

(2) LangChain을 활용한 RAG 파이프라인

이유: LangChain을 사용하면 RAG 파이프라인을 쉽게 구성하고 확장 가능

구성 요소:

RetrievalQA 체인을 활용하여 검색 + 생성 통합

벡터 데이터베이스와 LLM을 연결하여 금융 상품 정보 제공

프롬프트 엔지니어링을 통해 응답 품질 향상

(3) 벡터 데이터베이스 (FAISS)

이유: 대량의 금융 상품 데이터를 빠르게 검색하고 유사도를 비교하기 위해 사용됩니다.

구현: sentence-transformers를 활용하여 금융 상품 설명을 벡터로 변환 후 저장

(4) LLM (FinShibainu) 기반 자연어 처리

이유: 사용자의 투자 목적, 가입 조건, 자금 규모 등 다양한 요소를 고려하여 적절한 상품을 추천하기 위함

활용 기술: transformers 라이브러리 기반의 FinShibainu 모델

5. 데이터 학습 및 사용자 인터페이스

(1) 데이터 학습 및 파인튜닝

금융 상품 데이터셋을 전처리하여 텍스트 임베딩 생성

FAISS 데이터베이스에 벡터화하여 저장

추가적인 금융 도메인 데이터로 FinShibainu 모델 파인튜닝 가능

(2) 사용자 인터페이스 (UI)

Gradio 기반 웹 인터페이스 제공

사용자는 텍스트 입력창을 통해 금융 상품 추천 요청 가능

모델은 사용자 입력 → 금융 상품 검색 → LLM 생성 → 추천 결과 반환의 과정을 거쳐 응답 제공

6. 설치 및 실행 방법

(1) 환경 설정

# 프로젝트 클론
git clone https://github.com/your-repo/project_name.git
cd project_name

# 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 필요한 라이브러리 설치
pip install -r requirements.txt

(2) 시스템 실행

1️⃣ 데이터 전처리 및 벡터 생성

python src/data_processing.py
python src/embedding.py

2️⃣ 모델 로드 및 RAG 파이프라인 실행

python src/model_training.py
python src/rag_pipeline.py

3️⃣ 웹 인터페이스 실행

python src/app.py

7. 결론 및 기대 효과

이 시스템은 전통적인 금융 추천 시스템과 달리 최신 데이터를 활용하는 모듈러 RAG 기법을 사용하여 더욱 신뢰성 높은 금융 상품 추천을 제공합니다.

이를 통해 사용자는 본인의 투자 성향에 맞는 상품을 보다 정확하게 추천받을 수 있으며, 금융 기업은 개인 맞춤형 추천 서비스를 보다 효과적으로 운영할 수 있습니다.