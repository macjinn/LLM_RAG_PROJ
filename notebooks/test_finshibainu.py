from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os, time



# ----------------------------  RTX3080 10gb vram 기준  ----------------------------  

# # 모델 저장 디렉토리 설정
# MODEL_DIR = "/home/inseong/LLM_RAG_PROJ/models/FinShibainu_4bit" 

# # 모델 및 토크나이저 로드 (저장된 모델이 있으면 다운로드 생략)
# model_name = "aiqwe/FinShibainu"

# if not os.path.exists(MODEL_DIR):
#     print(f"모델이 저장된 경로가 없습니다. `{MODEL_DIR}`에 모델을 다운로드합니다...")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     torch_dtype="auto"
# )
#     # 모델을 지정된 폴더에 저장
#     tokenizer.save_pretrained(MODEL_DIR)
#     model.save_pretrained(MODEL_DIR)
#     print(f"모델이 `{MODEL_DIR}`에 저장.")
# else:
#     print("저장된 모델 로드 중...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
#     # 4-bit 양자화 적용 (VRAM 절약)
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,  # 4-bit 양자화 적용
#         bnb_4bit_compute_dtype=torch.float16,  # 계산은 FP16 사용
#         bnb_4bit_use_double_quant=True,  # 추가 양자화 적용 (메모리 절약)
#     )

#     # 모델 로드 (자동으로 GPU/CPU에 최적 분배)
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_DIR,
#         quantization_config=quantization_config,
#         device_map="auto"  # VRAM 부족 시 일부 CPU로 오프로드
#     )
#     # 모델을 GPU 또는 CPU로 로드
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"모델이 `{device}`에서 실행됩니다.")


# ----------------------------  RTX3090 24gb vram 기준  ----------------------------  


# 모델 저장 디렉토리 설정 (양자화된 버전과 구분)
MODEL_DIR = "/home/inseong/LLM_RAG_PROJ/models/FinShibainu_full"
model_name = "aiqwe/FinShibainu"

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")

if not os.path.exists(MODEL_DIR):
    print(f"모델이 `{MODEL_DIR}`에 없으므로 다운로드합니다...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    # 모델 저장
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    print(f"모델이 `{MODEL_DIR}`에 저장되었습니다.")
else:
    print("저장된 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    print(f"모델이 `{device}`에서 실행됩니다.")


# --------------------------------------------------------------------------------

# 입력 데이터
information = """
은행: KDB산업은행
상품명: KDB Hi 입출금통장
기본금리: 1.8
최고금리(우대금리포함): 1.8

은행: NH농협은행
상품명: NH1934우대통장
기본금리: 0.1
최고금리(우대금리포함): 3.0

은행: 신한은행
상품명: 신한 주거래 미래설계통장
기본금리: 0.1
최고금리(우대금리포함): 0.75

은행: SC제일은행
상품명: 내월급통장
기본금리: 0.6
최고금리(우대금리포함): 3.1
이자지급방식: 월지급
"""

# 프롬프트
# input_text = f"""
# 당신은 금융상품 전문가이며 금융상품을 추천하는 상담원 역할입니다.
# 다음 금융 상품 데이터에서 '최고금리(우대금리포함)' 값이 가장 높은 **단 하나의** 상품을 찾아 출력 형식 예시를 참고하여 출력하세요.
# 특히 답변에서 상품추천의 이유를 상세하게 설명해야 합니다. 이때 데이터 정보에서 근거를 들어 논리적이고 상세하게 설명하세요.

# 금융 상품 데이터:
# {information}

# 출력 형식 예시:
# [은행명]: [상품명]
# 상품설명: 
# 상품추천 이유: 

# """
# print(f"입력:\n{input_text}\n")



# input_text = f"""
#     You are a financial product expert and consultant who always responds in Korean. Your task is to analyze the given financial product data and recommend exactly one product that has the highest "최고금리(우대금리포함)" (highest interest rate including preferential rates).

#     Please follow these instructions carefully:
#     1. Use the provided data only. Do not add any information that is not present in the data.
#     2. If you do not know the answer or if the data does not contain sufficient information, simply respond with "모르겠습니다" (I don't know). Do not fabricate an answer.
#     3. Clearly extract and present the key details: Bank Name, Product Name, Basic Interest Rate, Highest Interest Rate (including preferential rate), and any relevant conditions or restrictions.
#     4. Provide a detailed recommendation reason based solely on the data, explaining why this product is the best choice.
#     5. Format your answer exactly as shown in the output format below.

#     Financial Product Data:
#     {information}

#     Output Format Example:
#     [은행명]: [상품명]
#     기본금리: 
#     최고금리(우대금리포함): 
#     가입조건/제한: 
#     추천 사유: 

#     Answer in Korean.
#     """

input_text = f"""

    You are a financial product expert and consultant who always responds in Korean.
    Your task is to analyze the user query and the given financial product data to recommend the most suitable financial product.

    ## User Query:
    청년을 위한 우대금리가 적용된 예금이 있을까?

    ## Financial Product Data:
    카테고리: 상품 개요
상품 특징: 병역의무수행자들의 전역 후 목돈 마련을 위한 고금리 자유적립식 정기적금 상품
출시일: 2018년 8월 29일
이자 혜택: 만기 해지 시 이자소득 비과세 및 1% 이자지원금 지급
매칭 지원금: 정부에서 지급, 병역법 시행령 제158조의2에 따라 결정

은행: BNK부산은행
상품명: 더(The) 특판 정기예금
기본금리(단리이자 %): 2.75
최고금리(우대금리포함, 단리이자 %): 3.2
은행 최종제공일: 2025-01-20
만기 후 금리: - 만기후1년내: 가입기간별 일반정기예금이율 x 50%,
- 만기후1년초과:가입기간별 일반정기예금이율 x 20%
가입방법: 인터넷뱅킹,스마트뱅킹
우대조건: * 우대이율 (최대 0.45%p)
가. 모바일뱅킹 금융정보 및 혜택알림 동의 우대이율 : 0.10%p
나. 이벤트 우대이율 : 최대 0.35%p 
1) 더(The) 특판 정기예금 신규고객 우대이율 : 0.20%p
2) 특판 우대이율 : 0.15%p
가입 제한조건: 제한없음
가입대상: 실명의 개인
기타 유의사항: 1. 가입금액 : 1백만원 이상 제한없음 (원단위)
2. 가입기간 : 1개월, 3개월, 6개월, 1년, 2년, 3년
3. 이자지급방식 : 만기일시지급식
최고한도: 정보 없음
전월취급평균금리(만기 12개월 기준): 3.1

은행: Sh수협은행
상품명: Sh평생주거래우대예금
(만기일시지급식)
기본금리(단리이자 %): 2.4
최고금리(우대금리포함, 단리이자 %): 2.8
은행 최종제공일: 2025-01-20
만기 후 금리: * 만기후 1년 이내
 - 만기당시 일반정기예금(월이자지급식) 계약기간별 기본금리 1/2
* 만기후 1년 초과
 - 만기당시 보통예금 기본금리
가입방법: 영업점
우대조건: * 최대우대금리 : 0.4%
1. 거래고객우대금리 : 최대0.1% (신규시) 
 -최초예적금고객/재예치/장기거래 각 0.05% 
2. 거래실적우대금리 : 최대0.3% (만기시)
 -급여,연금이체등/수협카드결제/공과금이체등 각0.1%
※단위:연%p
가입 제한조건: 제한없음
가입대상: 실명의 개인
기타 유의사항: - 1인 1계좌
- 최저 100만원 이상
최고한도: 정보 없음
전월취급평균금리(만기 12개월 기준): 2.4

    ## Instructions:
    1. Use only the provided data. Do not add any information that is not present in the data.
    2. If you do not know the answer or the data is insufficient, respond with "모르겠습니다".
    3. Clearly extract and present the following details:
       - Bank Name
       - Product Name
       - Basic Interest Rate
       - Highest Interest Rate (including preferential rate)
       - Conditions/Restrictions
       - Recommendation Reason

    Please provide your final answer after the tag [USER_ANSWER]: 
    
    """

# 입력을 토큰화
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

start_time = time.time()

# 답변 생성
print("모델이 답변을 생성하는 중...\n")
max_length = 2000 # 생성 토큰 수
with torch.no_grad():
    output = model.generate(**inputs, max_length=max_length)


# 결과 디코딩
response_text = tokenizer.decode(output[0], skip_special_tokens=True)

if "Answer in Korean." in response_text:
    response_text = response_text.split("Answer in Korean.")[-1].strip()
print(f"💬 모델 응답: {response_text}\n")

end_time = time.time()
print(f"생성한 토큰 수: {len(output[0])}")
print(f"모델 응답 소요 시간: {end_time - start_time:.3f}")



