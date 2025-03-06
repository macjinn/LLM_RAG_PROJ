from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os, time

# 모델 저장 디렉토리 설정
MODEL_DIR = "/home/inseong/LLM_RAG_PROJ/models/FinShibainu_4bit" 

# 모델 및 토크나이저 로드 (저장된 모델이 있으면 다운로드 생략)
model_name = "aiqwe/FinShibainu"

if not os.path.exists(MODEL_DIR):
    print(f"모델이 저장된 경로가 없습니다. `{MODEL_DIR}`에 모델을 다운로드합니다...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)
    # 모델을 지정된 폴더에 저장
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    print(f"모델이 `{MODEL_DIR}`에 저장.")
else:
    print("저장된 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    # 4-bit 양자화 적용 (VRAM 절약)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4-bit 양자화 적용
        bnb_4bit_compute_dtype=torch.float16,  # 계산은 FP16 사용
        bnb_4bit_use_double_quant=True,  # 추가 양자화 적용 (메모리 절약)
    )

    # 모델 로드 (자동으로 GPU/CPU에 최적 분배)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=quantization_config,
        device_map="auto"  # VRAM 부족 시 일부 CPU로 오프로드
    )
    # 모델을 GPU 또는 CPU로 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"모델이 `{device}`에서 실행됩니다.")


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
    대한민국 경제상황을 종합적으로 정리하고 앞으로의 경제 동향을 보고서의 형태로 작성해줘.

    Answer in Korean.
    """

# 입력을 토큰화
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

start_time = time.time()

# 답변 생성
print("모델이 답변을 생성하는 중...\n")
max_length = 1000 # 생성 토큰 수
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



