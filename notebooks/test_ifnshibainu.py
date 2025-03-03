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


# 입력
information = '은행: KDB산업은행\n상품명: KDB Hi 입출금통장\n기본금리: 1.8\n최고금리(우대금리포함): 1.8, 은행: NH농협은행\n상품명: NH1934우대통장\n기본금리: 0.1\n최고금리(우대금리포함): 3.0\n이자지급방식: 분기지급'
input_text = f"다음의 금융사품 정보를 참고하여 금리가 높은 적금 상품 1개만 추천해줘 \n 정보 \n {information}"
print(f"입력:{input_text}\n")

# 입력을 토큰화
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

start_time = time.time()

# 답변 생성
print("모델이 답변을 생성하는 중...\n")
max_length = 700 # 생성 토큰 수
with torch.no_grad():
    output = model.generate(**inputs, max_length=max_length)

# 결과 디코딩
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"💬 모델 응답: {response}\n")

end_time = time.time()
print(f"생성한 토큰 수: {len(response)}")
print(f"모델 응답 소요 시간: {end_time - start_time:.3f}")



