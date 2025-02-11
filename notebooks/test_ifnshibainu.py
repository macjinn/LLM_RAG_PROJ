from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# 모델 저장 디렉토리 설정
MODEL_DIR = "/home/s2019105385/2025_1/LLM_RAG_PROJ/models/FinShibainu" 

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
    print(f"저장된 모델 로드.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

# 모델을 GPU 또는 CPU로 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"모델이 `{device}`에서 실행됩니다.")

# 입력 예시
input_text = "금리가 높은 적금 상품 추천해줘"

# 입력을 토큰화
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer(input_text, return_tensors="pt").to(device)


# 모델 예측
with torch.no_grad():
    output = model.generate(**inputs, max_length=100)

# 결과 디코딩
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"💬 모델 응답: {response}")
