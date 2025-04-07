import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 모델 저장 경로 설정
model_dir = "./models/FinShibainu_4bit"

# 모델 및 토크나이저 로드 (Hugging Face Hub에서 다운로드)
model_name = "aiqwe/FinShibainu"

# models 폴더가 없으면 생성
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 토크나이저 로드 (로컬 저장 후 사용)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_dir)  # 모델 폴더에 저장

# BitsAndBytes 4-bit 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit 양자화
    llm_int8_enable_fp32_cpu_offload=True
)

# 모델 다운로드 및 로컬 저장
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)
model.save_pretrained(model_dir)  # 모델 폴더에 저장