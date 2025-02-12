from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

def load_model(model_name="aiqwe/FinShibainu"):
    """ FinShibainu 모델을 로드합니다. """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model
###
if __name__ == "__main__":
    tokenizer, model = load_model()
    print("모델이 성공적으로 로드되었습니다.")

