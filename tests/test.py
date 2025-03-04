import os, json

json_path = "/home/inseong/LLM_RAG_PROJ/data/processed/예금금리_입출금자유예금_20250213.json"

try:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        print("✅ JSON 파일이 정상적으로 로드됨!")
except json.JSONDecodeError as e:
    print(f"❌ JSON 파일이 깨졌거나 유효하지 않음: {e}")
except Exception as e:
    print(f"❌ 파일을 열 수 없음: {e}")