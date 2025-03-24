# data_processing_military_description.py
import os
import json
import re
import pandas as pd

def clean_text(text: str) -> str:
    """
    텍스트에서 불필요한 특수문자, HTML 태그, 개행문자 등을 제거합니다.
    """
    text = str(text)
    text = text.replace("▷", "").replace("<br />", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_data(raw_csv_path: str, processed_json_dir: str) -> None:
    """
    장병내일준비적금_상품설명 CSV 파일을 읽어, 각 카테고리별로 그룹화한 후,
    RAG 시스템 및 하이브리드 서치 알고리즘에 최적화된 JSON 문서로 저장합니다.
    
    최종 JSON 구조 예시:
    {
      "documents": [
        {
          "id": "military_desc_000",
          "bank": "정보 없음",
          "product_name": "<카테고리>",
          "type": "장병내일준비적금 상품설명",
          "content": "카테고리: <카테고리>\n<세부 항목1>: <내용1>\n<세부 항목2>: <내용2>\n...",
          "key_summary": "<세부 항목1>: <내용1 (최대50자)>, <세부 항목2>: <내용2 (최대50자)>",
          "metadata": {
            "<세부 항목1>": "<내용1>",
            "<세부 항목2>": "<내용2>",
            ...
          }
        },
        ...
      ]
    }
    """
    try:
        df = pd.read_csv(raw_csv_path, encoding="utf-8")
    except Exception as e:
        print(f"CSV 파일 읽기 오류: {e}")
        return

    # 결측치는 "정보 없음"으로 대체 및 각 열 정제
    df.fillna("정보 없음", inplace=True)
    df["카테고리"] = df["카테고리"].apply(lambda x: clean_text(x))
    df["세부 항목"] = df["세부 항목"].apply(lambda x: clean_text(x))
    df["내용"] = df["내용"].apply(lambda x: clean_text(x))
    
    documents = []
    grouped = df.groupby("카테고리")
    for idx, (category, group) in enumerate(grouped):
        # 내러티브 content 생성: 첫 줄은 카테고리, 이후 각 행의 세부 항목과 내용을 연결
        content_lines = [f"카테고리: {category}"]
        metadata = {}
        for _, row in group.iterrows():
            key = row["세부 항목"]
            value = row["내용"]
            content_lines.append(f"{key}: {value}")
            # 같은 세부 항목이 중복되면 연결
            if key in metadata:
                metadata[key] += " / " + value
            else:
                metadata[key] = value
        content = "\n".join(content_lines)
        
        # key_summary: 그룹의 상위 두 행을 선택하여, 각 행의 세부 항목과 내용(최대50자)을 간결하게 연결
        summary_items = []
        for _, row in group.head(2).iterrows():
            key = row["세부 항목"]
            value = row["내용"]
            short_value = value[:50] + ("..." if len(value) > 50 else "")
            summary_items.append(f"{key}: {short_value}")
        key_summary = ", ".join(summary_items) if summary_items else "정보 없음"
        metadata["key_summary"] = key_summary  # key_summary를 추가

        document = {
            "id": f"military_desc_{idx:03d}",
            "bank": "정보 없음",
            "product_name": category,
            "type": "장병내일준비적금 상품설명",
            "content": content,
            "key_summary": key_summary,
            "metadata": metadata
        }
        documents.append(document)
    
    output = {"documents": documents}
    json_filename = os.path.splitext(os.path.basename(raw_csv_path))[0] + ".json"
    processed_json_path = os.path.join(processed_json_dir, json_filename)
    os.makedirs(processed_json_dir, exist_ok=True)
    with open(processed_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"{len(documents)}개의 문서를 {processed_json_path}에 저장했습니다.")

if __name__ == "__main__":
    raw_csv = "/home/inseong/LLM_RAG_PROJ/data/raw/장병내일준비적금_상품설명.csv"
    processed_json_dir = "/home/inseong/LLM_RAG_PROJ/data/processed"
    process_data(raw_csv, processed_json_dir)
