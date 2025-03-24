# data_processing_military.py
import os
import json
import re
import pandas as pd
import numpy as np

def clean_text(text: str) -> str:
    """
    텍스트에서 불필요한 특수문자, HTML 태그, 개행문자를 제거합니다.
    """
    text = str(text)
    text = text.replace("▷", "").replace("<br />", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_column_names(columns):
    """
    불필요한 'Unnamed' 라벨을 제거하고, 올바른 컬럼명을 생성합니다.
    """
    cleaned_columns = []
    for col in columns:
        col = re.sub(r"Unnamed: \d+_level_\d+", "", col).strip()
        cleaned_columns.append(col)
    return cleaned_columns

def generate_key_summary(row: dict, keys: list) -> str:
    """
    row 데이터(dict)에서 지정한 키에 해당하는 값들을 "키: 값" 형식으로 결합합니다.
    값이 없거나 "정보 없음"이면 제외합니다.
    """
    summary_items = []
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value and value != "정보 없음":
            summary_items.append(f"{key}: {value}")
    return ", ".join(summary_items) if summary_items else "정보 없음"

def process_data(raw_csv_path: str, processed_json_dir: str) -> None:
    """
    장병내일준비적금금리_금리비교 CSV 파일을 읽어 각 행을 내러티브 문서로 변환한 후,
    하이브리드 서치 알고리즘에 최적화된 JSON 구조로 저장합니다.
    
    최종 JSON 구조 예시:
    {
      "documents": [
        {
          "id": "military_product_000",
          "bank": "NH농협은행",
          "product_name": "NH농협은행",
          "type": "장병내일준비적금",
          "content": "은행: NH농협은행\n제공 금리(%) 1개월 이상 ~ 3개월 미만: 2.70\n3개월 이상 ~ 6개월 미만: 2.80\n... (전체 내용)",
          "key_summary": "은행: NH농협은행, 제공 금리(%) 1개월 이상 ~ 3개월 미만: 2.70, 1년 6개월 이상 ~ 만기: 5.00, 우대금리 조건: (내용의 앞부분)",
          "metadata": { ... }   // 모든 컬럼 정보 + key_summary
        },
        ...
      ]
    }
    """
    try:
        # 첫 두 행을 헤더로 읽기
        df = pd.read_csv(raw_csv_path, header=[0, 1], encoding="utf-8")
    except Exception as e:
        print(f"CSV 파일 읽기 오류: {e}")
        return

    # 멀티 인덱스 컬럼을 하나의 문자열로 결합
    last_top = ""
    new_columns = []
    for top, sub in df.columns:
        if pd.isna(top) or top.strip() == "":
            top = last_top
        else:
            last_top = top
        if pd.notna(sub) and sub.strip() != "":
            combined = f"{top.strip()} {sub.strip()}"
        else:
            combined = top.strip()
        new_columns.append(combined)
    new_columns = clean_column_names(new_columns)
    df.columns = new_columns

    # 불필요한 'Unnamed' 컬럼 제거 및 결측치 처리
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.replace({np.nan: "정보 없음"}, inplace=True)

    documents = []
    # 도메인 전문가가 선정한 핵심 필드 리스트 (key_summary에 포함할 정보)
    key_fields = [
        "은행",
        "제공 금리(%) 1개월 이상 ~ 3개월 미만",
        "1년 6개월 이상 ~ 만기",
        "우대금리 조건"
    ]
    
    for idx, row in df.iterrows():
        # 각 행의 값을 clean_text를 이용해 정제
        cleaned_row = {col: clean_text(row[col]) for col in df.columns}
        
        bank = cleaned_row.get("은행", "정보 없음")
        # 상품명이 없으므로 product_name은 은행명으로 사용
        product_name = bank
        product_type = "장병내일준비적금"
        
        # 내러티브: 각 컬럼의 "컬럼명: 값" 형태로 연결
        content = "\n".join([f"{col}: {cleaned_row[col]}" for col in df.columns])
        
        # 핵심 필드 기반 key_summary 생성
        key_summary = generate_key_summary(cleaned_row, key_fields)
        
        # metadata: 전체 정제된 row 데이터에 key_summary 추가
        metadata = cleaned_row.copy()
        metadata["key_summary"] = key_summary
        
        document = {
            "id": f"military_product_{idx:03d}",
            "bank": bank,
            "product_name": product_name,
            "type": product_type,
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
    raw_csv = "/home/inseong/LLM_RAG_PROJ/data/raw/장병내일준비적금금리_금리비교_20250213.csv"
    processed_json_dir = "/home/inseong/LLM_RAG_PROJ/data/processed"
    process_data(raw_csv, processed_json_dir)
