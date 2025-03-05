import os
import json
import re
import pandas as pd
import numpy as np

def clean_text(text: str) -> str:
    """
    텍스트에서 불필요한 이모지, HTML 태그, 여분의 공백을 제거합니다.
    """
    text = str(text)
    text = text.replace("▷", "").replace("<br />", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_data(raw_csv_path: str, processed_json_dir: str) -> None:
    """
    장병내일준비적금금리_금리비교 (멀티 헤더 구조)을 읽어 각 행을 내러티브 형식의 텍스트 문서로 변환하고 JSON 파일로 저장합니다.
    
    Args:
        raw_csv_path (str): 원본 CSV 파일 경로.
        processed_json_dir (str): 전처리된 결과 JSON 파일을 저장할 디렉토리.
    """
    try:
        # 첫 두 행을 헤더로 읽기
        df = pd.read_csv(raw_csv_path, header=[0, 1], encoding="utf-8")
    except Exception as e:
        print(f"CSV 파일 읽기 오류: {e}")
        return

    # 멀티 인덱스 컬럼을 하나의 문자열로 결합 (상단 헤더가 비어있는 경우 전행 값을 채움)
    new_columns = []
    last_top = ""
    for top, sub in df.columns:
        if pd.isna(top) or top.strip() == "":
            top = last_top
        else:
            last_top = top
        # 서브 헤더가 있을 경우 결합
        if pd.notna(sub) and sub.strip() != "":
            combined = f"{top.strip()} {sub.strip()}"
        else:
            combined = top.strip()
        new_columns.append(combined)
    df.columns = new_columns

    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.replace({np.nan: "정보 없음"}, inplace=True)

    documents = []
    for idx, row in df.iterrows():
        # 각 컬럼에 대해 정제한 값을 내러티브 텍스트로 결합
        narrative_parts = []
        for col in df.columns:
            value = clean_text(row[col])
            narrative_parts.append(f"{col}: {value}")
        narrative = "\n".join(narrative_parts)

        # 메타데이터도 동일하게 정제된 값으로 구성
        metadata = {col: clean_text(row[col]) for col in df.columns}

        document = {
            "id": idx,
            "narrative": narrative,
            "metadata": metadata
        }
        documents.append(document)

    json_filename = os.path.splitext(os.path.basename(raw_csv_path))[0] + ".json"
    processed_json_path = os.path.join(processed_json_dir, json_filename)
    os.makedirs(processed_json_dir, exist_ok=True)

    # JSON 파일로 저장 (ensure_ascii=False로 한글이 깨지지 않도록)
    with open(processed_json_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"{len(documents)}개의 문서를 {processed_json_path}에 저장했습니다.")

if __name__ == "__main__":
    raw_csv = "C:\\Users\\insung\\LLM_RAG_PROJ\\data\\raw\\장병내일준비적금금리_금리비교_20250213.csv"
    processed_json_dir = "C:\\Users\\insung\\LLM_RAG_PROJ\\data\\processed"
    process_data(raw_csv, processed_json_dir)
