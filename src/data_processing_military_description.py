import os
import json
import re
import pandas as pd

def clean_text(text: str) -> str:
    """
    텍스트에서 불필요한 특수문자, HTML 태그, 개행문자 등을 제거하고 정리합니다.
    """
    text = str(text)
    text = text.replace("▷", "").replace("<br />", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_data(raw_csv_path: str, processed_json_dir: str) -> None:
    """
    장병내일준비적금_상품설명 CSV 파일을 읽어, 각 카테고리별로 그룹화한 후 내러티브와 메타데이터를 JSON 파일로 저장합니다.
    
    CSV 파일은 아래의 컬럼을 포함합니다:
      - 카테고리
      - 세부 항목
      - 내용

    각 JSON 문서는 하나의 카테고리에 해당하는 모든 정보를 포함하며,
    RAG 시스템의 검색 및 백터 서치에 최적화된 형태로 구성됩니다.
    
    Args:
        raw_csv_path (str): 원본 CSV 파일 경로.
        processed_json_dir (str): 전처리된 결과 JSON 파일을 저장할 디렉토리.
    """
    try:
        df = pd.read_csv(raw_csv_path, encoding="utf-8")
    except Exception as e:
        print(f"CSV 파일 읽기 오류: {e}")
        return
    

    df.fillna("정보 없음", inplace=True)

    df["카테고리"] = df["카테고리"].apply(clean_text)
    df["세부 항목"] = df["세부 항목"].apply(clean_text)
    df["내용"] = df["내용"].apply(clean_text)
    
    documents = []
    
    # '카테고리'별로 그룹화
    grouped = df.groupby("카테고리")
    for idx, (category, group) in enumerate(grouped):
        metadata_details = {}
        narrative_lines = [f"카테고리: {category}"]
        
        # 그룹 내 각 행에 대해 세부 항목과 내용을 내러티브 및 메타데이터에 추가
        for _, row in group.iterrows():
            sub = row["세부 항목"]
            content = row["내용"]
            # 동일 세부 항목이 여러 번 등장할 경우 연결
            if sub in metadata_details:
                metadata_details[sub] += f" / {content}"
            else:
                metadata_details[sub] = content
            narrative_lines.append(f"{sub}: {content}")
        
        narrative = "\n".join(narrative_lines)
        
        document = {
            "id": idx,
            "narrative": narrative,
            "metadata": {
                "카테고리": category,
                "세부 항목": metadata_details
            }
        }
        documents.append(document)
    
    json_filename = os.path.splitext(os.path.basename(raw_csv_path))[0] + ".json"
    processed_json_path = os.path.join(processed_json_dir, json_filename)
    os.makedirs(processed_json_dir, exist_ok=True)
    
    with open(processed_json_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"{len(documents)}개의 문서를 {processed_json_path}에 저장했습니다.")


if __name__ == "__main__":
    raw_csv = "C:\\Users\\insung\\LLM_RAG_PROJ\\data\\raw\\장병내일준비적금_상품설명.csv"
    processed_json_dir = "C:\\Users\\insung\\LLM_RAG_PROJ\\data\\processed"
    process_data(raw_csv, processed_json_dir)
