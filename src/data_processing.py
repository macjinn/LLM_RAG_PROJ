import os
import json
import pandas as pd
import numpy as np

def process_data(raw_csv_path: str, processed_json_dir: str) -> None:
    """
    입출금자유예금 파일을 읽어 각 행을 내러티브 형식의 텍스트 문서로 변환한 후,
    JSON 파일에 저장하는 함수입니다.
    
    Args:
        raw_csv_path (str): 원본 CSV 파일 경로.
        processed_json_dir (str): 전처리된 결과를 저장할 폴더 경로.
    """
    try:
        df = pd.read_csv(raw_csv_path, encoding="utf-8")
    except Exception as e:
        print(f"CSV 파일 읽기 오류: {e}")
        return

    # 불필요한 "Unnamed:"로 시작하는 컬럼 제거
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # 모든 NaN 값을 "정보 없음"으로 변환
    df.replace({np.nan: "정보 없음"}, inplace=True)

    documents = []
    
    # 각 행을 순회하며 내러티브 텍스트를 생성
    for idx, row in df.iterrows():
        try:
            narrative = (
                f"은행: {row['은행'] if row['은행'] is not None else '정보 없음'}\n"
                f"상품명: {row['상품명'] if row['상품명'] is not None else '정보 없음'}\n"
                f"기본금리: {row['기본금리( %)'] if row['기본금리( %)'] is not None else '정보 없음'}\n"
                f"최고금리(우대금리포함): {row['최고금리(우대금리포함,  %)'] if row['최고금리(우대금리포함,  %)'] is not None else '정보 없음'}\n"
                f"이자지급방식: {row['이자지급방식'] if row['이자지급방식'] is not None else '정보 없음'}\n"
                f"은행 최종제공일: {row['은행 최종제공일'] if row['은행 최종제공일'] is not None else '정보 없음'}\n"
                f"가입방법: {row['가입방법'] if row['가입방법'] is not None else '정보 없음'}\n"
                f"우대조건: {row['우대조건'] if row['우대조건'] is not None else '정보 없음'}\n"
                f"가입 제한조건: {row['가입 제한조건'] if row['가입 제한조건'] is not None else '정보 없음'}\n"
                f"가입대상: {row['가입대상'] if row['가입대상'] is not None else '정보 없음'}\n"
                f"기타 유의사항: {row['기타 유의사항'] if row['기타 유의사항'] is not None else '정보 없음'}\n"
                f"최고한도: {row['최고한도'] if row['최고한도'] is not None else '정보 없음'}"
            )
        except KeyError as ke:
            print(f"필수 컬럼이 누락되었습니다: {ke}")
            continue

        document = {
            "id": idx,
            "narrative": narrative,
            "metadata": row.to_dict()
        }
        documents.append(document)

    json_filename = os.path.splitext(os.path.basename(raw_csv_path))[0] + ".json"
    processed_json_path = os.path.join(processed_json_dir, json_filename)
    os.makedirs(processed_json_dir, exist_ok=True)

    # JSON 파일로 저장 (ensure_ascii=False를 사용해 한글이 깨지지 않도록 함)
    with open(processed_json_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"{len(documents)}개의 문서를 {processed_json_path}에 저장했습니다.")

if __name__ == "__main__":
    raw_csv = "/home/inseong/LLM_RAG_PROJ/data/raw/예금금리_입출금자유예금_20250213.csv"
    processed_json_dir = "/home/inseong/LLM_RAG_PROJ/data/processed"
    process_data(raw_csv, processed_json_dir)
