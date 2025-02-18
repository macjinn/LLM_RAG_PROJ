import os
import json
import pandas as pd

def process_data(raw_csv_path: str, processed_json_path: str) -> None:
    """
    CSV 파일을 읽어 각 행을 내러티브 형식의 텍스트 문서로 변환한 후,
    JSON 파일에 저장하는 함수입니다.
    
    Args:
        raw_csv_path (str): 원본 CSV 파일 경로.
        processed_json_path (str): 전처리된 결과를 저장할 JSON 파일 경로.
    """
    try:
        # CSV 파일을 읽습니다. (인코딩 문제 발생 시 encoding 파라미터 조정)
        df = pd.read_csv(raw_csv_path, encoding="utf-8")
    except Exception as e:
        print(f"CSV 파일 읽기 오류: {e}")
        return

    documents = []
    
    # 각 행을 순회하며 내러티브 텍스트를 생성합니다.
    for idx, row in df.iterrows():
        try:
            narrative = (
                f"은행: {row['은행']}\n"
                f"상품명: {row['상품명']}\n"
                f"기본금리: {row['기본금리( %)']}\n"
                f"최고금리(우대금리포함): {row['최고금리(우대금리포함,  %)']}\n"
                f"이자지급방식: {row['이자지급방식']}\n"
                f"은행 최종제공일: {row['은행 최종제공일']}\n"
                f"가입방법: {row['가입방법']}\n"
                f"우대조건: {row['우대조건']}\n"
                f"가입 제한조건: {row['가입 제한조건']}\n"
                f"가입대상: {row['가입대상']}\n"
                f"기타 유의사항: {row['기타 유의사항']}\n"
                f"최고한도: {row['최고한도']}"
            )
        except KeyError as ke:
            print(f"필수 컬럼이 누락되었습니다: {ke}")
            continue

        document = {
            "id": idx,
            "narrative": narrative,
            # 검색 후 원본 정보와 연계를 위해 메타데이터(전체 행)를 함께 저장합니다.
            "metadata": row.to_dict()
        }
        documents.append(document)

    # 저장 경로의 디렉토리가 없으면 생성합니다.
    os.makedirs(os.path.dirname(processed_json_path), exist_ok=True)
    
    # JSON 파일로 저장 (ensure_ascii=False를 사용해 한글이 깨지지 않도록 함)
    with open(processed_json_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"{len(documents)}개의 문서를 {processed_json_path}에 저장했습니다.")

#C:\Users\insung\LLM_RAG_PROJ\data\raw\예금금리_입출금자유예금_20250213.csv
if __name__ == "__main__":
    raw_csv = os.path.join("C:\\Users\\insung\\LLM_RAG_PROJ\\data\\raw\\예금금리_입출금자유예금_20250213.csv")
    processed_json = os.path.join("C:\\Users\\insung\\LLM_RAG_PROJ\\data\\processed")
    process_data(raw_csv, processed_json)
