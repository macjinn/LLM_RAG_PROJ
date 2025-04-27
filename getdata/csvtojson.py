import pandas as pd
import json
from pathlib import Path

# CSV 데이터를 수동 입력 대신 예제에 따라 생성
# 실제로는 pd.read_csv()로 불러와야 함

sample_data = pd.read_csv("C:/Users/sj/Documents/CAPD/LLM_RAG_PROJ/금융위원회_서민금융상품기본정보.csv")

sample_data_1 = [
    {
        "finPrdNm": "새희망홀씨Ⅱ",
        "ofrInstNm": "14개 취급은행",
        "lnLmt": "3500만원",
        "irtCtg": "변동금리",
        "irt": "은행별 상이",
        "maxTotLnTrm": "은행별 상이",
        "maxDfrmTrm": "-",
        "maxRdptTrm": "-",
        "rdptMthd": "원(리)금균등분할상환",
        "usge": "생계",
        "trgt": "근로자",
        "jnMthd": "14개 취급은행 신청방법 문의",
        "prftAddIrtCond": "성실상환자, 사회적 취약계층 ...",
        "etcRefSbjc": "-",
        "hdlInst": "취급 은행(14개)",
        "cnpl": "취급은행 콜센터, 서민금융콜센터 (국번없이)1397",
        "rltSite": "취급은행 홈페이지",
        "prdExisYn": "Y"
    }
]

print(type(sample_data_1))
print(type(sample_data))

json_docs = []
for idx, row in sample_data.iterrows():
    id_str = f"loan_{idx:05d}"
    product_name = row["finPrdNm"]
    bank = row["ofrInstNm"]
    summary = f"기관: {bank}, 상품명: {product_name}, 금리: {row['irt']}, 대출한도: {row['lnLmt']}, 기간: {row['maxTotLnTrm']}"

    doc = {
        "id": id_str,
        "bank": bank,
        "product_name": product_name,
        "type": "대출상품",
        "content": (
            f"기관명: {bank}\n상품명: {product_name}\n대출한도: {row['lnLmt']}\n"
            f"금리: {row['irtCtg']} ({row['irt']})\n대출기간: {row['maxTotLnTrm']} "
            f"(거치 {row['maxDfrmTrm']} / 상환 {row['maxRdptTrm']})\n상환방법: {row['rdptMthd']}\n"
            f"용도: {row['usge']}\n대상: {row['trgt']}\n가입방법: {row['jnMthd']}\n"
            f"우대/가산 조건: {row['prftAddIrtCond']}\n기타사항: {row['etcRefSbjc']}"
        ),
        "key_summary": {
            "요약": summary
        },
        "metadata": {
            "기관명": bank,
            "상품명": product_name,
            "대출한도": row["lnLmt"],
            "금리구분": row["irtCtg"],
            "금리": row["irt"],
            "총대출기간": row["maxTotLnTrm"],
            "거치기간": row["maxDfrmTrm"],
            "상환기간": row["maxRdptTrm"],
            "상환방법": row["rdptMthd"],
            "용도": row["usge"],
            "대상": row["trgt"],
            "가입방법": row["jnMthd"],
            "우대조건": row["prftAddIrtCond"],
            "기타참고사항": row["etcRefSbjc"],
            "취급기관": row["hdlInst"],
            "연락처": row["cnpl"],
            "관련사이트": row["rltSite"],
            "상품존재여부": row["prdExisYn"],
            "key_summary": summary
        }
    }

    json_docs.append(doc)

# 저장
output_path = Path("C:/Users/sj/Documents/CAPD/LLM_RAG_PROJ/data/processed/converted_loans.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(json_docs, f, ensure_ascii=False, indent=2)

output_path.name
