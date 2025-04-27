import requests
import xml.etree.ElementTree as ET
import pandas as pd
import urllib.parse

# ✅ 설정값
API_KEY = 'pouv9PVQTMTrZSt0KQplGtJlJnHTu9xoMA4E%2F5g78JpWxC2JfPiBf87rNaAxu6RkT6wCb0%2BMHwUcEdtKW%2BbiNg%3D%3D'  # 디코딩 전 키
BASE_URL = 'http://apis.data.go.kr/B190030/GetCardProductInfoService/getCardProductList'
NUM_OF_ROWS = 1000 # 페이지당 항목 수

# ✅ 서비스 키 인코딩
decoded_key = urllib.parse.unquote(API_KEY)

# ✅ 페이지 1 요청 → 전체 데이터 수 확인
params = {
    'serviceKey': decoded_key,
    'pageNo': 1,
    'numOfRows': NUM_OF_ROWS,
    'sBseDt' : '20210730',
    'eBseDt' : '20250330' 
}
response = requests.get(BASE_URL, params=params)
root = ET.fromstring(response.text)
print(ET.tostring(root, encoding='unicode'))


total_count = int(root.find('.//totalCount').text)
total_pages = (total_count // NUM_OF_ROWS) + (1 if total_count % NUM_OF_ROWS > 0 else 0)
print(f"총 {total_count}건, {total_pages}페이지 분량")

# ✅ 모든 데이터 수집
all_data = []

for page in range(1, total_pages + 1):
    print(f"📦 페이지 {page} 처리 중...")
    params['pageNo'] = page
    response = requests.get(BASE_URL, params=params)
    root = ET.fromstring(response.text)
    items = root.findall('.//item')

    for item in items:
        record = {}
        for child in item:
            record[child.tag] = child.text
        all_data.append(record)

# ✅ DataFrame으로 변환 후 CSV 저장
df = pd.DataFrame(all_data)
csv_file = '한국산업은행_카드상품 정보.csv'
df.to_csv(csv_file, index=False, encoding='utf-8-sig')
print(f"✅ CSV 저장 완료: {csv_file}")
