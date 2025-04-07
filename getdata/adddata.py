import ssl
import requests
import xml.etree.ElementTree as ET
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

# 수동으로 TLS 1.2 컨텍스트 생성
class TLS12Adapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2  # 명시적으로 TLS 1.2 이상
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

url = "http://apis.data.go.kr/1160100/service/GetMedicalReimbursementInsuranceInfoService/getInsuranceInfo"
params = {
    "serviceKey": "pouv9PVQTMTrZSt0KQplGtJlJnHTu9xoMA4E/5g78JpWxC2JfPiBf87rNaAxu6RkT6wCb0+MHwUcEdtKW+biNg==",
    "pageNo": 1,
    "numOfRows": 10
}

session = requests.Session()
session.mount("http://", TLS12Adapter())

try:
    response = session.get(url, params=params, timeout=10)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    print("✅ 연결 성공")

except requests.exceptions.SSLError as e:
    print("❌ SSL 오류 발생:", e)
except requests.exceptions.RequestException as e:
    print("❌ 요청 실패:", e)


root = ET.fromstring(response.content)

# header 확인
result_code = root.find('.//resultCode')
if result_code is not None and result_code.text != '00':
    print("❌ API 응답 오류 - 코드:", result_code.text)
else:
    print("✅ 정상 응답")
    

    # 아이템 데이터 추출
    items = root.findall('.//item')
    if not items:
        print("⚠️ 아이템이 없습니다.")
    else:
        # 예시: 첫 번째 item 내부 구조 확인
        items = root.findall(".//item")
        print(items[1])

        for i, item in enumerate(items, 1):
            name =  item.findtext('prdNm') #금융상품명
            institute =  item.findtext('cmpyNm') #기관명

            print(f"{i}. 상품명: {name} / 회사명: {institute}")