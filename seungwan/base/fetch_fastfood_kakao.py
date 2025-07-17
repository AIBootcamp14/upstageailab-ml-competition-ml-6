import requests
import pandas as pd
from time import sleep

# 카카오 REST API 키 입력 (본인 키로 교체)
KAKAO_API_KEY = '8695c5e4e2887c01f13f00dcd2d9035d'
headers = {
    'Authorization': f'KakaoAK {KAKAO_API_KEY}'
}

def search_kakao_places(query, page=1):
    url = 'https://dapi.kakao.com/v2/local/search/keyword.json'
    params = {
        'query': query,
        'page': page,
        'size': 10000
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f'Error: {response.status_code}')
        return None

def fetch_brand_in_seoul(brand):
    results = []
    for page in range(1, 46): 
        data = search_kakao_places(f'서울 {brand}', page=page)
        if data and 'documents' in data:
            results.extend(data['documents'])
            if not data['meta'].get('is_end', True):
                sleep(0.2) 
            else:
                break
        else:
            break
    return results

if __name__ == "__main__":
    brands = ['버거킹', '맥도날드', '맘스터치', '롯데리아', '케이에프씨']
    all_results = []
    for brand in brands:
        print(f'Fetching {brand}...')
        stores = fetch_brand_in_seoul(brand)
        for store in stores:
            store['brand'] = brand
        all_results.extend(stores)
        sleep(1)  # 브랜드별로 잠시 대기
    df = pd.DataFrame(all_results)
    if not df.empty:
        df = df[['brand', 'place_name', 'address_name', 'road_address_name', 'phone', 'x', 'y']]
        print(df.head())
        df.to_csv('seoul_fastfood_kakao.csv', index=False)
        print('CSV 파일로 저장 완료: seoul_fastfood_kakao.csv')
    else:
        print('검색 결과가 없습니다.') 