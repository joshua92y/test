import requests
from dotenv import load_dotenv
import os

load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
NAVER_SEARCH_CLIENT_ID = os.getenv("NAVER_SEARCH_CLIENT_ID")
NAVER_SEARCH_CLIENT_SECRET = os.getenv("NAVER_SEARCH_CLIENT_SECRET")

print(f"🔑 NAVER_CLIENT_ID: {NAVER_CLIENT_ID}")
print(f"🔑 NAVER_CLIENT_SECRET: {NAVER_CLIENT_SECRET}")
print(f"🔑 NAVER_SEARCH_CLIENT_ID: {NAVER_SEARCH_CLIENT_ID}")
print(f"🔑 NAVER_SEARCH_CLIENT_SECRET: {NAVER_SEARCH_CLIENT_SECRET}")

def test_search_api():
    """네이버 검색 API 테스트"""
    print("\n🔍 네이버 검색 API 테스트 시작...")
    
    url = "https://openapi.naver.com/v1/search/local.json"
    params = {
        "query": "메가커피",
        "display": 5
    }
    headers = {
        "X-Naver-Client-Id": NAVER_SEARCH_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_SEARCH_CLIENT_SECRET,
    }
    
    print(f"URL: {url}")
    print(f"Headers: {headers}")
    print(f"Params: {params}")
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        print(f"📡 응답 코드: {response.status_code}")
        print(f"📡 응답 내용: {response.text[:500]}...")
        
        if response.status_code == 200:
            print("✅ 검색 API 성공!")
        else:
            print("❌ 검색 API 실패!")
            
    except Exception as e:
        print(f"❌ 검색 API 오류: {e}")

def test_geocoding_api():
    """네이버 지오코딩 API 테스트"""
    print("\n🗺️ 네이버 지오코딩 API 테스트 시작...")
    
    url = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
    params = {"query": "서울특별시 강남구 테헤란로 123"}
    headers = {
        "x-ncp-apigw-api-key-id": NAVER_CLIENT_ID,
        "x-ncp-apigw-api-key": NAVER_CLIENT_SECRET,
        "Accept": "application/json"
    }
    
    print(f"URL: {url}")
    print(f"Headers: {headers}")
    print(f"Params: {params}")
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        print(f"📡 응답 코드: {response.status_code}")
        print(f"📡 응답 내용: {response.text[:500]}...")
        
        if response.status_code == 200:
            print("✅ 지오코딩 API 성공!")
        else:
            print("❌ 지오코딩 API 실패!")
            
    except Exception as e:
        print(f"❌ 지오코딩 API 오류: {e}")

def test_backend_api():
    """백엔드 API 테스트"""
    print("\n🚀 백엔드 API 테스트 시작...")
    
    # 검색 API 테스트
    try:
        response = requests.post(
            "http://localhost:5000/api/search",
            json={"query": "메가커피", "display": 5},
            timeout=10
        )
        print(f"📡 검색 API 응답 코드: {response.status_code}")
        print(f"📡 검색 API 응답 내용: {response.text[:300]}...")
        
        if response.status_code == 200:
            print("✅ 백엔드 검색 API 성공!")
        else:
            print("❌ 백엔드 검색 API 실패!")
            
    except Exception as e:
        print(f"❌ 백엔드 검색 API 오류: {e}")
    
    # 지오코딩 API 테스트
    try:
        response = requests.post(
            "http://localhost:5000/api/geocode",
            json={"address": "서울특별시 강남구 테헤란로 123"},
            timeout=10
        )
        print(f"📡 지오코딩 API 응답 코드: {response.status_code}")
        print(f"📡 지오코딩 API 응답 내용: {response.text[:300]}...")
        
        if response.status_code == 200:
            print("✅ 백엔드 지오코딩 API 성공!")
        else:
            print("❌ 백엔드 지오코딩 API 실패!")
            
    except Exception as e:
        print(f"❌ 백엔드 지오코딩 API 오류: {e}")

if __name__ == "__main__":
    test_search_api()
    test_geocoding_api()
    test_backend_api()
