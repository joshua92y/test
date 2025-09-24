#!/usr/bin/env python3
"""
백엔드 서버 테스트 스크립트
"""

import requests
import json
import base64
from PIL import Image
import io

def create_test_image():
    """테스트용 이미지 생성"""
    # 간단한 집 그림을 그린 이미지 생성
    img = Image.new('RGB', (400, 400), color='white')
    
    # 간단한 집 모양 그리기 (실제로는 더 복잡한 그림이어야 함)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # 집 기본 구조
    draw.rectangle([150, 200, 250, 350], outline='black', width=3)  # 벽
    draw.polygon([(120, 200), (200, 120), (280, 200)], outline='black', width=3)  # 지붕
    draw.rectangle([180, 250, 220, 350], outline='black', width=3)  # 문
    draw.rectangle([160, 220, 190, 250], outline='black', width=2)  # 창문1
    draw.rectangle([210, 220, 240, 250], outline='black', width=2)  # 창문2
    
    # 이미지를 Base64로 변환
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def test_health():
    """서버 상태 확인"""
    try:
        response = requests.get('http://localhost:5000/api/health')
        if response.status_code == 200:
            print("✅ 서버 상태: 정상")
            print(f"응답: {response.json()}")
            return True
        else:
            print(f"❌ 서버 상태: 오류 (상태 코드: {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
        return False

def test_analyze():
    """이미지 분석 테스트"""
    try:
        # 테스트 이미지 생성
        test_image = create_test_image()
        
        # 분석 요청
        data = {
            "image": test_image
        }
        
        response = requests.post(
            'http://localhost:5000/api/analyze',
            headers={'Content-Type': 'application/json'},
            data=json.dumps(data)
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 이미지 분석 성공")
            print(f"분석 결과: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ 이미지 분석 실패 (상태 코드: {response.status_code})")
            print(f"오류: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 분석 테스트 오류: {e}")
        return False

def test_models():
    """모델 목록 확인"""
    try:
        response = requests.get('http://localhost:5000/api/models')
        if response.status_code == 200:
            result = response.json()
            print("✅ 모델 목록 조회 성공")
            print(f"사용 가능한 모델: {len(result['models'])}개")
            for model in result['models']:
                print(f"  - {model['name']}: {model['description']}")
            return True
        else:
            print(f"❌ 모델 목록 조회 실패 (상태 코드: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ 모델 목록 테스트 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("MindCanvas Backend 서버 테스트")
    print("=" * 60)
    
    # 1. 서버 상태 확인
    print("\n1. 서버 상태 확인...")
    if not test_health():
        print("서버가 실행되지 않았습니다. 'python app.py'로 서버를 시작해주세요.")
        return
    
    # 2. 모델 목록 확인
    print("\n2. 모델 목록 확인...")
    test_models()
    
    # 3. 이미지 분석 테스트
    print("\n3. 이미지 분석 테스트...")
    test_analyze()
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
