#!/usr/bin/env python3
"""
모델 파일 다운로드 스크립트
YOLOv5 HTP 모델의 사전 훈련된 가중치 파일들을 다운로드합니다.
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
import hashlib

# 모델 다운로드 URL (실제 URL로 교체 필요)
MODEL_URLS = {
    "models.zip": "https://github.com/Bobgyu/mindcanvas_ver1/releases/download/v1.0/models.zip"
}

# 모델 파일 체크섬 (다운로드 검증용)
MODEL_CHECKSUMS = {
    "models.zip": "your_checksum_here"  # 실제 체크섬으로 교체 필요
}

def download_file(url, filepath, expected_checksum=None):
    """파일을 다운로드하고 체크섬을 검증합니다."""
    print(f"다운로드 중: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r진행률: {percent:.1f}%", end='', flush=True)
        
        print(f"\n다운로드 완료: {filepath}")
        
        # 체크섬 검증
        if expected_checksum:
            print("체크섬 검증 중...")
            with open(filepath, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            if file_hash == expected_checksum:
                print("✅ 체크섬 검증 성공")
                return True
            else:
                print(f"❌ 체크섬 검증 실패")
                print(f"예상: {expected_checksum}")
                print(f"실제: {file_hash}")
                return False
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 다운로드 실패: {e}")
        return False

def extract_models(zip_path, extract_to):
    """모델 파일들을 압축 해제합니다."""
    print(f"압축 해제 중: {zip_path}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ 압축 해제 완료")
        return True
    except zipfile.BadZipFile as e:
        print(f"❌ 압축 해제 실패: {e}")
        return False

def verify_models():
    """다운로드된 모델 파일들을 검증합니다."""
    model_path = Path("01modelcode/yolov5-htp-docker/pretrained-weights")
    
    required_files = [
        "House/exp/weights/best.pt",
        "House/exp/weights/last.pt",
        "PersonF/exp/weights/best.pt", 
        "PersonF/exp/weights/last.pt",
        "PersonM/exp/weights/best.pt",
        "PersonM/exp/weights/last.pt",
        "Tree/exp/weights/best.pt",
        "Tree/exp/weights/last.pt"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = model_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 누락된 모델 파일들:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    else:
        print("✅ 모든 모델 파일이 존재합니다")
        return True

def main():
    """메인 함수"""
    print("=" * 60)
    print("YOLOv5 HTP 모델 다운로드 스크립트")
    print("=" * 60)
    
    # 현재 디렉토리가 back인지 확인
    if not os.path.exists("app.py"):
        print("❌ 오류: back 디렉토리에서 실행해주세요")
        print("사용법: cd back && python download_models.py")
        sys.exit(1)
    
    # 모델 디렉토리 생성
    model_dir = Path("01modelcode/yolov5-htp-docker/pretrained-weights")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미 모델이 있는지 확인
    if verify_models():
        print("✅ 모델 파일들이 이미 존재합니다")
        response = input("다시 다운로드하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("다운로드를 건너뜁니다")
            return
    
    # 모델 다운로드
    zip_path = "models.zip"
    
    print("\n📥 모델 파일 다운로드 시작...")
    print("⚠️  주의: 모델 파일은 약 1GB 크기입니다")
    
    # 실제로는 GitHub Releases나 다른 서버에서 다운로드
    # 여기서는 예시 URL을 사용
    success = download_file(
        MODEL_URLS["models.zip"], 
        zip_path,
        MODEL_CHECKSUMS.get("models.zip")
    )
    
    if not success:
        print("❌ 모델 다운로드에 실패했습니다")
        print("\n대안 방법:")
        print("1. 수동으로 모델 파일들을 다운로드하여 다음 경로에 배치:")
        print("   back/01modelcode/yolov5-htp-docker/pretrained-weights/")
        print("2. 또는 관리자에게 문의하세요")
        sys.exit(1)
    
    # 압축 해제
    print("\n📦 모델 파일 압축 해제...")
    if extract_models(zip_path, model_dir):
        # 임시 파일 삭제
        os.remove(zip_path)
        print("✅ 임시 파일 삭제 완료")
        
        # 최종 검증
        if verify_models():
            print("\n🎉 모델 다운로드가 완료되었습니다!")
            print("이제 웹 애플리케이션을 실행할 수 있습니다:")
            print("  python app.py")
        else:
            print("❌ 모델 파일 검증에 실패했습니다")
            sys.exit(1)
    else:
        print("❌ 모델 압축 해제에 실패했습니다")
        sys.exit(1)

if __name__ == "__main__":
    main()
