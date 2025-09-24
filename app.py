#!/usr/bin/env python3
"""
MindCanvas Backend - YOLOv5 HTP 이미지 분석 API
Flask를 사용한 웹 인터페이스
"""

import os
import sys
import json
import base64
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import yolov5
import yolov5.models # Add this line to import models
from htp_analyzer import HTPAnalyzer

app = Flask(__name__)
CORS(app)  # CORS 활성화

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 최대 파일 크기
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# 업로드 및 출력 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

class YOLOv5HTPAnalyzer:
    def __init__(self):
        self.device = 'cpu'  # 웹에서는 CPU 사용
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """모든 YOLOv5 HTP 모델 로드"""
        # PyTorch 2.6+에서 모델 로딩 문제 해결
        torch.serialization.add_safe_globals([yolov5.models.yolo.Model])
        model_configs = {
            "House": {
                "weights": "01modelcode/yolov5-htp-docker/pretrained-weights/House/exp/weights/best.pt",
                "classes": ["집", "지붕", "문", "창문", "굴뚝", "연기", "울타리", "길", "연못", "산", "나무", "꽃", "잔디", "태양"]
            },
            "PersonF": {
                "weights": "01modelcode/yolov5-htp-docker/pretrained-weights/PersonF/exp/weights/best.pt",
                "classes": ["머리", "얼굴", "눈", "코", "입", "귀", "머리카락", "목", "상체", "팔", "손", "다리", "발", "단추", "주머니", "운동화", "여자구두"]
            },
            "PersonM": {
                "weights": "01modelcode/yolov5-htp-docker/pretrained-weights/PersonM/exp/weights/best.pt",
                "classes": ["머리", "얼굴", "눈", "코", "입", "귀", "머리카락", "목", "상체", "팔", "손", "다리", "발", "단추", "주머니", "운동화", "남자구두"]
            },
            "Tree": {
                "weights": "01modelcode/yolov5-htp-docker/pretrained-weights/Tree/exp/weights/best.pt",
                "classes": ["나무", "기둥", "수관", "가지", "뿌리", "나뭇잎", "꽃", "열매", "그네", "새", "다람쥐", "구름", "달", "별"]
            }
        }
        
        for model_name, config in model_configs.items():
            try:
                if os.path.exists(config["weights"]):
                    model = yolov5.load(config["weights"])
                    model.conf = 0.25  # 기본 신뢰도 임계값
                    model.iou = 0.45   # 기본 IoU 임계값
                    self.models[model_name] = {
                        "model": model,
                        "classes": config["classes"]
                    }
                    print(f"✅ {model_name} 모델 로드 완료")
                else:
                    print(f"❌ {model_name} 모델 파일을 찾을 수 없습니다: {config['weights']}")
            except Exception as e:
                print(f"❌ {model_name} 모델 로드 실패: {e}")
    
    def predict(self, image, model_name, conf_threshold=0.25, iou_threshold=0.45):
        """이미지에 대한 객체 탐지 수행"""
        if model_name not in self.models:
            raise ValueError(f"모델을 찾을 수 없습니다: {model_name}")
        
        model_info = self.models[model_name]
        model = model_info["model"]
        classes = model_info["classes"]
        
        # 모델 설정 업데이트
        model.conf = conf_threshold
        model.iou = iou_threshold
        
        # 예측 수행
        results = model(image)
        
        # 결과 파싱
        detections = []
        if len(results.pred[0]) > 0:
            for *box, conf, cls in results.pred[0]:
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls)
                confidence = float(conf)
                
                if class_id < len(classes):
                    detections.append({
                        "class": classes[class_id],
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2]
                    })
        
        return detections
    
    
    

# 전역 분석기 인스턴스
yolo_analyzer = YOLOv5HTPAnalyzer()
htp_analyzer = HTPAnalyzer()

def allowed_file(filename):
    """파일 확장자 검증"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def base64_to_image(base64_string):
    """Base64 문자열을 이미지로 변환"""
    try:
        # data:image/png;base64, 부분 제거
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Base64 디코딩
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # RGB로 변환 (RGBA인 경우)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        print(f"Base64 이미지 변환 오류: {e}")
        return None

@app.route('/api/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    loaded_models = list(yolo_analyzer.models.keys())
    htp_criteria_count = len(htp_analyzer.htp_criteria)
    return jsonify({
        "status": "healthy",
        "message": "MindCanvas Backend is running",
        "loaded_models": loaded_models,
        "total_models": len(loaded_models),
        "htp_criteria_count": htp_criteria_count
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """이미지 분석 API"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "error": "이미지 데이터가 필요합니다."
            }), 400
        
        # Base64 이미지 데이터 추출
        image_data = data['image']
        
        # Base64를 이미지로 변환
        image = base64_to_image(image_data)
        
        if image is None:
            return jsonify({
                "error": "이미지 변환에 실패했습니다."
            }), 400
        
        # YOLOv5로 객체 탐지
        house_detections = yolo_analyzer.predict(image, "House")
        
        # HTP 전문 분석기로 심리 분석
        analysis_result = htp_analyzer.analyze_house_drawing(house_detections)
        
        if analysis_result is None:
            return jsonify({
                "error": "이미지 분석에 실패했습니다."
            }), 500
        
        return jsonify({
            "success": True,
            "analysis": analysis_result,
            "message": "분석이 완료되었습니다."
        })
        
    except Exception as e:
        print(f"분석 API 오류: {e}")
        return jsonify({
            "error": f"서버 오류: {str(e)}"
        }), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """사용 가능한 모델 목록"""
    models_info = []
    for model_name, model_info in yolo_analyzer.models.items():
        models_info.append({
            "id": model_name.lower(),
            "name": f"{model_name} 분석",
            "description": f"{model_name} 그림을 분석하여 심리 상태를 파악합니다.",
            "status": "available",
            "classes": model_info["classes"]
        })
    
    return jsonify({
        "models": models_info,
        "htp_criteria_loaded": len(htp_analyzer.htp_criteria) > 0
    })

@app.route('/api/predict/<model_name>', methods=['POST'])
def predict_with_model(model_name):
    """특정 모델로 예측"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "error": "이미지 데이터가 필요합니다."
            }), 400
        
        # Base64 이미지 데이터 추출
        image_data = data['image']
        
        # Base64를 이미지로 변환
        image = base64_to_image(image_data)
        
        if image is None:
            return jsonify({
                "error": "이미지 변환에 실패했습니다."
            }), 400
        
        # 모델로 예측
        detections = yolo_analyzer.predict(image, model_name)
        
        return jsonify({
            "success": True,
            "model": model_name,
            "detections": detections,
            "message": f"{model_name} 모델 분석이 완료되었습니다."
        })
        
    except Exception as e:
        print(f"예측 API 오류: {e}")
        return jsonify({
            "error": f"서버 오류: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("MindCanvas Backend 서버를 시작합니다...")
    print("=" * 60)
    print(f"로드된 YOLOv5 모델: {list(yolo_analyzer.models.keys())}")
    print(f"로드된 HTP 분석 기준: {len(htp_analyzer.htp_criteria)}개")
    print("서버 주소: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)