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
from dotenv import load_dotenv
import openai
import httpx

# 환경변수 로드
load_dotenv()

# OpenAI API 설정
openai.api_key = os.getenv('OPENAI_API_KEY')

# 네이버 API 키 설정
NAVER_CLIENT_ID = os.getenv('NAVER_CLIENT_ID')
NAVER_CLIENT_SECRET = os.getenv('NAVER_CLIENT_SECRET')
NAVER_SEARCH_CLIENT_ID = os.getenv('NAVER_SEARCH_CLIENT_ID')
NAVER_SEARCH_CLIENT_SECRET = os.getenv('NAVER_SEARCH_CLIENT_SECRET')

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
        # PyTorch 2.8.0+에서 모델 로딩 문제 해결
        try:
            # YOLOv5 모델 클래스를 안전한 글로벌로 등록
            torch.serialization.add_safe_globals([yolov5.models.yolo.Model])
            print("✅ PyTorch 안전 글로벌 설정 완료")
        except Exception as e:
            print(f"PyTorch 안전 글로벌 설정 경고: {e}")
       
        # torch.load를 래핑하여 weights_only=False 설정
        original_torch_load = torch.load
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch.load = patched_torch_load
        
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

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    """OpenAI 챗봇 API"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "메시지가 필요합니다."
            }), 400
        
        user_message = data['message']
        
        if not openai.api_key:
            return jsonify({
                "error": "OpenAI API 키가 설정되지 않았습니다."
            }), 500
        
        # OpenAI API 호출
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 심리상담 전문가입니다. 사용자의 심리 상태를 분석하고 도움을 주는 역할을 합니다."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        bot_response = response.choices[0].message.content
        
        return jsonify({
            "success": True,
            "response": bot_response,
            "message": "챗봇 응답이 완료되었습니다."
        })
        
    except Exception as e:
        print(f"챗봇 API 오류: {e}")
        return jsonify({
            "error": f"서버 오류: {str(e)}"
        }), 500

@app.route('/api/search', methods=['POST'])
def search_places():
    """네이버 검색 API 프록시"""
    try:
        data = request.get_json()
        query = data.get("query", "")
        display = data.get("display", 10)
        
        print(f"🔍 검색 요청 받음: {query}")
        
        if not query:
            return jsonify({"error": "검색어가 필요합니다"}), 400
        
        if not NAVER_SEARCH_CLIENT_ID or not NAVER_SEARCH_CLIENT_SECRET:
            return jsonify({"error": "네이버 검색 API 키가 설정되지 않았습니다"}), 500
        
        with httpx.Client() as client:
            response = client.get(
                "https://openapi.naver.com/v1/search/local.json",
                params={
                    "query": query,
                    "display": display,
                    "start": 1,
                    "sort": "random"
                },
                headers={
                    "X-Naver-Client-Id": NAVER_SEARCH_CLIENT_ID,
                    "X-Naver-Client-Secret": NAVER_SEARCH_CLIENT_SECRET,
                },
                timeout=10.0
            )
            
            if response.status_code != 200:
                return jsonify({
                    "error": f"네이버 검색 API 오류: {response.text}"
                }), response.status_code
            
            data = response.json()
            
            # 검색 결과 파싱
            if data.get("items"):
                results = []
                for item in data["items"]:
                    results.append({
                        "title": item.get("title", "").replace("<b>", "").replace("</b>", ""),
                        "address": item.get("address", ""),
                        "roadAddress": item.get("roadAddress", ""),
                        "category": item.get("category", ""),
                        "description": item.get("description", "").replace("<b>", "").replace("</b>", ""),
                        "link": item.get("link", ""),
                        "telephone": item.get("telephone", "")
                    })
                
                return jsonify({
                    "success": True,
                    "data": results,
                    "total": data.get("total", 0),
                    "source": "naver_api"
                })
            else:
                return jsonify({
                    "success": True,
                    "data": [],
                    "total": 0,
                    "source": "naver_api"
                })
                
    except httpx.TimeoutException:
        return jsonify({"error": "API 요청 시간 초과"}), 408
    except httpx.RequestError as e:
        return jsonify({"error": f"API 요청 오류: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

@app.route('/api/geocode', methods=['POST'])
def geocode():
    """네이버 지오코딩 API 프록시"""
    try:
        data = request.get_json()
        address = data.get("address", "")
        
        print(f"🗺️ 지오코딩 요청 받음: {address}")
        
        if not address:
            return jsonify({"error": "주소가 필요합니다"}), 400
        
        if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
            return jsonify({"error": "네이버 지오코딩 API 키가 설정되지 않았습니다"}), 500
        
        with httpx.Client() as client:
            response = client.get(
                "https://maps.apigw.ntruss.com/map-geocode/v2/geocode",
                params={
                    "query": address,
                    "output": "json"
                },
                headers={
                    "x-ncp-apigw-api-key-id": NAVER_CLIENT_ID,
                    "x-ncp-apigw-api-key": NAVER_CLIENT_SECRET,
                    "Accept": "application/json"
                },
                timeout=10.0
            )
            
            if response.status_code != 200:
                return jsonify({
                    "error": f"네이버 지오코딩 API 오류: {response.text}"
                }), response.status_code
            
            data = response.json()
            
            if data.get("addresses") and len(data["addresses"]) > 0:
                address_info = data["addresses"][0]
                return jsonify({
                    "success": True,
                    "data": {
                        "lat": float(address_info.get("y", 0)),
                        "lng": float(address_info.get("x", 0)),
                        "address": address_info.get("roadAddress", ""),
                        "jibunAddress": address_info.get("jibunAddress", "")
                    },
                    "source": "naver_api"
                })
            else:
                return jsonify({"error": "주소를 찾을 수 없습니다"}), 404
                
    except httpx.TimeoutException:
        return jsonify({"error": "API 요청 시간 초과"}), 408
    except httpx.RequestError as e:
        return jsonify({"error": f"API 요청 오류: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

@app.route('/api/reverse-geocode', methods=['POST'])
def reverse_geocode():
    """네이버 역지오코딩 API 프록시"""
    try:
        data = request.get_json()
        lat = data.get("lat")
        lng = data.get("lng")
        
        print(f"🗺️ 역지오코딩 요청 받음: {lat}, {lng}")
        
        if not lat or not lng:
            return jsonify({"error": "위도와 경도가 필요합니다"}), 400
        
        if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
            return jsonify({"error": "네이버 지오코딩 API 키가 설정되지 않았습니다"}), 500
        
        with httpx.Client() as client:
            response = client.get(
                "https://maps.apigw.ntruss.com/map-reversegeocode/v2/gc",
                params={
                    "coords": f"{lng},{lat}",
                    "output": "json"
                },
                headers={
                    "x-ncp-apigw-api-key-id": NAVER_CLIENT_ID,
                    "x-ncp-apigw-api-key": NAVER_CLIENT_SECRET,
                    "Accept": "application/json"
                },
                timeout=10.0
            )
            
            if response.status_code != 200:
                return jsonify({
                    "error": f"네이버 역지오코딩 API 오류: {response.text}"
                }), response.status_code
            
            data = response.json()
            
            if data.get("results") and len(data["results"]) > 0:
                result = data["results"][0]
                region = result.get("region", {})
                land = result.get("land", {})
                
                address_parts = []
                if region.get("area1", {}).get("name"):
                    address_parts.append(region["area1"]["name"])
                if region.get("area2", {}).get("name"):
                    address_parts.append(region["area2"]["name"])
                if region.get("area3", {}).get("name"):
                    address_parts.append(region["area3"]["name"])
                
                full_address = " ".join(address_parts)
                
                return jsonify({
                    "success": True,
                    "data": {
                        "address": full_address,
                        "area1": region.get("area1", {}).get("name", ""),
                        "area2": region.get("area2", {}).get("name", ""),
                        "area3": region.get("area3", {}).get("name", ""),
                        "roadAddress": land.get("name", ""),
                        "jibunAddress": land.get("number1", "")
                    },
                    "source": "naver_api"
                })
            else:
                return jsonify({"error": "주소를 찾을 수 없습니다"}), 404
                
    except httpx.TimeoutException:
        return jsonify({"error": "API 요청 시간 초과"}), 408
    except httpx.RequestError as e:
        return jsonify({"error": f"API 요청 오류: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("MindCanvas Backend 서버를 시작합니다...")
    print("=" * 60)
    print(f"로드된 YOLOv5 모델: {list(yolo_analyzer.models.keys())}")
    print(f"로드된 HTP 분석 기준: {len(htp_analyzer.htp_criteria)}개")
    print("서버 주소: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)