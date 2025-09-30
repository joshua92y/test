# MindCanvas Backend

YOLOv5 HTP(House-Tree-Person) 이미지 분석을 위한 Flask 백엔드 서버입니다.

## 🚀 빠른 시작

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정
`.env` 파일을 생성하고 필요한 API 키를 설정하세요:
```bash
# .env 파일 생성
OPENAI_API_KEY=your_openai_api_key_here
NAVER_CLIENT_ID=your_naver_client_id_here
NAVER_CLIENT_SECRET=your_naver_client_secret_here
NAVER_SEARCH_CLIENT_ID=your_naver_search_client_id_here
NAVER_SEARCH_CLIENT_SECRET=your_naver_search_client_secret_here
```

### 3. 모델 파일 다운로드 (선택사항)
```bash
python download_models.py
```

### 4. 웹 서버 실행
```bash
python app.py
```

서버가 http://localhost:5000 에서 실행됩니다.

## 📁 구조

```
back/
├── app.py                    # Flask 메인 애플리케이션
├── download_models.py        # 모델 다운로드 스크립트
├── requirements.txt          # Python 의존성
├── README.md                # 이 파일
└── 01modelcode/
    └── yolov5-htp-docker/
        └── pretrained-weights/
            ├── House/        # 집 그림 분석 모델
            ├── PersonF/      # 여성 인물 그림 분석 모델
            ├── PersonM/      # 남성 인물 그림 분석 모델
            └── Tree/         # 나무 그림 분석 모델
```

## 🔧 API 엔드포인트

- `GET /api/health` - 서버 상태 확인
- `GET /api/models` - 사용 가능한 모델 목록
- `POST /api/analyze` - 집 그림 분석 (기본)
- `POST /api/predict/<model_name>` - 특정 모델로 예측

## 📝 API 사용법

### 이미지 분석 (집 그림)
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."}'
```

### 특정 모델로 예측
```bash
curl -X POST http://localhost:5000/api/predict/House \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."}'
```

### 응답 예시
```json
{
  "success": true,
  "analysis": {
    "detected_elements": [
      {
        "class": "집",
        "confidence": 0.85,
        "bbox": [100, 100, 300, 400]
      },
      {
        "class": "지붕",
        "confidence": 0.78,
        "bbox": [80, 80, 320, 120]
      }
    ],
    "house_elements": {
      "walls": "벽이 그려져 있습니다.",
      "roof": "지붕이 그려져 있습니다.",
      "door": "문이 명확하지 않습니다.",
      "windows": "창문이 그려져 있습니다."
    },
    "psychological_analysis": {
      "stability": "안정적인 집 구조로 보입니다. 안전감을 느끼는 것 같습니다.",
      "creativity": "창의적이고 풍부한 상상력을 가지고 있습니다.",
      "emotional_state": "긍정적이고 밝은 감정 상태로 보입니다."
    },
    "recommendations": [
      "문을 그려서 집에 들어갈 수 있는 입구를 만들어보세요.",
      "주변 환경(나무, 꽃, 태양 등)을 추가해보세요."
    ]
  },
  "message": "분석이 완료되었습니다."
}
```

## 🧠 지원하는 분석 모델

1. **House (집)**: 집, 지붕, 문, 창문, 굴뚝, 연기, 울타리, 길, 연못, 산, 나무, 꽃, 잔디, 태양
2. **PersonF (여성 인물)**: 머리, 얼굴, 눈, 코, 입, 귀, 머리카락, 목, 상체, 팔, 손, 다리, 발, 단추, 주머니, 운동화, 여자구두
3. **PersonM (남성 인물)**: 머리, 얼굴, 눈, 코, 입, 귀, 머리카락, 목, 상체, 팔, 손, 다리, 발, 단추, 주머니, 운동화, 남자구두
4. **Tree (나무)**: 나무, 기둥, 수관, 가지, 뿌리, 나뭇잎, 꽃, 열매, 그네, 새, 다람쥐, 구름, 달, 별

## 🧠 전문 HTP 심리 분석 시스템

본 시스템은 **interpretation 폴더의 전문 HTP 분석 기준**을 기반으로 한 정확한 심리 분석을 제공합니다.

### 📊 분석 데이터베이스

- **HTP 분석 기준**: 740개의 전문 분석 코드 (H1~H37, T1~T50, P1~P60)
- **그림 해석 가이드**: 전문 심리상담사용 해석 매뉴얼
- **YOLOv5 AI 모델**: 14개 집 요소 자동 탐지

### 🔍 분석 영역

1. **집 구조 분석**
   - **문 (H22)**: 생략 시 관계 회피, 고립, 위축
   - **창문 (H23, H24, H25)**: 개수와 형태로 사회적 개방성 측정
   - **지붕**: 보호 욕구와 안정감 지표
   - **굴뚝 (H27)**: 연기 유무로 가정 내 갈등 분석

2. **사회적 지표**
   - **울타리 (H31)**: 자기보호, 방어벽 설정
   - **길, 연못, 산**: 사회적 연결성과 외부 세계 관계
   - **문과 창문 조합**: 사회적 상호작용 경향성

3. **감정 상태 지표**
   - **태양 (H28, H29)**: 자신감, 자존감, 애정결핍
   - **자연 요소**: 꽃, 나무, 잔디 = 긍정적 감정
   - **부정적 요소**: 구름, 비 = 우울감

4. **심리적 지표**
   - **그림 복잡성**: 인지 능력과 상상력 수준
   - **창의성**: 예술적 요소의 다양성
   - **경계 설정**: 개인 공간과 사회적 개방성

### 🎯 전문 분석 예시

```json
{
  "house_elements": {
    "door": "문이 생략되었습니다. 관계 회피, 고립, 위축의 신호일 수 있습니다.",
    "windows": "창문이 3개로 많습니다. 불안의 보상심리, 개방과 환경적 접촉에 대한 갈망을 나타낼 수 있습니다.",
    "chimney": "굴뚝에 연기가 있습니다. 마음속 긴장, 가정 내 갈등, 정서 혼란을 나타낼 수 있습니다."
  },
  "psychological_analysis": {
    "social_openness": "창문은 있지만 문이 없습니다. 관찰은 하지만 직접적인 사회적 접촉을 꺼릴 수 있습니다.",
    "emotional_state": "밝고 긍정적인 감정 상태를 보입니다. 활력과 낙관적인 태도를 가지고 있습니다.",
    "boundary_setting": "울타리가 그려져 있습니다. 명확한 경계 설정을 원하고 개인 공간을 중시하는 경향이 있습니다."
  }
}
```

### ⚠️ 주의사항

- 이 분석은 **참고용**이며, 정확한 심리 진단을 위해서는 전문가와의 상담이 필요합니다.
- 그림의 해석은 개인의 문화적 배경, 나이, 상황에 따라 달라질 수 있습니다.
- 단일 그림으로는 개인의 전체적인 심리 상태를 완전히 파악하기 어렵습니다.
- 분석 결과는 HTP 심리 검사의 표준화된 기준을 따릅니다.

## ⚙️ 설정

- **신뢰도 임계값**: 0.25 (기본값)
- **IoU 임계값**: 0.45 (기본값)
- **최대 파일 크기**: 16MB
- **지원 형식**: PNG, JPG, JPEG, GIF, BMP
