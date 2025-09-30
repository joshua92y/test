#!/bin/bash
echo "MindCanvas Backend 서버를 시작합니다..."
echo ""
echo "의존성 설치 중..."
pip install -r requirements.txt
echo ""
echo "서버 시작 중..."
python app.py
