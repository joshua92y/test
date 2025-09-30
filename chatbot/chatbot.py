import os
import json
import requests
from typing import List, Dict, Any

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv가 설치되지 않았습니다. pip install python-dotenv")

# API KEYS
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load interpretation rules
def load_interpretation_rules():
    """이미지 분석 해석기준 JSON 파일을 로드합니다."""
    try:
        with open('interpretation/img_int.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("해석기준 파일을 찾을 수 없습니다.")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return None

# Load interpretation rules
interpretation_rules = load_interpretation_rules()

def call_openai_api(messages: List[Dict[str, str]]) -> str:
    """OpenAI API를 직접 호출합니다."""
    if not OPENAI_API_KEY:
        return "OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요."
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"API 오류: {response.status_code} - {response.text}"
    
    except Exception as e:
        return f"API 호출 중 오류가 발생했습니다: {str(e)}"

def analyze_image_features(image_analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """이미지 분석 결과를 HTP 해석기준에 따라 분석합니다."""
    if not interpretation_rules:
        return {"error": "해석기준을 로드할 수 없습니다."}
    
    analysis_result = {
        "objects": {},
        "total_score": 0,
        "interpretations": [],
        "risk_level": "normal"
    }
    
    htp_criteria = interpretation_rules.get("htp_criteria_detailed", {})
    
    # 각 객체별 분석 (집, 나무, 사람)
    for object_type in ["house", "tree", "person"]:
        if object_type not in image_analysis_result:
            continue
            
        object_features = image_analysis_result[object_type]
        object_criteria = htp_criteria.get(object_type, {})
        
        object_analysis = {
            "label": "집" if object_type == "house" else "나무" if object_type == "tree" else "사람",
            "features": {},
            "score": 0,
            "interpretations": []
        }
        
        # 각 특징별 분석
        for feature_name, feature_value in object_features.items():
            # 특징에 따른 해석 생성
            interpretation = generate_interpretation(object_type, feature_name, feature_value, "")
            if interpretation:
                object_analysis["interpretations"].append(interpretation)
                object_analysis["score"] += interpretation.get("score", 0)
                analysis_result["interpretations"].append(interpretation)
        
        analysis_result["objects"][object_type] = object_analysis
        analysis_result["total_score"] += object_analysis["score"]
    
    # 위험도 평가
    if analysis_result["total_score"] <= -5:
        analysis_result["risk_level"] = "high"
    elif analysis_result["total_score"] <= -1:
        analysis_result["risk_level"] = "moderate"
    elif analysis_result["total_score"] >= 4:
        analysis_result["risk_level"] = "positive"
    
    return analysis_result

def generate_interpretation(object_type: str, feature_name: str, feature_value: Any, criteria_text: str) -> Dict[str, Any]:
    """특징값을 기반으로 상세한 해석을 생성합니다."""
    if not interpretation_rules:
        return None
    
    detailed_criteria = interpretation_rules.get("htp_criteria_detailed", {})
    object_criteria = detailed_criteria.get(object_type, {})
    
    # 기본 해석 구조
    interpretation = {
        "feature": feature_name,
        "interpretation": "",
        "severity": "info",
        "score": 0,
        "reasoning": "",
        "threshold": "",
        "psychological_meaning": ""
    }
    
    # 크기 분석
    if feature_name == "size" and isinstance(feature_value, (int, float)):
        size_criteria = object_criteria.get("size", {})
        
        if feature_value >= size_criteria.get("very_large", {}).get("threshold", 0.8):
            criteria = size_criteria["very_large"]
            threshold = size_criteria.get("very_large", {}).get("threshold", 0.8)
            interpretation.update({
                "interpretation": criteria["interpretation"],
                "severity": criteria["severity"],
                "score": criteria["score"],
                "reasoning": f"크기 비율 {feature_value:.2f}이 임계값 {threshold} 이상으로 매우 큼",
                "threshold": f"임계값: {threshold} 이상",
                "psychological_meaning": "HTP 기준에 따르면 화지를 꽉 채우거나 밖으로 벗어날 정도의 큰 크기는 충동적이고 공격적인 성향을 나타냅니다. 이는 자아 통제력 부족이나 과도한 자기 표현 욕구를 의미할 수 있습니다."
            })
        elif feature_value <= size_criteria.get("small", {}).get("threshold", 0.25):
            criteria = size_criteria["small"]
            threshold = size_criteria.get("small", {}).get("threshold", 0.25)
            interpretation.update({
                "interpretation": criteria["interpretation"],
                "severity": criteria["severity"],
                "score": criteria["score"],
                "reasoning": f"크기 비율 {feature_value:.2f}이 임계값 {threshold} 이하로 매우 작음",
                "threshold": f"임계값: {threshold} 이하",
                "psychological_meaning": "HTP 기준에 따르면 1/4 이하의 작은 크기는 대인관계에서의 무력감, 열등감, 불안, 우울적 경향을 나타냅니다. 이는 자신감 부족이나 위축된 자아상을 반영할 수 있습니다."
            })
        else:
            criteria = size_criteria.get("normal", {})
            interpretation.update({
                "interpretation": criteria["interpretation"],
                "severity": criteria["severity"],
                "score": criteria["score"],
                "reasoning": f"크기 비율 {feature_value:.2f}이 정상 범위 내에 있음",
                "threshold": f"정상 범위: 0.25 < 크기 < 0.8",
                "psychological_meaning": "적절한 크기는 균형 잡힌 자아상과 현실적 인식을 나타냅니다."
            })
    
    # 위치 분석
    elif feature_name == "location" and isinstance(feature_value, (int, float)):
        position_criteria = object_criteria.get("position", {})
        
        if feature_value < 0.3:  # 상단
            if "top_view" in position_criteria:
                criteria = position_criteria["top_view"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"위치 비율 {feature_value:.3f}이 임계값 0.3 미만으로 상단에 위치",
                    "threshold": "위치 < 0.3 (상단)",
                    "psychological_meaning": "HTP 기준에 따르면 상단에 위치한 객체는 이상화 성향이나 현실 도피 경향을 나타냅니다. 이는 현실보다 이상적인 세계를 추구하는 심리를 의미할 수 있습니다."
                })
        elif feature_value > 0.7:  # 하단
            if "bottom_half" in position_criteria:
                criteria = position_criteria["bottom_half"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"위치 비율 {feature_value:.3f}이 임계값 0.7 초과로 하단에 위치",
                    "threshold": "위치 > 0.7 (하단)",
                    "psychological_meaning": "HTP 기준에 따르면 하단에 위치한 객체는 불안정감, 우울적 경향을 나타냅니다. 이는 기반 부족이나 불안정한 정서 상태를 의미할 수 있습니다."
                })
        elif feature_value < 0.5:  # 좌측
            if "left" in position_criteria:
                criteria = position_criteria["left"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"위치 비율 {feature_value:.3f}이 임계값 0.5 미만으로 좌측에 위치",
                    "threshold": "위치 < 0.5 (좌측)",
                    "psychological_meaning": "HTP 기준에 따르면 좌측에 위치한 객체는 내향적, 열등감을 나타냅니다. 이는 과거 지향적이거나 소극적인 성향을 의미할 수 있습니다."
                })
        else:  # 우측
            if "right" in position_criteria:
                criteria = position_criteria["right"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"위치 비율 {feature_value:.3f}이 임계값 0.5 이상으로 우측에 위치",
                    "threshold": "위치 >= 0.5 (우측)",
                    "psychological_meaning": "HTP 기준에 따르면 우측에 위치한 객체는 외향성, 활동성을 나타냅니다. 이는 미래 지향적이거나 적극적인 성향을 의미할 수 있습니다."
                })
    
    # 창문 분석
    elif feature_name == "window":
        window_criteria = object_criteria.get("window", {})
        
        if feature_value == 0:
            if "missing" in window_criteria:
                criteria = window_criteria["missing"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"창문 개수 {feature_value}개로 창문이 완전히 없음",
                    "threshold": "창문 0개",
                    "psychological_meaning": "HTP 기준 H23에 따르면 창문이 생략된 집은 폐쇄적 사고와 환경에 대한 관심 결여, 적의를 나타냅니다. 이는 사회적 교류 회피나 외부 세계에 대한 방어적 태도를 의미합니다."
                })
        elif feature_value >= 3:
            if "many" in window_criteria:
                criteria = window_criteria["many"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"창문 개수 {feature_value}개로 3개 이상의 많은 창문",
                    "threshold": "창문 3개 이상",
                    "psychological_meaning": "HTP 기준 H24에 따르면 3개 이상의 많은 창문은 불안의 보상심리와 개방, 환경적 접촉에 대한 갈망을 나타냅니다. 이는 내적 불안을 외적 개방성으로 보상하려는 시도일 수 있습니다."
                })
    
    # 문 분석
    elif feature_name == "door":
        door_criteria = object_criteria.get("door", {})
        
        if feature_value == 0:
            if "missing" in door_criteria:
                criteria = door_criteria["missing"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"문 크기 비율 {feature_value}으로 문이 완전히 없음",
                    "threshold": "문 0개 (완전 생략)",
                    "psychological_meaning": "HTP 기준 H22에 따르면 현관문이 생략된 집은 관계 회피, 고립, 위축을 나타냅니다. 이는 대인관계에서의 회피적 성향이나 사회적 고립을 의미합니다."
                })
        elif feature_value < 0.1:  # 매우 작은 문
            if "very_small" in door_criteria:
                criteria = door_criteria["very_small"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"문 크기 비율 {feature_value:.3f}이 임계값 0.1 미만으로 매우 작음",
                    "threshold": "문 크기 < 0.1",
                    "psychological_meaning": "HTP 기준 H19에 따르면 현관문이 집에 비해 과도하게 작은 경우 수줍음, 까다로움, 사회성 결핍, 현실도피를 나타냅니다. 이는 대인관계에서의 소극적 성향을 의미합니다."
                })
    
    # 굴뚝/연기 분석
    elif feature_name == "chimney":
        chimney_criteria = object_criteria.get("chimney", {})
        
        if feature_value == 1 or feature_value is True:
            if "with_smoke" in chimney_criteria:
                criteria = chimney_criteria["with_smoke"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"굴뚝 존재 여부 {feature_value}으로 굴뚝이 그려져 있음",
                    "threshold": "굴뚝 1개 (존재)",
                    "psychological_meaning": "HTP 기준 H27에 따르면 굴뚝의 연기 표현은 마음속 긴장, 가정 내 갈등, 정서 혼란을 나타냅니다. 이는 가정 내 불화나 내적 갈등의 표현일 수 있습니다."
                })
    
    # 나무 기둥 분석
    elif feature_name == "trunk" and isinstance(feature_value, (int, float)):
        trunk_criteria = object_criteria.get("trunk", {})
        
        if feature_value < 0.1:
            if "thin" in trunk_criteria:
                criteria = trunk_criteria["thin"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"나무 기둥 두께 비율 {feature_value:.3f}이 임계값 0.1 미만으로 매우 가늘음",
                    "threshold": "기둥 두께 < 0.1",
                    "psychological_meaning": "HTP 기준 T18에 따르면 나무기둥의 두께가 전체 나무 크기에 비해 얇은 경우 우울과 외로움을 나타냅니다. 이는 지지 기반의 약화나 불안정한 자아상을 의미합니다."
                })
    
    # 나무 가지 분석
    elif feature_name == "branches":
        branches_criteria = object_criteria.get("branches", {})
        
        if isinstance(feature_value, int):
            if feature_value >= 5:
                if "many" in branches_criteria:
                    criteria = branches_criteria["many"]
                    interpretation.update({
                        "interpretation": criteria["interpretation"],
                        "severity": criteria["severity"],
                        "score": criteria["score"],
                        "reasoning": f"가지 개수 {feature_value}개로 5개 이상의 많은 가지",
                        "threshold": "가지 5개 이상",
                        "psychological_meaning": "HTP 기준 T23에 따르면 수관에서 나뭇가지의 수가 지나치게 많은 표현은 하고 싶은 일이 많고, 대인관계가 활발하고 의욕이 과함을 나타냅니다. 이는 에너지와 활동성의 과도한 표현일 수 있습니다."
                    })
            elif feature_value <= 4:
                if "few" in branches_criteria:
                    criteria = branches_criteria["few"]
                    interpretation.update({
                        "interpretation": criteria["interpretation"],
                        "severity": criteria["severity"],
                        "score": criteria["score"],
                        "reasoning": f"가지 개수 {feature_value}개로 4개 이하의 적은 가지",
                        "threshold": "가지 4개 이하",
                        "psychological_meaning": "HTP 기준 T24에 따르면 수관에서 나뭇가지의 수가 4개 이하로 표현된 경우 세상과 상호작용에 억제적임, 위축과 우울감을 나타냅니다. 이는 사회적 활동의 제한이나 에너지 부족을 의미합니다."
                    })
    
    # 뿌리 분석
    elif feature_name == "roots":
        roots_criteria = object_criteria.get("roots", {})
        
        if feature_value == 1 or feature_value is True:
            if "underground_emphasized" in roots_criteria:
                criteria = roots_criteria["underground_emphasized"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"뿌리 존재 여부 {feature_value}으로 뿌리가 그려져 있음",
                    "threshold": "뿌리 1개 (존재)",
                    "psychological_meaning": "HTP 기준 T20에 따르면 땅속에 있는 뿌리를 강조하여 표현한 경우 현실적응의 장애, 예민함, 퇴행을 나타냅니다. 이는 안정감에 대한 과도한 욕구나 현실 도피 경향을 의미합니다."
                })
        elif feature_value == 0 or feature_value is False:
            if "exposed_no_ground" in roots_criteria:
                criteria = roots_criteria["exposed_no_ground"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"뿌리 존재 여부 {feature_value}으로 뿌리가 없음",
                    "threshold": "뿌리 0개 (없음)",
                    "psychological_meaning": "HTP 기준 T22에 따르면 지면선 없이 뿌리가 모두 노출된 표현은 유아기부터 지속된 불안, 우울의 표현을 나타냅니다. 이는 기반 부족이나 불안정한 정서 상태를 의미합니다."
                })
    
    # 잎 분석
    elif feature_name == "leaves" and isinstance(feature_value, (int, float)):
        leaves_criteria = object_criteria.get("leaves", {})
        
        if feature_value > 0.5:
            if "overly_detailed" in leaves_criteria:
                criteria = leaves_criteria["overly_detailed"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"잎 비율 {feature_value:.3f}이 임계값 0.5 이상으로 과도하게 상세함",
                    "threshold": "잎 비율 > 0.5",
                    "psychological_meaning": "HTP 기준 T28에 따르면 수관의 잎이 구체적으로 과도하게 크게 표현된 경우 충동적, 정열, 희망적, 자신감(힘의 욕구 강화)을 나타냅니다. 이는 활력과 에너지의 과도한 표현일 수 있습니다."
                })
        elif feature_value < 0.2:
            if "fallen" in leaves_criteria:
                criteria = leaves_criteria["fallen"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"잎 비율 {feature_value:.3f}이 임계값 0.2 미만으로 매우 적음",
                    "threshold": "잎 비율 < 0.2",
                    "psychological_meaning": "HTP 기준 T38에 따르면 떨어지거나 떨어진 잎의 표현은 우울, 외로움, 정서불안을 나타냅니다. 이는 활력 저하나 정서적 위축을 의미합니다."
                })
        elif feature_value == 0:
            if "bare_branches" in leaves_criteria:
                criteria = leaves_criteria["bare_branches"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"잎 비율 {feature_value}으로 잎이 전혀 없음 (겨울나무)",
                    "threshold": "잎 비율 = 0",
                    "psychological_meaning": "HTP 기준 T16에 따르면 마른 가지만 있는 수관의 표현(겨울나무)은 자아 통제력 상실, 외상경험, 무력감, 수동적 성향을 나타냅니다. 이는 심리적 위축이나 에너지 부족을 의미합니다."
                })
    
    # 구멍 분석
    elif feature_name == "hole":
        holes_criteria = object_criteria.get("holes", {})
        
        if feature_value == 1 or feature_value is True:
            if "present" in holes_criteria:
                criteria = holes_criteria["present"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"]
                })
    
    # 사람 얼굴 분석
    elif feature_name == "face":
        face_criteria = object_criteria.get("face", {})
        
        if feature_value == 0 or feature_value is False:
            if "missing_features" in face_criteria:
                criteria = face_criteria["missing_features"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"얼굴 특징 존재 여부 {feature_value}으로 얼굴 특징이 완전히 없음",
                    "threshold": "얼굴 특징 0개 (완전 생략)",
                    "psychological_meaning": "HTP 기준 P17에 따르면 얼굴의 눈, 코, 입이 생략된 경우 회피, 불안, 우울, 성적 갈등을 나타냅니다. 이는 정서표현 회피나 대인관계에서의 긴장을 의미합니다."
                })
    
    # 사람 손 분석
    elif feature_name == "hands":
        hands_criteria = object_criteria.get("hands", {})
        
        if feature_value == 0 or feature_value is False:
            if "missing" in hands_criteria:
                criteria = hands_criteria["missing"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"손 존재 여부 {feature_value}으로 손이 그려지지 않음",
                    "threshold": "손 0개 (생략)",
                    "psychological_meaning": "HTP 기준 P38에 따르면 팔이나 손의 생략은 죄의식, 우울, 무력감, 대인관계 기피, 과도한 업무를 나타냅니다. 이는 행동 통제의 어려움이나 사회적 유능감 저하를 의미합니다."
                })
        elif feature_value == 1 or feature_value is True:
            if "present" in hands_criteria:
                criteria = hands_criteria["present"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"손 존재 여부 {feature_value}으로 손이 그려져 있음",
                    "threshold": "손 1개 이상 (존재)",
                    "psychological_meaning": "손이 그려진 것은 행동 능력과 사회적 유능감을 나타냅니다. 이는 적극적인 행동 의지나 대인관계 능력을 의미할 수 있습니다."
                })
    
    # 사람 발 분석
    elif feature_name == "feet":
        legs_feet_criteria = object_criteria.get("legs_feet", {})
        
        if feature_value == 0 or feature_value is False:
            if "missing" in legs_feet_criteria:
                criteria = legs_feet_criteria["missing"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"발 존재 여부 {feature_value}으로 발이 그려지지 않음",
                    "threshold": "발 0개 (생략)",
                    "psychological_meaning": "HTP 기준 P43에 따르면 발을 표시하지 않은 경우나 절단된 다리 표현은 우울, 의기소침, 불안을 나타냅니다. 이는 현실 기반 부족이나 불안정한 정서 상태를 의미합니다."
                })
        elif feature_value == 1 or feature_value is True:
            if "present" in legs_feet_criteria:
                criteria = legs_feet_criteria["present"]
                interpretation.update({
                    "interpretation": criteria["interpretation"],
                    "severity": criteria["severity"],
                    "score": criteria["score"],
                    "reasoning": f"발 존재 여부 {feature_value}으로 발이 그려져 있음",
                    "threshold": "발 1개 이상 (존재)",
                    "psychological_meaning": "발이 그려진 것은 현실 기반과 안정감을 나타냅니다. 이는 현실적 지향이나 안정된 정서 상태를 의미할 수 있습니다."
                })
    
    return interpretation if interpretation["interpretation"] else None

# System prompt
def get_system_prompt():
    """img_int.json의 내용을 기반으로 시스템 프롬프트를 생성합니다."""
    if not interpretation_rules:
        return "HTP 해석기준을 로드할 수 없습니다."
    
    instructions = interpretation_rules.get("instructions", [])
    htp_criteria = interpretation_rules.get("htp_criteria_detailed", {})
    examples = interpretation_rules.get("examples", [])
    
    prompt = "당신은 HTP(House-Tree-Person) 그림 검사 해석 전문가입니다.\n\n"
    
    # 시스템 지시사항 추가
    for instruction in instructions:
        if instruction.get("role") == "system":
            prompt += instruction.get("content", "") + "\n\n"
    
    # HTP 해석 기준 추가
    prompt += "HTP 해석 기준:\n"
    for object_type, criteria in htp_criteria.items():
        if object_type == "house":
            prompt += "🏠 집 (House):\n"
        elif object_type == "tree":
            prompt += "🌳 나무 (Tree):\n"
        elif object_type == "person":
            prompt += "👤 사람 (Person):\n"
        
        for feature, description in criteria.items():
            prompt += f"- {feature}: {description}\n"
        prompt += "\n"
    
    # 예시 추가
    if examples:
        prompt += "예시 대화:\n"
        for example in examples[:3]:  # 처음 3개 예시만
            prompt += f"사용자: {example.get('user', '')}\n"
            prompt += f"상담사: {example.get('assistant', '')}\n\n"
    
    prompt += """당신의 역할:
1. 이미지 분석 결과를 받으면 HTP 기준에 따라 심리적 해석을 제공
2. 각 특징별로 점수를 계산하고 위험도를 평가
3. 구체적이고 실용적인 상담 조언 제공
4. 미술심리상담과 그림 해석 관련 질문만 답변"""
    
    return prompt

system_prompt = get_system_prompt()

def process_query(query, conversation_history, image_analysis_result=None):
    """간단한 챗봇 쿼리 처리 함수"""
    # 메시지 구성
    messages = [{"role": "system", "content": system_prompt}]
    
    # 기존 대화 기록 추가
    for msg in conversation_history:
        if isinstance(msg, tuple):
            messages.append({"role": "user", "content": msg[0]})
            messages.append({"role": "assistant", "content": msg[1]})
    
    # 이미지 분석 결과 처리
    enhanced_query = query
    if image_analysis_result:
        analysis_result = analyze_image_features(image_analysis_result)
        
        if "error" not in analysis_result:
            analysis_summary = f"""
이미지 분석 결과:

총 점수: {analysis_result['total_score']}
위험도: {analysis_result['risk_level']}

객체별 분석:
"""
            
            for obj_id, obj_data in analysis_result['objects'].items():
                analysis_summary += f"\n{obj_data['label']} (점수: {obj_data['score']}):\n"
                for interpretation in obj_data['interpretations']:
                    analysis_summary += f"- {interpretation['feature']}: {interpretation['interpretation']} (심각도: {interpretation['severity']})\n"
            
            enhanced_query = f"{query}\n\n{analysis_summary}"
        else:
            enhanced_query = f"{query}\n\n이미지 분석 중 오류가 발생했습니다: {analysis_result['error']}"
    
    # 현재 질문 추가
    messages.append({"role": "user", "content": enhanced_query})
    
    try:
        # OpenAI API 호출
        answer = call_openai_api(messages)
        
        # 대화 기록에 추가
        conversation_history.append((query, answer))
        return answer
    except Exception as e:
        print(f"Query processing error: {str(e)}")
        return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다." 

def process_image_analysis(image_analysis_result: Dict[str, Any]) -> str:
    """이미지 분석 결과를 해석하여 상세한 분석 보고서를 생성합니다."""
    analysis_result = analyze_image_features(image_analysis_result)
    
    if "error" in analysis_result:
        return f"이미지 분석 중 오류가 발생했습니다: {analysis_result['error']}"
    
    # 위험도별 라벨 매핑
    risk_labels = {
        'high': '🔴 위험 신호 다수',
        'moderate': '⚠️ 위험 단서 일부', 
        'normal': '📝 일반적 수준',
        'positive': '✅ 적응적 단서 우세'
    }
    
    report = f"""
=== HTP 그림 해석 분석 보고서 ===

📊 전체 평가
- 총 점수: {analysis_result['total_score']}
- 위험도: {risk_labels.get(analysis_result['risk_level'], '알 수 없음')}

📋 객체별 상세 분석
"""
    
    for obj_id, obj_data in analysis_result['objects'].items():
        if obj_id == 'house':
            emoji = '🏠'
        elif obj_id == 'tree':
            emoji = '🌳'
        elif obj_id == 'person':
            emoji = '👤'
        else:
            emoji = '📝'
            
        report += f"\n{emoji} {obj_data['label']} (점수: {obj_data['score']})\n"
        report += "=" * 30 + "\n"
        
        if obj_data['interpretations']:
            for interpretation in obj_data['interpretations']:
                severity_emoji = {
                    'info': 'ℹ️',
                    'low': '⚠️',
                    'moderate': '🔶',
                    'high': '🔴'
                }.get(interpretation['severity'], '📝')
                
                report += f"{severity_emoji} {interpretation['feature']}: {interpretation['interpretation']}\n"
                
                # 상세한 근거 설명 추가
                if 'reasoning' in interpretation:
                    report += f"   📊 근거: {interpretation['reasoning']}\n"
                if 'threshold' in interpretation:
                    report += f"   📏 기준: {interpretation['threshold']}\n"
                if 'psychological_meaning' in interpretation:
                    report += f"   🧠 심리적 의미: {interpretation['psychological_meaning']}\n"
                report += "\n"
        else:
            report += "특별한 특징이 감지되지 않았습니다.\n"
    
    # 위험도별 권장사항
    if analysis_result['risk_level'] == 'high':
        report += "\n🚨 권장사항: 전문가 상담이 필요합니다."
    elif analysis_result['risk_level'] == 'moderate':
        report += "\n⚠️ 권장사항: 추가 관찰이 필요합니다."
    elif analysis_result['risk_level'] == 'positive':
        report += "\n✅ 긍정적인 특징들이 관찰됩니다."
    else:
        report += "\n📝 일반적인 수준의 특징들이 관찰됩니다."
    
    return report
