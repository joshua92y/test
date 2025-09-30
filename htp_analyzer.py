#!/usr/bin/env python3
"""
HTP(House-Tree-Person) 전문 분석 모듈
interpretation 폴더의 데이터를 기반으로 한 정확한 심리 분석
"""

import json
import os
from typing import Dict, List, Any

class HTPAnalyzer:
    def __init__(self):
        self.htp_criteria = self.load_htp_criteria()
        self.interpretation_guide = self.load_interpretation_guide()
    
    def load_htp_criteria(self):
        """HTP 분석 기준 데이터 로드"""
        try:
            with open('interpretation/htp_criteria_full.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"HTP 기준 데이터 로드 오류: {e}")
            return []
    
    def load_interpretation_guide(self):
        """해석 가이드 로드"""
        try:
            with open('interpretation/img_int.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"해석 가이드 로드 오류: {e}")
            return {}
    
    def analyze_house_drawing(self, detections: List[Dict], image_size: tuple = (400, 680)):
        """집 그림 전문 분석"""
        analysis = {
            "detected_elements": detections,
            "house_elements": {},
            "psychological_analysis": {},
            "recommendations": []
        }
        
        # 탐지된 요소들 분석
        elements = [d["class"] for d in detections]
        element_counts = {}
        for element in elements:
            element_counts[element] = element_counts.get(element, 0) + 1
        
        # 집 구조 분석
        analysis["house_elements"] = self.analyze_house_structure(detections, element_counts)
        
        # 심리 분석
        analysis["psychological_analysis"] = self.analyze_psychological_indicators(detections, element_counts, image_size)
        
        # 추천사항
        analysis["recommendations"] = self.generate_recommendations(detections, element_counts)
        
        return analysis
    
    def analyze_house_structure(self, detections: List[Dict], element_counts: Dict):
        """집 구조 분석"""
        structure_analysis = {}
        
        # 기본 요소 확인
        has_house = any(d["class"] == "집" for d in detections)
        has_roof = any(d["class"] == "지붕" for d in detections)
        has_door = any(d["class"] == "문" for d in detections)
        has_windows = any(d["class"] == "창문" for d in detections)
        has_chimney = any(d["class"] == "굴뚝" for d in detections)
        has_smoke = any(d["class"] == "연기" for d in detections)
        has_fence = any(d["class"] == "울타리" for d in detections)
        has_sun = any(d["class"] == "태양" for d in detections)
        has_trees = any(d["class"] == "나무" for d in detections)
        has_flowers = any(d["class"] == "꽃" for d in detections)
        
        # 문 분석 (H22, H21)
        if not has_door:
            structure_analysis["door"] = "문이 생략되었습니다. 관계 회피, 고립, 위축의 신호일 수 있습니다."
        else:
            structure_analysis["door"] = "문이 그려져 있습니다. 사회적 접촉에 대한 의지가 있습니다."
        
        # 창문 분석 (H23, H24, H25)
        window_count = element_counts.get("창문", 0)
        if window_count == 0:
            structure_analysis["windows"] = "창문이 생략되었습니다. 폐쇄적 사고, 환경에 대한 관심 결여와 적의를 나타낼 수 있습니다."
        elif window_count >= 3:
            structure_analysis["windows"] = f"창문이 {window_count}개로 많습니다. 불안의 보상심리, 개방과 환경적 접촉에 대한 갈망을 나타낼 수 있습니다."
        else:
            structure_analysis["windows"] = f"창문이 {window_count}개 그려져 있습니다. 적당한 사회적 개방성을 보입니다."
        
        # 지붕 분석
        if has_roof:
            structure_analysis["roof"] = "지붕이 그려져 있습니다. 보호 욕구와 안정감을 나타냅니다."
        else:
            structure_analysis["roof"] = "지붕이 없습니다. 보호 욕구가 충족되지 않거나 현실 지향적일 수 있습니다."
        
        # 굴뚝과 연기 분석 (H27)
        if has_chimney and has_smoke:
            structure_analysis["chimney"] = "굴뚝에 연기가 있습니다. 마음속 긴장, 가정 내 갈등, 정서 혼란을 나타낼 수 있습니다."
        elif has_chimney:
            structure_analysis["chimney"] = "굴뚝이 그려져 있습니다. 가정의 따뜻함과 안정감을 나타냅니다."
        else:
            structure_analysis["chimney"] = "굴뚝이 없습니다. 기본적인 집 구조입니다."
        
        # 울타리 분석 (H31)
        if has_fence:
            structure_analysis["fence"] = "울타리가 그려져 있습니다. 자기보호, 방어벽을 나타냅니다."
        else:
            structure_analysis["fence"] = "울타리가 없습니다. 개방적이고 유연한 경계 설정을 선호합니다."
        
        return structure_analysis
    
    def analyze_psychological_indicators(self, detections: List[Dict], element_counts: Dict, image_size: tuple):
        """심리 지표 분석"""
        psychological_analysis = {}
        
        # 기본 요소 확인
        has_house = any(d["class"] == "집" for d in detections)
        has_door = any(d["class"] == "문" for d in detections)
        has_windows = any(d["class"] == "창문" for d in detections)
        has_sun = any(d["class"] == "태양" for d in detections)
        has_trees = any(d["class"] == "나무" for d in detections)
        has_flowers = any(d["class"] == "꽃" for d in detections)
        has_grass = any(d["class"] == "잔디" for d in detections)
        has_clouds = any(d["class"] == "구름" for d in detections)
        has_fence = any(d["class"] == "울타리" for d in detections)
        has_path = any(d["class"] == "길" for d in detections)
        has_pool = any(d["class"] == "연못" for d in detections)
        has_mountain = any(d["class"] == "산" for d in detections)
        
        # 사회적 개방성 분석
        if has_door and has_windows:
            psychological_analysis["social_openness"] = "문과 창문이 모두 있습니다. 사회적 상호작용에 적극적이고 개방적인 성향을 보입니다."
        elif has_door and not has_windows:
            psychological_analysis["social_openness"] = "문은 있지만 창문이 없습니다. 사회적 접촉은 원하지만 내면을 보호하려는 경향이 있습니다."
        elif has_windows and not has_door:
            psychological_analysis["social_openness"] = "창문은 있지만 문이 없습니다. 관찰은 하지만 직접적인 사회적 접촉을 꺼릴 수 있습니다."
        else:
            psychological_analysis["social_openness"] = "문과 창문이 모두 없습니다. 사회적 고립감이나 회피 경향을 보일 수 있습니다."
        
        # 감정 상태 분석
        positive_elements = sum([has_sun, has_flowers, has_trees, has_grass])
        negative_elements = sum([has_clouds])
        
        if positive_elements >= 3 and negative_elements == 0:
            psychological_analysis["emotional_state"] = "밝고 긍정적인 감정 상태를 보입니다. 활력과 낙관적인 태도를 가지고 있습니다."
        elif positive_elements >= 2 and negative_elements <= 1:
            psychological_analysis["emotional_state"] = "전반적으로 안정적이고 균형 잡힌 감정 상태입니다."
        elif positive_elements >= 1:
            psychological_analysis["emotional_state"] = "보통의 감정 상태를 보이지만, 더 밝은 요소들을 추가해볼 수 있습니다."
        else:
            psychological_analysis["emotional_state"] = "우울하거나 답답한 감정 상태일 수 있습니다. 전문가와의 상담을 고려해보세요."
        
        # 경계 설정 분석
        if has_fence:
            psychological_analysis["boundary_setting"] = "울타리가 그려져 있습니다. 명확한 경계 설정을 원하고 개인 공간을 중시하는 경향이 있습니다."
        else:
            psychological_analysis["boundary_setting"] = "울타리가 없습니다. 개방적이고 유연한 경계 설정을 선호하는 경향이 있습니다."
        
        # 사회적 연결성 분석
        connection_elements = sum([has_path, has_pool, has_mountain])
        if connection_elements >= 2:
            psychological_analysis["social_connection"] = "다양한 연결 요소들이 있습니다. 사회적 관계와 외부 세계와의 연결을 중시합니다."
        elif connection_elements == 1:
            psychological_analysis["social_connection"] = "일부 연결 요소가 있습니다. 적당한 사회적 연결을 원합니다."
        else:
            psychological_analysis["social_connection"] = "연결 요소가 적습니다. 내향적이거나 고립을 선호할 수 있습니다."
        
        # 인지 복잡성 분석
        total_elements = len(detections)
        unique_elements = len(set([d["class"] for d in detections]))
        
        if total_elements >= 8 and unique_elements >= 6:
            psychological_analysis["cognitive_complexity"] = "그림이 매우 상세하고 복잡합니다. 높은 인지 능력과 상상력을 보여줍니다."
        elif total_elements >= 5 and unique_elements >= 4:
            psychological_analysis["cognitive_complexity"] = "적당한 수준의 상세함을 보입니다. 균형 잡힌 인지 능력을 가지고 있습니다."
        elif total_elements >= 3:
            psychological_analysis["cognitive_complexity"] = "기본적인 요소들을 포함하고 있습니다. 더 자세한 표현을 시도해볼 수 있습니다."
        else:
            psychological_analysis["cognitive_complexity"] = "그림이 단순합니다. 스트레스나 집중력 부족의 신호일 수 있습니다."
        
        # 창의성 분석
        creative_elements = ["꽃", "나무", "태양", "구름", "새", "나비", "달", "별"]
        creative_count = sum(1 for d in detections if d["class"] in creative_elements)
        
        if creative_count >= 4:
            psychological_analysis["creativity"] = "매우 창의적이고 풍부한 상상력을 가지고 있습니다. 예술적 감각이 뛰어납니다."
        elif creative_count >= 2:
            psychological_analysis["creativity"] = "적당한 창의성을 보여줍니다. 상상력이 풍부한 편입니다."
        elif creative_count == 1:
            psychological_analysis["creativity"] = "기본적인 창의성을 보입니다. 더 다양한 요소를 추가해보세요."
        else:
            psychological_analysis["creativity"] = "창의적 요소가 부족합니다. 상상력을 발휘해 다양한 요소를 그려보세요."
        
        return psychological_analysis
    
    def generate_recommendations(self, detections: List[Dict], element_counts: Dict):
        """추천사항 생성"""
        recommendations = []
        
        # 기본 요소 확인
        has_house = any(d["class"] == "집" for d in detections)
        has_roof = any(d["class"] == "지붕" for d in detections)
        has_door = any(d["class"] == "문" for d in detections)
        has_windows = any(d["class"] == "창문" for d in detections)
        has_sun = any(d["class"] == "태양" for d in detections)
        has_trees = any(d["class"] == "나무" for d in detections)
        has_flowers = any(d["class"] == "꽃" for d in detections)
        
        # 구조적 추천
        if not has_house:
            recommendations.append("집의 기본 구조를 더 명확하게 그려보세요.")
        if not has_roof:
            recommendations.append("지붕을 추가하여 집을 완성해보세요.")
        if not has_door:
            recommendations.append("문을 그려서 집에 들어갈 수 있는 입구를 만들어보세요.")
        if not has_windows:
            recommendations.append("창문을 추가하여 집이 더 생동감 있게 보이도록 해보세요.")
        
        # 창의적 요소 추천
        creative_elements = sum(1 for d in detections if d["class"] in ["꽃", "나무", "태양", "구름", "새", "나비", "달", "별"])
        if creative_elements < 2:
            recommendations.append("주변 환경(나무, 꽃, 태양 등)을 추가해보세요.")
        
        # 사회적 요소 추천
        if not has_door and not has_windows:
            recommendations.append("문이나 창문을 추가하여 사회적 연결을 표현해보세요.")
        
        if not recommendations:
            recommendations.append("훌륭한 그림입니다! 더 자세한 요소들을 추가해보세요.")
        
        return recommendations
    
    def get_htp_criteria_by_code(self, code: str):
        """코드로 HTP 기준 조회"""
        for criteria in self.htp_criteria:
            if criteria.get("코드") == code:
                return criteria
        return None
    
    def get_interpretation_by_element(self, element: str):
        """요소별 해석 조회"""
        # HTP 기준에서 해당 요소와 관련된 해석 찾기
        related_criteria = []
        for criteria in self.htp_criteria:
            if element in criteria.get("그림표상", ""):
                related_criteria.append(criteria)
        return related_criteria
