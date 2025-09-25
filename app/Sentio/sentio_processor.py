# -*- coding: utf-8 -*-
"""
Sentio 텍스트 프로세서
HR 텍스트의 감정 분석, 키워드 추출, 퇴직 위험 신호 탐지
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class SentioTextProcessor:
    """Sentio 텍스트 처리 및 분석 클래스"""
    
    def __init__(self, analyzer=None):
        """텍스트 프로세서 초기화"""
        self.attrition_keywords = self._initialize_attrition_keywords()
        self.stopwords = self._initialize_stopwords()
        self.analyzer = analyzer  # JD-R 기반 분석기
        
    def _initialize_attrition_keywords(self) -> Dict[str, Dict[str, List[str]]]:
        """퇴직원인별 키워드 매핑 초기화"""
        return {
            "불공정한_보상과_평가": {
                "보상": {
                    "주요_키워드": ["급여", "연봉", "월급", "보상", "임금"],
                    "동의어": ["페이", "돈", "수당", "성과급", "인센티브", "보너스", "연봉협상"],
                    "고위험_신호": ["일한 만큼 못 받는다", "연봉이 너무 짜다", "성과급 기준이 불투명하다"]
                },
                "복리후생": {
                    "주요_키워드": ["복지", "복리후생", "혜택"],
                    "동의어": ["베네핏", "지원", "제도"],
                    "고위험_신호": ["복지가 거의 없다", "명절 상여금도 안 나온다", "다른 회사는 다 해주는데"]
                },
                "지각된_공정성": {
                    "주요_키워드": ["공정", "불공평", "차별", "형평성"],
                    "동의어": ["편애", "특혜", "연고", "정치", "불공정", "비교"],
                    "고위험_신호": ["같은 일을 하는데 A만 인정받는다", "팀장님이 특정 팀원만 챙긴다", "성과 평가가 불공평하다"]
                }
            },
            "성장_정체와_동기부여_상실": {
                "성장_및_발전": {
                    "주요_키워드": ["성장", "발전", "경력", "커리어", "교육", "학습"],
                    "동의어": ["승진", "기회", "비전", "배움", "개발"],
                    "고위험_신호": ["여기서는 더 배울 게 없다", "성장할 수 있다는 느낌이 안 든다", "승진 기회가 막혀있다"]
                },
                "업무_자체": {
                    "주요_키워드": ["업무", "일", "직무", "도전", "의미"],
                    "동의어": ["재미", "흥미", "매력", "가치", "기여"],
                    "고위험_신호": ["일이 너무 단조롭고 지루하다", "의미 없는 일을 반복하는 기분", "도전적인 과제가 없다"]
                },
                "인정_및_성취": {
                    "주요_키워드": ["인정", "칭찬", "성과", "성취", "보람"],
                    "동의어": ["피드백", "격려", "무시", "알아주다"],
                    "고위험_신호": ["아무리 잘해도 알아주지 않는다", "성과를 가로채는 상사", "이 일을 왜 하는지 모르겠다"]
                },
                "책임_및_자율성": {
                    "주요_키워드": ["자율성", "권한", "책임", "오너십", "마이크로매니징"],
                    "동의어": ["간섭", "통제", "재량", "믿음", "위임"],
                    "고위험_신호": ["사사건건 간섭해서 일을 할 수가 없다", "아무런 권한 없이 책임만 진다", "내 방식대로 할 수 있는 게 없다"]
                }
            },
            "과도한_업무부담과_번아웃": {
                "업무량_및_압박": {
                    "주요_키워드": ["업무량", "과부하", "압박", "스트레스"],
                    "동의어": ["부담", "강도", "업무강도", "실적 압박"],
                    "고위험_신호": ["일이 끝나지 않는다", "업무량이 감당이 안 된다", "매일 압박감에 시달린다"]
                },
                "번아웃_및_소진": {
                    "주요_키워드": ["번아웃", "소진", "탈진", "지쳤다", "무기력"],
                    "동의어": ["방전", "모든 걸 쏟았다", "에너지가 없다", "냉소적"],
                    "고위험_신호": ["완전히 번아웃됐다", "아침에 일어나는 게 고통스럽다", "일에 대한 열정이 사라졌다"]
                },
                "일과_삶의_균형": {
                    "주요_키워드": ["워라밸", "개인 생활", "휴가", "퇴근 후"],
                    "동의어": ["저녁 있는 삶", "사생활", "연차", "연락"],
                    "고위험_신호": ["퇴근 후에도 계속 업무 연락이 온다", "개인 생활이 전혀 없다", "휴가 쓰는 것도 눈치 보인다"]
                },
                "역할_모호성_갈등": {
                    "주요_키워드": ["역할", "R&R", "책임", "혼란", "지시"],
                    "동의어": ["업무 범위", "가이드라인", "상충"],
                    "고위험_신호": ["내 역할이 뭔지 모르겠다", "A팀장과 B팀장의 지시가 다르다", "책임 소재가 불분명하다"]
                }
            },
            "건강하지_못한_조직문화와_관계": {
                "관리자_리더십": {
                    "주요_키워드": ["팀장", "리더", "상사", "관리자", "매니저"],
                    "동의어": ["멘토", "코칭", "피드백", "방향성", "의사결정"],
                    "고위험_신호": ["팀장님은 팀원들에게 관심이 없다", "비전 없이 시키는 일만 한다", "피드백을 준 적이 없다"]
                },
                "유해한_문화": {
                    "주요_키워드": ["문화", "갑질", "괴롭힘", "정치", "꼰대"],
                    "동의어": ["뒷담화", "따돌림", "폭언", "성희롱", "불통"],
                    "고위험_신호": ["윗사람에게 찍히면 끝이다", "사내 정치가 너무 심하다", "의견을 말하면 묵살당한다"]
                },
                "심리적_안정감": {
                    "주요_키워드": ["심리적 안정감", "실수", "의견", "질문", "안전"],
                    "동의어": ["두려움", "눈치", "솔직함", "취약성", "리스크"],
                    "고위험_신호": ["실수하면 크게 질책받는 분위기다", "다른 의견을 내는 것이 두렵다", "어려운 문제를 솔직하게 말할 수 없다"]
                },
                "동료_및_팀워크": {
                    "주요_키워드": ["동료", "팀원", "팀워크", "협업", "분위기"],
                    "동의어": ["관계", "소통", "갈등", "이기주의", "비협조적"],
                    "고위험_신호": ["팀원들끼리 서로 돕지 않는다", "뒷담화가 너무 심하다", "협업이 전혀 되지 않는다"]
                },
                "조직_지원_신뢰": {
                    "주요_키워드": ["지원", "신뢰", "존중", "믿음", "배려"],
                    "동의어": ["투자", "관심", "공감", "투명성"],
                    "고위험_신호": ["회사는 직원을 부품으로 생각한다", "문제가 생겨도 회사가 지켜주지 않는다", "경영진을 신뢰할 수 없다"]
                }
            },
            "불안정한_고용_및_비효율적_시스템": {
                "고용_안정성": {
                    "주요_키워드": ["고용 안정", "계약", "구조조정", "불안"],
                    "동의어": ["정규직", "비정규직", "해고", "권고사직", "조직개편"],
                    "고위험_신호": ["언제 잘릴지 몰라 불안하다", "계약 연장이 될지 모르겠다", "회사가 어렵다는 소문이 돈다"]
                },
                "회사_정책_행정": {
                    "주요_키워드": ["정책", "규정", "규칙", "제도", "시스템"],
                    "동의어": ["프로세스", "절차", "방침", "보고라인", "비효율"],
                    "고위험_신호": ["쓸데없는 보고가 너무 많다", "규정이 너무 빡빡해서 숨 막힌다", "시스템이 구식이다"]
                },
                "근무_조건": {
                    "주요_키워드": ["근무 환경", "사무실", "장비", "야근", "근무시간"],
                    "동의어": ["현장", "시설", "오피스", "초과근무", "워라밸", "출퇴근"],
                    "고위험_신호": ["사무실이 너무 덥고 환기도 안 된다", "매일 야근하는 문화", "업무 일정이 불규칙하다"]
                }
            }
        }
    
    def _initialize_stopwords(self) -> set:
        """불용어 초기화"""
        return set([
            # 조사
            '은', '는', '이', '가', '을', '를', '에', '에서', '로', '으로', '와', '과', '의', '도',
            '만', '부터', '까지', '에게', '한테', '께', '에게서', '한테서', '께서', '라', '야',
            
            # 어미
            '다', '습니다', '입니다', '했습니다', '있습니다', '없습니다', '됩니다',
            
            # 부사
            '아직', '이미', '벌써', '곧', '바로', '즉시', '항상', '늘', '자주', '가끔',
            '그냥', '좀', '조금', '많이', '너무', '매우', '정말', '진짜', '완전',
            
            # 대명사
            '그', '이', '저', '것', '그것', '이것', '저것', '여기', '거기', '저기',
            
            # 일반적인 명사
            '것', '거', '게', '걸', '건', '곳', '데', '때', '분', '번', '개', '명'
        ])
    
    def extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출"""
        if pd.isna(text) or not text:
            return []
        
        # 텍스트 정제
        clean_text = re.sub(r'[^\w\s가-힣]', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # 한글 단어 추출 (2글자 이상)
        words = re.findall(r'[가-힣]{2,}', clean_text)
        
        # 불용어 제거 및 필터링
        keywords = []
        for word in words:
            if word not in self.stopwords and len(word) >= 2:
                keywords.append(word)
        
        return keywords
    
    def calculate_sentiment_score(self, text: str, keywords: List[str]) -> float:
        """감정 점수 계산 (간단한 키워드 기반)"""
        if not text or not keywords:
            return 0.5  # 중립
        
        positive_keywords = [
            "만족", "좋", "훌륭", "성공", "성장", "발전", "기쁨", "행복", "즐거움",
            "보람", "성취", "인정", "칭찬", "격려", "지원", "도움", "협력"
        ]
        
        negative_keywords = [
            "불만", "나쁨", "실패", "스트레스", "압박", "부담", "피로", "소진",
            "번아웃", "불공정", "차별", "무시", "비판", "갈등", "문제", "어려움"
        ]
        
        positive_count = sum(1 for keyword in keywords if any(pos in keyword for pos in positive_keywords))
        negative_count = sum(1 for keyword in keywords if any(neg in keyword for neg in negative_keywords))
        
        total_count = len(keywords)
        if total_count == 0:
            return 0.5
        
        # 0.0 (매우 부정) ~ 1.0 (매우 긍정)
        sentiment_score = (positive_count - negative_count + total_count) / (2 * total_count)
        return max(0.0, min(1.0, sentiment_score))
    
    def _calculate_risk_level(self, attrition_risk_score: float) -> str:
        """퇴직 위험 점수를 기반으로 위험 레벨 계산"""
        if attrition_risk_score >= 0.7:
            return "HIGH"
        elif attrition_risk_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def calculate_attrition_risk_score(self, text: str, keywords: List[str]) -> Tuple[float, List[str]]:
        """퇴직 위험 점수 계산"""
        if not text or not keywords:
            return 0.0, []
        
        risk_factors = []
        total_risk_score = 0.0
        
        # 각 퇴직 원인 카테고리별로 위험 점수 계산
        for category, subcategories in self.attrition_keywords.items():
            category_risk = 0.0
            category_factors = []
            
            for subcat, keyword_data in subcategories.items():
                # 키워드 매칭 점수
                main_keywords = keyword_data.get("주요_키워드", [])
                synonyms = keyword_data.get("동의어", [])
                high_risk_signals = keyword_data.get("고위험_신호", [])
                
                # 매칭된 키워드 찾기
                matched_main = [k for k in keywords if any(mk in k for mk in main_keywords)]
                matched_synonyms = [k for k in keywords if any(syn in k for syn in synonyms)]
                matched_high_risk = [phrase for phrase in high_risk_signals if phrase in text]
                
                # 점수 계산
                subcat_score = 0.0
                if matched_main:
                    subcat_score += len(matched_main) * 0.3
                    category_factors.extend(matched_main)
                
                if matched_synonyms:
                    subcat_score += len(matched_synonyms) * 0.2
                    category_factors.extend(matched_synonyms)
                
                if matched_high_risk:
                    subcat_score += len(matched_high_risk) * 0.8  # 고위험 신호는 높은 가중치
                    category_factors.extend(matched_high_risk)
                
                category_risk += subcat_score
            
            if category_risk > 0:
                total_risk_score += category_risk
                risk_factors.extend(category_factors)
        
        # 정규화 (0.0 ~ 1.0)
        normalized_score = min(1.0, total_risk_score / 10.0)  # 10.0으로 나누어 정규화
        
        return normalized_score, list(set(risk_factors))  # 중복 제거
    
    def analyze_text(self, text: str, employee_id: str = "unknown", text_type: str = "general") -> Dict[str, Any]:
        """종합 텍스트 분석 (개선된 JD-R 모델 기반)"""
        try:
            # JD-R 기반 분석 사용 (analyzer가 있는 경우)
            if hasattr(self, 'analyzer') and self.analyzer:
                jdr_result = self.analyzer.analyze_employee_text(
                    employee_id=employee_id,
                    self_review=text,
                    peer_feedback="",
                    weekly_survey=""
                )
                
                # jdr_result가 딕셔너리인지 확인
                if isinstance(jdr_result, dict):
                    return {
                        "keywords": jdr_result.get('detected_keywords', []),
                        "sentiment_score": jdr_result.get('sentiment_score', 0.5),
                        "attrition_risk_score": jdr_result.get('psychological_risk_score', 0.5),
                        "risk_factors": jdr_result.get('job_demands_matches', []) + jdr_result.get('job_resources_deficiency_matches', []),
                        "keyword_count": len(jdr_result.get('detected_keywords', [])),
                        "text_length": len(text),
                        "jd_r_indicators": jdr_result.get('jd_r_indicators', {}),
                        "risk_level": jdr_result.get('risk_level', 'MEDIUM'),
                        "attrition_prediction": jdr_result.get('attrition_prediction', 0)
                    }
                else:
                    # jdr_result가 딕셔너리가 아닌 경우 로그 출력하고 fallback 사용
                    logger.warning(f"JD-R 분석 결과가 예상과 다른 타입입니다: {type(jdr_result)}, 값: {jdr_result}")
                    # fallback으로 넘어감
            
            # 기존 방식 (fallback)
            else:
                # 키워드 추출
                keywords = self.extract_keywords(text)
                
                # 감정 점수 계산
                sentiment_score = self.calculate_sentiment_score(text, keywords)
                
                # 퇴직 위험 점수 계산
                attrition_risk_score, risk_factors = self.calculate_attrition_risk_score(text, keywords)
                
                # 위험 레벨 계산
                risk_level = self._calculate_risk_level(attrition_risk_score)
                
                return {
                    "keywords": keywords,
                    "sentiment_score": sentiment_score,
                    "attrition_risk_score": attrition_risk_score,
                    "risk_factors": risk_factors,
                    "risk_level": risk_level,
                    "keyword_count": len(keywords),
                    "text_length": len(text)
                }
            
        except Exception as e:
            logger.error(f"텍스트 분석 오류: {str(e)}")
            return {
                "keywords": [],
                "sentiment_score": 0.5,
                "attrition_risk_score": 0.5,
                "risk_factors": [],
                "risk_level": "MEDIUM",  # 기본값으로 MEDIUM 설정
                "keyword_count": 0,
                "text_length": len(text) if text else 0
            }
    
    def analyze_attrition_risk(self, text: str, employee_id: str = "unknown") -> Dict[str, Any]:
        """퇴직 위험 전용 분석"""
        keywords = self.extract_keywords(text)
        risk_score, risk_factors = self.calculate_attrition_risk_score(text, keywords)
        
        # 위험 수준 분류
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "keywords_detected": keywords
        }
