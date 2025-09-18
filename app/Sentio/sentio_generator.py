# -*- coding: utf-8 -*-
"""
Sentio 텍스트 생성기 (키워드 기반)
퇴직원인 키워드를 반영한 HR 텍스트 생성 (페르소나 정보 제거)
"""

import pandas as pd
import openai
import json
import time
import os
from typing import Dict, List
import random
import logging

logger = logging.getLogger(__name__)

class SentioTextGenerator:
    """Sentio HR 텍스트 생성기 (키워드 기반)"""
    
    def __init__(self, api_key: str, csv_file_path: str = None):
        """
        HR 텍스트 데이터 생성기 초기화
        
        Args:
            api_key: OpenAI API 키
            csv_file_path: 선택적 - 페르소나 정보 없이 동작
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.df = None
        if csv_file_path and os.path.exists(csv_file_path):
            self.df = pd.read_csv(csv_file_path)
        self.generated_texts = []
        self.attrition_keywords = self._initialize_attrition_keywords()
        
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

    def get_all_available_keywords(self) -> Dict[str, List[str]]:
        """사용 가능한 모든 키워드 카테고리 반환"""
        categories = {}
        for category, subcategories in self.attrition_keywords.items():
            category_keywords = []
            for subcategory, keyword_data in subcategories.items():
                category_keywords.extend(keyword_data.get("주요_키워드", []))
                category_keywords.extend(keyword_data.get("동의어", []))
            categories[category] = list(set(category_keywords))  # 중복 제거
        return categories
    
    def get_keywords_by_category(self, category: str) -> List[str]:
        """특정 카테고리의 키워드 반환"""
        if category not in self.attrition_keywords:
            return []
        
        keywords = []
        for subcategory, keyword_data in self.attrition_keywords[category].items():
            keywords.extend(keyword_data.get("주요_키워드", []))
            keywords.extend(keyword_data.get("동의어", []))
        return list(set(keywords))  # 중복 제거

    def create_keyword_based_prompt(self, keywords: List[str], text_type: str) -> str:
        """Sentio.ipynb의 full_prompt 형식을 따른 키워드 기반 프롬프트 생성"""
        
        text_type_descriptions = {
            "SELF_REVIEW": "자기 성과 평가 - 지난 1년간 자신의 주요 성과, 어려웠던 점, 그리고 향후 커리어 목표에 대해 작성하는 공식적인 문서",
            "PEER_FEEDBACK": "동료 피드백 - 함께 프로젝트를 수행한 동료가 해당 직원의 강점, 개선점, 협업 태도에 대해 작성하는 리뷰",
            "WEEKLY_SURVEY": "주간 익명 설문 - '이번 주 업무 만족도는 어떠셨나요? 그 이유는 무엇인가요?'와 같은 간단한 질문에 대해 직원이 익명으로 짧게 답변하는 텍스트"
        }
        
        keyword_text = ", ".join(keywords[:10])  # 상위 10개 키워드만 사용
        
        # Sentio.ipynb의 full_prompt 형식을 따른 구조화된 프롬프트
        full_prompt = f"""
당신은 조직 내 개인의 심리와 감정을 깊이 이해하는 HR 텍스트 생성 전문가입니다. 지금부터 {text_type_descriptions[text_type]}을 작성해야 합니다.

**분석 대상 키워드:**
- 주요 감지 키워드: {keyword_text}

**작성 지침:**
1. 위 키워드들을 자연스럽게 포함하여 작성하세요
2. 키워드를 직접적으로 나열하지 말고, 텍스트의 맥락에 자연스럽게 녹여서 사용하세요
3. 약 4-5문장 내외로 작성해주세요
4. 한국어로 작성하며, 자연스럽고 진정성 있는 표현을 사용하세요
5. 구체적인 점수, 등급, 수치는 언급하지 마세요
6. 퇴사 의사나 이직 계획에 대해서는 직접적으로 언급하지 마세요

**요청사항:**
위 지침에 따라 {text_type_descriptions[text_type]}을 전문적이면서도 실용적인 톤으로 작성해주세요.

한국어로 응답하고, 전문적이면서도 실용적인 톤으로 작성해주세요.
"""
        return full_prompt

    def generate_text_from_keywords(self, keywords: List[str], text_type: str) -> str:
        """키워드 기반 텍스트 생성"""
        
        # 키워드 기반 프롬프트 생성
        prompt = self.create_keyword_based_prompt(keywords, text_type)
        
        try:
            # 표준 OpenAI Chat Completions API 사용
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 조직 내 개인의 심리와 감정을 깊이 이해하는 HR 분석 전문가입니다."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            generated_text = response.choices[0].message.content.strip()
            return generated_text
            
        except Exception as e:
            logger.error(f"API 호출 오류: {str(e)}")
            return "텍스트 생성 중 오류가 발생했습니다."

    def generate_batch_texts_from_keywords(self, keyword_batches: List[Dict], text_types: List[str] = None):
        """키워드 배치 기반 텍스트 생성"""
        
        if text_types is None:
            text_types = ["SELF_REVIEW", "PEER_FEEDBACK", "WEEKLY_SURVEY"]
            
        total_requests = len(keyword_batches) * len(text_types)
        current_request = 0
        
        for batch_data in keyword_batches:
            batch_texts = {
                'batch_id': batch_data.get('batch_id', 'unknown'),
                'keywords': batch_data.get('keywords', [])
            }
            
            for text_type in text_types:
                current_request += 1
                logger.info(f"처리 중... ({current_request}/{total_requests}) 배치 {batch_data.get('batch_id', 'unknown')} - {text_type}")
                
                generated_text = self.generate_text_from_keywords(
                    keywords=batch_data.get('keywords', []),
                    text_type=text_type
                )
                batch_texts[f"{text_type}_text"] = generated_text
                
                # API 호출 제한을 위한 대기
                time.sleep(0.5)
            
            self.generated_texts.append(batch_texts)
    
    def save_to_csv(self, filename: str = "generated_hr_texts.csv"):
        """결과를 CSV 파일로 저장"""
        if not self.generated_texts:
            logger.warning("생성된 텍스트가 없습니다. generate_batch_texts_from_keywords()를 먼저 실행해주세요.")
            return
            
        result_df = pd.DataFrame(self.generated_texts)
        result_df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"결과가 {filename}에 저장되었습니다.")
        
    def save_to_json(self, filename: str = "generated_hr_texts.json"):
        """결과를 JSON 파일로 저장"""
        if not self.generated_texts:
            logger.warning("생성된 텍스트가 없습니다. generate_batch_texts_from_keywords()를 먼저 실행해주세요.")
            return
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.generated_texts, f, ensure_ascii=False, indent=2)
        logger.info(f"결과가 {filename}에 저장되었습니다.")