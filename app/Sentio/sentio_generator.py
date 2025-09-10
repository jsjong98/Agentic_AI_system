# -*- coding: utf-8 -*-
"""
Sentio 텍스트 생성기
퇴직원인 키워드를 반영한 페르소나 기반 HR 텍스트 생성
"""

import pandas as pd
import openai
import json
import time
from typing import Dict, List
import random
import logging

logger = logging.getLogger(__name__)

class SentioTextGenerator:
    """Sentio HR 텍스트 생성기"""
    
    def __init__(self, api_key: str, csv_file_path: str):
        """
        HR 텍스트 데이터 생성기 초기화
        
        Args:
            api_key: OpenAI API 키
            csv_file_path: IBM HR 데이터셋 CSV 파일 경로
        """
        self.client = openai.OpenAI(api_key=api_key)
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

    def get_persona_instructions(self, persona_code: str) -> Dict[str, str]:
        """페르소나별 세부 지시사항 반환"""
        
        instructions = {
            "P01": {  # 번아웃 및 소진 페르소나
                "tone": "전반적으로 지쳐있고, 무기력하며, 때로는 냉소적인 톤을 유지하세요. 긍정적인 성과를 언급하더라도, 그 성과를 위해 치른 '희생'을 암시하는 표현을 덧붙여야 합니다.",
                "keywords": "'번아웃', '소진', '탈진', '지쳤다', '무기력', '방전', '에너지가 없다', '냉소적', '업무량이 감당이 안 된다', '매일 압박감에 시달린다', '완전히 번아웃됐다', '아침에 일어나는 게 고통스럽다', '일에 대한 열정이 사라졌다', '퇴근 후에도 계속 업무 연락이 온다', '개인 생활이 전혀 없다' 등의 표현을 사용하세요.",
                "perspective": "모든 것을 노력과 시간의 '소모' 관점에서 바라봅니다. 워라밸 부재와 과도한 업무 부담으로 인한 피로감을 중심으로 서술하세요. 일과 삶의 균형이 무너진 상태를 암시하는 표현을 포함하세요."
            },
            "P02": {  # 조직 부적응 페르소나
                "tone": "불안하고, 자신감 없으며, 망설이는 톤을 사용하세요. 문장이 명확하게 끝나지 않거나, 추측성 표현을 자주 사용하여 혼란스러운 상태를 드러내세요.",
                "keywords": "'심리적 안정감', '실수', '의견', '질문', '두려움', '눈치', '어색한', '아직 파악 중인', '기대와는 다른', '도움을 받기 어려운', '혼자 해결해야 하는', '실수하면 크게 질책받는 분위기다', '다른 의견을 내는 것이 두렵다', '어려운 문제를 솔직하게 말할 수 없다', '팀원들끼리 서로 돕지 않는다', '협업이 전혀 되지 않는다' 등의 표현을 사용하세요.",
                "perspective": "'나' vs '회사/팀'의 구도가 미묘하게 드러납니다. 조직 내부자로서의 발언이 아닌, 외부 관찰자처럼 겉도는 시각을 유지하세요. 심리적 안전감 부족과 동료 관계의 어려움을 암시하세요."
            },
            "P03": {  # 성장 정체 페르소나
                "tone": "열정이 식고 다소 체념적인, 혹은 미묘한 불만이 담긴 톤을 유지하세요. 과거의 성과에 대해서는 언급할 수 있지만, 현재와 미래에 대해서는 비전이 부재함을 드러내야 합니다.",
                "keywords": "'성장', '발전', '경력', '커리어', '교육', '학습', '승진', '기회', '비전', '배움', '개발', '반복적인', '정체된', '새로운 것을 배우고 싶은', '성장의 기회가 없는', '더 이상의 발전은 없는', '여기서는 더 배울 게 없다', '성장할 수 있다는 느낌이 안 든다', '승진 기회가 막혀있다', '일이 너무 단조롭고 지루하다', '의미 없는 일을 반복하는 기분', '도전적인 과제가 없다' 등의 표현을 사용하세요.",
                "perspective": "시간을 '현재의 반복'과 '불투명한 미래'로 인식합니다. 성장과 발전에 대한 갈증과 현재 상황에 대한 답답함을 표현하세요. 업무의 의미와 도전에 대한 아쉬움을 드러내세요."
            },
            "P04": {  # 불공정 대우 페르소나
                "tone": "객관적인 사실을 나열하는 척하지만, 그 안에 미묘한 불만과 억울함이 담긴 톤을 사용하세요. 직접적인 비난보다는, 객관적인 비교를 통해 불공정함을 암시하는 방식이 효과적입니다.",
                "keywords": "'급여', '연봉', '보상', '복지', '복리후생', '공정', '불공평', '차별', '형평성', '편애', '특혜', '인정', '칭찬', '성과', '성취', '기여도에 비해', '시장 상황과 비교하면', '공정한 평가인지 의문이 드는', '정당한 보상', '일한 만큼 못 받는다', '성과급 기준이 불투명하다', '복지가 거의 없다', '같은 일을 하는데 A만 인정받는다', '성과 평가가 불공평하다', '아무리 잘해도 알아주지 않는다' 등의 표현을 사용하세요.",
                "perspective": "'나의 가치' vs '회사의 평가'라는 프레임으로 상황을 해석합니다. 자신의 성과를 구체적 사례로 강조하며, 보상과 인정의 불공정함을 부각하세요. 복리후생과 평가 시스템에 대한 불만을 암시하세요."
            },
            "S01": {
                "tone": "차분하고, 안정적이며, 높은 책임감이 느껴지는 톤을 유지하세요. 개인의 성과보다는 '팀'이나 '회사'의 성공을 함께 언급하며 공동체 의식을 드러내세요.",
                "keywords": "'꾸준히 기여', '장기적인 관점에서', '조직의 성공을 위해', '안정적인 성과', '책임감을 가지고' 등의 표현을 통해 신뢰감을 주세요.",
                "perspective": "개인의 커리어를 조직의 발전과 함께 바라봅니다. 단기적인 성과보다는 장기적인 기여와 지속 가능성을 중시하는 시각을 보여주세요."
            },
            "S02": {
                "tone": "매우 긍정적이고, 에너지 넘치며, 미래지향적인 톤을 사용하세요. 자신의 성과를 구체적인 수치나 결과 중심으로 명확하게 제시하며 자신감을 표현해야 합니다.",
                "keywords": "'새로운 도전을 통해 성장', '주도적으로 해결', '결과를 초과 달성', '빠르게 배우고 적용하며', '팀에 긍정적인 영향' 등의 표현을 사용하세요.",
                "perspective": "성장을 '기회'와 '도전'으로 인식합니다. 현재의 성공에 안주하지 않고, 다음 단계의 목표와 비전을 적극적으로 언급하세요."
            },
            "S03": {
                "tone": "차분하지만 깊은 몰입과 전문성이 느껴지는 톤을 사용하세요. 성과를 자랑하기보다는, 그 과정에서 겪었던 '기술적 도전'이나 '흥미로운 문제'에 대해 이야기하는 것을 선호합니다.",
                "keywords": "'업무 자체의 보람', '기술적 깊이를 더하는', '어려운 문제를 해결하는 즐거움', '최적의 솔루션을 찾아내어', '전문성을 발휘하여' 등의 표현을 사용하세요.",
                "perspective": "일을 '보상의 수단'이 아닌 '지적 유희'나 '자기실현의 과정'으로 바라봅니다. 사람이나 정치적 문제보다는 '과업' 그 자체에 초점을 맞춥니다."
            },
            "N01": {
                "tone": "감정이 거의 드러나지 않는, 다소 형식적이고 무미건조한 톤을 사용하세요. 텍스트가 추상적이고 일반적인 표현으로 채워져, 구체적인 내용이나 깊이 있는 고민이 부족하게 느껴져야 합니다.",
                "keywords": "'주어진 역할에 충실', '문제없이 진행', '지난 분기와 유사하게', '특별한 이슈 없음', '무난했습니다' 등의 표현을 반복적으로 사용하세요.",
                "perspective": "일을 '해야만 하는 의무'로만 인식합니다. 성장, 발전, 기여와 같은 단어보다는 '유지', '안정', '현상 유지'에 초점을 둡니다."
            },
            "N02": {
                "tone": "객관적이고 논리적인 척하지만, 문장 곳곳에 비판적인 시각이나 냉소적인 뉘앙스가 묻어나오는 톤을 구사해야 합니다. 성과에 대한 칭찬과 문제점에 대한 지적을 한 문단 안에서 교차시키는 것이 효과적입니다.",
                "keywords": "(성과) '데이터 기반으로 증명', '문제의 본질을 파악하여 해결', '효율을 개선' / (불만) '하지만 비효율적인 회의 문화는 여전합니다', '개인의 역량에만 의존하는 경향이 아쉽습니다'",
                "perspective": "'나(유능함)' vs '그들(비합리적임)'의 구도를 명확히 합니다. 자신을 문제 해결사로 포지셔닝하는 동시에, 조직의 비합리성을 지적하는 제3자적 관점을 유지하세요."
            },
            "N03": {
                "tone": "차분하고 현실적인 톤을 유지하세요. 때로는 피로감이 묻어날 수 있습니다. '효율성'과 '시간 관리'를 중요한 가치로 여기는 모습을 보여줘야 합니다.",
                "keywords": "'업무와 육아를 병행하며', '시간 관리가 중요해진', '유연 근무 제도가 큰 도움이 되는', '집중해야 할 때와 아닐 때를 구분하여', '가족과의 시간' 등의 표현을 사용하세요.",
                "perspective": "모든 것을 '시간'과 '에너지'라는 한정된 자원의 분배 문제로 바라봅니다. 회사의 지원(유연근무, 재택 등)에 대한 감사나 필요성을 언급하며, 워라밸의 중요성을 강조하세요."
            }
        }
        
        return instructions.get(persona_code, instructions["N01"])

    def get_attrition_keywords_for_persona(self, persona_code: str) -> List[str]:
        """페르소나별 퇴직 위험 키워드 반환"""
        attrition_mapping = {
            "P01": ["과도한_업무부담과_번아웃"],  # 번아웃 및 소진
            "P02": ["건강하지_못한_조직문화와_관계"],  # 조직 부적응
            "P03": ["성장_정체와_동기부여_상실"],  # 성장 정체
            "P04": ["불공정한_보상과_평가"],  # 불공정 대우
            "S01": [],  # 안정형 - 퇴직 위험 낮음
            "S02": [],  # 성장형 - 퇴직 위험 낮음  
            "S03": [],  # 몰입형 - 퇴직 위험 낮음
            "N01": ["불안정한_고용_및_비효율적_시스템"],  # 중립형 - 시스템 불만
            "N02": ["건강하지_못한_조직문화와_관계"],  # 비판형 - 조직 문화 불만
            "N03": ["과도한_업무부담과_번아웃"]  # 균형형 - 워라밸 이슈
        }
        
        relevant_categories = attrition_mapping.get(persona_code, [])
        keywords = []
        
        for category in relevant_categories:
            if category in self.attrition_keywords:
                for subcategory, keyword_data in self.attrition_keywords[category].items():
                    keywords.extend(keyword_data.get("주요_키워드", []))
                    keywords.extend(keyword_data.get("동의어", []))
                    # 고위험 신호는 직접적이므로 선별적으로 사용
                    if persona_code in ["P01", "P02", "P03", "P04"]:  # 퇴직 위험 높은 페르소나만
                        keywords.extend(keyword_data.get("고위험_신호", [])[:2])  # 상위 2개만 선택
        
        return list(set(keywords))  # 중복 제거

    def create_base_prompt_template(self, employee_data: pd.Series, text_type: str) -> str:
        """기본 프롬프트 템플릿 생성"""
        
        text_type_descriptions = {
            "SELF_REVIEW": "자기 성과 평가 - 지난 1년간 자신의 주요 성과, 어려웠던 점, 그리고 향후 커리어 목표에 대해 작성하는 공식적인 문서",
            "PEER_FEEDBACK": "동료 피드백 - 함께 프로젝트를 수행한 동료가 해당 직원의 강점, 개선점, 협업 태도에 대해 작성하는 리뷰",
            "WEEKLY_SURVEY": "주간 익명 설문 - '이번 주 업무 만족도는 어떠셨나요? 그 이유는 무엇인가요?'와 같은 간단한 질문에 대해 직원이 익명으로 짧게 답변하는 텍스트"
        }
        
        # 성과/만족도를 자연스러운 표현으로 변환
        performance_context = {
            1: "개선이 필요한 상황",
            2: "기대에 미치지 못하는 상황", 
            3: "기대 수준을 충족하는 상황",
            4: "기대를 초과하는 우수한 상황"
        }
        
        satisfaction_context = {
            1: "업무에 대한 불만족이 높은 상황",
            2: "업무에 대한 만족도가 낮은 상황",
            3: "업무에 대해 보통 수준의 만족을 느끼는 상황", 
            4: "업무에 대해 높은 만족을 느끼는 상황"
        }
        
        involvement_context = {
            1: "업무 몰입도가 낮은 상황",
            2: "업무에 대한 관심이 제한적인 상황",
            3: "업무에 적절히 몰입하고 있는 상황",
            4: "업무에 깊이 몰입하고 있는 상황"
        }
        
        base_template = f"""
당신은 조직 내 개인의 심리와 감정을 깊이 이해하는 HR 분석 전문가입니다. 지금부터 {text_type_descriptions[text_type]}을 작성해야 합니다.

당신이 분석하고 묘사할 대상 직원의 현재 상황은 다음과 같습니다:

- **직무 역할**: {employee_data['JobRole']} 
- **조직 경험**: 현 회사에서 {employee_data['YearsAtCompany']}년간 근무 (조직 문화에 대한 이해도와 적응 수준을 나타냄)
- **현재 성과 상황**: {performance_context.get(employee_data['PerformanceRating'], '보통 수준')}
- **업무에 대한 감정**: {satisfaction_context.get(employee_data['JobSatisfaction'], '보통 수준')}
- **업무 참여도**: {involvement_context.get(employee_data['JobInvolvement'], '보통 수준')}

**중요한 작성 지침:**
1. 구체적인 점수, 등급, 수치는 절대 언급하지 마세요
2. 직원 번호나 개인 식별 정보는 사용하지 마세요  
3. 퇴사 의사나 이직 계획에 대해서는 직접적으로 언급하지 마세요
4. 연봉, 보상 수준 등 구체적인 금액은 언급하지 마세요
5. 자연스럽고 일반적인 업무 상황으로 표현해주세요

이제 아래의 [페르소나별 심층 지시사항]을 가장 중요한 지침으로 삼아 반드시 준수하여, 대상 직원의 입장에서 또는 대상 직원에 대해 약 4문장 내외로 {text_type_descriptions[text_type]}을 한국어로 작성해주세요.

텍스트는 위에 제시된 상황을 기반으로 하되, 지시사항에 명시된 페르소나의 내면 심리와 감정이 자연스럽게 드러나야 합니다.
"""
        return base_template

    def generate_text_for_employee(self, employee_data: pd.Series, text_type: str) -> str:
        """개별 직원에 대한 텍스트 생성"""
        
        # 페르소나 코드 확인
        persona_code = employee_data['softmax_Persona_Code']
        persona_instructions = self.get_persona_instructions(persona_code)
        
        # 기본 프롬프트 생성
        base_prompt = self.create_base_prompt_template(employee_data, text_type)
        
        # 페르소나별 퇴직 위험 키워드 가져오기
        attrition_keywords = self.get_attrition_keywords_for_persona(persona_code)
        attrition_keyword_text = ""
        if attrition_keywords:
            # 키워드를 랜덤하게 선택해서 자연스럽게 포함
            selected_keywords = random.sample(attrition_keywords, min(5, len(attrition_keywords)))
            attrition_keyword_text = f"""

**퇴직 위험 신호 키워드 (자연스럽게 포함할 것):** {', '.join(selected_keywords)}
- 이 키워드들을 직접적으로 나열하지 말고, 텍스트의 맥락에 자연스럽게 녹여서 사용하세요.
- 퇴직 의사를 직접 언급하지 말고, 이러한 키워드들이 담긴 상황이나 감정을 간접적으로 표현하세요."""

        # 페르소나별 지시사항 추가
        full_prompt = base_prompt + f"""

---
**[페르소나별 심층 지시사항]:**

**어조:** {persona_instructions['tone']}

**키워드:** {persona_instructions['keywords']}

**관점:** {persona_instructions['perspective']}
{attrition_keyword_text}

---

위 지시사항을 철저히 준수하여 해당 페르소나의 특성이 잘 드러나는 텍스트를 생성해주세요.
"""
        
        try:
            # ChatGPT API 호출 (gpt-4 사용)
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 HR 텍스트 생성 전문가입니다. 주어진 지시사항을 정확히 따라 자연스러운 한국어 텍스트를 생성해주세요."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            generated_text = response.choices[0].message.content.strip()
            return generated_text
            
        except Exception as e:
            logger.error(f"API 호출 오류 (직원 {employee_data['EmployeeNumber']}): {str(e)}")
            return "텍스트 생성 중 오류가 발생했습니다."

    def generate_all_texts(self, text_types: List[str] = None, sample_size: int = None):
        """모든 직원에 대해 모든 텍스트 유형 생성"""
        
        if text_types is None:
            text_types = ["SELF_REVIEW", "PEER_FEEDBACK", "WEEKLY_SURVEY"]
            
        # 샘플링 (테스트용)
        df_to_process = self.df.sample(n=sample_size) if sample_size else self.df
        
        total_requests = len(df_to_process) * len(text_types)
        current_request = 0
        
        for _, employee_data in df_to_process.iterrows():
            employee_texts = {
                'EmployeeNumber': employee_data['EmployeeNumber'],
                'JobRole': employee_data['JobRole'],
                'Persona_Code': employee_data['softmax_Persona_Code'],
                'Persona_Name': employee_data['softmax_Persona'],
                'Attrition': employee_data['Attrition']
            }
            
            for text_type in text_types:
                current_request += 1
                logger.info(f"처리 중... ({current_request}/{total_requests}) 직원 {employee_data['EmployeeNumber']} - {text_type}")
                
                generated_text = self.generate_text_for_employee(employee_data, text_type)
                employee_texts[f"{text_type}_text"] = generated_text
                
                # API 호출 제한을 위한 대기
                time.sleep(0.5)
            
            self.generated_texts.append(employee_texts)
    
    def save_to_csv(self, filename: str = "generated_hr_texts.csv"):
        """결과를 CSV 파일로 저장"""
        if not self.generated_texts:
            logger.warning("생성된 텍스트가 없습니다. generate_all_texts()를 먼저 실행해주세요.")
            return
            
        result_df = pd.DataFrame(self.generated_texts)
        result_df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"결과가 {filename}에 저장되었습니다.")
        
    def save_to_json(self, filename: str = "generated_hr_texts.json"):
        """결과를 JSON 파일로 저장"""
        if not self.generated_texts:
            logger.warning("생성된 텍스트가 없습니다. generate_all_texts()를 먼저 실행해주세요.")
            return
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.generated_texts, f, ensure_ascii=False, indent=2)
        logger.info(f"결과가 {filename}에 저장되었습니다.")
