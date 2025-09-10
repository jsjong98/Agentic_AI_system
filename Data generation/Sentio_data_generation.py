import pandas as pd
import openai
import json
import time
from typing import Dict, List
import random
from dotenv import load_dotenv
import os

class HRTextGenerator:
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
        
    def get_persona_instructions(self, persona_code: str) -> Dict[str, str]:
        """페르소나별 세부 지시사항 반환"""
        
        instructions = {
            "P01": {
                "tone": "전반적으로 지쳐있고, 무기력하며, 때로는 냉소적인 톤을 유지하세요. 긍정적인 성과를 언급하더라도, 그 성과를 위해 치른 '희생'을 암시하는 표현을 덧붙여야 합니다.",
                "keywords": "'소진', '한계', '탈진', '기계적으로', '의미 없는', '끝이 없는' 등의 단어를 사용하여 에너지 고갈 상태를 명확히 표현하세요.",
                "perspective": "모든 것을 노력과 시간의 '소모' 관점에서 바라봅니다. 미래에 대한 기대나 희망보다는 현재의 압박감과 피로를 중심으로 서술하세요."
            },
            "P02": {
                "tone": "불안하고, 자신감 없으며, 망설이는 톤을 사용하세요. 문장이 명확하게 끝나지 않거나, 추측성 표현을 자주 사용하여 혼란스러운 상태를 드러내세요.",
                "keywords": "'아직 파악 중인', '기대와는 다른', '도움을 받기 어려운', '혼자 해결해야 하는', '어색한' 등의 표현으로 조직에 융화되지 못하고 있음을 암시하세요.",
                "perspective": "'나' vs '회사/팀'의 구도가 미묘하게 드러납니다. 조직 내부자로서의 발언이 아닌, 외부 관찰자처럼 겉도는 시각을 유지하세요."
            },
            "P03": {
                "tone": "열정이 식고 다소 체념적인, 혹은 미묘한 불만이 담긴 톤을 유지하세요. 과거의 성과에 대해서는 언급할 수 있지만, 현재와 미래에 대해서는 비전이 부재함을 드러내야 합니다.",
                "keywords": "'반복적인', '정체된', '새로운 것을 배우고 싶은', '성장의 기회가 없는', '더 이상의 발전은 없는' 등의 표현을 통해 경력에 대한 답답함을 표현하세요.",
                "perspective": "시간을 '현재의 반복'과 '불투명한 미래'로 인식합니다. 새로운 기술이나 트렌드에 대한 갈증을 간접적으로 표현할 수 있습니다."
            },
            "P04": {
                "tone": "객관적인 사실을 나열하는 척하지만, 그 안에 미묘한 불만과 억울함이 담긴 톤을 사용하세요. 직접적인 비난보다는, 객관적인 비교를 통해 불공정함을 암시하는 방식이 효과적입니다.",
                "keywords": "'기여도에 비해', '시장 상황과 비교하면', '공정한 평가인지 의문이 드는', '정당한 보상', '연봉 인상률이 아쉬운' 등의 표현을 사용하세요.",
                "perspective": "'나의 가치' vs '회사의 평가'라는 프레임으로 상황을 해석합니다. 자신의 성과를 수치나 구체적 사례를 들어 강조하며, 보상과의 연결고리가 약하다는 점을 부각하세요."
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
        
        # 페르소나별 지시사항 추가
        full_prompt = base_prompt + f"""

---
**[페르소나별 심층 지시사항]:**

**어조:** {persona_instructions['tone']}

**키워드:** {persona_instructions['keywords']}

**관점:** {persona_instructions['perspective']}

---

위 지시사항을 철저히 준수하여 해당 페르소나의 특성이 잘 드러나는 텍스트를 생성해주세요.
"""
        
        try:
            # 새로운 responses.create API 사용
            response = self.client.responses.create(
                model="gpt-5-nano-2025-08-07",
                input=full_prompt,
                reasoning={"effort": "medium"},
                text={"verbosity": "medium"}
            )
            
            generated_text = response.output_text.strip()
            return generated_text
            
        except Exception as e:
            print(f"API 호출 오류 (직원 {employee_data['EmployeeNumber']}): {str(e)}")
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
                print(f"처리 중... ({current_request}/{total_requests}) 직원 {employee_data['EmployeeNumber']} - {text_type}")
                
                generated_text = self.generate_text_for_employee(employee_data, text_type)
                employee_texts[f"{text_type}_text"] = generated_text
                
                # API 호출 제한을 위한 대기 (안정적인 처리를 위해 조금 더 대기)
                time.sleep(0.5)
            
            self.generated_texts.append(employee_texts)
    
    def save_to_csv(self, filename: str = "generated_hr_texts.csv"):
        """결과를 CSV 파일로 저장"""
        if not self.generated_texts:
            print("생성된 텍스트가 없습니다. generate_all_texts()를 먼저 실행해주세요.")
            return
            
        result_df = pd.DataFrame(self.generated_texts)
        result_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"결과가 {filename}에 저장되었습니다.")
        
    def save_to_json(self, filename: str = "generated_hr_texts.json"):
        """결과를 JSON 파일로 저장"""
        if not self.generated_texts:
            print("생성된 텍스트가 없습니다. generate_all_texts()를 먼저 실행해주세요.")
            return
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.generated_texts, f, ensure_ascii=False, indent=2)
        print(f"결과가 {filename}에 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    # 설정
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CSV_FILE_PATH = "data/IBM_HR_personas_assigned.csv"  # CSV 파일 경로

    # API 키가 제대로 로드되었는지 확인합니다.
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
    
    if OPENAI_API_KEY != "your-api-key-here":
        # 텍스트 생성기 초기화
        generator = HRTextGenerator(OPENAI_API_KEY, CSV_FILE_PATH)
        
        print("\n=== 실제 텍스트 생성 시작 ===")
        print("실제 테스트: 1470명에 대해서 생성합니다...")
        
        # 전체 데ㅐ 테스트로 실행
        generator.generate_all_texts(sample_size=1470)
        
        # 결과 저장
        generator.save_to_csv("data/IBM_HR_text.csv")
        generator.save_to_json("data/IBM_HR_text.json")
        
        print("텍스트 생성 완료!")
        print(f"총 {len(generator.generated_texts)}명의 텍스트가 생성되었습니다.")
        
        # 결과 확인
        if generator.generated_texts:
            for i, result in enumerate(generator.generated_texts):
                print(f"\n=== 샘플 {i+1} (직원 {result['EmployeeNumber']}) ===")
                print(f"페르소나: {result['Persona_Name']} ({result['Persona_Code']})")
                print(f"퇴사여부: {result['Attrition']}")
                print(f"자기평가: {result['SELF_REVIEW_text'][:150]}...")
    else:
        print("\n실제 텍스트 생성을 위해서는 OPENAI_API_KEY를 설정해주세요.")