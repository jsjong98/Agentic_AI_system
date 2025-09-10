import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (필요시)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('default')

# 데이터 읽기
data = pd.read_csv('data/IBM_HR_text.csv')

# 기본 정보 출력
print("데이터 형태:", data.shape)
print("\n컬럼 목록:")
print(data.columns.tolist())

# Attrition 분포
print("\nAttrition 분포:")
print(data['Attrition'].value_counts())
print(f"퇴사율: {data['Attrition'].value_counts()['Yes'] / len(data) * 100:.2f}%")

# 텍스트 컬럼 확인
text_columns = ['SELF_REVIEW_text', 'PEER_FEEDBACK_text', 'WEEKLY_SURVEY_text']
print(f"\n사용 가능한 텍스트 컬럼: {text_columns}")

# 텍스트 데이터 샘플 확인
print("\n=== 텍스트 데이터 샘플 ===")
for i, col in enumerate(text_columns):
    print(f"\n{col} 샘플 (첫 150자):")
    sample_text = data[col].iloc[0] if pd.notna(data[col].iloc[0]) else "결측값"
    print(f"'{sample_text[:150]}...'")

# Persona 정보 확인
print(f"\n페르소나 분포:")
print(data['Persona_Name'].value_counts())

# 셀 3: 통계적으로 검증된 키워드 사전
VALIDATED_JDR_KEYWORDS = {
    'high_risk_indicators': {
        # 통계적으로 입증된 고위험 키워드
        'burnout': {
            'keywords': ['소진', '번아웃', '탈진', '에너지 고갈', '지쳐간다'],
            'weight': 2.37,  # 오즈비 기반 가중치
            'evidence': 'p<0.001, 매우 강한 통계적 유의성'
        },
        'pressure_stress': {
            'keywords': ['압박', '압박감', '스트레스', '부담감', '긴장감'],
            'weight': 2.35,  # 오즈비 기반 가중치
            'evidence': 'p<0.001, 매우 강한 통계적 유의성'
        },
        'dissatisfaction': {
            'keywords': ['불만', '불만족', '짜증', '화나다', '실망'],
            'weight': 1.50,  # 오즈비 기반 가중치
            'evidence': 'p<0.05, 통계적 유의성'
        }
    },
    
    'medium_risk_indicators': {
        # 경계선상 키워드들 (통계적 유의성은 없지만 오즈비 1 이상)
        'recognition_issues': {
            'keywords': ['인정', '저평가', '과소평가', '무시'],
            'weight': 1.44,
            'evidence': 'p=0.11, 경계선상 유의성'
        },
        'workload_issues': {
            'keywords': ['과로', '초과근무', '업무량', '과중'],
            'weight': 1.24,
            'evidence': 'p=0.21, 경계선상'
        }
    },
    
    'protective_factors': {
        # 퇴사 위험을 낮추는 긍정적 요인들
        'work_life_balance': {
            'keywords': ['워라밸', '일과 삶', '균형', '밸런스'],
            'weight': 0.40,  # 보호 효과 (낮을수록 좋음)
            'evidence': 'p<0.001, 매우 강한 보호 효과'
        },
        'growth_opportunities': {
            'keywords': ['성장', '발전', '승진', '기회', '미래'],
            'weight': 0.58,
            'evidence': 'p<0.001, 강한 보호 효과'
        },
        'social_support': {
            'keywords': ['도움', '지원', '서포트', '협력'],
            'weight': 0.75,
            'evidence': 'p<0.05, 보호 효과'
        }
    }
}

print("=== 통계적으로 검증된 키워드 사전 생성 완료 ===")
print("JD-R 모델 + 실제 데이터 검증 결합")

# 셀 4: 문서 기반 OpenAI API 통합 ValidatedSentioAgent
import openai
import time
import os
from dotenv import load_dotenv

class ValidatedSentioAgent:
    """문서 기반 OpenAI API 통합 Sentio 에이전트"""
    
    def __init__(self, openai_api_key=None):
        self.keywords = VALIDATED_JDR_KEYWORDS
        self.optimal_threshold = 0.45
        
        # OpenAI API 설정
        if openai_api_key:
            self.client = openai.OpenAI(api_key=openai_api_key)
            self.use_api = True
        else:
            # .env 파일에서 API 키 로드 시도
            load_dotenv()
            api_key_from_env = os.getenv("OPENAI_API_KEY")
            if api_key_from_env and api_key_from_env != "your-api-key-here":
                self.client = openai.OpenAI(api_key=api_key_from_env)
                self.use_api = True
                print("✅ .env 파일에서 OpenAI API 키를 성공적으로 로드했습니다.")
            else:
                self.client = None
                self.use_api = False
                print("⚠️  OpenAI API 키가 없어서 기본 템플릿 기반 해석을 사용합니다.")
        
    def calculate_jdr_scores(self, text):
        """JD-R 점수 계산 (기존과 동일)"""
        if pd.isna(text) or not isinstance(text, str):
            return {
                'job_demands_score': 0.0,
                'job_resources_deficiency_score': 0.5,
                'detected_keywords': []
            }
        
        detected_keywords = []
        
        # 직무 요구 점수
        job_demands_raw = 0.0
        demands_categories_found = set()
        
        for category_name, category_data in self.keywords['high_risk_indicators'].items():
            if category_name not in demands_categories_found:
                for keyword in category_data['keywords']:
                    if keyword in text:
                        job_demands_raw += category_data['weight']
                        demands_categories_found.add(category_name)
                        detected_keywords.append(keyword)
                        break
        
        for category_name, category_data in self.keywords['medium_risk_indicators'].items():
            if category_name not in demands_categories_found:
                for keyword in category_data['keywords']:
                    if keyword in text:
                        job_demands_raw += category_data['weight']
                        demands_categories_found.add(category_name)
                        detected_keywords.append(keyword)
                        break
        
        # 직무 자원 결핍 점수
        base_deficiency = 0.72
        protection_found = 0.0
        protection_categories_found = set()
        
        for category_name, category_data in self.keywords['protective_factors'].items():
            if category_name not in protection_categories_found:
                for keyword in category_data['keywords']:
                    if keyword in text:
                        protection_strength = (1 - category_data['weight']) * 1.2
                        protection_found += protection_strength
                        protection_categories_found.add(category_name)
                        detected_keywords.append(keyword)
                        break
        
        protection_found = min(protection_found, 0.6)
        job_resources_deficiency_score = max(0.1, base_deficiency - protection_found)
        
        # 점수 정규화
        max_possible_demands = 6.5
        job_demands_score = min(job_demands_raw / max_possible_demands, 1.0)
        job_resources_deficiency_score = min(max(job_resources_deficiency_score, 0.0), 1.0)
        
        return {
            'job_demands_score': job_demands_score,
            'job_resources_deficiency_score': job_resources_deficiency_score,
            'detected_keywords': list(set(detected_keywords))
        }
    
    def generate_llm_interpretation_with_api(self, jdr_indicators, detected_keywords, psychological_risk_score, original_text):
        """문서 방식의 OpenAI API를 활용한 전문적 LLM 해석 생성"""
        
        # 위험 등급 판정
        if psychological_risk_score > 0.55:
            risk_level = "고위험"
            risk_context = "즉각적인 개입이 필요한 상황"
        elif psychological_risk_score > 0.35:
            risk_level = "잠재위험"
            risk_context = "지속적인 모니터링이 필요한 상황"
        elif psychological_risk_score > 0.2:
            risk_level = "중간위험"
            risk_context = "예방적 관리가 권장되는 상황"
        else:
            risk_level = "저위험"
            risk_context = "현재 상태 유지가 바람직한 상황"
        
        # 상세한 프롬프트 구성 (문서 스타일 반영)
        prompt = f"""
당신은 직장 내 직원의 심리적 상태를 분석하는 전문 HR 컨설턴트이자 조직 심리학 전문가입니다. 
JD-R(Job Demands-Resources) 모델을 기반으로 한 과학적 분석 결과를 바탕으로 해당 직원의 상태에 대한 전문적이고 실용적인 해석을 제공해주세요.

**분석 대상 직원의 심리적 상태 데이터:**
- 종합 위험 점수: {psychological_risk_score:.4f} (0~1 척도, 1에 가까울수록 높은 위험)
- 위험 등급: {risk_level} ({risk_context})
- 직무 요구(Job Demands) 점수: {jdr_indicators['job_demands_score']:.4f}
- 직무 자원 결핍(Job Resources Deficiency) 점수: {jdr_indicators['job_resources_deficiency_score']:.4f}
- 주요 감지 키워드: {', '.join(detected_keywords[:8]) if detected_keywords else '없음'}

**원본 텍스트 샘플 (맥락 이해용):**
"{original_text[:300]}..."

**전문적 해석 요청사항:**
1. **현재 상태 진단**: 해당 직원의 심리적 상태를 JD-R 모델 관점에서 한 문장으로 명확히 진단해주세요.

2. **핵심 위험/보호 요인**: 감지된 키워드를 바탕으로 주요 위험 요인이나 긍정적 보호 요인을 구체적으로 2-3개 언급해주세요.

3. **실무적 권장사항**: HR 담당자나 직속 관리자가 취할 수 있는 구체적이고 실행 가능한 조치를 제안해주세요.

**작성 가이드라인:**
- 전체 응답은 4-5문장 이내로 간결하게 작성
- 전문 용어보다는 실무진이 이해하기 쉬운 표현 사용
- 객관적이고 건설적인 톤 유지
- 직원의 프라이버시와 존엄성을 존중하는 표현 사용

한국어로 응답하고, 조직 심리학적 근거에 기반한 실용적 조언을 제공해주세요.
"""
        
        try:
            # 문서에서 사용한 방식을 표준 API로 변환
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # 비용 효율적이고 실제 존재하는 모델
                messages=[
                    {
                        "role": "system", 
                        "content": "당신은 조직 심리학과 HR 데이터 분석 전문가입니다. JD-R 모델을 기반으로 직원의 심리적 상태와 퇴사 위험을 평가하며, 실무진에게 객관적이고 실용적인 조언을 제공합니다. 항상 직원의 존엄성을 존중하며 건설적인 해석을 제공합니다."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=400,
                temperature=0.3,  # 일관성 있는 전문적 응답을 위해 낮게 설정
                top_p=0.9
            )
            
            generated_interpretation = response.choices[0].message.content.strip()
            
            # 응답 품질 검증 (너무 짧거나 비어있는 경우 처리)
            if len(generated_interpretation) < 50:
                print(f"⚠️  API 응답이 너무 짧습니다. 백업 해석을 사용합니다.")
                return self.generate_fallback_interpretation(jdr_indicators, detected_keywords, psychological_risk_score)
                
            return generated_interpretation
            
        except openai.APIError as e:
            print(f"OpenAI API 오류: {str(e)}")
            return self.generate_fallback_interpretation(jdr_indicators, detected_keywords, psychological_risk_score)
        except openai.RateLimitError as e:
            print(f"API 호출 제한 초과: {str(e)}. 잠시 대기 후 백업 해석을 사용합니다.")
            time.sleep(2)
            return self.generate_fallback_interpretation(jdr_indicators, detected_keywords, psychological_risk_score)
        except Exception as e:
            print(f"예상치 못한 오류: {str(e)}")
            return self.generate_fallback_interpretation(jdr_indicators, detected_keywords, psychological_risk_score)
    
    def generate_fallback_interpretation(self, jdr_indicators, detected_keywords, psychological_risk_score):
        """API 실패 시 사용할 백업 해석 (기존 방식 개선)"""
        
        if psychological_risk_score > 0.55:
            risk_level = "고위험"
            action = "즉각적인 상담 및 개입이 필요한 상황으로 판단됩니다."
        elif psychological_risk_score > 0.35:
            risk_level = "잠재위험"
            action = "지속적인 모니터링과 예방적 관리가 권장됩니다."
        elif psychological_risk_score > 0.2:
            risk_level = "중간위험"
            action = "정기적인 체크업과 스트레스 관리가 도움이 될 것입니다."
        else:
            risk_level = "저위험"
            action = "현재 상태를 지속적으로 유지하시기 바랍니다."
        
        # 위험 요인 분석
        risk_factors = []
        if jdr_indicators['job_demands_score'] > 0.25:
            risk_factors.append("높은 직무 요구")
        if jdr_indicators['job_resources_deficiency_score'] > 0.5:
            risk_factors.append("직무 자원 부족")
            
        # 키워드 분류
        unique_keywords = list(set(detected_keywords))
        high_risk_keywords = [kw for kw in unique_keywords 
                             if any(danger in kw for danger in ['소진', '탈진', '압박', '스트레스', '불만', '과로'])]
        protective_keywords = [kw for kw in unique_keywords 
                              if any(protect in kw for protect in ['성장', '발전', '균형', '워라밸', '도움', '지원', '협력'])]
        
        # 해석 생성
        interpretation = f"분석 결과 {risk_level} 상태로 평가됩니다. "
        
        if high_risk_keywords:
            interpretation += f"주요 위험 요인으로 {', '.join(high_risk_keywords[:2])} 등이 감지되었습니다. "
            
        if protective_keywords:
            interpretation += f"긍정적 요인으로는 {', '.join(protective_keywords[:2])} 등이 확인됩니다. "
            
        if risk_factors:
            interpretation += f"특히 {', '.join(risk_factors)}에 대한 관리가 필요합니다. "
            
        interpretation += action
            
        return interpretation
    
    def analyze_employee_text(self, employee_id, self_review, peer_feedback, weekly_survey):
        """직원 텍스트 분석 (문서 기반 OpenAI API 통합)"""
        
        # 텍스트 결합
        texts = [self_review, peer_feedback, weekly_survey]
        combined_text = ' '.join([str(text) for text in texts if pd.notna(text)])
        
        if not combined_text.strip():
            return {
                'employee_id': employee_id,
                'psychological_risk_score': 0.5,
                'jd_r_indicators': {
                    'job_demands_score': 0.0,
                    'job_resources_deficiency_score': 0.5
                },
                'detected_keywords': [],
                'llm_interpretation': "분석 결과 중간위험 상태로 평가됩니다. 텍스트 데이터가 부족하여 정확한 분석이 어렵습니다. 추가적인 피드백 수집을 권장합니다.",
                'attrition_pred': 0
            }
        
        # JD-R 지표 계산
        jdr_result = self.calculate_jdr_scores(combined_text)
        
        # 심리적 위험 점수 계산
        psychological_risk_score = (
            jdr_result['job_demands_score'] * 0.75 +
            jdr_result['job_resources_deficiency_score'] * 0.25
        )
        psychological_risk_score = min(max(psychological_risk_score, 0.0), 1.0)
        
        # LLM 해석 생성 (API 또는 백업)
        if self.use_api:
            llm_interpretation = self.generate_llm_interpretation_with_api(
                {
                    'job_demands_score': jdr_result['job_demands_score'],
                    'job_resources_deficiency_score': jdr_result['job_resources_deficiency_score']
                },
                jdr_result['detected_keywords'],
                psychological_risk_score,
                combined_text
            )
            time.sleep(0.1)  # API 호출 제한 고려
        else:
            llm_interpretation = self.generate_fallback_interpretation(
                {
                    'job_demands_score': jdr_result['job_demands_score'],
                    'job_resources_deficiency_score': jdr_result['job_resources_deficiency_score']
                },
                jdr_result['detected_keywords'],
                psychological_risk_score
            )
        
        # 이진 예측
        attrition_pred = 1 if psychological_risk_score > self.optimal_threshold else 0
        
        return {
            'employee_id': employee_id,
            'psychological_risk_score': psychological_risk_score,
            'jd_r_indicators': {
                'job_demands_score': jdr_result['job_demands_score'],
                'job_resources_deficiency_score': jdr_result['job_resources_deficiency_score']
            },
            'detected_keywords': jdr_result['detected_keywords'],
            'llm_interpretation': llm_interpretation,
            'attrition_pred': attrition_pred
        }

print("문서 기반 OpenAI API 통합 ValidatedSentioAgent 클래스 구현 완료")

# 방법 1: API 키를 직접 전달
OPENAI_API_KEY = "your-actual-api-key-here"  # 실제 API 키 입력
sentio_agent = ValidatedSentioAgent(openai_api_key=OPENAI_API_KEY)

# 방법 2: .env 파일 사용 (권장)
# .env 파일에 OPENAI_API_KEY=your-actual-key 저장 후
sentio_agent = ValidatedSentioAgent()

# 분석 실행 (기존과 동일)
print("고품질 LLM 해석이 포함된 Sentio 분석 실행 중...")
nlp_reports = []

for idx, row in data.iterrows():
    analysis_result = sentio_agent.analyze_employee_text(
        employee_id=row['EmployeeNumber'],
        self_review=row['SELF_REVIEW_text'],
        peer_feedback=row['PEER_FEEDBACK_text'],
        weekly_survey=row['WEEKLY_SURVEY_text']
    )
    nlp_reports.append(analysis_result)
    
    if (idx + 1) % 100 == 0:
        print(f"처리 완료: {idx + 1}/{len(data)} 건")

print("LLM 해석 통합 분석 완료!")

# 셀 5: 페르소나 정보 없이 실제 데이터 분석 실행
sentio_agent = ValidatedSentioAgent()

print("실무용 Sentio 에이전트로 분석 실행 중 (페르소나 정보 미사용)...")
nlp_reports = []

for idx, row in data.iterrows():
    employee_id = row['EmployeeNumber']
    
    # 페르소나 정보 제거 - 텍스트만 사용
    analysis_result = sentio_agent.analyze_employee_text(
        employee_id=employee_id,
        self_review=row['SELF_REVIEW_text'],
        peer_feedback=row['PEER_FEEDBACK_text'],
        weekly_survey=row['WEEKLY_SURVEY_text']
    )
    
    nlp_reports.append(analysis_result)
    
    if (idx + 1) % 100 == 0:
        print(f"처리 완료: {idx + 1}/{len(data)} 건")

print("분석 완료!")

# 결과를 DataFrame으로 변환
nlp_results_df = pd.DataFrame(nlp_reports)
data['Attrition_numeric'] = data['Attrition'].map({'Yes': 1, 'No': 0})

print(f"분석 결과 형태: {nlp_results_df.shape}")
print(f"심리적 위험 점수 통계:")
print(nlp_results_df['psychological_risk_score'].describe())

from collections import Counter

# 상세 점수 분석 (문서 스펙 준수 버전) - 수정됨
print("=== 상세 점수 분석 (문서 스펙 준수 버전) ===")

# 1. 위험 점수 분포 분석
print(f"\n1. 위험 점수 분포:")
print(f"평균 위험 점수: {nlp_results_df['psychological_risk_score'].mean():.4f}")
print(f"표준편차: {nlp_results_df['psychological_risk_score'].std():.4f}")

# 위험 등급별 분포
risk_bins = [0, 0.3, 0.7, 1.0]
risk_labels = ['저위험', '잠재위험', '고위험']
nlp_results_df['risk_level'] = pd.cut(nlp_results_df['psychological_risk_score'], 
                                     bins=risk_bins, labels=risk_labels, include_lowest=True)

print(f"\n위험 등급별 분포:")
risk_distribution = nlp_results_df['risk_level'].value_counts()
for level in risk_labels:
    count = risk_distribution.get(level, 0)
    percentage = count / len(nlp_results_df) * 100
    print(f"{level}: {count}명 ({percentage:.1f}%)")

# 2. 실제 퇴사자들의 위험 점수 분석
print(f"\n2. 실제 퇴사 vs 위험 점수:")
attrition_scores = nlp_results_df[data['Attrition'] == 'Yes']['psychological_risk_score']
no_attrition_scores = nlp_results_df[data['Attrition'] == 'No']['psychological_risk_score']

print(f"퇴사자 평균 위험 점수: {attrition_scores.mean():.4f}")
print(f"비퇴사자 평균 위험 점수: {no_attrition_scores.mean():.4f}")
print(f"차이: {attrition_scores.mean() - no_attrition_scores.mean():.4f}")

# 3. JD-R 지표별 기여도 분석 (문서 스펙)
print(f"\n3. JD-R 지표별 점수 기여도:")

# JD-R 지표 추출
jdr_data = pd.DataFrame(list(nlp_results_df['jd_r_indicators']))

print(f"직무 요구(Job Demands) 점수:")
print(f"  평균: {jdr_data['job_demands_score'].mean():.4f}")
print(f"  표준편차: {jdr_data['job_demands_score'].std():.4f}")
print(f"  위험 수준 (>0.25): {(jdr_data['job_demands_score'] > 0.25).sum()}명")

print(f"직무 자원 결핍(Job Resources Deficiency) 점수:")
print(f"  평균: {jdr_data['job_resources_deficiency_score'].mean():.4f}")
print(f"  표준편차: {jdr_data['job_resources_deficiency_score'].std():.4f}")
print(f"  위험 수준 (>0.5): {(jdr_data['job_resources_deficiency_score'] > 0.5).sum()}명")

# 퇴사자 vs 비퇴사자 JD-R 지표 비교
print(f"\n퇴사자 vs 비퇴사자 JD-R 지표 비교:")
attrition_mask = data['Attrition'] == 'Yes'
no_attrition_mask = data['Attrition'] == 'No'

attrition_jdr = jdr_data[attrition_mask]
no_attrition_jdr = jdr_data[no_attrition_mask]

print(f"직무 요구 점수:")
print(f"  퇴사자: {attrition_jdr['job_demands_score'].mean():.4f}")
print(f"  비퇴사자: {no_attrition_jdr['job_demands_score'].mean():.4f}")

print(f"직무 자원 결핍 점수:")
print(f"  퇴사자: {attrition_jdr['job_resources_deficiency_score'].mean():.4f}")
print(f"  비퇴사자: {no_attrition_jdr['job_resources_deficiency_score'].mean():.4f}")

# 4. 가장 위험한 직원들 분석
print(f"\n4. 최고 위험 직원들 (상위 20명):")
top_risk = nlp_results_df.nlargest(20, 'psychological_risk_score')

print("순위 | Employee ID | 실제퇴사 | 위험점수 | 직무요구 | 자원결핍 | 주요 키워드")
print("-" * 90)
for i, (idx, row) in enumerate(top_risk.iterrows()):
    emp_id = row['employee_id']
    actual_attrition = data[data['EmployeeNumber'] == emp_id]['Attrition'].iloc[0]
    risk_score = row['psychological_risk_score']
    job_demands = row['jd_r_indicators']['job_demands_score']
    resource_deficiency = row['jd_r_indicators']['job_resources_deficiency_score']
    keywords = ', '.join(row['detected_keywords'][:3])  # 상위 3개
    print(f"{i+1:3d}  | {emp_id:11} | {actual_attrition:7} | {risk_score:8.4f} | {job_demands:8.4f} | {resource_deficiency:8.4f} | {keywords}")

# 5. 키워드별 실제 퇴사율 검증
print(f"\n5. 주요 키워드별 실제 퇴사율:")

# 모든 키워드 수집
all_keywords = []
for keywords_list in nlp_results_df['detected_keywords']:
    all_keywords.extend(keywords_list)

keyword_frequency = Counter(all_keywords)

# 키워드가 검출된 직원들의 실제 퇴사율 계산
keyword_attrition_rates = {}

for keyword, count in keyword_frequency.most_common(15):
    # 해당 키워드가 검출된 직원들 찾기
    employees_with_keyword = []
    for i, keywords_list in enumerate(nlp_results_df['detected_keywords']):
        if keyword in keywords_list:
            employees_with_keyword.append(nlp_results_df.iloc[i]['employee_id'])
    
    if employees_with_keyword:
        # 실제 퇴사율 계산
        actual_attritions = data[data['EmployeeNumber'].isin(employees_with_keyword)]['Attrition']
        attrition_rate = (actual_attritions == 'Yes').sum() / len(actual_attritions)
        keyword_attrition_rates[keyword] = {
            'count': count,
            'attrition_rate': attrition_rate,
            'employees': len(employees_with_keyword)
        }

print("키워드 | 검출횟수 | 직원수 | 실제퇴사율 | 전체퇴사율대비")
print("-" * 65)
overall_attrition_rate = (data['Attrition'] == 'Yes').sum() / len(data)

for keyword, stats in sorted(keyword_attrition_rates.items(), key=lambda x: x[1]['attrition_rate'], reverse=True):
    ratio = stats['attrition_rate'] / overall_attrition_rate if overall_attrition_rate > 0 else 0
    print(f"{keyword:15} | {stats['count']:8d} | {stats['employees']:6d} | {stats['attrition_rate']:9.1%} | {ratio:11.2f}x")

# 6. LLM 해석 분석
print(f"\n6. LLM 해석 분석:")

# 위험 등급별 해석 샘플
for risk_level in ['고위험', '잠재위험', '저위험']:
    mask = nlp_results_df['risk_level'] == risk_level
    sample_interpretations = nlp_results_df[mask]['llm_interpretation'].head(3)
    
    print(f"\n{risk_level} 직원 해석 샘플:")
    for i, interpretation in enumerate(sample_interpretations):
        print(f"  {i+1}. {interpretation}")

# 7. JD-R 모델 검증 (점수 계산 로직) - 수정됨
print(f"\n7. JD-R 모델 기반 점수 계산 검증 (상위 10명 샘플):")
print("Employee ID | 최종점수 | 직무요구 | 자원결핍 | 계산식(75:25) | 검출키워드")
print("-" * 100)

for i in range(10):
    emp_data = nlp_results_df.iloc[i]
    emp_id = emp_data['employee_id']
    final_score = emp_data['psychological_risk_score']
    job_demands = emp_data['jd_r_indicators']['job_demands_score']
    resource_def = emp_data['jd_r_indicators']['job_resources_deficiency_score']
    
    # 점수 계산 공식 검증 (올바른 75:25 비율) - 수정됨
    calculated_score = job_demands * 0.75 + resource_def * 0.25
    keywords_str = ', '.join(emp_data['detected_keywords'][:2])
    
    print(f"{emp_id:11} | {final_score:8.4f} | {job_demands:8.4f} | {resource_def:8.4f} | {calculated_score:11.4f} | {keywords_str}")

# 8. 페르소나별 위험도 분석
print(f"\n8. 페르소나별 위험도 분석:")

persona_analysis = []
for persona in data['Persona_Name'].unique():
    persona_mask = data['Persona_Name'] == persona
    persona_indices = data[persona_mask].index
    
    persona_scores = nlp_results_df.iloc[persona_indices]['psychological_risk_score']
    persona_jdr = jdr_data.iloc[persona_indices]
    actual_attrition_rate = (data[persona_mask]['Attrition'] == 'Yes').sum() / persona_mask.sum()
    
    persona_analysis.append({
        'persona': persona,
        'count': persona_mask.sum(),
        'avg_risk_score': persona_scores.mean(),
        'avg_job_demands': persona_jdr['job_demands_score'].mean(),
        'avg_resource_def': persona_jdr['job_resources_deficiency_score'].mean(),
        'actual_attrition_rate': actual_attrition_rate
    })

# 위험도 순으로 정렬
persona_analysis.sort(key=lambda x: x['avg_risk_score'], reverse=True)

print("페르소나 | 인원 | 평균위험점수 | 직무요구 | 자원결핍 | 실제퇴사율")
print("-" * 75)
for analysis in persona_analysis:
    print(f"{analysis['persona'][:15]:15} | {analysis['count']:4d} | {analysis['avg_risk_score']:10.4f} | {analysis['avg_job_demands']:8.4f} | {analysis['avg_resource_def']:8.4f} | {analysis['actual_attrition_rate']:9.1%}")