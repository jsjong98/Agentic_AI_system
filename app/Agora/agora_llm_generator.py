# -*- coding: utf-8 -*-
"""
Agora LLM Generator
시장 분석 결과에 대한 LLM 기반 해석 생성 모듈
"""

import openai
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class AgoraLLMGenerator:
    """LLM 기반 시장 분석 해석 생성 클래스"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # 실제 사용 가능한 모델로 변경
        
        # 직무별 시장 컨텍스트 (Agora.ipynb와 동일한 구조)
        self.market_context = {
            'Sales': {
                'market_volatility': 'high',
                'competition_level': 'intense',
                'opportunity_keywords': ['신규 시장 개척', '고객사 확대', '매출 성장'],
                'risk_factors': ['실적 압박', '경쟁 심화', '목표 달성 부담']
            },
            'Research and Development': {
                'market_volatility': 'medium',
                'competition_level': 'moderate',  
                'opportunity_keywords': ['기술 혁신', '연구 개발', '특허 출원'],
                'risk_factors': ['프로젝트 불확실성', '기술 트렌드 변화', '장기 투자 필요']
            },
            'Human Resources': {
                'market_volatility': 'low',
                'competition_level': 'moderate',
                'opportunity_keywords': ['조직 개발', '인사 제도 개선', '문화 혁신'],
                'risk_factors': ['조직 갈등', '제도 변화 저항', '인건비 압박']
            },
            'Manufacturing': {
                'market_volatility': 'medium', 
                'competition_level': 'high',
                'opportunity_keywords': ['생산성 향상', '자동화 도입', '품질 개선'],
                'risk_factors': ['원자재 가격 변동', '생산 효율성 압박', '기술 혁신 요구']
            }
        }
        
        logger.info("Agora LLM Generator 초기화 완료")
    
    def generate_market_interpretation(self, analysis_result: Dict, use_llm: bool = True) -> str:
        """시장 분석 결과에 대한 해석 생성"""
        
        if not use_llm:
            return self._generate_rule_based_interpretation(analysis_result)
        
        try:
            # LLM 프롬프트 구성
            prompt = self._build_market_analysis_prompt(analysis_result)
            
            # 표준 OpenAI Chat Completions API 사용
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 시장 분석 및 외부 기회 평가 전문가입니다."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            interpretation = response.choices[0].message.content.strip()
            
            # API 호출 제한 고려
            time.sleep(0.1)
            
            logger.debug("LLM 기반 시장 해석 생성 완료")
            return interpretation
            
        except Exception as e:
            logger.error(f"LLM 해석 생성 실패: {e}")
            # 실패 시 규칙 기반 해석으로 대체
            return self._generate_rule_based_interpretation(analysis_result)
    
    def _build_market_analysis_prompt(self, analysis_result: Dict) -> str:
        """Agora.ipynb의 create_market_analysis_prompt 형식을 따른 프롬프트 구성"""
        
        employee_id = analysis_result.get('employee_id', 'unknown')
        job_role = analysis_result.get('job_role', '')
        department = analysis_result.get('department', '')
        job_level = analysis_result.get('job_level', 1)
        current_salary = analysis_result.get('current_salary', 0)
        years_at_company = analysis_result.get('years_at_company', 0)
        market_pressure = analysis_result.get('market_pressure_index', 0)
        compensation_gap = analysis_result.get('compensation_gap', 0)
        job_postings = analysis_result.get('job_postings_count', 0)
        market_data = analysis_result.get('market_data', {})
        
        # 직무별 컨텍스트 추가
        job_category = self._get_job_category(job_role)
        dept_context = self.market_context.get(job_category, self.market_context.get('Sales', {}))
        
        # 위험도 레벨 계산 (Agora.ipynb와 동일한 방식)
        overall_risk = (market_pressure * 0.6 + compensation_gap * 0.4)
        if overall_risk > 0.7:
            risk_level = "높음"
            risk_description = "퇴사 위험이 높은 상황"
        elif overall_risk > 0.4:
            risk_level = "보통" 
            risk_description = "주의 깊은 모니터링이 필요한 상황"
        else:
            risk_level = "낮음"
            risk_description = "안정적인 상황"
        
        # Agora.ipynb의 정확한 프롬프트 구조 적용
        prompt = f"""
당신은 HR 데이터 분석 전문가입니다. 아래 정보를 바탕으로 해당 직원의 외부 시장 분석 결과를 자연스럽고 실무적인 한국어로 해석해주세요.

**직원 기본 정보:**
- 직무: {job_role} ({department} 부서)
- 직급: Level {job_level}
- 현재 월급: ${current_salary:,}
- 재직 기간: {years_at_company}년

**시장 분석 결과:**
- 관련 채용 공고 수: {job_postings}개
- 시장 평균 월급: ${market_data.get('market_avg_salary', 'N/A'):,}
- 시장 압력 지수: {market_pressure} (0~1, 높을수록 외부 기회 많음)
- 보상 격차 지수: {compensation_gap} (0~1, 높을수록 시장 대비 불리)
- 종합 퇴사 위험도: {risk_level} ({overall_risk:.3f})

**업계 특성:**
- 시장 변동성: {dept_context.get('market_volatility', 'medium')}
- 경쟁 수준: {dept_context.get('competition_level', 'medium')}
- 주요 기회 요인: {', '.join(dept_context.get('opportunity_keywords', []))}
- 리스크 요인: {', '.join(dept_context.get('risk_factors', []))}

**작성 지침:**
1. 약 4-6문장으로 구성해주세요
2. 구체적인 수치보다는 해석과 시사점에 집중해주세요
3. HR 담당자가 실무에서 활용할 수 있는 관점으로 작성해주세요
4. 직원 개인정보나 평가 점수는 언급하지 마세요
5. 퇴사 가능성을 직접적으로 언급하기보다는 "관심 필요", "모니터링 권장" 같은 표현을 사용해주세요

위 정보를 종합하여 이 직원의 외부 시장 상황과 HR 관점에서의 시사점을 자연스럽게 해석해주세요.
"""
        
        return prompt.strip()
    
    def _get_job_category(self, job_role: str) -> str:
        """직무를 카테고리로 분류 (Agora.ipynb와 동일한 방식)"""
        
        job_role_lower = job_role.lower()
        
        if 'sales' in job_role_lower or 'executive' in job_role_lower:
            return 'Sales'
        elif 'research' in job_role_lower or 'scientist' in job_role_lower:
            return 'Research and Development'
        elif 'manufacturing' in job_role_lower or 'technician' in job_role_lower:
            return 'Manufacturing'
        elif 'human' in job_role_lower or 'hr' in job_role_lower:
            return 'Human Resources'
        else:
            return 'Sales'  # 기본값을 Sales로 설정 (Agora.ipynb와 동일)
    
    def _generate_rule_based_interpretation(self, analysis_result: Dict) -> str:
        """규칙 기반 해석 생성 (LLM 대체)"""
        
        employee_id = analysis_result.get('employee_id', 'unknown')
        job_role = analysis_result.get('job_role', '')
        market_pressure = analysis_result.get('market_pressure_index', 0)
        compensation_gap = analysis_result.get('compensation_gap', 0)
        job_postings = analysis_result.get('job_postings_count', 0)
        risk_level = analysis_result.get('risk_level', 'MEDIUM')
        market_data = analysis_result.get('market_data', {})
        
        interpretation = f"""
=== 직원 {employee_id} ({job_role}) 시장 분석 해석 ===

📊 현재 시장 상황:
{job_role} 직무의 채용 시장에서 현재 {job_postings}개의 관련 공고가 활발히 게시되어 있습니다. 
시장 트렌드는 '{market_data.get('market_trend', 'STABLE')}'이며, 경쟁 수준은 '{market_data.get('competition_level', 'MEDIUM')}'입니다.

🎯 이직 위험도 분석:
종합 위험도는 '{risk_level}'로 평가됩니다.
- 시장 압력 지수: {market_pressure:.3f}/1.0
- 보상 격차: {compensation_gap:.3f}
"""
        
        # 위험 수준별 상세 분석
        if risk_level == "HIGH":
            interpretation += """
⚠️ 높은 이직 위험 상황입니다:
- 외부 시장에서 해당 직무에 대한 수요가 매우 높습니다
- 현재 보상 수준이 시장 평균보다 낮을 가능성이 있습니다
- 즉각적인 관심과 조치가 필요한 상황입니다

🔧 권장 조치:
1. 급여 및 복리후생 패키지 재검토
2. 상급자와의 긴급 면담 실시
3. 경력 개발 기회 및 승진 계획 논의
4. 업무 환경 개선 방안 검토
5. 정기적인 만족도 조사 실시

📈 모니터링 포인트:
- 월 1회 만족도 체크
- 시장 급여 수준 분기별 모니터링
- 업무 성과 및 참여도 변화 관찰"""
        
        elif risk_level == "MEDIUM":
            interpretation += """
⚠️ 중간 수준의 이직 위험이 있습니다:
- 시장 상황을 지속적으로 모니터링해야 합니다
- 예방적 차원의 관리가 필요합니다

🔧 권장 조치:
1. 정기적인 피드백 세션 진행
2. 업무 만족도 및 성장 계획 논의
3. 스킬 개발 및 교육 기회 제공
4. 팀 내 역할 및 책임 재검토
5. 분기별 성과 리뷰 강화

📈 모니터링 포인트:
- 분기별 만족도 조사
- 시장 동향 반기별 점검
- 성과 및 참여도 변화 추적"""
        
        else:
            interpretation += """
✅ 현재 이직 위험이 낮은 안정적인 상태입니다:
- 현재 조건이 시장 대비 경쟁력이 있습니다
- 지속적인 동기부여에 집중하면 됩니다

🔧 권장 조치:
1. 현재 수준의 보상 및 복리후생 유지
2. 장기적 성장 계획 수립 및 공유
3. 멘토링 및 리더십 개발 기회 제공
4. 새로운 도전 과제 부여
5. 조직 내 영향력 확대 기회 제공

📈 모니터링 포인트:
- 반기별 만족도 확인
- 연간 시장 동향 점검
- 장기 경력 목표 달성도 추적"""
        
        # 시장 트렌드별 추가 조언
        trend = market_data.get('market_trend', 'STABLE')
        if trend == 'HOT':
            interpretation += "\n\n🔥 추가 고려사항: 시장이 매우 활발하여 인재 유치 경쟁이 치열합니다. 특별한 주의가 필요합니다."
        elif trend == 'GROWING':
            interpretation += "\n\n📈 추가 고려사항: 성장하는 시장으로 새로운 기회가 많습니다. 경쟁력 있는 조건 제시를 고려하세요."
        elif trend == 'DECLINING':
            interpretation += "\n\n📉 추가 고려사항: 시장이 둔화되고 있어 현재 포지션의 가치를 강조할 좋은 시점입니다."
        
        return interpretation.strip()
    
    def generate_batch_summary(self, batch_results: List[Dict]) -> str:
        """배치 분석 결과 요약 생성"""
        
        if not batch_results:
            return "분석 결과가 없습니다."
        
        # 통계 계산
        total_count = len(batch_results)
        high_risk_count = sum(1 for r in batch_results if r.get('risk_level') == 'HIGH')
        medium_risk_count = sum(1 for r in batch_results if r.get('risk_level') == 'MEDIUM')
        low_risk_count = sum(1 for r in batch_results if r.get('risk_level') == 'LOW')
        
        avg_market_pressure = sum(r.get('market_pressure_index', 0) for r in batch_results) / total_count
        avg_compensation_gap = sum(r.get('compensation_gap', 0) for r in batch_results) / total_count
        
        # 직무별 위험도 분석
        job_risk_analysis = {}
        for result in batch_results:
            job_role = result.get('job_role', 'Unknown')
            risk_level = result.get('risk_level', 'MEDIUM')
            
            if job_role not in job_risk_analysis:
                job_risk_analysis[job_role] = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'total': 0}
            
            job_risk_analysis[job_role][risk_level] += 1
            job_risk_analysis[job_role]['total'] += 1
        
        # 요약 보고서 생성
        summary = f"""
=== 배치 시장 분석 종합 보고서 ===

📊 전체 현황:
- 분석 대상: {total_count}명
- 고위험군: {high_risk_count}명 ({high_risk_count/total_count*100:.1f}%)
- 중위험군: {medium_risk_count}명 ({medium_risk_count/total_count*100:.1f}%)
- 저위험군: {low_risk_count}명 ({low_risk_count/total_count*100:.1f}%)

📈 평균 지표:
- 평균 시장 압력 지수: {avg_market_pressure:.3f}/1.0
- 평균 보상 격차: {avg_compensation_gap:.3f}

🎯 직무별 위험도 분석:
"""
        
        # 직무별 상세 분석
        for job_role, stats in job_risk_analysis.items():
            high_pct = stats['HIGH'] / stats['total'] * 100
            summary += f"\n• {job_role}: 고위험 {stats['HIGH']}명 ({high_pct:.1f}%), 총 {stats['total']}명"
        
        # 우선 조치 대상
        high_risk_jobs = [job for job, stats in job_risk_analysis.items() 
                         if stats['HIGH'] / stats['total'] > 0.3]
        
        if high_risk_jobs:
            summary += f"\n\n🚨 우선 조치 필요 직무: {', '.join(high_risk_jobs)}"
        
        summary += f"\n\n📅 분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return summary.strip()
