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
        self.model = "gpt-3.5-turbo"
        
        # 직무별 시장 컨텍스트
        self.market_context = {
            'Sales': {
                'market_volatility': 'high',
                'competition_level': 'intense',
                'key_factors': ['실적', '고객관계', '영업스킬', '시장지식']
            },
            'Research': {
                'market_volatility': 'medium',
                'competition_level': 'high',
                'key_factors': ['기술력', '논문실적', '연구경험', '혁신능력']
            },
            'Manufacturing': {
                'market_volatility': 'low',
                'competition_level': 'medium',
                'key_factors': ['생산효율', '품질관리', '안전관리', '프로세스개선']
            },
            'Healthcare': {
                'market_volatility': 'low',
                'competition_level': 'medium',
                'key_factors': ['전문지식', '환자케어', '의료기술', '규정준수']
            },
            'Human Resources': {
                'market_volatility': 'medium',
                'competition_level': 'medium',
                'key_factors': ['인사관리', '채용', '교육기획', '노무관리']
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
            
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 HR 시장 분석 전문가입니다. 주어진 데이터를 바탕으로 직원의 이직 위험도와 시장 상황을 분석하여 한국어로 명확하고 실용적인 해석을 제공해주세요."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=800,
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
        """시장 분석 프롬프트 구성"""
        
        employee_id = analysis_result.get('employee_id', 'unknown')
        job_role = analysis_result.get('job_role', '')
        department = analysis_result.get('department', '')
        market_pressure = analysis_result.get('market_pressure_index', 0)
        compensation_gap = analysis_result.get('compensation_gap', 0)
        job_postings = analysis_result.get('job_postings_count', 0)
        risk_level = analysis_result.get('risk_level', 'MEDIUM')
        market_data = analysis_result.get('market_data', {})
        
        # 직무별 컨텍스트 추가
        job_category = self._get_job_category(job_role)
        context = self.market_context.get(job_category, {})
        
        prompt = f"""
다음 직원의 시장 분석 결과를 해석해주세요:

=== 직원 정보 ===
- 직원 ID: {employee_id}
- 직무: {job_role}
- 부서: {department}

=== 시장 분석 결과 ===
- 시장 압력 지수: {market_pressure:.3f}/1.0
- 보상 격차: {compensation_gap:.3f} ({'시장 대비 낮음' if compensation_gap > 0 else '시장 대비 높음'})
- 관련 채용 공고 수: {job_postings}개
- 위험 수준: {risk_level}

=== 시장 현황 ===
- 평균 급여: {market_data.get('avg_salary', 0):,}원
- 시장 트렌드: {market_data.get('market_trend', 'STABLE')}
- 경쟁 수준: {market_data.get('competition_level', 'MEDIUM')}

=== 직무 특성 ===
- 시장 변동성: {context.get('market_volatility', 'medium')}
- 경쟁 강도: {context.get('competition_level', 'medium')}
- 핵심 요소: {', '.join(context.get('key_factors', []))}

다음 형식으로 해석해주세요:
1. 현재 시장 상황 요약
2. 이직 위험도 분석
3. 주요 위험 요인
4. 구체적인 권장 조치 (회사 관점)
5. 향후 모니터링 포인트

한국어로 작성하고, 실무진이 바로 활용할 수 있도록 구체적이고 실용적인 내용으로 구성해주세요.
"""
        
        return prompt.strip()
    
    def _get_job_category(self, job_role: str) -> str:
        """직무를 카테고리로 분류"""
        
        job_role_lower = job_role.lower()
        
        if 'sales' in job_role_lower or 'executive' in job_role_lower:
            return 'Sales'
        elif 'research' in job_role_lower or 'scientist' in job_role_lower:
            return 'Research'
        elif 'manufacturing' in job_role_lower or 'technician' in job_role_lower:
            return 'Manufacturing'
        elif 'healthcare' in job_role_lower or 'representative' in job_role_lower:
            return 'Healthcare'
        elif 'human' in job_role_lower or 'hr' in job_role_lower:
            return 'Human Resources'
        else:
            return 'General'
    
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
