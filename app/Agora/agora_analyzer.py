# -*- coding: utf-8 -*-
"""
Agora Market Analyzer
시장 분석 및 직원별 위험도 평가 모듈
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from agora_processor import AgoraMarketProcessor

logger = logging.getLogger(__name__)

class AgoraMarketAnalyzer:
    """시장 분석 및 직원별 위험도 평가 클래스"""
    
    def __init__(self, hr_data_path: str):
        self.hr_data_path = hr_data_path
        self.hr_data = None
        self.market_processor = AgoraMarketProcessor()
        
        # 분석 결과 저장소
        self.analysis_results = []
        
        # HR 데이터 로드
        self._load_hr_data()
        
        logger.info("Agora Market Analyzer 초기화 완료")
    
    def _load_hr_data(self):
        """HR 데이터 로드"""
        try:
            if Path(self.hr_data_path).exists():
                self.hr_data = pd.read_csv(self.hr_data_path)
                logger.info(f"HR 데이터 로드 완료: {len(self.hr_data)}명")
            else:
                logger.warning(f"HR 데이터 파일을 찾을 수 없습니다: {self.hr_data_path}")
        except Exception as e:
            logger.error(f"HR 데이터 로드 실패: {e}")
    
    def analyze_employee_market(self, employee_data: Dict, include_llm: bool = False) -> Dict:
        """개별 직원 시장 분석"""
        
        try:
            employee_id = employee_data.get('EmployeeNumber', 'unknown')
            job_role = employee_data.get('JobRole', '')
            department = employee_data.get('Department', '')
            
            # 숫자 필드들의 안전한 타입 변환
            try:
                monthly_income = float(employee_data.get('MonthlyIncome', 0))
            except (ValueError, TypeError):
                logger.warning(f"MonthlyIncome 변환 실패: {employee_data.get('MonthlyIncome')}, 기본값 0 사용")
                monthly_income = 0
            
            try:
                years_at_company = int(employee_data.get('YearsAtCompany', 0))
            except (ValueError, TypeError):
                logger.warning(f"YearsAtCompany 변환 실패: {employee_data.get('YearsAtCompany')}, 기본값 0 사용")
                years_at_company = 0
            
            try:
                job_satisfaction = int(employee_data.get('JobSatisfaction', 3))
            except (ValueError, TypeError):
                logger.warning(f"JobSatisfaction 변환 실패: {employee_data.get('JobSatisfaction')}, 기본값 3 사용")
                job_satisfaction = 3
            
            logger.debug(f"직원 {employee_id} 시장 분석 시작")
            
            # 시장 압력 지수 계산
            market_pressure_index = self.market_processor.calculate_market_pressure_index(
                job_role, monthly_income
            )
            
            # 보상 격차 계산
            compensation_gap = self.market_processor.calculate_compensation_gap(
                job_role, monthly_income
            )
            
            # 채용 공고 수 조회
            market_data = self.market_processor.collect_job_postings(job_role)
            job_postings_count = market_data.get('job_postings', 0)
            
            # 시장 경쟁력 평가
            market_competitiveness = self._evaluate_market_competitiveness(
                market_pressure_index, compensation_gap, job_postings_count
            )
            
            # 위험 수준 및 점수 계산
            risk_result = self._determine_risk_level(
                market_pressure_index, compensation_gap, job_satisfaction, years_at_company
            )
            
            # 분석 결과 구성
            analysis_result = {
                'employee_id': str(employee_id),
                'job_role': job_role,
                'department': department,
                'market_pressure_index': round(market_pressure_index, 3),
                'compensation_gap': round(compensation_gap, 3),
                'job_postings_count': job_postings_count,
                'market_competitiveness': market_competitiveness,
                'agora_score': round(risk_result['risk_score'], 3),  # 0~1 범위 점수
                'risk_level': risk_result['risk_level'],
                'market_data': {
                    'avg_salary': market_data.get('avg_salary', 0),
                    'salary_range': market_data.get('salary_range', {}),
                    'market_trend': market_data.get('market_trend', 'STABLE'),
                    'competition_level': market_data.get('competition_level', 'MEDIUM')
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # LLM 해석 추가 (선택적)
            if include_llm:
                analysis_result['llm_interpretation'] = self._generate_rule_based_interpretation(
                    analysis_result
                )
            
            logger.debug(f"직원 {employee_id} 시장 분석 완료: 위험도 {risk_result['risk_level']}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"직원 시장 분석 실패: {e}")
            raise
    
    def _evaluate_market_competitiveness(self, market_pressure: float, compensation_gap: float, 
                                       job_postings: int) -> str:
        """시장 경쟁력 평가"""
        
        # 종합 점수 계산
        competitiveness_score = (
            market_pressure * 0.4 +           # 시장 압력 40%
            max(0, compensation_gap) * 0.4 +  # 보상 격차 40% (양수만)
            min(job_postings / 200.0, 1.0) * 0.2  # 채용 활성도 20%
        )
        
        if competitiveness_score >= 0.7:
            return "VERY_HIGH"
        elif competitiveness_score >= 0.5:
            return "HIGH"
        elif competitiveness_score >= 0.3:
            return "MEDIUM"
        elif competitiveness_score >= 0.1:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _determine_risk_level(self, market_pressure: float, compensation_gap: float, 
                            job_satisfaction: int, years_at_company: int) -> Dict:
        """위험 수준 및 점수 계산"""
        
        # 위험 점수 계산 (0~1 범위)
        risk_score = 0.0
        
        # 시장 압력 (0-0.4)
        risk_score += market_pressure * 0.4
        
        # 보상 격차 (0-0.3)
        if compensation_gap > 0:  # 시장 대비 낮은 급여
            risk_score += compensation_gap * 0.3
        
        # 직무 만족도 (0-0.2)
        satisfaction_risk = (5 - job_satisfaction) / 4.0  # 1-5 → 1-0
        risk_score += satisfaction_risk * 0.2
        
        # 재직 기간 (0-0.1)
        if years_at_company < 2:
            tenure_risk = 0.1  # 신입은 높은 위험
        elif years_at_company > 10:
            tenure_risk = 0.05  # 장기 근속자는 중간 위험
        else:
            tenure_risk = 0.02  # 중간 경력은 낮은 위험
        
        risk_score += tenure_risk
        
        # 0~1 범위로 정규화
        risk_score = max(0.0, min(1.0, risk_score))
        
        # 위험 수준 분류
        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
            
        return {
            'risk_score': risk_score,  # 0~1 범위의 agora_score
            'risk_level': risk_level   # 문자열 레벨
        }
    
    def _generate_rule_based_interpretation(self, analysis_result: Dict) -> str:
        """규칙 기반 해석 생성"""
        
        employee_id = analysis_result['employee_id']
        job_role = analysis_result['job_role']
        market_pressure = analysis_result['market_pressure_index']
        compensation_gap = analysis_result['compensation_gap']
        job_postings = analysis_result['job_postings_count']
        risk_level = analysis_result['risk_level']
        market_data = analysis_result['market_data']
        
        interpretation = f"""
=== 직원 {employee_id} ({job_role}) 시장 분석 결과 ===

📊 시장 현황:
- 관련 채용 공고: {job_postings}개
- 시장 평균 급여: {market_data['avg_salary']:,}원
- 시장 트렌드: {market_data['market_trend']}
- 경쟁 수준: {market_data['competition_level']}

📈 위험도 분석:
- 시장 압력 지수: {market_pressure:.3f}/1.0
- 보상 격차: {compensation_gap:.3f} ({'시장 대비 낮음' if compensation_gap > 0 else '시장 대비 높음'})
- 종합 위험도: {risk_level}

💡 해석 및 권장사항:
"""
        
        # 위험 수준별 해석
        if risk_level == "HIGH":
            interpretation += """🚨 높은 이직 위험:
- 시장에서 해당 직무에 대한 수요가 높습니다
- 현재 보상 수준이 시장 대비 낮을 가능성이 있습니다
- 적극적인 리텐션 전략이 필요합니다

권장 조치:
- 급여 및 복리후생 재검토
- 경력 개발 기회 제공
- 정기적인 만족도 조사 실시"""
        
        elif risk_level == "MEDIUM":
            interpretation += """⚠️ 중간 수준 위험:
- 시장 상황을 지속적으로 모니터링해야 합니다
- 예방적 차원의 관리가 필요합니다

권장 조치:
- 정기적인 피드백 세션
- 업무 환경 개선 검토
- 스킬 개발 지원"""
        
        else:
            interpretation += """✅ 낮은 이직 위험:
- 현재 상태가 안정적입니다
- 지속적인 동기부여에 집중하세요

권장 조치:
- 현재 수준 유지
- 장기적 성장 계획 수립
- 멘토링 기회 제공"""
        
        # 시장 트렌드별 추가 조언
        trend = market_data['market_trend']
        if trend == 'HOT':
            interpretation += "\n\n🔥 시장이 매우 활발합니다. 인재 유지에 특별한 주의가 필요합니다."
        elif trend == 'GROWING':
            interpretation += "\n\n📈 성장하는 시장입니다. 경쟁력 있는 조건 제시를 고려하세요."
        elif trend == 'DECLINING':
            interpretation += "\n\n📉 시장이 둔화되고 있습니다. 현재 포지션의 가치를 강조하세요."
        
        return interpretation.strip()
    
    def batch_analyze_market(self, employees_data: List[Dict], include_llm: bool = False) -> List[Dict]:
        """배치 시장 분석"""
        
        logger.info(f"배치 시장 분석 시작: {len(employees_data)}명")
        
        results = []
        
        try:
            # 순차 처리 (API 제한 고려)
            for i, employee_data in enumerate(employees_data):
                try:
                    result = self.analyze_employee_market(employee_data, include_llm)
                    results.append(result)
                    
                    # 진행 상황 로깅
                    if (i + 1) % 10 == 0:
                        logger.info(f"배치 분석 진행: {i + 1}/{len(employees_data)} 완료")
                    
                except Exception as e:
                    employee_id = employee_data.get('EmployeeNumber', 'unknown')
                    logger.error(f"직원 {employee_id} 분석 실패: {e}")
                    
                    # 실패한 직원에 대한 기본 결과 생성
                    failed_result = {
                        'employee_id': str(employee_id),
                        'job_role': employee_data.get('JobRole', 'Unknown'),
                        'department': employee_data.get('Department', 'Unknown'),
                        'market_pressure_index': 0.0,
                        'compensation_gap': 0.0,
                        'job_postings_count': 0,
                        'market_competitiveness': 'UNKNOWN',
                        'agora_score': 0.0,
                        'risk_level': 'UNKNOWN',
                        'error': str(e),
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                    results.append(failed_result)
                    continue
            
            logger.info(f"배치 시장 분석 완료: {len(results)}/{len(employees_data)} 성공")
            return results
            
        except Exception as e:
            logger.error(f"배치 시장 분석 실패: {e}")
            raise
    
    def analyze_market_trends(self) -> Dict:
        """전체 시장 트렌드 분석"""
        
        try:
            if self.hr_data is None:
                raise ValueError("HR 데이터가 로드되지 않았습니다")
            
            # 직무별 시장 데이터 수집
            job_roles = self.hr_data['JobRole'].unique()
            market_trends = {}
            
            for job_role in job_roles:
                try:
                    market_data = self.market_processor.collect_job_postings(job_role)
                    market_trends[job_role] = {
                        'job_postings': market_data['job_postings'],
                        'avg_salary': market_data['avg_salary'],
                        'market_trend': market_data['market_trend'],
                        'competition_level': market_data['competition_level']
                    }
                except Exception as e:
                    logger.warning(f"직무 {job_role} 시장 데이터 수집 실패: {e}")
                    continue
            
            # 전체 통계 계산
            total_postings = sum(data['job_postings'] for data in market_trends.values())
            avg_salary_overall = np.mean([data['avg_salary'] for data in market_trends.values()])
            
            # 트렌드 분포
            trend_distribution = {}
            for data in market_trends.values():
                trend = data['market_trend']
                trend_distribution[trend] = trend_distribution.get(trend, 0) + 1
            
            # 경쟁 수준 분포
            competition_distribution = {}
            for data in market_trends.values():
                competition = data['competition_level']
                competition_distribution[competition] = competition_distribution.get(competition, 0) + 1
            
            # 결과 구성
            trends_analysis = {
                'overall_statistics': {
                    'total_job_postings': total_postings,
                    'average_salary': round(avg_salary_overall, 0),
                    'analyzed_job_roles': len(market_trends)
                },
                'trend_distribution': trend_distribution,
                'competition_distribution': competition_distribution,
                'job_role_details': market_trends,
                'market_insights': self._generate_market_insights(market_trends),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return trends_analysis
            
        except Exception as e:
            logger.error(f"시장 트렌드 분석 실패: {e}")
            raise
    
    def _generate_market_insights(self, market_trends: Dict) -> List[str]:
        """시장 인사이트 생성"""
        
        insights = []
        
        # 가장 활발한 직무
        most_active_job = max(market_trends.items(), key=lambda x: x[1]['job_postings'])
        insights.append(f"가장 활발한 채용 시장: {most_active_job[0]} ({most_active_job[1]['job_postings']}개 공고)")
        
        # 가장 높은 급여 직무
        highest_salary_job = max(market_trends.items(), key=lambda x: x[1]['avg_salary'])
        insights.append(f"최고 급여 직무: {highest_salary_job[0]} (평균 {highest_salary_job[1]['avg_salary']:,}원)")
        
        # HOT 트렌드 직무 수
        hot_jobs = [job for job, data in market_trends.items() if data['market_trend'] == 'HOT']
        if hot_jobs:
            insights.append(f"급성장 직무: {', '.join(hot_jobs)}")
        
        # 높은 경쟁 직무 수
        high_competition_jobs = [job for job, data in market_trends.items() if data['competition_level'] in ['HIGH', 'VERY_HIGH']]
        if high_competition_jobs:
            insights.append(f"고경쟁 직무: {', '.join(high_competition_jobs[:3])}{'...' if len(high_competition_jobs) > 3 else ''}")
        
        return insights
    
    def analyze_competitiveness(self, employee_data: Dict) -> Dict:
        """개별 직원 경쟁력 분석"""
        
        try:
            job_role = employee_data.get('JobRole', '')
            monthly_income = employee_data.get('MonthlyIncome', 0)
            years_at_company = employee_data.get('YearsAtCompany', 0)
            education = employee_data.get('Education', 3)
            job_satisfaction = employee_data.get('JobSatisfaction', 3)
            
            # 시장 데이터 수집
            market_data = self.market_processor.collect_job_postings(job_role)
            
            # 경쟁력 지표 계산
            salary_competitiveness = monthly_income / market_data['avg_salary'] if market_data['avg_salary'] > 0 else 1.0
            experience_score = min(years_at_company / 10.0, 1.0)  # 10년을 만점으로
            education_score = education / 5.0  # 5를 만점으로
            satisfaction_score = job_satisfaction / 4.0  # 4를 만점으로
            
            # 종합 경쟁력 점수
            competitiveness_score = (
                salary_competitiveness * 0.3 +
                experience_score * 0.3 +
                education_score * 0.2 +
                satisfaction_score * 0.2
            )
            
            # 경쟁력 등급
            if competitiveness_score >= 0.8:
                competitiveness_grade = "EXCELLENT"
            elif competitiveness_score >= 0.6:
                competitiveness_grade = "GOOD"
            elif competitiveness_score >= 0.4:
                competitiveness_grade = "AVERAGE"
            else:
                competitiveness_grade = "BELOW_AVERAGE"
            
            # 분석 결과
            analysis = {
                'employee_id': str(employee_data.get('EmployeeNumber', 'unknown')),
                'job_role': job_role,
                'competitiveness_score': round(competitiveness_score, 3),
                'competitiveness_grade': competitiveness_grade,
                'detailed_scores': {
                    'salary_competitiveness': round(salary_competitiveness, 3),
                    'experience_score': round(experience_score, 3),
                    'education_score': round(education_score, 3),
                    'satisfaction_score': round(satisfaction_score, 3)
                },
                'market_comparison': {
                    'current_salary': monthly_income,
                    'market_avg_salary': market_data['avg_salary'],
                    'salary_gap': market_data['avg_salary'] - monthly_income,
                    'market_trend': market_data['market_trend']
                },
                'recommendations': self._generate_competitiveness_recommendations(
                    competitiveness_score, salary_competitiveness, experience_score
                ),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"경쟁력 분석 실패: {e}")
            raise
    
    def _generate_competitiveness_recommendations(self, overall_score: float, 
                                                salary_score: float, experience_score: float) -> List[str]:
        """경쟁력 개선 권장사항 생성"""
        
        recommendations = []
        
        if overall_score < 0.6:
            recommendations.append("전반적인 경쟁력 강화가 필요합니다.")
        
        if salary_score < 0.8:
            recommendations.append("급여 수준이 시장 대비 낮습니다. 급여 조정을 검토하세요.")
        
        if experience_score < 0.5:
            recommendations.append("경력 개발 기회를 제공하여 전문성을 강화하세요.")
        
        if overall_score >= 0.8:
            recommendations.append("우수한 경쟁력을 보유하고 있습니다. 현재 수준을 유지하세요.")
        
        recommendations.append("정기적인 시장 동향 모니터링을 권장합니다.")
        
        return recommendations
