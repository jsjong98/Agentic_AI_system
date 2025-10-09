"""
LLM 기반 퇴사 레포트 생성기
각 직원별로 종합적인 퇴사 위험도 분석 레포트를 생성합니다.
"""

import pandas as pd
import numpy as np
import json
import time
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class ReportGenerator:
    """LLM 기반 퇴사 레포트 생성 클래스"""
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.employee_data = None
        self.agent_scores = {}
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.7,
            'high': 1.0
        }
        
        # LangChain LLM 초기화 (강력한 gpt-5 모델 사용)
        self.llm = llm or ChatOpenAI(
            model="gpt-5",
            temperature=0.2,
            max_tokens=2000
        )
        
        logger.info(f"✅ Integration LLM 초기화 완료 - 모델: gpt-5")
        
        # 분석 프롬프트 템플릿
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_analysis_system_prompt()),
            ("human", self._get_analysis_human_prompt())
        ])
        
        # 에이전트별 분석 컨텍스트
        self.agent_contexts = {
            'agora_score': {
                'name': 'Agora (업무 성과)',
                'description': '업무 성과 및 생산성 분석',
                'focus_areas': ['업무 성과', '목표 달성', '생산성', '업무 효율성'],
                'high_risk_indicators': ['낮은 업무 성과', '생산성 저하', '목표 달성률 부족', '업무 품질 문제']
            },
            'chronos_score': {
                'name': 'Chronos (시계열 패턴)',
                'description': '근무 패턴 및 시간 관리 분석',
                'focus_areas': ['근무 패턴', '출퇴근 시간', '야근 빈도', '휴가 사용'],
                'high_risk_indicators': ['불규칙한 근무 패턴', '잦은 지각', '초과 근무 증가', '휴가 미사용']
            },
            'cognita_score': {
                'name': 'Cognita (관계 네트워크)',
                'description': '동료와의 관계 및 네트워크 분석',
                'focus_areas': ['팀워크', '의사소통', '협업', '사회적 관계'],
                'high_risk_indicators': ['사회적 고립', '팀워크 부족', '의사소통 문제', '갈등 상황']
            },
            'sentio_score': {
                'name': 'Sentio (감정 분석)',
                'description': '감정 상태 및 만족도 분석',
                'focus_areas': ['직무 만족도', '감정 상태', '스트레스 수준', '동기부여'],
                'high_risk_indicators': ['부정적 감정 증가', '스트레스 수준 상승', '직무 불만족', '동기부여 저하']
            },
            'structura_score': {
                'name': 'Structura (구조적 요인)',
                'description': '조직 구조 및 환경적 요인 분석',
                'focus_areas': ['조직 적응', '역할 명확성', '승진 기회', '조직 문화'],
                'high_risk_indicators': ['조직 적응 어려움', '역할 모호성', '승진 기회 부족', '조직 문화 부적응']
            }
        }
        
    def load_employee_data(self, data_path: str) -> bool:
        """직원 데이터 로드"""
        try:
            if data_path.endswith('.csv'):
                self.employee_data = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.employee_data = pd.DataFrame(data)
            else:
                raise ValueError("지원하지 않는 파일 형식입니다. CSV 또는 JSON 파일을 사용하세요.")
            
            print(f"✅ 직원 데이터 로드 완료: {len(self.employee_data)}명")
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {str(e)}")
            return False
    
    def set_agent_scores(self, employee_id: str, scores: Dict[str, float]):
        """특정 직원의 에이전트 점수 설정"""
        self.agent_scores[employee_id] = scores
    
    def get_risk_level(self, score: float) -> tuple:
        """점수를 기반으로 위험도 레벨 결정"""
        if score < self.risk_thresholds['low']:
            return "안전군", "LOW", "🟢"
        elif score < self.risk_thresholds['medium']:
            return "주의군", "MEDIUM", "🟡"
        else:
            return "고위험군", "HIGH", "🔴"
    
    def analyze_agent_scores(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """에이전트별 점수 분석"""
        analysis = {}
        
        for agent, score in scores.items():
            if agent in self.agent_contexts:
                risk_level, risk_code, emoji = self.get_risk_level(score)
                context = self.agent_contexts[agent]
                
                analysis[agent] = {
                    'score': score,
                    'risk_level': risk_level,
                    'risk_code': risk_code,
                    'emoji': emoji,
                    'name': context['name'],
                    'description': context['description'],
                    'focus_areas': context['focus_areas'],
                    'indicators': context['high_risk_indicators'] if score >= self.risk_thresholds['medium'] else []
                }
        
        return analysis
    
    def calculate_integrated_risk(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """통합 위험도 계산"""
        if not scores:
            return {'integrated_score': 0, 'risk_level': '안전군', 'confidence': 0}
        
        # 가중 평균 계산 (모든 에이전트 동일 가중치)
        integrated_score = np.mean(list(scores.values()))
        risk_level, risk_code, emoji = self.get_risk_level(integrated_score)
        
        # 신뢰도 계산 (점수 분산이 낮을수록 높은 신뢰도)
        score_variance = np.var(list(scores.values()))
        confidence = max(0, 1 - score_variance)  # 분산이 낮을수록 신뢰도 높음
        
        return {
            'integrated_score': round(integrated_score, 3),
            'risk_level': risk_level,
            'risk_code': risk_code,
            'emoji': emoji,
            'confidence': round(confidence, 3),
            'score_variance': round(score_variance, 3)
        }
    
    def generate_recommendations(self, analysis: Dict[str, Any], integrated_risk: Dict[str, Any]) -> List[str]:
        """맞춤형 권장사항 생성"""
        recommendations = []
        
        risk_code = integrated_risk['risk_code']
        
        if risk_code == 'HIGH':
            recommendations.append("🚨 즉시 개입이 필요한 고위험 직원입니다.")
            recommendations.append("📞 상급자와의 긴급 면담을 권장합니다.")
            
        elif risk_code == 'MEDIUM':
            recommendations.append("⚠️ 주의 깊은 관찰과 지원이 필요합니다.")
            recommendations.append("📋 정기적인 1:1 미팅을 통한 상태 점검을 권장합니다.")
        
        # 에이전트별 구체적 권장사항
        high_risk_agents = [agent for agent, data in analysis.items() 
                          if data['risk_code'] == 'HIGH']
        
        if 'agora_score' in high_risk_agents:
            recommendations.append("📈 업무 성과 개선을 위한 교육 및 멘토링 프로그램 참여")
            recommendations.append("🎯 명확한 목표 설정 및 성과 관리 시스템 도입")
        
        if 'chronos_score' in high_risk_agents:
            recommendations.append("⏰ 근무 패턴 정상화를 위한 시간 관리 교육")
            recommendations.append("🏠 유연 근무제 또는 재택근무 옵션 검토")
        
        if 'cognita_score' in high_risk_agents:
            recommendations.append("👥 팀 빌딩 활동 및 사회적 네트워크 구축 지원")
            recommendations.append("🤝 멘토-멘티 프로그램 참여 권장")
        
        if 'sentio_score' in high_risk_agents:
            recommendations.append("💚 심리 상담 및 스트레스 관리 프로그램 제공")
            recommendations.append("😊 직무 만족도 향상을 위한 업무 조정 검토")
        
        if 'structura_score' in high_risk_agents:
            recommendations.append("🏢 조직 내 역할 명확화 및 커리어 패스 제시")
            recommendations.append("📚 역량 개발 기회 및 승진 경로 안내")
        
        if not recommendations:
            recommendations.append("✅ 현재 안정적인 상태를 유지하고 있습니다.")
            recommendations.append("📊 정기적인 모니터링을 통한 지속적 관리 권장")
        
        return recommendations
    
    def _get_analysis_system_prompt(self) -> str:
        """Integration 분석을 위한 시스템 프롬프트"""
        return """
당신은 조직 내 개인의 심리와 감정을 깊이 이해하는 HR 통합 분석 전문가입니다. 다중 에이전트 시스템(Sentio, Agora, Chronos, Structura, Cognita)을 통한 종합적인 분석 결과를 바탕으로 해당 직원의 퇴사 위험도에 대한 전문적이고 실용적인 해석을 제공해야 합니다.

**분석 맥락:**
- 심리적 상태 (Sentio): 직무 요구-자원 모델 기반 분석
- 시장 상황 (Agora): 외부 기회 및 보상 경쟁력 분석  
- 시계열 패턴 (Chronos): 행동 변화 및 트렌드 분석
- 이탈 위험 (Structura): 구조적 요인 기반 예측
- 네트워크 영향 (Cognita): 조직 내 관계 및 영향력 분석

**작성 가이드라인:**
- 전문적이면서도 실무진이 이해하기 쉬운 표현 사용
- 객관적이고 건설적인 톤 유지
- 직원의 프라이버시와 존엄성을 존중하는 표현 사용
- 각 섹션은 2-3문장으로 간결하게 작성

한국어로 응답하고, 전문적이면서도 실용적인 톤으로 작성해주세요.
"""
    
    def _get_analysis_human_prompt(self) -> str:
        """Integration 분석을 위한 휴먼 프롬프트"""
        return """
**직원 기본 정보:**
- 직원 ID: {employee_id}
- 종합 위험 점수: {risk_score:.4f} (0~1 척도, 1에 가까울수록 높은 위험)
- 위험 등급: {risk_level} ({risk_context})
- 분석 신뢰도: {confidence:.1%}

**다중 에이전트 분석 결과:**
{agent_details}

**요청사항:**
1. 현재 상태를 한 문장으로 간결하게 평가해주세요
2. 가장 중요한 위험 요인 2-3개를 구체적으로 언급해주세요
3. HR 담당자나 직속 관리자가 취할 수 있는 구체적인 조치를 제안해주세요
4. 지속적으로 모니터링해야 할 핵심 지표들을 제시해주세요
5. 전체 응답은 다음 형식으로 구성해주세요:

## 종합 분석
[현재 직원의 전반적인 상황 분석]

## 주요 위험 요인
[가장 중요한 위험 요인들과 그 원인 분석]

## 개선 방안
[구체적이고 실행 가능한 개선 방안들]

## 모니터링 포인트
[지속적으로 관찰해야 할 지표들과 예상 결과]
"""
    
    def _prepare_analysis_input(self, employee_id: str, analysis: Dict[str, Any], 
                               integrated_risk: Dict[str, Any]) -> Dict[str, Any]:
        """분석 입력 데이터 준비"""
        # 위험도 레벨 판정
        risk_score = integrated_risk['integrated_score']
        if risk_score > 0.7:
            risk_level = "고위험"
            risk_context = "즉각적인 개입이 필요한 상황"
        elif risk_score > 0.5:
            risk_level = "잠재위험"
            risk_context = "지속적인 모니터링이 필요한 상황"
        elif risk_score > 0.3:
            risk_level = "중간위험"
            risk_context = "예방적 관리가 권장되는 상황"
        else:
            risk_level = "저위험"
            risk_context = "현재 상태 유지가 바람직한 상황"
        
        # 에이전트별 상세 정보 구성
        agent_details = []
        for agent, data in analysis.items():
            agent_details.append(f"- {data['name']}: {data['score']:.4f} ({data['risk_level']})")
            if data['indicators']:
                agent_details.append(f"  주요 지표: {', '.join(data['indicators'][:3])}")
        
        return {
            'employee_id': employee_id,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_context': risk_context,
            'confidence': integrated_risk['confidence'],
            'agent_details': chr(10).join(agent_details)
        }
    
    async def generate_llm_insights(self, employee_id: str, analysis: Dict[str, Any], 
                             integrated_risk: Dict[str, Any], use_llm: bool = True) -> Dict[str, str]:
        """LLM을 사용한 심층 분석 및 인사이트 생성"""
        
        if not use_llm:
            return self._generate_rule_based_insights(analysis, integrated_risk)
        
        try:
            # 분석 입력 데이터 준비
            analysis_input = self._prepare_analysis_input(employee_id, analysis, integrated_risk)
            
            # LangChain 방식으로 분석 수행
            chain = self.analysis_prompt | self.llm
            response = await chain.ainvoke(analysis_input)
            
            llm_response = response.content.strip()
            
            # 응답 품질 검증
            if len(llm_response) < 30:
                logger.warning(f"⚠️ API 응답이 너무 짧습니다. 백업 해석을 사용합니다.")
                return self._generate_rule_based_insights(analysis, integrated_risk)
            
            # 응답을 섹션별로 분리
            insights = self._parse_llm_response(llm_response)
            
            # API 호출 제한 고려
            time.sleep(0.1)
            
            logger.info(f"LLM 인사이트 생성 완료: {employee_id}")
            return insights
            
        except Exception as e:
            logger.error(f"API 호출 오류 (직원 분석 중): {str(e)}")
            # API 실패 시 백업 해석 사용
            return self._generate_rule_based_insights(analysis, integrated_risk)
    
    def _build_analysis_prompt(self, employee_id: str, analysis: Dict[str, Any], 
                              integrated_risk: Dict[str, Any]) -> str:
        """Sentio/Agora와 일관된 구조화된 분석 프롬프트 구성"""
        
        # 위험도 레벨 판정
        risk_score = integrated_risk['integrated_score']
        if risk_score > 0.7:
            risk_level = "고위험"
            risk_context = "즉각적인 개입이 필요한 상황"
        elif risk_score > 0.5:
            risk_level = "잠재위험"
            risk_context = "지속적인 모니터링이 필요한 상황"
        elif risk_score > 0.3:
            risk_level = "중간위험"
            risk_context = "예방적 관리가 권장되는 상황"
        else:
            risk_level = "저위험"
            risk_context = "현재 상태 유지가 바람직한 상황"
        
        # 에이전트별 상세 정보 구성
        agent_details = []
        for agent, data in analysis.items():
            agent_details.append(f"- {data['name']}: {data['score']:.4f} ({data['risk_level']})")
            if data['indicators']:
                agent_details.append(f"  주요 지표: {', '.join(data['indicators'][:3])}")
        
        # Sentio/Agora와 일관된 구조화된 프롬프트
        full_prompt = f"""
당신은 조직 내 개인의 심리와 감정을 깊이 이해하는 HR 통합 분석 전문가입니다. 다중 에이전트 시스템(Sentio, Agora, Chronos, Structura, Cognita)을 통한 종합적인 분석 결과를 바탕으로 해당 직원의 퇴사 위험도에 대한 전문적이고 실용적인 해석을 제공해야 합니다.

**직원 기본 정보:**
- 직원 ID: {employee_id}
- 종합 위험 점수: {risk_score:.4f} (0~1 척도, 1에 가까울수록 높은 위험)
- 위험 등급: {risk_level} ({risk_context})
- 분석 신뢰도: {integrated_risk['confidence']:.1%}

**다중 에이전트 분석 결과:**
{chr(10).join(agent_details)}

**분석 맥락:**
- 심리적 상태 (Sentio): 직무 요구-자원 모델 기반 분석
- 시장 상황 (Agora): 외부 기회 및 보상 경쟁력 분석  
- 시계열 패턴 (Chronos): 행동 변화 및 트렌드 분석
- 이탈 위험 (Structura): 구조적 요인 기반 예측
- 네트워크 영향 (Cognita): 조직 내 관계 및 영향력 분석

**요청사항:**
1. 현재 상태를 한 문장으로 간결하게 평가해주세요
2. 가장 중요한 위험 요인 2-3개를 구체적으로 언급해주세요
3. HR 담당자나 직속 관리자가 취할 수 있는 구체적인 조치를 제안해주세요
4. 지속적으로 모니터링해야 할 핵심 지표들을 제시해주세요
5. 전체 응답은 다음 형식으로 구성해주세요:

## 종합 분석
[현재 직원의 전반적인 상황 분석]

## 주요 위험 요인
[가장 중요한 위험 요인들과 그 원인 분석]

## 개선 방안
[구체적이고 실행 가능한 개선 방안들]

## 모니터링 포인트
[지속적으로 관찰해야 할 지표들과 예상 결과]

**작성 가이드라인:**
- 전문적이면서도 실무진이 이해하기 쉬운 표현 사용
- 객관적이고 건설적인 톤 유지
- 직원의 프라이버시와 존엄성을 존중하는 표현 사용
- 각 섹션은 2-3문장으로 간결하게 작성

한국어로 응답하고, 전문적이면서도 실용적인 톤으로 작성해주세요.
"""
        
        return full_prompt.strip()
    
    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """LLM 응답을 섹션별로 파싱"""
        
        sections = {
            'comprehensive_analysis': '',
            'risk_factors': '',
            'improvement_plans': '',
            'monitoring_points': ''
        }
        
        current_section = None
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if '## 종합 분석' in line or '종합분석' in line:
                current_section = 'comprehensive_analysis'
            elif '## 주요 위험 요인' in line or '위험 요인' in line or '위험요인' in line:
                current_section = 'risk_factors'
            elif '## 개선 방안' in line or '개선방안' in line or '해결방안' in line:
                current_section = 'improvement_plans'
            elif '## 모니터링 포인트' in line or '모니터링' in line:
                current_section = 'monitoring_points'
            elif current_section and line and not line.startswith('##'):
                sections[current_section] += line + '\n'
        
        # 빈 섹션 처리
        for key, value in sections.items():
            sections[key] = value.strip() if value.strip() else "분석 결과가 없습니다."
        
        return sections
    
    def _generate_rule_based_insights(self, analysis: Dict[str, Any], 
                                    integrated_risk: Dict[str, Any]) -> Dict[str, str]:
        """규칙 기반 인사이트 생성 (LLM 사용 불가 시 대체)"""
        
        risk_code = integrated_risk['risk_code']
        high_risk_agents = [agent for agent, data in analysis.items() if data['risk_code'] == 'HIGH']
        medium_risk_agents = [agent for agent, data in analysis.items() if data['risk_code'] == 'MEDIUM']
        
        # 종합 분석
        if risk_code == 'HIGH':
            comprehensive = "현재 직원은 높은 퇴사 위험도를 보이고 있어 즉시 개입이 필요한 상황입니다."
        elif risk_code == 'MEDIUM':
            comprehensive = "주의가 필요한 상황으로, 예방적 차원의 관리가 필요합니다."
        else:
            comprehensive = "현재 안정적인 상태를 유지하고 있으나 지속적인 모니터링이 필요합니다."
        
        # 위험 요인
        risk_factors = []
        for agent in high_risk_agents:
            if agent in analysis:
                context = self.agent_contexts.get(agent, {})
                risk_factors.append(f"- {context.get('name', agent)}: {context.get('description', '')}")
        
        if not risk_factors:
            risk_factors = ["현재 특별한 위험 요인은 발견되지 않았습니다."]
        
        # 개선 방안
        improvement_plans = []
        if 'agora_score' in high_risk_agents:
            improvement_plans.append("- 업무 성과 향상을 위한 교육 및 멘토링 프로그램 제공")
        if 'sentio_score' in high_risk_agents:
            improvement_plans.append("- 스트레스 관리 및 심리 상담 프로그램 제공")
        if 'cognita_score' in high_risk_agents:
            improvement_plans.append("- 팀 빌딩 및 커뮤니케이션 개선 프로그램 참여")
        
        if not improvement_plans:
            improvement_plans = ["현재 상태 유지를 위한 정기적 면담 및 피드백"]
        
        # 모니터링 포인트
        monitoring = [
            "- 월 1회 정기 면담을 통한 상태 점검",
            "- 에이전트별 점수 변화 추이 모니터링",
            "- 직무 만족도 및 스트레스 수준 정기 조사"
        ]
        
        return {
            'comprehensive_analysis': comprehensive,
            'risk_factors': '\n'.join(risk_factors),
            'improvement_plans': '\n'.join(improvement_plans),
            'monitoring_points': '\n'.join(monitoring)
        }
    
    def generate_employee_report(self, employee_id: str, use_llm: bool = True) -> Dict[str, Any]:
        """개별 직원 레포트 생성"""
        if employee_id not in self.agent_scores:
            return {'error': f'직원 ID {employee_id}의 점수 데이터가 없습니다.'}
        
        # 기본 정보 조회
        employee_info = {}
        if self.employee_data is not None:
            employee_row = self.employee_data[self.employee_data['employee_id'] == employee_id]
            if not employee_row.empty:
                employee_info = employee_row.iloc[0].to_dict()
        
        # 점수 분석
        scores = self.agent_scores[employee_id]
        agent_analysis = self.analyze_agent_scores(scores)
        integrated_risk = self.calculate_integrated_risk(scores)
        recommendations = self.generate_recommendations(agent_analysis, integrated_risk)
        
        # LLM 기반 심층 인사이트 생성
        llm_insights = self.generate_llm_insights(employee_id, agent_analysis, integrated_risk, use_llm)
        
        # 레포트 생성
        report = {
            'employee_id': employee_id,
            'employee_info': employee_info,
            'analysis_date': datetime.now().isoformat(),
            'agent_scores': scores,
            'agent_analysis': agent_analysis,
            'integrated_risk': integrated_risk,
            'recommendations': recommendations,
            'llm_insights': llm_insights,
            'summary': {
                'total_agents': len(scores),
                'high_risk_agents': len([a for a in agent_analysis.values() if a['risk_code'] == 'HIGH']),
                'medium_risk_agents': len([a for a in agent_analysis.values() if a['risk_code'] == 'MEDIUM']),
                'low_risk_agents': len([a for a in agent_analysis.values() if a['risk_code'] == 'LOW']),
                'llm_enabled': use_llm and self.client is not None
            }
        }
        
        return report
    
    def generate_text_report(self, employee_id: str, use_llm: bool = True) -> str:
        """텍스트 형태의 레포트 생성"""
        report_data = self.generate_employee_report(employee_id, use_llm)
        
        if 'error' in report_data:
            return f"❌ 레포트 생성 실패: {report_data['error']}"
        
        # 텍스트 레포트 구성
        text_report = []
        
        # 헤더
        text_report.append("=" * 80)
        text_report.append("🏢 직원 퇴사 위험도 분석 레포트")
        text_report.append("=" * 80)
        text_report.append(f"📅 분석 일시: {report_data['analysis_date'][:19]}")
        text_report.append(f"👤 직원 ID: {employee_id}")
        
        # 기본 정보
        if report_data['employee_info']:
            text_report.append("\n📋 직원 기본 정보")
            text_report.append("-" * 40)
            for key, value in report_data['employee_info'].items():
                if key != 'employee_id':
                    text_report.append(f"   {key}: {value}")
        
        # 통합 위험도
        integrated = report_data['integrated_risk']
        text_report.append(f"\n🎯 통합 위험도 분석")
        text_report.append("-" * 40)
        text_report.append(f"   {integrated['emoji']} 위험도: {integrated['risk_level']}")
        text_report.append(f"   📊 통합 점수: {integrated['integrated_score']}")
        text_report.append(f"   🎲 신뢰도: {integrated['confidence']:.1%}")
        
        # 에이전트별 분석
        text_report.append(f"\n🔍 에이전트별 상세 분석")
        text_report.append("-" * 40)
        
        for agent, analysis in report_data['agent_analysis'].items():
            text_report.append(f"\n   {analysis['emoji']} {analysis['name']}")
            text_report.append(f"      점수: {analysis['score']:.3f} ({analysis['risk_level']})")
            text_report.append(f"      설명: {analysis['description']}")
            
            if analysis['indicators']:
                text_report.append("      ⚠️ 위험 지표:")
                for indicator in analysis['indicators']:
                    text_report.append(f"         • {indicator}")
        
        # LLM 인사이트 (있는 경우)
        if 'llm_insights' in report_data and report_data['summary'].get('llm_enabled', False):
            insights = report_data['llm_insights']
            text_report.append(f"\n🤖 AI 심층 분석")
            text_report.append("=" * 40)
            
            text_report.append(f"\n📋 종합 분석:")
            text_report.append(f"   {insights.get('comprehensive_analysis', 'N/A')}")
            
            text_report.append(f"\n⚠️ 주요 위험 요인:")
            for line in insights.get('risk_factors', '').split('\n'):
                if line.strip():
                    text_report.append(f"   {line}")
            
            text_report.append(f"\n💡 개선 방안:")
            for line in insights.get('improvement_plans', '').split('\n'):
                if line.strip():
                    text_report.append(f"   {line}")
            
            text_report.append(f"\n📊 모니터링 포인트:")
            for line in insights.get('monitoring_points', '').split('\n'):
                if line.strip():
                    text_report.append(f"   {line}")
        
        # 권장사항
        text_report.append(f"\n💡 기본 권장사항")
        text_report.append("-" * 40)
        for i, recommendation in enumerate(report_data['recommendations'], 1):
            text_report.append(f"   {i}. {recommendation}")
        
        # 요약 통계
        summary = report_data['summary']
        text_report.append(f"\n📊 분석 요약")
        text_report.append("-" * 40)
        text_report.append(f"   전체 에이전트 수: {summary['total_agents']}")
        text_report.append(f"   🔴 고위험 에이전트: {summary['high_risk_agents']}")
        text_report.append(f"   🟡 주의 에이전트: {summary['medium_risk_agents']}")
        text_report.append(f"   🟢 안전 에이전트: {summary['low_risk_agents']}")
        text_report.append(f"   🤖 AI 분석 사용: {'예' if summary.get('llm_enabled', False) else '아니오'}")
        
        text_report.append("\n" + "=" * 80)
        
        return "\n".join(text_report)
    
    def save_report(self, employee_id: str, output_dir: str = "reports", 
                   format_type: str = "both") -> Dict[str, str]:
        """레포트를 파일로 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = {}
        
        try:
            if format_type in ["json", "both"]:
                # JSON 형태 저장
                report_data = self.generate_employee_report(employee_id)
                json_file = os.path.join(output_dir, f"report_{employee_id}_{timestamp}.json")
                
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
                
                saved_files['json'] = json_file
            
            if format_type in ["text", "both"]:
                # 텍스트 형태 저장
                text_report = self.generate_text_report(employee_id)
                text_file = os.path.join(output_dir, f"report_{employee_id}_{timestamp}.txt")
                
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text_report)
                
                saved_files['text'] = text_file
            
            return saved_files
            
        except Exception as e:
            return {'error': f'파일 저장 실패: {str(e)}'}
    
    def generate_comprehensive_report(self, employee_id, comprehensive_report, agent_data, 
                                     employee_info, analysis_summary, visualization_files):
        """저장된 파일 데이터로부터 종합 보고서 생성 (XAI, 원인 분석, LLM 인사이트 포함)"""
        try:
            import os
            import openai
            
            # 1. 기본 정보 추출
            overall_assessment = comprehensive_report.get('comprehensive_assessment', {})
            risk_score = overall_assessment.get('overall_risk_score', 0)
            risk_level = overall_assessment.get('overall_risk_level', 'UNKNOWN')
            risk_indicator = overall_assessment.get('risk_indicator', '❓')
            confidence = overall_assessment.get('confidence_level', 'LOW')
            
            # 위험도 레벨 한글 변환
            risk_level_kr = {
                'HIGH': '🔴 고위험군',
                'MEDIUM': '🟡 주의군',
                'LOW': '🟢 안전군',
                'UNKNOWN': '❓ 미분류'
            }.get(risk_level, '❓ 미분류')
            
            # 직원 기본 정보
            emp_data = employee_info.get('employee_data', {})
            department = emp_data.get('Department', '미분류')
            job_role = emp_data.get('JobRole', '미분류')
            age = emp_data.get('Age', 'N/A')
            years_at_company = emp_data.get('YearsAtCompany', 'N/A')
            job_satisfaction = emp_data.get('JobSatisfaction', 'N/A')
            work_life_balance = emp_data.get('WorkLifeBalance', 'N/A')
            
            # 2. 에이전트별 점수 추출 (개선된 로직)
            agent_scores = self._extract_agent_scores(comprehensive_report, agent_data)
            
            # 3. XAI 분석
            xai_analysis = self._analyze_xai(agent_data, employee_id)
            
            # 4. 근본 원인 분석
            root_cause = self._analyze_root_cause(agent_data, employee_info, department, job_role)
            
            # 5. GPT-5-Nano LLM 인사이트 생성
            llm_insights = self._generate_gpt5_nano_insights(
                employee_id, department, job_role, age, years_at_company,
                job_satisfaction, work_life_balance, agent_scores, 
                risk_level, risk_score, xai_analysis, root_cause
            )
            
            # 6. 보고서 생성
            rule_based_interpretation = comprehensive_report.get('rule_based_interpretation', '')
            
            report = self._format_comprehensive_report(
                employee_id=employee_id,
                department=department,
                job_role=job_role,
                age=age,
                years_at_company=years_at_company,
                job_satisfaction=job_satisfaction,
                work_life_balance=work_life_balance,
                risk_indicator=risk_indicator,
                risk_level_kr=risk_level_kr,
                risk_score=risk_score,
                confidence=confidence,
                agent_scores=agent_scores,
                rule_based_interpretation=rule_based_interpretation,
                xai_analysis=xai_analysis,
                root_cause=root_cause,
                llm_insights=llm_insights,
                visualization_files=visualization_files,
                comprehensive_report=comprehensive_report
            )
            
            return report
            
        except Exception as e:
            import traceback
            logger.error(f"종합 보고서 생성 실패: {str(e)}")
            logger.error(traceback.format_exc())
            return f"보고서 생성 중 오류가 발생했습니다: {str(e)}"
    
    def _extract_agent_scores(self, comprehensive_report, agent_data):
        """에이전트별 점수 추출"""
        worker_contributions = comprehensive_report.get('worker_contributions', {})
        scores = {}
        
        # Structura
        if 'structura' in worker_contributions:
            scores['structura'] = worker_contributions['structura'].get('score', 0)
        elif 'structura' in agent_data and 'prediction' in agent_data['structura']:
            scores['structura'] = agent_data['structura']['prediction'].get('attrition_probability', 0)
        else:
            scores['structura'] = 0
        
        # Chronos
        if 'chronos' in worker_contributions:
            scores['chronos'] = worker_contributions['chronos'].get('score', 0)
        elif 'chronos' in agent_data and 'prediction' in agent_data['chronos']:
            scores['chronos'] = agent_data['chronos']['prediction'].get('risk_score', 0)
        else:
            scores['chronos'] = 0
        
        # Cognita
        if 'cognita' in worker_contributions:
            scores['cognita'] = worker_contributions['cognita'].get('score', 0)
        elif 'cognita' in agent_data and 'risk_analysis' in agent_data['cognita']:
            scores['cognita'] = agent_data['cognita']['risk_analysis'].get('overall_risk_score', 0)
        else:
            scores['cognita'] = 0
        
        # Sentio
        if 'sentio' in worker_contributions:
            scores['sentio'] = worker_contributions['sentio'].get('score', 0)
        elif 'sentio' in agent_data:
            sentio_data = agent_data['sentio']
            scores['sentio'] = sentio_data.get('psychological_risk_score',
                                             sentio_data.get('risk_score', 0))
        else:
            scores['sentio'] = 0
        
        # Agora
        if 'agora' in worker_contributions:
            scores['agora'] = worker_contributions['agora'].get('score', 0)
        elif 'agora' in agent_data:
            agora_data = agent_data['agora']
            if 'market_analysis' in agora_data:
                scores['agora'] = agora_data['market_analysis'].get('risk_score', 0)
            else:
                scores['agora'] = agora_data.get('agora_score', 0)
        else:
            scores['agora'] = 0
        
        return scores
    
    def _analyze_xai(self, agent_data, employee_id):
        """XAI 분석 (Structura, Chronos)"""
        xai_summary = ""
        
        # Structura XAI
        if 'structura' in agent_data and 'explanation' in agent_data['structura']:
            explanation = agent_data['structura']['explanation']
            xai_summary += "📊 **Structura XAI 분석:**\n"
            
            if 'feature_importance' in explanation:
                xai_summary += "  주요 영향 변수:\n"
                for feat in explanation['feature_importance'][:5]:
                    xai_summary += f"    - {feat.get('feature', 'N/A')}: {feat.get('importance', 0):.3f}\n"
            
            if 'individual_explanation' in explanation:
                ind_exp = explanation['individual_explanation']
                if 'top_risk_factors' in ind_exp:
                    xai_summary += "  주요 위험 요인:\n"
                    for factor in ind_exp['top_risk_factors'][:3]:
                        xai_summary += f"    - {factor.get('factor', 'N/A')}\n"
            
            xai_summary += "\n"
        
        return xai_summary.strip() if xai_summary else "XAI 데이터가 저장되지 않았습니다."
    
    def _analyze_root_cause(self, agent_data, employee_info, department, job_role):
        """근본 원인 분석 (Sentio, Agora, Cognita)"""
        root_cause = ""
        
        # Sentio
        if 'sentio' in agent_data:
            sentio_data = agent_data['sentio']
            root_cause += "🧠 **Sentio 심리·감정 분석:**\n"
            if 'sentiment_analysis' in sentio_data:
                sentiment = sentio_data['sentiment_analysis']
                root_cause += f"  감정 상태: {sentiment.get('sentiment_label', 'N/A')}\n"
            if 'jd_r_analysis' in sentio_data:
                jdr = sentio_data['jd_r_analysis']
                root_cause += f"  직무 요구/자원 균형: {jdr.get('balance_status', 'N/A')}\n"
            root_cause += "\n"
        
        # Agora
        if 'agora' in agent_data:
            agora_data = agent_data['agora']
            root_cause += "🌍 **Agora 시장 분석:**\n"
            if 'market_analysis' in agora_data:
                market = agora_data['market_analysis']
                root_cause += f"  시장 압력: {market.get('market_pressure_index', 0):.3f}\n"
                root_cause += f"  보상 격차: {market.get('compensation_gap', 0):.3f}\n"
            root_cause += "\n"
        
        return root_cause.strip() if root_cause else "원인 분석 데이터가 부족합니다."
    
    def _generate_gpt5_nano_insights(self, employee_id, department, job_role, age, years_at_company,
                                    job_satisfaction, work_life_balance, agent_scores, 
                                    risk_level, risk_score, xai_analysis, root_cause):
        """GPT-5-Nano를 사용한 LLM 인사이트 생성"""
        try:
            import os
            import openai
            
            if not os.getenv('OPENAI_API_KEY'):
                return ""
            
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            prompt = f"""직원 {employee_id}의 퇴사 위험도 분석 결과입니다.

**기본 정보:**
- 부서: {department}, 직무: {job_role}
- 나이: {age}세, 재직: {years_at_company}년
- 만족도: {job_satisfaction}/4, 워라밸: {work_life_balance}/4

**AI 에이전트 분석:**
- Structura: {agent_scores.get('structura', 0):.1%}
- Chronos: {agent_scores.get('chronos', 0):.1%}
- Cognita: {agent_scores.get('cognita', 0):.1%}
- Sentio: {agent_scores.get('sentio', 0):.1%}
- Agora: {agent_scores.get('agora', 0):.1%}

**XAI 분석:** {xai_analysis}
**원인 분석:** {root_cause}

위 정보를 바탕으로 다음을 제공해주세요:
1. 주요 위험 요인 3가지와 구체적 설명
2. 즉시 실행 가능한 개선 방안 3가지
3. 장기적 관리 전략 2가지"""

            response = client.responses.create(
                model="gpt-5-nano-2025-08-07",
                input=prompt,
                reasoning={"effort": "medium"},
                text={"verbosity": "medium"}
            )
            
            return "\n\n🤖 LLM 기반 심층 분석 (GPT-5-Nano)\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" + response.output_text
            
        except Exception as e:
            logger.error(f"LLM 인사이트 생성 실패: {e}")
            return ""
    
    def _format_comprehensive_report(self, **kwargs):
        """종합 보고서 포맷팅"""
        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         📊 직원 퇴사 위험도 종합 분석 보고서
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 기본 정보
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 직원 ID        : {kwargs['employee_id']}
• 소속 부서      : {kwargs['department']}
• 직무           : {kwargs['job_role']}
• 나이           : {kwargs['age']}세
• 재직 기간      : {kwargs['years_at_company']}년
• 직무 만족도    : {kwargs['job_satisfaction']}/4
• 워라밸         : {kwargs['work_life_balance']}/4

🎯 종합 위험도 평가
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{kwargs['risk_indicator']} 위험도 등급  : {kwargs['risk_level_kr']}
📊 종합 위험 점수 : {kwargs['risk_score']:.3f} / 1.0 ({kwargs['risk_score']*100:.1f}%)
🎲 신뢰도 수준    : {kwargs['confidence']}

📈 다중 에이전트 분석 결과
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏢 Structura: {kwargs['agent_scores']['structura']:.3f} ({kwargs['agent_scores']['structura']*100:.1f}%)
⏰ Chronos: {kwargs['agent_scores']['chronos']:.3f} ({kwargs['agent_scores']['chronos']*100:.1f}%)
🔗 Cognita: {kwargs['agent_scores']['cognita']:.3f} ({kwargs['agent_scores']['cognita']*100:.1f}%)
🧠 Sentio: {kwargs['agent_scores']['sentio']:.3f} ({kwargs['agent_scores']['sentio']*100:.1f}%)
🌍 Agora: {kwargs['agent_scores']['agora']:.3f} ({kwargs['agent_scores']['agora']*100:.1f}%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 AI 분석 인사이트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{kwargs['rule_based_interpretation']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 XAI 설명 가능한 AI 분석
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{kwargs['xai_analysis']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 근본 원인 분석
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{kwargs['root_cause']}
{kwargs['llm_insights']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 시각화 자료
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{"• 총 " + str(len(kwargs['visualization_files'])) + "개의 시각화 차트" if kwargs['visualization_files'] else "• 시각화 자료 없음"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 보고서 생성: {kwargs['comprehensive_report'].get('analysis_timestamp', 'N/A')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    def generate_llm_based_report(self, employee_id, department, risk_level, risk_score, 
                                 agent_scores, agent_data, employee_info):
        """LLM 기반 보고서 생성 (하위 호환성)"""
        # ReportGenerator의 기존 메서드 활용
        return self.generate_text_report(employee_id, use_llm=True)
    
    def generate_batch_reports(self, employee_ids: List[str], output_dir: str = "reports") -> Dict[str, Any]:
        """여러 직원의 레포트를 일괄 생성"""
        results = {
            'success': [],
            'failed': [],
            'total': len(employee_ids),
            'output_directory': output_dir
        }
        
        for employee_id in employee_ids:
            try:
                saved_files = self.save_report(employee_id, output_dir, "both")
                
                if 'error' in saved_files:
                    results['failed'].append({
                        'employee_id': employee_id,
                        'error': saved_files['error']
                    })
                else:
                    results['success'].append({
                        'employee_id': employee_id,
                        'files': saved_files
                    })
                    
            except Exception as e:
                results['failed'].append({
                    'employee_id': employee_id,
                    'error': str(e)
                })
        
        return results


# 사용 예시
if __name__ == "__main__":
    # 레포트 생성기 초기화 (API 키 없이 테스트)
    generator = ReportGenerator()
    
    # 샘플 데이터로 테스트
    sample_scores = {
        'agora_score': 0.75,
        'chronos_score': 0.45,
        'cognita_score': 0.82,
        'sentio_score': 0.65,
        'structura_score': 0.38
    }
    
    generator.set_agent_scores('EMP001', sample_scores)
    
    # 텍스트 레포트 생성 (LLM 없이)
    text_report = generator.generate_text_report('EMP001', use_llm=False)
    print(text_report)
    
    # 파일로 저장
    saved_files = generator.save_report('EMP001')
    print(f"\n저장된 파일: {saved_files}")
        
    # LLM 사용 예시 (API 키가 있는 경우)
    print("\n" + "="*50)
    print("LLM 사용 예시:")
    print("generator = ReportGenerator(api_key='your-api-key')")
    print("llm_report = generator.generate_text_report('EMP001', use_llm=True)")
    print("="*50)
