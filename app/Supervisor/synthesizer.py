"""
Synthesizer Agent - 최종 종합 분석 에이전트
모든 워커 결과를 종합하여 AHP 기반 최종 보고서 생성
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import numpy as np

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .agent_state import WorkerType, FinalReport


class SynthesizerAgent:
    """종합 분석을 담당하는 에이전트"""
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        종합 분석 에이전트 초기화
        
        Args:
            llm: LangChain LLM 인스턴스 (gpt-5 사용)
        """
        # LangChain LLM 초기화 (강력한 gpt-5 모델 사용)
        self.llm = llm or ChatOpenAI(
            model="gpt-5",
            temperature=0.2,
            max_tokens=2000
        )
        
        self.logger = logging.getLogger(__name__)
        
        # AHP 가중치 (기본값)
        self.ahp_weights = {
            WorkerType.STRUCTURA: 0.35,  # 정형 데이터 - 가장 높은 가중치
            WorkerType.COGNITA: 0.20,    # 관계 분석
            WorkerType.CHRONOS: 0.25,    # 시계열 패턴
            WorkerType.SENTIO: 0.15,     # 감성 분석
            WorkerType.AGORA: 0.05       # 외부 시장 - 참고용
        }
        
        # 종합 분석 프롬프트 템플릿
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_synthesis_system_prompt()),
            ("human", self._get_synthesis_human_prompt())
        ])
    
    def _get_synthesis_system_prompt(self) -> str:
        """Sentio/Agora와 일관된 구조화된 종합 분석 시스템 프롬프트"""
        return """
당신은 조직 내 개인의 심리와 감정을 깊이 이해하는 HR Attrition 예측 시스템의 최종 종합 분석 전문가입니다. 다중 에이전트 시스템의 분석 결과를 통합하여 직원의 이탈 위험도를 평가하는 전문가입니다.

**다중 에이전트 분석 시스템:**
여러 워커 에이전트의 분석 결과를 종합하여 직원의 이탈 위험도를 평가해야 합니다.

**워커 에이전트별 전문 영역:**
- STRUCTURA: 정형 데이터 분석 (성과, 급여, 근속년수, 구조적 요인)
- COGNITA: 관계 분석 (팀 내 위치, 프로젝트 연결성, 네트워크 영향)  
- CHRONOS: 시계열 분석 (근무 패턴, 행동 변화, 이상 행동 탐지)
- SENTIO: 감성 분석 (피드백, 설문 텍스트 분석, 심리적 상태)
- AGORA: 외부 시장 분석 (채용 시장, 연봉 벤치마크, 시장 압력)

**종합 분석 기준:**
1. 각 워커의 결과를 AHP 가중치에 따라 종합
2. 일관성 있는 패턴과 상충하는 신호 식별
3. 위험 요인의 우선순위 결정
4. 실행 가능한 권장사항 제시
5. 직원의 프라이버시와 존엄성을 존중하는 분석

**출력 형식:**
- risk_score: 0-100 점수 (높을수록 이탈 위험 높음)
- risk_grade: A(매우 높음), B(높음), C(보통), D(낮음), E(매우 낮음)
- attrition_probability: 0-1 확률값
- summary: 종합 분석 요약 (한국어, 200-300자)
- recommendations: 구체적 권장사항 리스트 (3-5개)
- confidence_score: 분석 신뢰도 (0-1)

**작성 가이드라인:**
- 전문적이면서도 실무진이 이해하기 쉬운 표현 사용
- 객관적이고 건설적인 톤 유지
- 각 에이전트의 분석 결과를 균형 있게 고려
- JSON 형식으로 정확하게 응답

JSON 형식으로 응답해주세요.
"""
    
    def _get_synthesis_human_prompt(self) -> str:
        """종합 분석을 위한 인간 프롬프트"""
        return """
다음 워커 분석 결과를 종합하여 최종 이탈 위험도를 평가해주세요:

직원 ID: {employee_id}
분석 일시: {analysis_time}

워커 분석 결과:
{worker_results}

AHP 가중치:
{ahp_weights}

추가 정보:
- 부분 분석 여부: {is_partial}
- 타임아웃 발생: {timeout_occurred}
- 사용 가능한 워커: {available_workers}
- 메타데이터: {metadata}

위 정보를 종합하여 최종 분석 결과를 JSON 형식으로 제공해주세요.
"""
    
    async def synthesize_analysis(self, 
                                employee_id: str,
                                worker_results: Dict[WorkerType, Dict[str, Any]],
                                metadata: Optional[Dict[str, Any]] = None,
                                is_partial: bool = False,
                                timeout_occurred: bool = False) -> FinalReport:
        """
        워커 결과를 종합하여 최종 보고서 생성
        
        Args:
            employee_id: 직원 ID
            worker_results: 워커별 분석 결과
            metadata: 추가 메타데이터
            is_partial: 부분 분석 여부
            timeout_occurred: 타임아웃 발생 여부
            
        Returns:
            최종 보고서
        """
        try:
            self.logger.info(f"Starting synthesis for employee {employee_id}")
            
            # 입력 데이터 준비
            synthesis_input = self._prepare_synthesis_input(
                employee_id, worker_results, metadata, is_partial, timeout_occurred
            )
            
            # LLM을 통한 종합 분석 (LangChain 방식)
            chain = self.synthesis_prompt | self.llm
            response = await chain.ainvoke(synthesis_input)
            
            # JSON 파싱
            try:
                analysis_result = json.loads(response.content)
            except json.JSONDecodeError:
                self.logger.error("Failed to parse LLM response as JSON")
                # 기본값으로 폴백
                analysis_result = self._create_fallback_analysis(
                    employee_id, worker_results, is_partial
                )
            
            # 수치적 검증 및 보정
            analysis_result = self._validate_and_correct_analysis(analysis_result)
            
            # FinalReport 객체 생성
            final_report = FinalReport(
                employee_id=employee_id,
                risk_score=analysis_result["risk_score"],
                risk_grade=analysis_result["risk_grade"],
                attrition_probability=analysis_result["attrition_probability"],
                summary=analysis_result["summary"],
                recommendations=analysis_result["recommendations"],
                analysis_breakdown=self._create_analysis_breakdown(worker_results),
                confidence_score=analysis_result["confidence_score"]
            )
            
            self.logger.info(f"Synthesis completed for employee {employee_id}")
            return final_report
            
        except Exception as e:
            self.logger.error(f"Error in synthesis: {e}")
            # 에러 발생 시 기본 보고서 생성
            return self._create_error_report(employee_id, str(e), worker_results)
    
    def _prepare_synthesis_input(self, 
                               employee_id: str,
                               worker_results: Dict[WorkerType, Dict[str, Any]],
                               metadata: Optional[Dict[str, Any]],
                               is_partial: bool,
                               timeout_occurred: bool) -> Dict[str, Any]:
        """종합 분석 입력 데이터 준비"""
        
        # 워커 결과 요약
        worker_summaries = []
        for worker_type, result in worker_results.items():
            summary = f"**{worker_type.value.upper()}**:\n"
            
            # 결과 요약 (워커별 특화)
            if worker_type == WorkerType.STRUCTURA:
                summary += self._summarize_structura_result(result)
            elif worker_type == WorkerType.COGNITA:
                summary += self._summarize_cognita_result(result)
            elif worker_type == WorkerType.CHRONOS:
                summary += self._summarize_chronos_result(result)
            elif worker_type == WorkerType.SENTIO:
                summary += self._summarize_sentio_result(result)
            elif worker_type == WorkerType.AGORA:
                summary += self._summarize_agora_result(result)
            else:
                summary += f"결과: {str(result)[:200]}..."
            
            worker_summaries.append(summary)
        
        return {
            "employee_id": employee_id,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "worker_results": "\n\n".join(worker_summaries),
            "ahp_weights": json.dumps(
                {k.value: v for k, v in self.ahp_weights.items()}, 
                indent=2
            ),
            "is_partial": is_partial,
            "timeout_occurred": timeout_occurred,
            "available_workers": [w.value for w in worker_results.keys()],
            "metadata": json.dumps(metadata or {}, indent=2)
        }
    
    def _summarize_structura_result(self, result: Dict[str, Any]) -> str:
        """Structura 결과 요약"""
        try:
            if "prediction" in result:
                prediction = result["prediction"]
                probability = result.get("probability", 0)
                return f"이탈 예측: {prediction}, 확률: {probability:.3f}"
            else:
                return f"정형 데이터 분석 결과: {str(result)[:100]}..."
        except:
            return "결과 파싱 오류"
    
    def _summarize_cognita_result(self, result: Dict[str, Any]) -> str:
        """Cognita 결과 요약"""
        try:
            if "relationship_score" in result:
                rel_score = result["relationship_score"]
                return f"관계 점수: {rel_score:.3f}, 네트워크 위치: {result.get('network_position', 'N/A')}"
            else:
                return f"관계 분석 결과: {str(result)[:100]}..."
        except:
            return "결과 파싱 오류"
    
    def _summarize_chronos_result(self, result: Dict[str, Any]) -> str:
        """Chronos 결과 요약"""
        try:
            if "anomaly_score" in result:
                anomaly = result["anomaly_score"]
                return f"이상 점수: {anomaly:.3f}, 패턴: {result.get('pattern_type', 'N/A')}"
            else:
                return f"시계열 분석 결과: {str(result)[:100]}..."
        except:
            return "결과 파싱 오류"
    
    def _summarize_sentio_result(self, result: Dict[str, Any]) -> str:
        """Sentio 결과 요약"""
        try:
            if "sentiment_score" in result:
                sentiment = result["sentiment_score"]
                return f"감성 점수: {sentiment:.3f}, 주요 키워드: {result.get('keywords', [])[:3]}"
            else:
                return f"감성 분석 결과: {str(result)[:100]}..."
        except:
            return "결과 파싱 오류"
    
    def _summarize_agora_result(self, result: Dict[str, Any]) -> str:
        """Agora 결과 요약"""
        try:
            if "market_risk" in result:
                market_risk = result["market_risk"]
                return f"시장 위험도: {market_risk:.3f}, 외부 기회: {result.get('external_opportunities', 'N/A')}"
            else:
                return f"시장 분석 결과: {str(result)[:100]}..."
        except:
            return "결과 파싱 오류"
    
    def _validate_and_correct_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과 검증 및 보정"""
        
        # risk_score 검증 (0-100)
        if "risk_score" not in analysis or not isinstance(analysis["risk_score"], (int, float)):
            analysis["risk_score"] = 50.0
        else:
            analysis["risk_score"] = max(0, min(100, float(analysis["risk_score"])))
        
        # attrition_probability 검증 (0-1)
        if "attrition_probability" not in analysis or not isinstance(analysis["attrition_probability"], (int, float)):
            analysis["attrition_probability"] = analysis["risk_score"] / 100
        else:
            analysis["attrition_probability"] = max(0, min(1, float(analysis["attrition_probability"])))
        
        # confidence_score 검증 (0-1)
        if "confidence_score" not in analysis or not isinstance(analysis["confidence_score"], (int, float)):
            analysis["confidence_score"] = 0.7
        else:
            analysis["confidence_score"] = max(0, min(1, float(analysis["confidence_score"])))
        
        # risk_grade 검증
        valid_grades = ["A", "B", "C", "D", "E"]
        if "risk_grade" not in analysis or analysis["risk_grade"] not in valid_grades:
            # risk_score 기반으로 등급 결정
            score = analysis["risk_score"]
            if score >= 80:
                analysis["risk_grade"] = "A"
            elif score >= 60:
                analysis["risk_grade"] = "B"
            elif score >= 40:
                analysis["risk_grade"] = "C"
            elif score >= 20:
                analysis["risk_grade"] = "D"
            else:
                analysis["risk_grade"] = "E"
        
        # summary 검증
        if "summary" not in analysis or not isinstance(analysis["summary"], str):
            analysis["summary"] = f"직원의 이탈 위험도는 {analysis['risk_grade']}등급으로 평가됩니다."
        
        # recommendations 검증
        if "recommendations" not in analysis or not isinstance(analysis["recommendations"], list):
            analysis["recommendations"] = ["정기적인 면담 실시", "업무 만족도 조사", "경력 개발 계획 수립"]
        
        return analysis
    
    def _create_analysis_breakdown(self, worker_results: Dict[WorkerType, Dict[str, Any]]) -> Dict[str, Any]:
        """분석 세부 내역 생성"""
        breakdown = {
            "worker_contributions": {},
            "weighted_scores": {},
            "total_weighted_score": 0
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for worker_type, result in worker_results.items():
            weight = self.ahp_weights.get(worker_type, 0.1)
            
            # 워커별 위험 점수 추출 (간단한 휴리스틱)
            worker_score = self._extract_risk_score_from_result(result)
            
            breakdown["worker_contributions"][worker_type.value] = {
                "raw_score": worker_score,
                "weight": weight,
                "weighted_score": worker_score * weight
            }
            
            breakdown["weighted_scores"][worker_type.value] = worker_score * weight
            
            total_weight += weight
            weighted_sum += worker_score * weight
        
        if total_weight > 0:
            breakdown["total_weighted_score"] = weighted_sum / total_weight
        
        return breakdown
    
    def _extract_risk_score_from_result(self, result: Dict[str, Any]) -> float:
        """워커 결과에서 위험 점수 추출"""
        try:
            # 다양한 필드명으로 점수 추출 시도
            score_fields = [
                "risk_score", "attrition_probability", "probability",
                "anomaly_score", "sentiment_score", "market_risk"
            ]
            
            for field in score_fields:
                if field in result:
                    value = result[field]
                    if isinstance(value, (int, float)):
                        # 0-1 범위면 100배, 이미 0-100 범위면 그대로
                        return float(value * 100 if value <= 1 else value)
            
            # 예측 결과가 문자열인 경우
            if "prediction" in result:
                pred = str(result["prediction"]).lower()
                if "yes" in pred or "high" in pred or "위험" in pred:
                    return 75.0
                elif "no" in pred or "low" in pred or "안전" in pred:
                    return 25.0
            
            # 기본값
            return 50.0
            
        except:
            return 50.0
    
    def _create_fallback_analysis(self, 
                                employee_id: str,
                                worker_results: Dict[WorkerType, Dict[str, Any]],
                                is_partial: bool) -> Dict[str, Any]:
        """폴백 분석 결과 생성"""
        
        # 간단한 수치적 종합
        scores = []
        for worker_type, result in worker_results.items():
            score = self._extract_risk_score_from_result(result)
            weight = self.ahp_weights.get(worker_type, 0.1)
            scores.append(score * weight)
        
        avg_score = sum(scores) / len(scores) if scores else 50.0
        
        return {
            "risk_score": avg_score,
            "risk_grade": "C",
            "attrition_probability": avg_score / 100,
            "summary": f"{'부분적인 ' if is_partial else ''}분석 결과, 직원 {employee_id}의 이탈 위험도는 보통 수준으로 평가됩니다.",
            "recommendations": [
                "정기적인 성과 면담 실시",
                "업무 만족도 모니터링",
                "경력 개발 기회 제공"
            ],
            "confidence_score": 0.6 if is_partial else 0.8
        }
    
    def _create_error_report(self, 
                           employee_id: str,
                           error_message: str,
                           worker_results: Dict[WorkerType, Dict[str, Any]]) -> FinalReport:
        """에러 발생 시 기본 보고서 생성"""
        
        return FinalReport(
            employee_id=employee_id,
            risk_score=50.0,
            risk_grade="C",
            attrition_probability=0.5,
            summary=f"분석 중 오류가 발생했습니다: {error_message[:100]}...",
            recommendations=[
                "시스템 오류로 인한 재분석 필요",
                "수동 검토 권장",
                "기술팀 문의"
            ],
            analysis_breakdown={
                "error": error_message,
                "available_results": len(worker_results)
            },
            confidence_score=0.1
        )
