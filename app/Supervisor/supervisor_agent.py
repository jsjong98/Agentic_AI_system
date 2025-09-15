"""
Supervisor Agent - 워크플로우 오케스트레이터
LangGraph 기반의 지능형 조건부 라우터
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .agent_state import AgentState, WorkerType, AnalysisStatus, ErrorLog


class SupervisorAgent:
    """
    슈퍼바이저 에이전트 - 워크플로우의 중추 신경계 역할
    """
    
    def __init__(self, 
                 llm: Optional[ChatOpenAI] = None,
                 max_retry_count: int = 3,
                 timeout_minutes: int = 30):
        """
        슈퍼바이저 에이전트 초기화
        
        Args:
            llm: LangChain LLM 인스턴스
            max_retry_count: 최대 재시도 횟수
            timeout_minutes: 워커 타임아웃 (분)
        """
        self.llm = llm or ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            max_tokens=1000
        )
        self.max_retry_count = max_retry_count
        self.timeout_minutes = timeout_minutes
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 라우팅 프롬프트 템플릿
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_routing_system_prompt()),
            ("human", self._get_routing_human_prompt())
        ])
        
        # 에러 분석 프롬프트 템플릿
        self.error_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_error_analysis_system_prompt()),
            ("human", self._get_error_analysis_human_prompt())
        ])
    
    def _get_routing_system_prompt(self) -> str:
        """라우팅 결정을 위한 시스템 프롬프트"""
        return """
당신은 HR Attrition 예측 시스템의 슈퍼바이저 에이전트입니다.
다음 워커 에이전트들의 분석 상태를 검토하여 워크플로우의 다음 단계를 결정해야 합니다:

워커 에이전트들:
- STRUCTURA: 정형 데이터 분석 (HR 기본 정보, 성과 데이터)
- COGNITA: 지식 그래프 분석 (조직 관계, 프로젝트 연결성)
- CHRONOS: 시계열 분석 (근무 패턴, 이상 탐지)
- SENTIO: NLP 감성 분석 (피드백, 설문 텍스트)
- AGORA: 외부 시장 분석 (채용 시장, 연봉 벤치마크)

가능한 다음 단계:
- CONTINUE_ANALYSIS: 아직 완료되지 않은 분석이 있음
- RETRY_FAILED: 실패한 분석을 재시도해야 함
- FINALIZE: 모든 분석이 완료되어 최종 종합 분석 단계로 진행
- ERROR_CRITICAL: 치명적 오류로 인한 중단 (인간 개입 필요)
- TIMEOUT: 시간 초과로 인한 중단

결정 기준:
1. 모든 워커가 SUCCESS 상태면 -> FINALIZE
2. PENDING 또는 IN_PROGRESS 워커가 있으면 -> CONTINUE_ANALYSIS
3. ERROR 워커가 있고 재시도 가능하면 -> RETRY_FAILED
4. 재시도 횟수 초과 또는 치명적 오류 -> ERROR_CRITICAL
5. 전체 실행 시간이 제한을 초과하면 -> TIMEOUT

반드시 위의 5가지 옵션 중 하나만 반환하세요.
"""
    
    def _get_routing_human_prompt(self) -> str:
        """라우팅을 위한 인간 프롬프트"""
        return """
현재 분석 상태를 검토하여 다음 단계를 결정해주세요:

직원 ID: {employee_id}
세션 ID: {session_id}
시작 시간: {started_at}
현재 시간: {current_time}
경과 시간: {elapsed_minutes}분

분석 체크리스트:
{analysis_checklist}

워커 결과 요약:
{worker_results_summary}

오류 로그 ({error_count}개):
{error_log_summary}

완료 상태:
- 전체 워커: {total_workers}개
- 완료된 워커: {completed_workers}개
- 실패한 워커: {failed_workers}개
- 완료율: {completion_rate:.1%}

다음 단계를 결정해주세요: CONTINUE_ANALYSIS, RETRY_FAILED, FINALIZE, ERROR_CRITICAL, TIMEOUT 중 하나
"""
    
    def _get_error_analysis_system_prompt(self) -> str:
        """에러 분석을 위한 시스템 프롬프트"""
        return """
당신은 시스템 오류를 분석하는 전문가입니다.
워커 에이전트의 오류를 분석하여 재시도 가능성과 해결 방안을 제시해주세요.

분석 기준:
1. 일시적 오류 (네트워크, API 제한) -> 재시도 권장
2. 데이터 오류 (형식, 누락) -> 데이터 검증 후 재시도
3. 시스템 오류 (메모리, 타임아웃) -> 리소스 조정 후 재시도
4. 치명적 오류 (인증, 권한, 코드 버그) -> 인간 개입 필요

JSON 형식으로 응답해주세요:
{
    "is_retryable": true/false,
    "error_category": "network/data/system/critical",
    "recommended_action": "retry/skip/human_intervention",
    "explanation": "오류 원인과 해결 방안 설명"
}
"""
    
    def _get_error_analysis_human_prompt(self) -> str:
        """에러 분석을 위한 인간 프롬프트"""
        return """
다음 워커 에이전트의 오류를 분석해주세요:

워커 타입: {worker_type}
오류 메시지: {error_message}
재시도 횟수: {retry_count}
발생 시간: {timestamp}

이 오류의 재시도 가능성과 해결 방안을 분석해주세요.
"""
    
    async def route_workflow(self, state: AgentState) -> str:
        """
        워크플로우 라우팅 결정
        
        Args:
            state: 현재 AgentState
            
        Returns:
            다음 단계 문자열
        """
        try:
            # 현재 시간과 경과 시간 계산
            current_time = datetime.now()
            elapsed_time = current_time - state.started_at
            elapsed_minutes = elapsed_time.total_seconds() / 60
            
            # 타임아웃 체크
            if elapsed_minutes > self.timeout_minutes:
                self.logger.warning(f"Workflow timeout for employee {state.employee_id}")
                return "TIMEOUT"
            
            # 완료 상태 확인
            completion_status = state.get_completion_status()
            
            # 모든 워커가 성공했으면 최종화
            if completion_status["is_all_completed"]:
                self.logger.info(f"All workers completed for employee {state.employee_id}")
                return "FINALIZE"
            
            # 치명적 오류 체크
            critical_errors = [log for log in state.error_log if log.is_critical]
            if critical_errors:
                self.logger.error(f"Critical errors detected for employee {state.employee_id}")
                return "ERROR_CRITICAL"
            
            # 재시도 가능한 실패 워커 체크
            failed_workers = state.get_failed_workers()
            retryable_workers = [w for w in failed_workers if state.can_retry_worker(w)]
            
            if retryable_workers:
                self.logger.info(f"Retryable workers found: {retryable_workers}")
                return "RETRY_FAILED"
            
            # 대기 중인 워커가 있으면 계속 진행
            pending_workers = state.get_pending_workers()
            if pending_workers:
                self.logger.info(f"Pending workers: {pending_workers}")
                return "CONTINUE_ANALYSIS"
            
            # LLM을 통한 지능적 라우팅 결정
            routing_input = self._prepare_routing_input(state, current_time, elapsed_minutes)
            
            chain = self.routing_prompt | self.llm
            response = await chain.ainvoke(routing_input)
            
            decision = response.content.strip().upper()
            
            # 유효한 결정인지 확인
            valid_decisions = ["CONTINUE_ANALYSIS", "RETRY_FAILED", "FINALIZE", "ERROR_CRITICAL", "TIMEOUT"]
            if decision not in valid_decisions:
                self.logger.warning(f"Invalid routing decision: {decision}, defaulting to CONTINUE_ANALYSIS")
                decision = "CONTINUE_ANALYSIS"
            
            self.logger.info(f"Routing decision for employee {state.employee_id}: {decision}")
            return decision
            
        except Exception as e:
            self.logger.error(f"Error in route_workflow: {e}")
            return "ERROR_CRITICAL"
    
    def _prepare_routing_input(self, state: AgentState, current_time: datetime, elapsed_minutes: float) -> Dict[str, Any]:
        """라우팅 입력 데이터 준비"""
        completion_status = state.get_completion_status()
        
        # 워커 결과 요약
        worker_results_summary = []
        for worker_type, result in state.worker_results.items():
            summary = f"- {worker_type}: {result.status}"
            if result.execution_time:
                summary += f" ({result.execution_time:.2f}s)"
            if result.error_message:
                summary += f" - {result.error_message[:100]}..."
            worker_results_summary.append(summary)
        
        # 오류 로그 요약
        error_log_summary = []
        for error in state.error_log[-5:]:  # 최근 5개만
            summary = f"- {error.worker_type}: {error.error_message[:100]}... (재시도: {error.retry_count})"
            error_log_summary.append(summary)
        
        return {
            "employee_id": state.employee_id,
            "session_id": state.session_id,
            "started_at": state.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_minutes": elapsed_minutes,
            "analysis_checklist": json.dumps(dict(state.analysis_checklist), indent=2),
            "worker_results_summary": "\n".join(worker_results_summary) or "없음",
            "error_log_summary": "\n".join(error_log_summary) or "없음",
            "error_count": len(state.error_log),
            "total_workers": completion_status["total_workers"],
            "completed_workers": completion_status["completed_workers"],
            "failed_workers": completion_status["failed_workers"],
            "completion_rate": completion_status["completion_rate"]
        }
    
    async def analyze_error(self, worker_type: WorkerType, error_message: str, 
                          retry_count: int) -> Dict[str, Any]:
        """
        오류 분석 및 재시도 가능성 판단
        
        Args:
            worker_type: 워커 타입
            error_message: 오류 메시지
            retry_count: 현재 재시도 횟수
            
        Returns:
            오류 분석 결과
        """
        try:
            error_input = {
                "worker_type": worker_type,
                "error_message": error_message,
                "retry_count": retry_count,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            chain = self.error_analysis_prompt | self.llm
            response = await chain.ainvoke(error_input)
            
            # JSON 파싱 시도
            try:
                analysis = json.loads(response.content)
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 기본값
                analysis = {
                    "is_retryable": retry_count < self.max_retry_count,
                    "error_category": "unknown",
                    "recommended_action": "retry" if retry_count < self.max_retry_count else "human_intervention",
                    "explanation": f"오류 분석 실패: {response.content[:200]}..."
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in analyze_error: {e}")
            return {
                "is_retryable": False,
                "error_category": "critical",
                "recommended_action": "human_intervention",
                "explanation": f"오류 분석 중 예외 발생: {str(e)}"
            }
    
    def update_state_for_retry(self, state: AgentState, worker_type: WorkerType) -> AgentState:
        """
        재시도를 위한 상태 업데이트
        
        Args:
            state: 현재 상태
            worker_type: 재시도할 워커 타입
            
        Returns:
            업데이트된 상태
        """
        # 워커 상태를 RETRY로 변경
        state.update_worker_status(worker_type, AnalysisStatus.RETRY)
        
        # 현재 단계 업데이트
        state.current_step = f"RETRY_{worker_type.upper()}"
        
        # 메타데이터에 재시도 정보 추가
        retry_key = f"{worker_type}_retry_count"
        current_retry = state.metadata.get(retry_key, 0)
        state.metadata[retry_key] = current_retry + 1
        state.metadata[f"{worker_type}_last_retry"] = datetime.now().isoformat()
        
        self.logger.info(f"Updated state for retry: {worker_type}, attempt {current_retry + 1}")
        
        return state
    
    def create_error_report(self, state: AgentState) -> Dict[str, Any]:
        """
        오류 보고서 생성
        
        Args:
            state: 현재 상태
            
        Returns:
            오류 보고서
        """
        failed_workers = state.get_failed_workers()
        critical_errors = [log for log in state.error_log if log.is_critical]
        
        report = {
            "employee_id": state.employee_id,
            "session_id": state.session_id,
            "timestamp": datetime.now().isoformat(),
            "failed_workers": failed_workers,
            "critical_errors": len(critical_errors),
            "total_errors": len(state.error_log),
            "completion_status": state.get_completion_status(),
            "error_summary": []
        }
        
        # 워커별 오류 요약
        for worker_type in failed_workers:
            worker_errors = [log for log in state.error_log if log.worker_type == worker_type]
            if worker_errors:
                latest_error = worker_errors[-1]
                report["error_summary"].append({
                    "worker_type": worker_type,
                    "error_message": latest_error.error_message,
                    "retry_count": latest_error.retry_count,
                    "is_critical": latest_error.is_critical,
                    "timestamp": latest_error.timestamp.isoformat()
                })
        
        return report
    
    def get_workflow_status(self, state: AgentState) -> Dict[str, Any]:
        """
        워크플로우 상태 요약 반환
        
        Args:
            state: 현재 상태
            
        Returns:
            상태 요약
        """
        current_time = datetime.now()
        elapsed_time = current_time - state.started_at
        
        return {
            "employee_id": state.employee_id,
            "session_id": state.session_id,
            "current_step": state.current_step,
            "is_completed": state.is_completed,
            "started_at": state.started_at.isoformat(),
            "elapsed_time_minutes": elapsed_time.total_seconds() / 60,
            "completion_status": state.get_completion_status(),
            "analysis_checklist": dict(state.analysis_checklist),
            "error_count": len(state.error_log),
            "has_final_report": state.final_report is not None,
            "next_action": state.next_action,
            "metadata": state.metadata
        }


class SupervisorRouter:
    """
    조건부 라우팅을 위한 라우터 클래스
    LangGraph의 add_conditional_edges에서 사용
    """
    
    def __init__(self, supervisor: SupervisorAgent):
        self.supervisor = supervisor
        self.logger = logging.getLogger(__name__)
    
    async def route_by_status(self, state: AgentState) -> str:
        """
        상태 기반 라우팅 함수
        
        Args:
            state: 현재 AgentState
            
        Returns:
            다음 노드 이름
        """
        try:
            decision = await self.supervisor.route_workflow(state)
            
            # 결정에 따른 노드 매핑
            node_mapping = {
                "CONTINUE_ANALYSIS": "worker_dispatcher",
                "RETRY_FAILED": "retry_handler",
                "FINALIZE": "synthesizer",
                "ERROR_CRITICAL": "error_handler",
                "TIMEOUT": "timeout_handler"
            }
            
            next_node = node_mapping.get(decision, "error_handler")
            
            # 상태 업데이트
            state.next_action = decision
            state.current_step = f"ROUTING_TO_{next_node.upper()}"
            
            self.logger.info(f"Routing decision: {decision} -> {next_node}")
            
            return next_node
            
        except Exception as e:
            self.logger.error(f"Error in route_by_status: {e}")
            state.add_error(WorkerType.STRUCTURA, f"Routing error: {str(e)}", is_critical=True)
            return "error_handler"
