"""
LangGraph 워크플로우 구현
슈퍼바이저-워커 아키텍처의 핵심 워크플로우
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from .agent_state import AgentState, WorkerType, AnalysisStatus, FinalReport
from .supervisor_agent import SupervisorAgent, SupervisorRouter
from .worker_integrator import WorkerIntegrator, DEFAULT_WORKER_CONFIGS
from .synthesizer import SynthesizerAgent


class WorkflowNodes:
    """워크플로우 노드 구현 클래스"""
    
    def __init__(self, worker_integrator: WorkerIntegrator, 
                 supervisor: SupervisorAgent,
                 synthesizer: SynthesizerAgent):
        self.worker_integrator = worker_integrator
        self.supervisor = supervisor
        self.synthesizer = synthesizer
        self.logger = logging.getLogger(__name__)
    
    async def start_node(self, state: AgentState) -> AgentState:
        """
        워크플로우 시작 노드
        초기 상태 설정 및 검증
        """
        self.logger.info(f"Starting workflow for employee {state.employee_id}")
        
        # 세션 ID가 없으면 생성
        if not state.session_id:
            state.session_id = str(uuid.uuid4())
        
        # 시작 시간 설정
        if not state.started_at:
            state.started_at = datetime.now()
        
        # 현재 단계 업데이트
        state.current_step = "INITIALIZED"
        
        # 워커 상태 확인
        health_status = await self.worker_integrator.health_check_all()
        unavailable_workers = [worker for worker, is_healthy in health_status.items() if not is_healthy]
        
        if unavailable_workers:
            self.logger.warning(f"Unavailable workers: {unavailable_workers}")
            # 사용 불가능한 워커는 SKIPPED로 설정
            for worker in unavailable_workers:
                state.update_worker_status(worker, AnalysisStatus.SKIPPED)
                state.add_error(worker, "Worker service unavailable", is_critical=False)
        
        # 메타데이터 업데이트
        state.metadata.update({
            "workflow_version": "1.0.0",
            "available_workers": [w.value for w, h in health_status.items() if h],
            "unavailable_workers": [w.value for w in unavailable_workers]
        })
        
        self.logger.info(f"Workflow initialized for employee {state.employee_id}, session {state.session_id}")
        return state
    
    async def worker_dispatcher_node(self, state: AgentState) -> AgentState:
        """
        워커 디스패처 노드
        사용 가능한 워커들에게 병렬로 작업 분배
        """
        self.logger.info(f"Dispatching workers for employee {state.employee_id}")
        
        # 대기 중인 워커들 찾기
        pending_workers = state.get_pending_workers()
        
        if not pending_workers:
            self.logger.info("No pending workers found")
            state.current_step = "NO_PENDING_WORKERS"
            return state
        
        # 워커 상태를 IN_PROGRESS로 업데이트
        for worker in pending_workers:
            state.update_worker_status(worker, AnalysisStatus.IN_PROGRESS)
        
        state.current_step = "DISPATCHING_WORKERS"
        
        # 병렬 분석 실행
        try:
            analysis_results = await self.worker_integrator.analyze_employee_parallel(
                employee_id=state.employee_id,
                target_workers=pending_workers,
                additional_data={
                    "session_id": state.session_id,
                    "analysis_type": state.analysis_type
                }
            )
            
            # 결과를 상태에 업데이트
            state = self.worker_integrator.update_agent_state(state, analysis_results)
            
            # 성공/실패 통계
            successful_workers = [w for w, r in analysis_results.items() if r["success"]]
            failed_workers = [w for w, r in analysis_results.items() if not r["success"]]
            
            self.logger.info(f"Worker dispatch completed - Success: {len(successful_workers)}, Failed: {len(failed_workers)}")
            
            state.current_step = "WORKERS_DISPATCHED"
            
        except Exception as e:
            self.logger.error(f"Error in worker dispatch: {e}")
            # 모든 대기 중인 워커를 ERROR로 설정
            for worker in pending_workers:
                state.set_worker_error(worker, f"Dispatch error: {str(e)}")
            
            state.current_step = "DISPATCH_ERROR"
        
        return state
    
    async def retry_handler_node(self, state: AgentState) -> AgentState:
        """
        재시도 처리 노드
        실패한 워커들을 재시도
        """
        self.logger.info(f"Handling retries for employee {state.employee_id}")
        
        # 재시도 가능한 실패 워커들 찾기
        failed_workers = state.get_failed_workers()
        retryable_workers = [w for w in failed_workers if state.can_retry_worker(w)]
        
        if not retryable_workers:
            self.logger.info("No retryable workers found")
            state.current_step = "NO_RETRYABLE_WORKERS"
            return state
        
        state.current_step = "RETRYING_WORKERS"
        
        # 재시도 워커들의 상태를 RETRY로 업데이트
        for worker in retryable_workers:
            state = self.supervisor.update_state_for_retry(state, worker)
        
        try:
            # 재시도 실행
            retry_results = await self.worker_integrator.analyze_employee_parallel(
                employee_id=state.employee_id,
                target_workers=retryable_workers,
                additional_data={
                    "session_id": state.session_id,
                    "analysis_type": state.analysis_type,
                    "is_retry": True
                }
            )
            
            # 결과를 상태에 업데이트
            state = self.worker_integrator.update_agent_state(state, retry_results)
            
            # 재시도 결과 로깅
            successful_retries = [w for w, r in retry_results.items() if r["success"]]
            failed_retries = [w for w, r in retry_results.items() if not r["success"]]
            
            self.logger.info(f"Retry completed - Success: {len(successful_retries)}, Failed: {len(failed_retries)}")
            
            state.current_step = "RETRIES_COMPLETED"
            
        except Exception as e:
            self.logger.error(f"Error in retry handler: {e}")
            for worker in retryable_workers:
                state.add_error(worker, f"Retry error: {str(e)}", is_critical=True)
            
            state.current_step = "RETRY_ERROR"
        
        return state
    
    async def synthesizer_node(self, state: AgentState) -> AgentState:
        """
        종합 분석 노드
        모든 워커 결과를 종합하여 최종 보고서 생성
        """
        self.logger.info(f"Synthesizing results for employee {state.employee_id}")
        
        state.current_step = "SYNTHESIZING"
        
        try:
            # 성공한 워커들의 결과 수집
            successful_results = {}
            for worker_type in state.get_successful_workers():
                if worker_type in state.worker_results:
                    successful_results[worker_type] = state.worker_results[worker_type].result
            
            if not successful_results:
                raise Exception("No successful worker results to synthesize")
            
            # 종합 분석 실행
            final_report = await self.synthesizer.synthesize_analysis(
                employee_id=state.employee_id,
                worker_results=successful_results,
                metadata=state.metadata
            )
            
            # 최종 보고서 설정
            state.final_report = final_report
            state.is_completed = True
            state.current_step = "COMPLETED"
            
            self.logger.info(f"Synthesis completed for employee {state.employee_id}")
            
        except Exception as e:
            self.logger.error(f"Error in synthesizer: {e}")
            state.add_error(WorkerType.STRUCTURA, f"Synthesis error: {str(e)}", is_critical=True)
            state.current_step = "SYNTHESIS_ERROR"
        
        return state
    
    async def error_handler_node(self, state: AgentState) -> AgentState:
        """
        오류 처리 노드
        치명적 오류 발생 시 처리
        """
        self.logger.error(f"Handling critical errors for employee {state.employee_id}")
        
        state.current_step = "ERROR_HANDLING"
        
        # 오류 보고서 생성
        error_report = self.supervisor.create_error_report(state)
        
        # 메타데이터에 오류 정보 추가
        state.metadata.update({
            "error_report": error_report,
            "error_handled_at": datetime.now().isoformat(),
            "requires_human_intervention": True
        })
        
        # 부분 결과라도 있으면 종합 시도
        successful_workers = state.get_successful_workers()
        if successful_workers:
            self.logger.info(f"Attempting partial synthesis with {len(successful_workers)} successful workers")
            try:
                successful_results = {}
                for worker_type in successful_workers:
                    if worker_type in state.worker_results:
                        successful_results[worker_type] = state.worker_results[worker_type].result
                
                partial_report = await self.synthesizer.synthesize_analysis(
                    employee_id=state.employee_id,
                    worker_results=successful_results,
                    metadata=state.metadata,
                    is_partial=True
                )
                
                state.final_report = partial_report
                state.current_step = "PARTIAL_COMPLETION"
                
            except Exception as e:
                self.logger.error(f"Partial synthesis also failed: {e}")
                state.current_step = "CRITICAL_ERROR"
        else:
            state.current_step = "CRITICAL_ERROR"
        
        return state
    
    async def timeout_handler_node(self, state: AgentState) -> AgentState:
        """
        타임아웃 처리 노드
        """
        self.logger.warning(f"Handling timeout for employee {state.employee_id}")
        
        state.current_step = "TIMEOUT_HANDLING"
        
        # 타임아웃 정보 추가
        elapsed_time = datetime.now() - state.started_at
        state.metadata.update({
            "timeout_occurred": True,
            "elapsed_time_minutes": elapsed_time.total_seconds() / 60,
            "timeout_handled_at": datetime.now().isoformat()
        })
        
        # 완료된 워커가 있으면 부분 결과 생성
        successful_workers = state.get_successful_workers()
        if successful_workers:
            try:
                successful_results = {}
                for worker_type in successful_workers:
                    if worker_type in state.worker_results:
                        successful_results[worker_type] = state.worker_results[worker_type].result
                
                timeout_report = await self.synthesizer.synthesize_analysis(
                    employee_id=state.employee_id,
                    worker_results=successful_results,
                    metadata=state.metadata,
                    is_partial=True,
                    timeout_occurred=True
                )
                
                state.final_report = timeout_report
                state.current_step = "TIMEOUT_PARTIAL_COMPLETION"
                
            except Exception as e:
                self.logger.error(f"Timeout partial synthesis failed: {e}")
                state.current_step = "TIMEOUT_ERROR"
        else:
            state.current_step = "TIMEOUT_NO_RESULTS"
        
        return state


class SupervisorWorkflow:
    """LangGraph 기반 슈퍼바이저 워크플로우"""
    
    def __init__(self, 
                 worker_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                 llm: Optional[ChatOpenAI] = None,
                 max_retry_count: int = 3,
                 timeout_minutes: int = 30):
        """
        워크플로우 초기화
        
        Args:
            worker_configs: 워커 설정
            llm: LangChain LLM 인스턴스 (gpt-5 사용)
            max_retry_count: 최대 재시도 횟수
            timeout_minutes: 타임아웃 (분)
        """
        self.logger = logging.getLogger(__name__)
        
        # 기본 설정
        if worker_configs is None:
            worker_configs = DEFAULT_WORKER_CONFIGS
        
        if llm is None:
            # OpenAI API 키가 없으면 None으로 설정
            import os
            if os.getenv("OPENAI_API_KEY"):
                llm = ChatOpenAI(model="gpt-5", temperature=0.1)
            else:
                llm = None
        
        # 컴포넌트 초기화
        self.worker_integrator = WorkerIntegrator(worker_configs)
        self.supervisor = SupervisorAgent(llm, max_retry_count, timeout_minutes)
        self.synthesizer = SynthesizerAgent(llm)
        
        # 라우터 초기화
        self.router = SupervisorRouter(self.supervisor)
        
        # 노드 초기화
        self.nodes = WorkflowNodes(
            self.worker_integrator,
            self.supervisor,
            self.synthesizer
        )
        
        # 워크플로우 그래프 구축
        self.workflow = self._build_workflow()
        
        # 메모리 체크포인터
        self.checkpointer = MemorySaver()
        
        # 컴파일된 그래프
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
        
        self.logger.info("Supervisor workflow initialized")
    
    def _build_workflow(self) -> StateGraph:
        """워크플로우 그래프 구축"""
        
        # StateGraph 생성
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("start", self.nodes.start_node)
        workflow.add_node("worker_dispatcher", self.nodes.worker_dispatcher_node)
        workflow.add_node("retry_handler", self.nodes.retry_handler_node)
        workflow.add_node("synthesizer", self.nodes.synthesizer_node)
        workflow.add_node("error_handler", self.nodes.error_handler_node)
        workflow.add_node("timeout_handler", self.nodes.timeout_handler_node)
        
        # 시작점 설정
        workflow.set_entry_point("start")
        
        # 조건부 엣지 추가 (슈퍼바이저 라우팅)
        workflow.add_conditional_edges(
            "start",
            self.router.route_by_status,
            {
                "worker_dispatcher": "worker_dispatcher",
                "error_handler": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "worker_dispatcher",
            self.router.route_by_status,
            {
                "worker_dispatcher": "worker_dispatcher",
                "retry_handler": "retry_handler", 
                "synthesizer": "synthesizer",
                "error_handler": "error_handler",
                "timeout_handler": "timeout_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "retry_handler",
            self.router.route_by_status,
            {
                "worker_dispatcher": "worker_dispatcher",
                "retry_handler": "retry_handler",
                "synthesizer": "synthesizer", 
                "error_handler": "error_handler",
                "timeout_handler": "timeout_handler"
            }
        )
        
        # 종료 엣지
        workflow.add_edge("synthesizer", END)
        workflow.add_edge("error_handler", END)
        workflow.add_edge("timeout_handler", END)
        
        return workflow
    
    async def analyze_employee(self, employee_id: str, 
                             session_id: Optional[str] = None,
                             analysis_type: str = 'batch') -> Dict[str, Any]:
        """
        직원 분석 실행
        
        Args:
            employee_id: 직원 ID
            session_id: 세션 ID (선택적)
            
        Returns:
            분석 결과
        """
        try:
            # 초기 상태 생성
            initial_state = AgentState(
                employee_id=employee_id,
                session_id=session_id or str(uuid.uuid4()),
                analysis_type=analysis_type
            )
            
            self.logger.info(f"Starting analysis for employee {employee_id}")
            
            # 워크플로우 실행
            config = {"configurable": {"thread_id": initial_state.session_id}}
            
            final_state = None
            async for state in self.app.astream(initial_state, config=config):
                final_state = state
                self.logger.debug(f"Workflow step: {state.current_step}")
            
            if final_state is None:
                raise Exception("Workflow execution failed - no final state")
            
            # 결과 반환
            result = {
                "success": final_state.is_completed,
                "employee_id": employee_id,
                "session_id": final_state.session_id,
                "final_report": final_state.final_report.dict() if final_state.final_report else None,
                "workflow_status": self.supervisor.get_workflow_status(final_state),
                "execution_summary": {
                    "started_at": final_state.started_at.isoformat(),
                    "completed_at": datetime.now().isoformat(),
                    "total_workers": len(final_state.analysis_checklist),
                    "successful_workers": len(final_state.get_successful_workers()),
                    "failed_workers": len(final_state.get_failed_workers()),
                    "error_count": len(final_state.error_log)
                }
            }
            
            self.logger.info(f"Analysis completed for employee {employee_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis failed for employee {employee_id}: {e}")
            return {
                "success": False,
                "employee_id": employee_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_workflow_status(self, session_id: str) -> Dict[str, Any]:
        """워크플로우 상태 조회"""
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = await self.app.aget_state(config)
            
            if state and state.values:
                agent_state = AgentState(**state.values)
                return self.supervisor.get_workflow_status(agent_state)
            else:
                return {"error": "Session not found"}
                
        except Exception as e:
            self.logger.error(f"Error getting workflow status: {e}")
            return {"error": str(e)}
    
    def get_available_workers(self) -> List[str]:
        """사용 가능한 워커 목록 반환"""
        return [worker.value for worker in self.worker_integrator.get_available_workers()]
