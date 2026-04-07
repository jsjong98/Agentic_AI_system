"""
MCP Adapter - SupervisorAgent를 MCP와 통합
기존 SupervisorAgent를 MCP 프로토콜과 연결하는 어댑터 레이어
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from supervisor_agent import SupervisorAgent, SupervisorRouter
from agent_state import AgentState, WorkerType, AnalysisStatus
from mcp_server import MCPServer
from mcp_tools import (
    MCPToolRegistry,
    StructuraTool,
    CognitaTool,
    ChronosTool,
    SentioTool,
    AgoraTool
)


class SupervisorMCPAdapter:
    """
    Supervisor Agent를 MCP 프로토콜과 통합하는 어댑터
    
    기능:
    - SupervisorAgent와 MCP 서버 간의 브릿지 역할
    - 워커 실행을 MCP 도구 호출로 변환
    - 상태 관리와 MCP 컨텍스트 동기화
    """
    
    def __init__(self, 
                 supervisor: Optional[SupervisorAgent] = None,
                 enable_mcp_server: bool = True):
        """
        어댑터 초기화
        
        Args:
            supervisor: SupervisorAgent 인스턴스
            enable_mcp_server: MCP 서버 활성화 여부
        """
        self.supervisor = supervisor or SupervisorAgent()
        self.enable_mcp_server = enable_mcp_server
        self.logger = logging.getLogger(__name__)
        
        # MCP 컴포넌트 초기화
        self.mcp_server: Optional[MCPServer] = None
        self.tool_registry = MCPToolRegistry()
        
        # 세션 관리
        self.active_sessions: Dict[str, AgentState] = {}
        
        # 워커 클라이언트 맵 (실제 API 연결)
        self.worker_clients: Dict[WorkerType, Any] = {}
        
        self._initialize_mcp_components()
        
        self.logger.info("SupervisorMCPAdapter initialized")
    
    def _initialize_mcp_components(self):
        """MCP 컴포넌트 초기화"""
        # 도구 레지스트리에 워커 도구 등록
        self.tool_registry.initialize_default_tools()
        
        # MCP 서버 초기화
        if self.enable_mcp_server:
            self.mcp_server = MCPServer("supervisor-mcp-adapter")
            self.mcp_server.set_supervisor(self.supervisor)
            
            # 커스텀 핸들러 등록
            self._register_custom_handlers()
            
            self.logger.info("MCP components initialized")
    
    def _register_custom_handlers(self):
        """커스텀 MCP 핸들러 등록"""
        if not self.mcp_server:
            return
        
        # 라우팅 핸들러
        self.mcp_server.register_tool(
            "route_workflow",
            self._mcp_route_workflow
        )
        
        # 상태 조회 핸들러
        self.mcp_server.register_tool(
            "get_workflow_status",
            self._mcp_get_workflow_status
        )
        
        # 오류 보고서 핸들러
        self.mcp_server.register_tool(
            "create_error_report",
            self._mcp_create_error_report
        )
        
        # 워커 실행 핸들러
        for worker_type in WorkerType:
            tool_name = f"analyze_{worker_type.lower()}"
            self.mcp_server.register_tool(
                tool_name,
                self._create_worker_handler(worker_type)
            )
    
    def _create_worker_handler(self, worker_type: WorkerType) -> callable:
        """워커별 MCP 핸들러 생성"""
        async def handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
            return await self._mcp_execute_worker(worker_type, arguments)
        return handler
    
    async def _mcp_execute_worker(self, 
                                   worker_type: WorkerType, 
                                   arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP를 통한 워커 실행
        
        Args:
            worker_type: 워커 타입
            arguments: 실행 인자
            
        Returns:
            실행 결과
        """
        employee_id = arguments.get("employee_id")
        session_id = arguments.get("session_id")
        
        self.logger.info(f"MCP executing worker {worker_type} for employee {employee_id}")
        
        try:
            # 세션 상태 가져오기 또는 생성
            if session_id and session_id in self.active_sessions:
                state = self.active_sessions[session_id]
            else:
                state = AgentState(employee_id=employee_id)
                if session_id:
                    state.session_id = session_id
                    self.active_sessions[session_id] = state
            
            # 워커 상태 업데이트
            state.update_worker_status(worker_type, AnalysisStatus.IN_PROGRESS)
            
            # 도구 실행
            tool_name = f"analyze_{worker_type.lower()}"
            result = await self.tool_registry.execute_tool(tool_name, arguments)
            
            # 결과를 상태에 반영
            if result.get("status") == "SUCCESS":
                state.update_worker_status(worker_type, AnalysisStatus.SUCCESS)
                state.worker_results[worker_type] = result
            else:
                state.update_worker_status(worker_type, AnalysisStatus.ERROR)
                state.add_error(
                    worker_type,
                    result.get("error", "Unknown error"),
                    is_critical=False
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing worker {worker_type}: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "worker_type": worker_type,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _mcp_route_workflow(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP를 통한 워크플로우 라우팅
        
        Args:
            arguments: 라우팅 인자 (session_id 포함)
            
        Returns:
            라우팅 결정
        """
        session_id = arguments.get("session_id")
        
        if not session_id or session_id not in self.active_sessions:
            return {
                "status": "ERROR",
                "error": "Invalid or missing session_id",
                "timestamp": datetime.now().isoformat()
            }
        
        state = self.active_sessions[session_id]
        
        try:
            decision = await self.supervisor.route_workflow(state)
            
            return {
                "status": "SUCCESS",
                "decision": decision,
                "session_id": session_id,
                "reasoning": f"Supervisor 결정: {decision}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error routing workflow: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _mcp_get_workflow_status(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP를 통한 워크플로우 상태 조회
        
        Args:
            arguments: 조회 인자 (session_id 포함)
            
        Returns:
            워크플로우 상태
        """
        session_id = arguments.get("session_id")
        
        if not session_id or session_id not in self.active_sessions:
            return {
                "status": "ERROR",
                "error": "Invalid or missing session_id",
                "timestamp": datetime.now().isoformat()
            }
        
        state = self.active_sessions[session_id]
        
        try:
            status = self.supervisor.get_workflow_status(state)
            
            return {
                "status": "SUCCESS",
                **status
            }
            
        except Exception as e:
            self.logger.error(f"Error getting workflow status: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _mcp_create_error_report(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP를 통한 오류 보고서 생성
        
        Args:
            arguments: 보고서 생성 인자 (session_id 포함)
            
        Returns:
            오류 보고서
        """
        session_id = arguments.get("session_id")
        
        if not session_id or session_id not in self.active_sessions:
            return {
                "status": "ERROR",
                "error": "Invalid or missing session_id",
                "timestamp": datetime.now().isoformat()
            }
        
        state = self.active_sessions[session_id]
        
        try:
            report = self.supervisor.create_error_report(state)
            
            return {
                "status": "SUCCESS",
                "report": report
            }
            
        except Exception as e:
            self.logger.error(f"Error creating error report: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def register_worker_client(self, worker_type: WorkerType, client: Any):
        """
        워커 클라이언트 등록 (실제 API 연결)
        
        Args:
            worker_type: 워커 타입
            client: 워커 클라이언트 인스턴스
        """
        self.worker_clients[worker_type] = client
        
        # 도구 레지스트리의 해당 도구에도 클라이언트 연결
        tool_name = f"analyze_{worker_type.lower()}"
        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            tool.worker_client = client
            
        self.logger.info(f"Worker client registered: {worker_type}")
    
    async def execute_workflow(self, 
                              employee_id: str,
                              use_mcp: bool = True) -> Dict[str, Any]:
        """
        워크플로우 실행 (MCP 사용 여부 선택 가능)
        
        Args:
            employee_id: 직원 ID
            use_mcp: MCP 프로토콜 사용 여부
            
        Returns:
            실행 결과
        """
        state = AgentState(employee_id=employee_id)
        session_id = state.session_id
        self.active_sessions[session_id] = state
        
        self.logger.info(f"Starting workflow for employee {employee_id} (MCP: {use_mcp})")
        
        try:
            if use_mcp and self.enable_mcp_server:
                # MCP를 통한 실행
                results = await self._execute_workflow_via_mcp(state)
            else:
                # 기존 방식 실행
                results = await self._execute_workflow_legacy(state)
            
            return {
                "status": "SUCCESS",
                "session_id": session_id,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Workflow execution error: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "session_id": session_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            # 완료된 세션 정리 (선택적)
            if state.is_completed:
                self.logger.info(f"Session {session_id} completed")
    
    async def _execute_workflow_via_mcp(self, state: AgentState) -> Dict[str, Any]:
        """MCP를 통한 워크플로우 실행"""
        results = {}
        
        # 각 워커를 MCP 도구로 실행
        for worker_type in WorkerType:
            tool_name = f"analyze_{worker_type.lower()}"
            arguments = {
                "employee_id": state.employee_id,
                "session_id": state.session_id
            }
            
            result = await self.tool_registry.execute_tool(tool_name, arguments)
            results[worker_type] = result
            
            # 상태 업데이트
            if result.get("status") == "SUCCESS":
                state.update_worker_status(worker_type, AnalysisStatus.SUCCESS)
                state.worker_results[worker_type] = result
            else:
                state.update_worker_status(worker_type, AnalysisStatus.ERROR)
        
        # 라우팅 결정
        decision = await self.supervisor.route_workflow(state)
        results["routing_decision"] = decision
        
        return results
    
    async def _execute_workflow_legacy(self, state: AgentState) -> Dict[str, Any]:
        """기존 방식으로 워크플로우 실행 (MCP 미사용)"""
        # 기존 SupervisorAgent의 로직 사용
        decision = await self.supervisor.route_workflow(state)
        
        return {
            "routing_decision": decision,
            "worker_results": dict(state.worker_results)
        }
    
    async def start_mcp_server(self):
        """MCP 서버 시작"""
        if not self.mcp_server:
            self.logger.warning("MCP server not initialized")
            return
        
        self.logger.info("Starting MCP server...")
        await self.mcp_server.run()
    
    def get_session(self, session_id: str) -> Optional[AgentState]:
        """세션 상태 조회"""
        return self.active_sessions.get(session_id)
    
    def cleanup_session(self, session_id: str):
        """세션 정리"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.logger.info(f"Session {session_id} cleaned up")


async def main():
    """테스트용 메인 함수"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 어댑터 초기화
    adapter = SupervisorMCPAdapter(enable_mcp_server=True)
    
    # 테스트 워크플로우 실행
    result = await adapter.execute_workflow(
        employee_id="EMP001",
        use_mcp=True
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())

