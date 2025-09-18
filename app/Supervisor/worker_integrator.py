"""
Worker Integrator - 워커 에이전트 통합 모듈
각 워커 에이전트(Structura, Cognita, Chronos, Sentio, Agora)와의 통신을 담당
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import traceback

from .agent_state import AgentState, WorkerType, AnalysisStatus


class WorkerClient:
    """개별 워커 에이전트와의 통신을 담당하는 클라이언트"""
    
    def __init__(self, worker_type: WorkerType, base_url: str, timeout: int = 300):
        """
        워커 클라이언트 초기화
        
        Args:
            worker_type: 워커 타입
            base_url: 워커 서버 기본 URL
            timeout: 요청 타임아웃 (초)
        """
        self.worker_type = worker_type
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.logger = logging.getLogger(f"{__name__}.{worker_type}")
    
    async def health_check(self) -> bool:
        """워커 서버 상태 확인"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                # 먼저 /health를 시도하고, 실패하면 /api/health를 시도
                health_paths = ['/health', '/api/health']
                
                for health_path in health_paths:
                    try:
                        async with session.get(f"{self.base_url}{health_path}") as response:
                            if response.status == 200:
                                return True
                    except Exception:
                        continue
                        
                return False
        except Exception as e:
            self.logger.error(f"Health check failed for {self.worker_type}: {e}")
            return False
    
    async def analyze_employee(self, employee_id: str, additional_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        직원 분석 요청
        
        Args:
            employee_id: 직원 ID
            additional_data: 추가 데이터
            
        Returns:
            분석 결과
        """
        start_time = datetime.now()
        
        try:
            # 요청 데이터 준비
            request_data = {
                "employee_id": employee_id,
                "timestamp": start_time.isoformat()
            }
            
            if additional_data:
                request_data.update(additional_data)
            
            # 워커별 특화 엔드포인트 결정
            endpoint = self._get_analysis_endpoint()
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(
                    f"{self.base_url}{endpoint}",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # 성공 결과 구조화
                        return {
                            "success": True,
                            "worker_type": self.worker_type,
                            "employee_id": employee_id,
                            "result": result,
                            "execution_time": execution_time,
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                        
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                "success": False,
                "worker_type": self.worker_type,
                "employee_id": employee_id,
                "error": f"Timeout after {execution_time:.1f}s",
                "error_type": "timeout",
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            
            # 에러 타입 분류
            error_type = self._classify_error(error_msg)
            
            return {
                "success": False,
                "worker_type": self.worker_type,
                "employee_id": employee_id,
                "error": error_msg,
                "error_type": error_type,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "traceback": traceback.format_exc()
            }
    
    def _get_analysis_endpoint(self) -> str:
        """워커 타입별 분석 엔드포인트 반환"""
        endpoint_mapping = {
            WorkerType.STRUCTURA: "/predict_attrition",
            WorkerType.COGNITA: "/analyze_relationships",
            WorkerType.CHRONOS: "/detect_anomalies",
            WorkerType.SENTIO: "/analyze_sentiment",
            WorkerType.AGORA: "/market_analysis"
        }
        return endpoint_mapping.get(self.worker_type, "/analyze")
    
    def _classify_error(self, error_message: str) -> str:
        """에러 메시지를 기반으로 에러 타입 분류"""
        error_msg_lower = error_message.lower()
        
        if "timeout" in error_msg_lower or "timed out" in error_msg_lower:
            return "timeout"
        elif "connection" in error_msg_lower or "network" in error_msg_lower:
            return "network"
        elif "404" in error_msg_lower or "not found" in error_msg_lower:
            return "endpoint_not_found"
        elif "500" in error_msg_lower or "internal server error" in error_msg_lower:
            return "server_error"
        elif "401" in error_msg_lower or "unauthorized" in error_msg_lower:
            return "authentication"
        elif "403" in error_msg_lower or "forbidden" in error_msg_lower:
            return "authorization"
        elif "rate limit" in error_msg_lower or "too many requests" in error_msg_lower:
            return "rate_limit"
        elif "data" in error_msg_lower or "format" in error_msg_lower:
            return "data_error"
        else:
            return "unknown"


class WorkerIntegrator:
    """워커 에이전트들을 통합 관리하는 클래스"""
    
    def __init__(self, worker_configs: Dict[str, Dict[str, Any]]):
        """
        워커 통합기 초기화
        
        Args:
            worker_configs: 워커별 설정 딕셔너리
                {
                    "structura": {"base_url": "http://localhost:5001", "timeout": 300},
                    "cognita": {"base_url": "http://localhost:5002", "timeout": 300},
                    ...
                }
        """
        self.logger = logging.getLogger(__name__)
        self.workers: Dict[WorkerType, WorkerClient] = {}
        
        # 워커 클라이언트 초기화
        for worker_name, config in worker_configs.items():
            try:
                worker_type = WorkerType(worker_name.lower())
                self.workers[worker_type] = WorkerClient(
                    worker_type=worker_type,
                    base_url=config["base_url"],
                    timeout=config.get("timeout", 300)
                )
                self.logger.info(f"Initialized worker client: {worker_type}")
            except ValueError:
                self.logger.error(f"Invalid worker type: {worker_name}")
    
    async def health_check_all(self) -> Dict[WorkerType, bool]:
        """모든 워커의 상태 확인"""
        health_results = {}
        
        tasks = []
        for worker_type, client in self.workers.items():
            tasks.append(self._check_worker_health(worker_type, client))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (worker_type, _) in enumerate(self.workers.items()):
            if isinstance(results[i], Exception):
                health_results[worker_type] = False
                self.logger.error(f"Health check error for {worker_type}: {results[i]}")
            else:
                health_results[worker_type] = results[i]
        
        return health_results
    
    async def _check_worker_health(self, worker_type: WorkerType, client: WorkerClient) -> bool:
        """개별 워커 상태 확인"""
        try:
            return await client.health_check()
        except Exception as e:
            self.logger.error(f"Health check failed for {worker_type}: {e}")
            return False
    
    async def analyze_employee_parallel(self, employee_id: str, 
                                      target_workers: Optional[List[WorkerType]] = None,
                                      additional_data: Optional[Dict] = None) -> Dict[WorkerType, Dict[str, Any]]:
        """
        여러 워커에서 병렬로 직원 분석 수행
        
        Args:
            employee_id: 직원 ID
            target_workers: 대상 워커 리스트 (None이면 모든 워커)
            additional_data: 추가 데이터
            
        Returns:
            워커별 분석 결과
        """
        if target_workers is None:
            target_workers = list(self.workers.keys())
        
        self.logger.info(f"Starting parallel analysis for employee {employee_id} with workers: {target_workers}")
        
        # 병렬 작업 생성
        tasks = []
        for worker_type in target_workers:
            if worker_type in self.workers:
                client = self.workers[worker_type]
                task = self._analyze_with_worker(client, employee_id, additional_data)
                tasks.append((worker_type, task))
            else:
                self.logger.warning(f"Worker {worker_type} not available")
        
        # 병렬 실행
        results = {}
        if tasks:
            task_list = [task for _, task in tasks]
            completed_results = await asyncio.gather(*task_list, return_exceptions=True)
            
            for i, (worker_type, _) in enumerate(tasks):
                if isinstance(completed_results[i], Exception):
                    results[worker_type] = {
                        "success": False,
                        "worker_type": worker_type,
                        "employee_id": employee_id,
                        "error": str(completed_results[i]),
                        "error_type": "exception",
                        "timestamp": datetime.now().isoformat()
                    }
                    self.logger.error(f"Analysis failed for {worker_type}: {completed_results[i]}")
                else:
                    results[worker_type] = completed_results[i]
        
        self.logger.info(f"Completed parallel analysis for employee {employee_id}")
        return results
    
    async def _analyze_with_worker(self, client: WorkerClient, employee_id: str, 
                                 additional_data: Optional[Dict] = None) -> Dict[str, Any]:
        """개별 워커로 분석 수행"""
        return await client.analyze_employee(employee_id, additional_data)
    
    async def analyze_employee_sequential(self, employee_id: str, 
                                        target_workers: Optional[List[WorkerType]] = None,
                                        additional_data: Optional[Dict] = None) -> Dict[WorkerType, Dict[str, Any]]:
        """
        여러 워커에서 순차적으로 직원 분석 수행
        
        Args:
            employee_id: 직원 ID
            target_workers: 대상 워커 리스트 (None이면 모든 워커)
            additional_data: 추가 데이터
            
        Returns:
            워커별 분석 결과
        """
        if target_workers is None:
            target_workers = list(self.workers.keys())
        
        self.logger.info(f"Starting sequential analysis for employee {employee_id} with workers: {target_workers}")
        
        results = {}
        for worker_type in target_workers:
            if worker_type in self.workers:
                client = self.workers[worker_type]
                self.logger.info(f"Analyzing with {worker_type}...")
                
                result = await client.analyze_employee(employee_id, additional_data)
                results[worker_type] = result
                
                if result["success"]:
                    self.logger.info(f"Analysis completed for {worker_type}")
                else:
                    self.logger.error(f"Analysis failed for {worker_type}: {result.get('error', 'Unknown error')}")
            else:
                self.logger.warning(f"Worker {worker_type} not available")
                results[worker_type] = {
                    "success": False,
                    "worker_type": worker_type,
                    "employee_id": employee_id,
                    "error": "Worker not available",
                    "error_type": "not_available",
                    "timestamp": datetime.now().isoformat()
                }
        
        self.logger.info(f"Completed sequential analysis for employee {employee_id}")
        return results
    
    def update_agent_state(self, state: AgentState, 
                          analysis_results: Dict[WorkerType, Dict[str, Any]]) -> AgentState:
        """
        분석 결과를 AgentState에 업데이트
        
        Args:
            state: 현재 AgentState
            analysis_results: 워커별 분석 결과
            
        Returns:
            업데이트된 AgentState
        """
        for worker_type, result in analysis_results.items():
            if result["success"]:
                # 성공한 경우
                state.set_worker_result(
                    worker_type=worker_type,
                    result=result["result"],
                    execution_time=result.get("execution_time")
                )
                self.logger.info(f"Updated state with successful result from {worker_type}")
                
            else:
                # 실패한 경우
                retry_count = state.get_worker_retry_count(worker_type)
                state.set_worker_error(
                    worker_type=worker_type,
                    error_message=result["error"],
                    retry_count=retry_count
                )
                self.logger.error(f"Updated state with error from {worker_type}: {result['error']}")
        
        return state
    
    def get_available_workers(self) -> List[WorkerType]:
        """사용 가능한 워커 목록 반환"""
        return list(self.workers.keys())
    
    def get_worker_status(self) -> Dict[str, Any]:
        """워커 상태 요약 반환"""
        return {
            "total_workers": len(self.workers),
            "available_workers": [worker.value for worker in self.workers.keys()],
            "worker_configs": {
                worker.value: {
                    "base_url": client.base_url,
                    "timeout": client.timeout
                }
                for worker, client in self.workers.items()
            }
        }


# 기본 워커 설정
DEFAULT_WORKER_CONFIGS = {
    "structura": {
        "base_url": "http://localhost:5001",
        "timeout": 300
    },
    "cognita": {
        "base_url": "http://localhost:5002", 
        "timeout": 300
    },
    "chronos": {
        "base_url": "http://localhost:5003",
        "timeout": 300
    },
    "sentio": {
        "base_url": "http://localhost:5004",
        "timeout": 300
    },
    "agora": {
        "base_url": "http://localhost:5005",
        "timeout": 300
    }
}
