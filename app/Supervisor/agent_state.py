"""
AgentState - 글로벌 상태 객체 정의
워크플로우 전반에 걸쳐 전달되는 핵심 데이터 구조
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class AnalysisStatus(str, Enum):
    """분석 상태 열거형"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    RETRY = "RETRY"
    SKIPPED = "SKIPPED"


class WorkerType(str, Enum):
    """워커 에이전트 타입"""
    STRUCTURA = "structura"  # 정형 데이터 분석
    COGNITA = "cognita"      # 지식 그래프 분석
    CHRONOS = "chronos"      # 시계열 분석
    SENTIO = "sentio"        # NLP 감성 분석
    AGORA = "agora"          # 외부 시장 분석


class WorkerResult(BaseModel):
    """워커 에이전트 결과 구조"""
    worker_type: WorkerType
    status: AnalysisStatus
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    retry_count: int = 0


class ErrorLog(BaseModel):
    """오류 로그 구조"""
    worker_type: WorkerType
    error_message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    retry_count: int
    is_critical: bool = False


class FinalReport(BaseModel):
    """최종 보고서 구조"""
    employee_id: str
    risk_score: float
    risk_grade: str
    attrition_probability: float
    summary: str
    recommendations: List[str]
    analysis_breakdown: Dict[str, Any]
    confidence_score: float
    generated_at: datetime = Field(default_factory=datetime.now)


class AgentState(BaseModel):
    """
    글로벌 상태 객체 - 모든 노드에서 공유되는 상태
    """
    
    # === 기본 정보 ===
    employee_id: str = Field(..., description="분석 대상 직원의 고유 식별자")
    session_id: str = Field(..., description="분석 세션 고유 식별자")
    analysis_type: str = Field(default="batch", description="분석 타입 (batch/post)")
    started_at: datetime = Field(default_factory=datetime.now)
    
    # === 분석 체크리스트 ===
    analysis_checklist: Dict[str, AnalysisStatus] = Field(
        default_factory=lambda: {
            WorkerType.STRUCTURA: AnalysisStatus.PENDING,
            WorkerType.COGNITA: AnalysisStatus.PENDING,
            WorkerType.CHRONOS: AnalysisStatus.PENDING,
            WorkerType.SENTIO: AnalysisStatus.PENDING,
            WorkerType.AGORA: AnalysisStatus.PENDING,
        },
        description="각 분석의 완료 여부를 추적하는 딕셔너리"
    )
    
    # === 워커 결과 저장소 ===
    worker_results: Dict[str, WorkerResult] = Field(
        default_factory=dict,
        description="각 워커 에이전트의 결과를 저장"
    )
    
    # === 개별 분석 결과 (하위 호환성을 위해 유지) ===
    st_report: Optional[Dict[str, Any]] = Field(None, description="Structura 정형 데이터 분석 결과")
    kg_report: Optional[Dict[str, Any]] = Field(None, description="Cognita 지식 그래프 분석 결과")
    ts_anomaly_report: Optional[Dict[str, Any]] = Field(None, description="Chronos 시계열 이상 탐지 결과")
    nlp_sentiment_report: Optional[Dict[str, Any]] = Field(None, description="Sentio NLP 감성 분석 결과")
    market_intelligence_report: Optional[Dict[str, Any]] = Field(None, description="Agora 시장 분석 결과")
    
    # === 오류 관리 ===
    error_log: List[ErrorLog] = Field(default_factory=list, description="오류 기록 리스트")
    max_retry_count: int = Field(default=3, description="최대 재시도 횟수")
    
    # === 최종 결과 ===
    final_report: Optional[FinalReport] = Field(None, description="최종 종합 보고서")
    
    # === 워크플로우 제어 ===
    current_step: str = Field(default="START", description="현재 워크플로우 단계")
    next_action: Optional[str] = Field(None, description="다음 수행할 액션")
    is_completed: bool = Field(default=False, description="전체 분석 완료 여부")
    
    # === 메타데이터 ===
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")
    
    class Config:
        """Pydantic 설정"""
        use_enum_values = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def get_completion_status(self) -> Dict[str, Any]:
        """분석 완료 상태 요약 반환"""
        total_workers = len(self.analysis_checklist)
        completed_workers = sum(1 for status in self.analysis_checklist.values() 
                              if status == AnalysisStatus.SUCCESS)
        failed_workers = sum(1 for status in self.analysis_checklist.values() 
                           if status == AnalysisStatus.ERROR)
        
        return {
            "total_workers": total_workers,
            "completed_workers": completed_workers,
            "failed_workers": failed_workers,
            "completion_rate": completed_workers / total_workers if total_workers > 0 else 0,
            "is_all_completed": completed_workers == total_workers,
            "has_failures": failed_workers > 0
        }
    
    def add_error(self, worker_type: WorkerType, error_message: str, 
                  retry_count: int = 0, is_critical: bool = False):
        """오류 로그 추가"""
        error_log = ErrorLog(
            worker_type=worker_type,
            error_message=error_message,
            retry_count=retry_count,
            is_critical=is_critical
        )
        self.error_log.append(error_log)
    
    def get_worker_retry_count(self, worker_type: WorkerType) -> int:
        """특정 워커의 재시도 횟수 반환"""
        worker_errors = [log for log in self.error_log if log.worker_type == worker_type]
        return len(worker_errors)
    
    def can_retry_worker(self, worker_type: WorkerType) -> bool:
        """워커 재시도 가능 여부 확인"""
        return self.get_worker_retry_count(worker_type) < self.max_retry_count
    
    def update_worker_status(self, worker_type: WorkerType, status: AnalysisStatus):
        """워커 상태 업데이트"""
        self.analysis_checklist[worker_type] = status
    
    def set_worker_result(self, worker_type: WorkerType, result: Dict[str, Any], 
                         execution_time: Optional[float] = None):
        """워커 결과 설정"""
        worker_result = WorkerResult(
            worker_type=worker_type,
            status=AnalysisStatus.SUCCESS,
            result=result,
            execution_time=execution_time
        )
        self.worker_results[worker_type] = worker_result
        self.update_worker_status(worker_type, AnalysisStatus.SUCCESS)
        
        # 하위 호환성을 위한 개별 결과 설정
        if worker_type == WorkerType.STRUCTURA:
            self.st_report = result
        elif worker_type == WorkerType.COGNITA:
            self.kg_report = result
        elif worker_type == WorkerType.CHRONOS:
            self.ts_anomaly_report = result
        elif worker_type == WorkerType.SENTIO:
            self.nlp_sentiment_report = result
        elif worker_type == WorkerType.AGORA:
            self.market_intelligence_report = result
    
    def set_worker_error(self, worker_type: WorkerType, error_message: str, 
                        retry_count: int = 0):
        """워커 오류 설정"""
        worker_result = WorkerResult(
            worker_type=worker_type,
            status=AnalysisStatus.ERROR,
            error_message=error_message,
            retry_count=retry_count
        )
        self.worker_results[worker_type] = worker_result
        self.update_worker_status(worker_type, AnalysisStatus.ERROR)
        self.add_error(worker_type, error_message, retry_count)
    
    def get_pending_workers(self) -> List[WorkerType]:
        """대기 중인 워커 목록 반환"""
        return [worker_type for worker_type, status in self.analysis_checklist.items() 
                if status == AnalysisStatus.PENDING]
    
    def get_failed_workers(self) -> List[WorkerType]:
        """실패한 워커 목록 반환"""
        return [worker_type for worker_type, status in self.analysis_checklist.items() 
                if status == AnalysisStatus.ERROR]
    
    def get_successful_workers(self) -> List[WorkerType]:
        """성공한 워커 목록 반환"""
        return [worker_type for worker_type, status in self.analysis_checklist.items() 
                if status == AnalysisStatus.SUCCESS]
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """상태 요약 딕셔너리 반환"""
        completion_status = self.get_completion_status()
        
        return {
            "employee_id": self.employee_id,
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "current_step": self.current_step,
            "is_completed": self.is_completed,
            "completion_status": completion_status,
            "analysis_checklist": dict(self.analysis_checklist),
            "error_count": len(self.error_log),
            "has_final_report": self.final_report is not None,
            "metadata": self.metadata
        }


# 타입 별칭
StateDict = Dict[str, Any]
