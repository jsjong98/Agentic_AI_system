# -*- coding: utf-8 -*-
"""
Agentic AI Master Server
워커 에이전트들을 통합 관리하는 마스터 서버

현재 구현된 워커 에이전트:
- 워커 에이전트 1: 정형 데이터 분석 (Structura)
- 워커 에이전트 2: 관계형 데이터 분석 (Cognita)

향후 확장 예정:
- 워커 에이전트 3: 시계열 데이터 분석
- 워커 에이전트 4: 자연어 데이터 분석  
- 워커 에이전트 5: 외부 시장 분석
- Supervisor 에이전트: 전체 조정
- 최종 종합 에이전트: 결과 통합
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import logging
import os
import json
import threading
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from queue import Queue
import sys
from pathlib import Path

# 워커 에이전트 import
sys.path.append(str(Path(__file__).parent / "Structura"))
sys.path.append(str(Path(__file__).parent / "Cognita"))

try:
    from Structura.structura_flask_backend import StructuraHRPredictor
    STRUCTURA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Structura 워커 에이전트 import 실패: {e}")
    STRUCTURA_AVAILABLE = False

try:
    from Cognita.cognita_flask_backend import CognitaRiskAnalyzer, Neo4jManager
    COGNITA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cognita 워커 에이전트 import 실패: {e}")
    COGNITA_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------
# 데이터 모델 정의
# ------------------------------------------------------

@dataclass
class WorkerStatus:
    """워커 에이전트 상태"""
    agent_id: str
    agent_name: str
    status: str  # 'running', 'stopped', 'error', 'busy'
    last_heartbeat: str
    tasks_completed: int
    current_task: Optional[str]
    error_message: Optional[str] = None

@dataclass
class AgenticTask:
    """에이전틱 작업"""
    task_id: str
    task_type: str  # 'individual_analysis', 'department_analysis', 'combined_analysis'
    employee_data: Optional[Dict] = None
    department_name: Optional[str] = None
    sample_size: Optional[int] = None
    use_structura: bool = True
    use_cognita: bool = True
    priority: int = 1  # 1=높음, 2=보통, 3=낮음
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

@dataclass
class AgenticResult:
    """에이전틱 분석 결과"""
    task_id: str
    task_type: str
    structura_result: Optional[Dict] = None
    cognita_result: Optional[Dict] = None
    combined_analysis: Optional[Dict] = None
    execution_time: float = 0.0
    status: str = "completed"  # 'completed', 'partial', 'failed'
    error_message: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

# ------------------------------------------------------
# 워커 에이전트 관리자
# ------------------------------------------------------

class WorkerAgentManager:
    """워커 에이전트 관리자"""
    
    def __init__(self):
        self.workers = {}
        self.task_queue = Queue()
        self.result_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 워커 에이전트 초기화
        self._initialize_workers()
    
    def _initialize_workers(self):
        """워커 에이전트 초기화"""
        logger.info("워커 에이전트 초기화 시작...")
        
        # 워커 에이전트 1: Structura (정형 데이터 분석)
        if STRUCTURA_AVAILABLE:
            try:
                structura_predictor = StructuraHRPredictor()
                self.workers['structura'] = {
                    'agent': structura_predictor,
                    'status': WorkerStatus(
                        agent_id='structura',
                        agent_name='정형 데이터 분석 에이전트',
                        status='running',
                        last_heartbeat=datetime.now().isoformat(),
                        tasks_completed=0,
                        current_task=None
                    ),
                    'type': 'structured_data'
                }
                logger.info("✅ Structura 워커 에이전트 초기화 완료")
            except Exception as e:
                logger.error(f"❌ Structura 워커 에이전트 초기화 실패: {e}")
                self.workers['structura'] = {
                    'agent': None,
                    'status': WorkerStatus(
                        agent_id='structura',
                        agent_name='정형 데이터 분석 에이전트',
                        status='error',
                        last_heartbeat=datetime.now().isoformat(),
                        tasks_completed=0,
                        current_task=None,
                        error_message=str(e)
                    ),
                    'type': 'structured_data'
                }
        
        # 워커 에이전트 2: Cognita (관계형 데이터 분석)
        if COGNITA_AVAILABLE:
            try:
                # Neo4j 연결 설정
                neo4j_config = {
                    "uri": os.getenv("NEO4J_URI", "bolt://YOUR_NEO4J_HOST:7687"),
                    "username": os.getenv("NEO4J_USERNAME", "neo4j"),
                    "password": os.getenv("NEO4J_PASSWORD", "YOUR_NEO4J_PASSWORD")
                }
                
                neo4j_manager = Neo4jManager(
                    neo4j_config['uri'],
                    neo4j_config['username'],
                    neo4j_config['password']
                )
                
                cognita_analyzer = CognitaRiskAnalyzer(neo4j_manager)
                
                self.workers['cognita'] = {
                    'agent': cognita_analyzer,
                    'status': WorkerStatus(
                        agent_id='cognita',
                        agent_name='관계형 데이터 분석 에이전트',
                        status='running',
                        last_heartbeat=datetime.now().isoformat(),
                        tasks_completed=0,
                        current_task=None
                    ),
                    'type': 'relational_data'
                }
                logger.info("✅ Cognita 워커 에이전트 초기화 완료")
            except Exception as e:
                logger.error(f"❌ Cognita 워커 에이전트 초기화 실패: {e}")
                self.workers['cognita'] = {
                    'agent': None,
                    'status': WorkerStatus(
                        agent_id='cognita',
                        agent_name='관계형 데이터 분석 에이전트',
                        status='error',
                        last_heartbeat=datetime.now().isoformat(),
                        tasks_completed=0,
                        current_task=None,
                        error_message=str(e)
                    ),
                    'type': 'relational_data'
                }
        
        logger.info(f"워커 에이전트 초기화 완료: {len(self.workers)}개 에이전트")
    
    def get_worker_status(self) -> Dict[str, WorkerStatus]:
        """모든 워커 에이전트 상태 조회"""
        status_dict = {}
        for worker_id, worker_info in self.workers.items():
            # 하트비트 업데이트
            worker_info['status'].last_heartbeat = datetime.now().isoformat()
            status_dict[worker_id] = worker_info['status']
        return status_dict
    
    def execute_task(self, task: AgenticTask) -> AgenticResult:
        """에이전틱 작업 실행"""
        logger.info(f"작업 실행 시작: {task.task_id} ({task.task_type})")
        
        start_time = time.time()
        result = AgenticResult(
            task_id=task.task_id,
            task_type=task.task_type
        )
        
        try:
            # 병렬 실행을 위한 Future 리스트
            futures = []
            
            # Structura 워커 실행
            if task.use_structura and 'structura' in self.workers:
                if self.workers['structura']['agent'] is not None:
                    future = self.executor.submit(self._execute_structura_task, task)
                    futures.append(('structura', future))
                    
                    # 워커 상태 업데이트
                    self.workers['structura']['status'].status = 'busy'
                    self.workers['structura']['status'].current_task = task.task_id
            
            # Cognita 워커 실행
            if task.use_cognita and 'cognita' in self.workers:
                if self.workers['cognita']['agent'] is not None:
                    future = self.executor.submit(self._execute_cognita_task, task)
                    futures.append(('cognita', future))
                    
                    # 워커 상태 업데이트
                    self.workers['cognita']['status'].status = 'busy'
                    self.workers['cognita']['status'].current_task = task.task_id
            
            # 결과 수집
            for worker_name, future in futures:
                try:
                    worker_result = future.result(timeout=60)  # 60초 타임아웃
                    
                    if worker_name == 'structura':
                        result.structura_result = worker_result
                    elif worker_name == 'cognita':
                        result.cognita_result = worker_result
                    
                    # 워커 상태 업데이트
                    self.workers[worker_name]['status'].status = 'running'
                    self.workers[worker_name]['status'].current_task = None
                    self.workers[worker_name]['status'].tasks_completed += 1
                    
                except Exception as e:
                    logger.error(f"워커 {worker_name} 실행 실패: {e}")
                    
                    # 워커 상태 업데이트 (에러)
                    self.workers[worker_name]['status'].status = 'error'
                    self.workers[worker_name]['status'].current_task = None
                    self.workers[worker_name]['status'].error_message = str(e)
            
            # 결합 분석 수행
            if result.structura_result and result.cognita_result:
                result.combined_analysis = self._combine_analysis_results(
                    result.structura_result, 
                    result.cognita_result,
                    task
                )
            
            result.execution_time = time.time() - start_time
            result.status = 'completed'
            
            # 결과 캐시에 저장
            self.result_cache[task.task_id] = result
            
            logger.info(f"작업 완료: {task.task_id} (소요시간: {result.execution_time:.2f}초)")
            
        except Exception as e:
            result.status = 'failed'
            result.error_message = str(e)
            result.execution_time = time.time() - start_time
            logger.error(f"작업 실행 실패: {task.task_id} - {e}")
        
        return result
    
    def _execute_structura_task(self, task: AgenticTask) -> Dict:
        """Structura 워커 작업 실행"""
        structura_agent = self.workers['structura']['agent']
        
        if task.task_type == 'individual_analysis' and task.employee_data:
            # 개별 직원 분석
            prediction_result = structura_agent.predict_single(task.employee_data)
            explanation_result = structura_agent.explain_prediction(task.employee_data)
            
            return {
                'prediction': prediction_result.to_dict(),
                'explanation': explanation_result.to_dict(),
                'agent_type': 'structura',
                'analysis_type': 'individual'
            }
        
        elif task.task_type == 'department_analysis' and task.department_name:
            # 부서 분석 (Structura는 개별 직원 기반이므로 시뮬레이션)
            return {
                'message': 'Structura는 개별 직원 분석에 특화되어 있습니다.',
                'agent_type': 'structura',
                'analysis_type': 'department',
                'recommendation': '부서별 분석을 위해서는 개별 직원 데이터가 필요합니다.'
            }
        
        else:
            raise ValueError(f"지원하지 않는 작업 유형: {task.task_type}")
    
    def _execute_cognita_task(self, task: AgenticTask) -> Dict:
        """Cognita 워커 작업 실행"""
        cognita_agent = self.workers['cognita']['agent']
        
        if task.task_type == 'individual_analysis' and task.employee_data:
            # 개별 직원 분석 (employee_id 필요)
            employee_id = task.employee_data.get('employee_id') or task.employee_data.get('EmployeeNumber', '1')
            
            risk_metrics = cognita_agent.analyze_employee_risk(str(employee_id))
            
            return {
                'risk_analysis': asdict(risk_metrics),
                'agent_type': 'cognita',
                'analysis_type': 'individual'
            }
        
        elif task.task_type == 'department_analysis' and task.department_name:
            # 부서 분석
            sample_size = task.sample_size or 20
            risk_analyses = cognita_agent.batch_analyze_department(task.department_name, sample_size)
            
            if risk_analyses:
                report = cognita_agent.generate_risk_report(risk_analyses)
                return {
                    'department_report': report,
                    'agent_type': 'cognita',
                    'analysis_type': 'department',
                    'analyzed_employees': len(risk_analyses)
                }
            else:
                return {
                    'message': f'부서 "{task.department_name}"에서 분석할 직원을 찾을 수 없습니다.',
                    'agent_type': 'cognita',
                    'analysis_type': 'department'
                }
        
        else:
            raise ValueError(f"지원하지 않는 작업 유형: {task.task_type}")
    
    def _combine_analysis_results(self, structura_result: Dict, cognita_result: Dict, task: AgenticTask) -> Dict:
        """두 워커 에이전트 결과 결합 분석"""
        
        combined = {
            'analysis_type': 'combined',
            'task_type': task.task_type,
            'structura_insights': [],
            'cognita_insights': [],
            'integrated_assessment': {},
            'recommendations': []
        }
        
        if task.task_type == 'individual_analysis':
            # 개별 직원 통합 분석
            
            # Structura 인사이트
            if 'prediction' in structura_result:
                pred = structura_result['prediction']
                combined['structura_insights'] = [
                    f"이직 확률: {pred['attrition_probability']:.1%}",
                    f"위험 범주: {pred['risk_category']}",
                    f"신뢰도: {pred['confidence_score']:.1%}"
                ]
            
            # Cognita 인사이트
            if 'risk_analysis' in cognita_result:
                risk = cognita_result['risk_analysis']
                combined['cognita_insights'] = [
                    f"종합 위험도: {risk['overall_risk_score']:.3f}",
                    f"사회적 고립: {risk['social_isolation_index']:.3f}",
                    f"네트워크 중심성: {risk['network_centrality_score']:.3f}"
                ]
            
            # 통합 평가
            structura_prob = structura_result.get('prediction', {}).get('attrition_probability', 0)
            cognita_risk = cognita_result.get('risk_analysis', {}).get('overall_risk_score', 0)
            
            # 가중 평균 (Structura 60%, Cognita 40%)
            integrated_risk = (structura_prob * 0.6) + (cognita_risk * 0.4)
            
            combined['integrated_assessment'] = {
                'integrated_risk_score': integrated_risk,
                'risk_level': 'HIGH' if integrated_risk >= 0.7 else 'MEDIUM' if integrated_risk >= 0.4 else 'LOW',
                'structura_weight': 0.6,
                'cognita_weight': 0.4,
                'consensus': 'HIGH' if structura_prob > 0.6 and cognita_risk > 0.6 else 'MIXED'
            }
            
            # 통합 권장사항
            if integrated_risk >= 0.7:
                combined['recommendations'] = [
                    "즉시 1:1 면담 실시",
                    "업무 환경 개선 방안 검토",
                    "팀 내 역할 및 관계 개선"
                ]
            elif integrated_risk >= 0.4:
                combined['recommendations'] = [
                    "정기적 모니터링 강화",
                    "멘토링 프로그램 참여 검토",
                    "업무 만족도 개선 방안 모색"
                ]
            else:
                combined['recommendations'] = [
                    "현재 상태 유지",
                    "지속적 관찰",
                    "긍정적 요인 강화"
                ]
        
        return combined

# ------------------------------------------------------
# Flask 애플리케이션 생성
# ------------------------------------------------------

def create_app():
    """Flask 애플리케이션 팩토리"""
    
    app = Flask(__name__)
    
    # CORS 설정 (React 연동)
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # 설정
    app.config['JSON_AS_ASCII'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    
    # 워커 에이전트 관리자
    worker_manager = None
    
    # ------------------------------------------------------
    # 애플리케이션 초기화
    # ------------------------------------------------------
    
    @app.before_first_request
    def initialize_services():
        """첫 요청 전 서비스 초기화"""
        nonlocal worker_manager
        
        try:
            logger.info("Agentic AI Master Server 초기화 중...")
            
            # 워커 에이전트 관리자 초기화
            worker_manager = WorkerAgentManager()
            
            # Flask 앱에 저장
            app.worker_manager = worker_manager
            
            logger.info("Agentic AI Master Server 준비 완료")
            
        except Exception as e:
            logger.error(f"서비스 초기화 실패: {str(e)}")
            raise
    
    # ------------------------------------------------------
    # 유틸리티 함수
    # ------------------------------------------------------
    
    def get_worker_manager():
        """워커 관리자 가져오기"""
        if not hasattr(app, 'worker_manager') or app.worker_manager is None:
            return None
        return app.worker_manager
    
    # ------------------------------------------------------
    # 에러 핸들러
    # ------------------------------------------------------
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Not Found",
            "message": "요청한 리소스를 찾을 수 없습니다",
            "status_code": 404
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "error": "Internal Server Error", 
            "message": "서버 내부 오류가 발생했습니다",
            "status_code": 500
        }), 500
    
    # ------------------------------------------------------
    # API 라우트
    # ------------------------------------------------------
    
    @app.route('/')
    def index():
        """루트 엔드포인트"""
        return jsonify({
            "service": "Agentic AI Master Server",
            "version": "1.0.0",
            "status": "running",
            "description": "워커 에이전트들을 통합 관리하는 마스터 서버",
            "architecture": {
                "supervisor_agent": "미구현 (향후 확장)",
                "worker_agents": {
                    "agent_1": "정형 데이터 분석 (Structura)",
                    "agent_2": "관계형 데이터 분석 (Cognita)",
                    "agent_3": "시계열 데이터 분석 (미구현)",
                    "agent_4": "자연어 데이터 분석 (미구현)",
                    "agent_5": "외부 시장 분석 (미구현)"
                },
                "final_synthesis_agent": "미구현 (향후 확장)"
            },
            "endpoints": {
                "health": "/api/health",
                "workers_status": "/api/workers/status",
                "analyze_individual": "/api/analyze/individual",
                "analyze_department": "/api/analyze/department",
                "task_status": "/api/task/{task_id}/status",
                "task_result": "/api/task/{task_id}/result"
            }
        })
    
    @app.route('/api/health')
    def health_check():
        """헬스체크 엔드포인트"""
        
        worker_mgr = get_worker_manager()
        
        if not worker_mgr:
            return jsonify({
                "status": "error",
                "message": "워커 관리자가 초기화되지 않았습니다",
                "timestamp": datetime.now().isoformat()
            }), 503
        
        # 워커 상태 확인
        worker_status = worker_mgr.get_worker_status()
        
        # 전체 시스템 상태 결정
        all_running = all(status.status == 'running' for status in worker_status.values())
        any_error = any(status.status == 'error' for status in worker_status.values())
        
        system_status = "healthy" if all_running else "degraded" if not any_error else "error"
        
        return jsonify({
            "status": system_status,
            "worker_count": len(worker_status),
            "workers": {worker_id: asdict(status) for worker_id, status in worker_status.items()},
            "capabilities": {
                "structura_available": STRUCTURA_AVAILABLE,
                "cognita_available": COGNITA_AVAILABLE
            },
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route('/api/workers/status')
    def get_workers_status():
        """워커 에이전트 상태 조회"""
        
        worker_mgr = get_worker_manager()
        if not worker_mgr:
            return jsonify({"error": "워커 관리자가 초기화되지 않았습니다"}), 503
        
        worker_status = worker_mgr.get_worker_status()
        
        return jsonify({
            "workers": {worker_id: asdict(status) for worker_id, status in worker_status.items()},
            "summary": {
                "total_workers": len(worker_status),
                "running": len([s for s in worker_status.values() if s.status == 'running']),
                "busy": len([s for s in worker_status.values() if s.status == 'busy']),
                "error": len([s for s in worker_status.values() if s.status == 'error'])
            },
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route('/api/analyze/individual', methods=['POST'])
    def analyze_individual():
        """개별 직원 통합 분석"""
        
        worker_mgr = get_worker_manager()
        if not worker_mgr:
            return jsonify({"error": "워커 관리자가 초기화되지 않았습니다"}), 503
        
        try:
            # 요청 데이터 파싱
            data = request.get_json()
            if not data:
                return jsonify({"error": "분석할 직원 데이터가 필요합니다"}), 400
            
            # 작업 생성
            task_id = f"individual_{int(time.time() * 1000)}"
            task = AgenticTask(
                task_id=task_id,
                task_type='individual_analysis',
                employee_data=data,
                use_structura=data.get('use_structura', True),
                use_cognita=data.get('use_cognita', True)
            )
            
            # 작업 실행
            result = worker_mgr.execute_task(task)
            
            return jsonify(asdict(result))
            
        except Exception as e:
            logger.error(f"개별 분석 실패: {str(e)}")
            return jsonify({"error": f"개별 분석 실패: {str(e)}"}), 500
    
    @app.route('/api/analyze/department', methods=['POST'])
    def analyze_department():
        """부서별 통합 분석"""
        
        worker_mgr = get_worker_manager()
        if not worker_mgr:
            return jsonify({"error": "워커 관리자가 초기화되지 않았습니다"}), 503
        
        try:
            # 요청 데이터 파싱
            data = request.get_json()
            if not data or not data.get('department_name'):
                return jsonify({"error": "부서명이 필요합니다"}), 400
            
            # 작업 생성
            task_id = f"department_{int(time.time() * 1000)}"
            task = AgenticTask(
                task_id=task_id,
                task_type='department_analysis',
                department_name=data['department_name'],
                sample_size=data.get('sample_size', 20),
                use_structura=data.get('use_structura', True),
                use_cognita=data.get('use_cognita', True)
            )
            
            # 작업 실행
            result = worker_mgr.execute_task(task)
            
            return jsonify(asdict(result))
            
        except Exception as e:
            logger.error(f"부서 분석 실패: {str(e)}")
            return jsonify({"error": f"부서 분석 실패: {str(e)}"}), 500
    
    @app.route('/api/task/<task_id>/result')
    def get_task_result(task_id):
        """작업 결과 조회"""
        
        worker_mgr = get_worker_manager()
        if not worker_mgr:
            return jsonify({"error": "워커 관리자가 초기화되지 않았습니다"}), 503
        
        if task_id in worker_mgr.result_cache:
            result = worker_mgr.result_cache[task_id]
            return jsonify(asdict(result))
        else:
            return jsonify({"error": f"작업 ID '{task_id}'를 찾을 수 없습니다"}), 404
    
    return app

# ------------------------------------------------------
# 서버 실행 함수
# ------------------------------------------------------

def run_server(host='0.0.0.0', port=8000, debug=True):
    """Agentic AI Master Server 실행"""
    app = create_app()
    
    print("=" * 70)
    print("🤖 Agentic AI Master Server 시작")
    print("=" * 70)
    print(f"📡 서버 주소: http://{host}:{port}")
    print(f"🔗 React 연동: http://localhost:3000에서 접근 가능")
    print(f"🔄 디버그 모드: {'활성화' if debug else '비활성화'}")
    print()
    print("🏗️ 아키텍처:")
    print("  📊 워커 에이전트 1: 정형 데이터 분석 (Structura)")
    print("  🕸️  워커 에이전트 2: 관계형 데이터 분석 (Cognita)")
    print("  ⏳ 워커 에이전트 3-5: 향후 확장 예정")
    print("  ⏳ Supervisor 에이전트: 향후 확장 예정")
    print("  ⏳ 최종 종합 에이전트: 향후 확장 예정")
    print()
    print("주요 엔드포인트:")
    print(f"  • 헬스체크: http://{host}:{port}/api/health")
    print(f"  • 워커 상태: http://{host}:{port}/api/workers/status")
    print(f"  • 개별 분석: http://{host}:{port}/api/analyze/individual")
    print(f"  • 부서 분석: http://{host}:{port}/api/analyze/department")
    print()
    print("워커 에이전트 상태:")
    print(f"  • Structura: {'✅' if STRUCTURA_AVAILABLE else '❌'}")
    print(f"  • Cognita: {'✅' if COGNITA_AVAILABLE else '❌'}")
    print()
    print("서버를 중지하려면 Ctrl+C를 누르세요.")
    print("=" * 70)
    
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_server()
