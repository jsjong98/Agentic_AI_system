"""
Supervisor Flask 백엔드
슈퍼바이저 에이전트의 REST API 서버
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
import os
import json

from langchain_openai import ChatOpenAI

from .langgraph_workflow import SupervisorWorkflow
from .worker_integrator import DEFAULT_WORKER_CONFIGS
from .agent_state import AgentState

# Flask 앱 생성
app = Flask(__name__)
CORS(app)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 전역 변수
supervisor_workflow: Optional[SupervisorWorkflow] = None
active_sessions: Dict[str, Dict[str, Any]] = {}


def initialize_supervisor():
    """슈퍼바이저 워크플로우 초기화"""
    global supervisor_workflow
    
    try:
        # 환경변수에서 설정 로드
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
        
        # LLM 초기화
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            api_key=openai_api_key
        )
        
        # 워커 설정 (환경변수에서 오버라이드 가능)
        worker_configs = DEFAULT_WORKER_CONFIGS.copy()
        
        # 환경변수에서 워커 URL 오버라이드
        for worker_name in worker_configs.keys():
            env_key = f"{worker_name.upper()}_URL"
            env_url = os.getenv(env_key)
            if env_url:
                worker_configs[worker_name]["base_url"] = env_url
                logger.info(f"Using {env_key}: {env_url}")
        
        # 슈퍼바이저 워크플로우 초기화
        supervisor_workflow = SupervisorWorkflow(
            worker_configs=worker_configs,
            llm=llm,
            max_retry_count=int(os.getenv("MAX_RETRY_COUNT", "3")),
            timeout_minutes=int(os.getenv("TIMEOUT_MINUTES", "30"))
        )
        
        logger.info("Supervisor workflow initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize supervisor workflow: {e}")
        logger.error(traceback.format_exc())
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    try:
        status = {
            'status': 'healthy',
            'service': 'Supervisor',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        if supervisor_workflow:
            status['workflow_initialized'] = True
            status['available_workers'] = supervisor_workflow.get_available_workers()
        else:
            status['workflow_initialized'] = False
            status['error'] = 'Workflow not initialized'
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/analyze_employee', methods=['POST'])
def analyze_employee():
    """직원 분석 요청"""
    try:
        if not supervisor_workflow:
            return jsonify({
                'success': False,
                'error': 'Supervisor workflow not initialized'
            }), 500
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        employee_id = data.get('employee_id')
        if not employee_id:
            return jsonify({
                'success': False,
                'error': 'employee_id is required'
            }), 400
        
        session_id = data.get('session_id')
        
        logger.info(f"Starting analysis for employee {employee_id}")
        
        # 비동기 분석 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                supervisor_workflow.analyze_employee(employee_id, session_id)
            )
            
            # 활성 세션에 결과 저장
            if result.get('session_id'):
                active_sessions[result['session_id']] = {
                    'employee_id': employee_id,
                    'result': result,
                    'created_at': datetime.now().isoformat(),
                    'status': 'completed' if result['success'] else 'failed'
                }
            
            logger.info(f"Analysis completed for employee {employee_id}")
            return jsonify(result)
            
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Analysis error for employee {data.get('employee_id', 'unknown')}: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/get_workflow_status/<session_id>', methods=['GET'])
def get_workflow_status(session_id: str):
    """워크플로우 상태 조회"""
    try:
        if not supervisor_workflow:
            return jsonify({
                'success': False,
                'error': 'Supervisor workflow not initialized'
            }), 500
        
        # 비동기 상태 조회
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            status = loop.run_until_complete(
                supervisor_workflow.get_workflow_status(session_id)
            )
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'status': status
            })
            
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Status query error for session {session_id}: {e}")
        
        return jsonify({
            'success': False,
            'error': str(e),
            'session_id': session_id
        }), 500


@app.route('/list_active_sessions', methods=['GET'])
def list_active_sessions():
    """활성 세션 목록 조회"""
    try:
        sessions = []
        for session_id, session_data in active_sessions.items():
            sessions.append({
                'session_id': session_id,
                'employee_id': session_data['employee_id'],
                'status': session_data['status'],
                'created_at': session_data['created_at']
            })
        
        return jsonify({
            'success': True,
            'active_sessions': sessions,
            'total_sessions': len(sessions)
        })
        
    except Exception as e:
        logger.error(f"List sessions error: {e}")
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/get_session_result/<session_id>', methods=['GET'])
def get_session_result(session_id: str):
    """세션 결과 조회"""
    try:
        if session_id not in active_sessions:
            return jsonify({
                'success': False,
                'error': 'Session not found'
            }), 404
        
        session_data = active_sessions[session_id]
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'employee_id': session_data['employee_id'],
            'status': session_data['status'],
            'created_at': session_data['created_at'],
            'result': session_data['result']
        })
        
    except Exception as e:
        logger.error(f"Get session result error for {session_id}: {e}")
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/worker_health_check', methods=['GET'])
def worker_health_check():
    """워커 에이전트 상태 확인"""
    try:
        if not supervisor_workflow:
            return jsonify({
                'success': False,
                'error': 'Supervisor workflow not initialized'
            }), 500
        
        # 비동기 헬스체크
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            health_status = loop.run_until_complete(
                supervisor_workflow.worker_integrator.health_check_all()
            )
            
            # 결과 변환
            worker_status = {}
            for worker_type, is_healthy in health_status.items():
                worker_status[worker_type.value] = {
                    'healthy': is_healthy,
                    'status': 'online' if is_healthy else 'offline'
                }
            
            total_workers = len(worker_status)
            healthy_workers = sum(1 for status in worker_status.values() if status['healthy'])
            
            return jsonify({
                'success': True,
                'worker_status': worker_status,
                'summary': {
                    'total_workers': total_workers,
                    'healthy_workers': healthy_workers,
                    'unhealthy_workers': total_workers - healthy_workers,
                    'health_rate': healthy_workers / total_workers if total_workers > 0 else 0
                },
                'timestamp': datetime.now().isoformat()
            })
            
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Worker health check error: {e}")
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/system_info', methods=['GET'])
def system_info():
    """시스템 정보 조회"""
    try:
        info = {
            'service': 'Supervisor',
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'workflow_initialized': supervisor_workflow is not None,
            'active_sessions_count': len(active_sessions),
            'environment': {
                'max_retry_count': os.getenv('MAX_RETRY_COUNT', '3'),
                'timeout_minutes': os.getenv('TIMEOUT_MINUTES', '30'),
                'openai_api_key_configured': bool(os.getenv('OPENAI_API_KEY'))
            }
        }
        
        if supervisor_workflow:
            info['available_workers'] = supervisor_workflow.get_available_workers()
            info['worker_configs'] = supervisor_workflow.worker_integrator.get_worker_status()
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"System info error: {e}")
        
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/clear_sessions', methods=['POST'])
def clear_sessions():
    """세션 정리"""
    try:
        data = request.get_json() or {}
        older_than_hours = data.get('older_than_hours', 24)
        
        current_time = datetime.now()
        cleared_count = 0
        
        sessions_to_remove = []
        for session_id, session_data in active_sessions.items():
            created_at = datetime.fromisoformat(session_data['created_at'])
            age_hours = (current_time - created_at).total_seconds() / 3600
            
            if age_hours > older_than_hours:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del active_sessions[session_id]
            cleared_count += 1
        
        return jsonify({
            'success': True,
            'cleared_sessions': cleared_count,
            'remaining_sessions': len(active_sessions),
            'older_than_hours': older_than_hours
        })
        
    except Exception as e:
        logger.error(f"Clear sessions error: {e}")
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """배치 분석 (여러 직원 동시 분석)"""
    try:
        if not supervisor_workflow:
            return jsonify({
                'success': False,
                'error': 'Supervisor workflow not initialized'
            }), 500
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        employee_ids = data.get('employee_ids', [])
        if not employee_ids or not isinstance(employee_ids, list):
            return jsonify({
                'success': False,
                'error': 'employee_ids list is required'
            }), 400
        
        max_batch_size = int(os.getenv('MAX_BATCH_SIZE', '10'))
        if len(employee_ids) > max_batch_size:
            return jsonify({
                'success': False,
                'error': f'Batch size exceeds maximum ({max_batch_size})'
            }), 400
        
        logger.info(f"Starting batch analysis for {len(employee_ids)} employees")
        
        # 비동기 배치 분석
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 병렬 분석 실행
            tasks = []
            for employee_id in employee_ids:
                task = supervisor_workflow.analyze_employee(employee_id)
                tasks.append(task)
            
            results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            
            # 결과 정리
            batch_results = []
            successful_count = 0
            
            for i, result in enumerate(results):
                employee_id = employee_ids[i]
                
                if isinstance(result, Exception):
                    batch_results.append({
                        'employee_id': employee_id,
                        'success': False,
                        'error': str(result)
                    })
                else:
                    batch_results.append(result)
                    if result.get('success'):
                        successful_count += 1
                    
                    # 세션 저장
                    if result.get('session_id'):
                        active_sessions[result['session_id']] = {
                            'employee_id': employee_id,
                            'result': result,
                            'created_at': datetime.now().isoformat(),
                            'status': 'completed' if result['success'] else 'failed'
                        }
            
            logger.info(f"Batch analysis completed: {successful_count}/{len(employee_ids)} successful")
            
            return jsonify({
                'success': True,
                'batch_results': batch_results,
                'summary': {
                    'total_employees': len(employee_ids),
                    'successful_analyses': successful_count,
                    'failed_analyses': len(employee_ids) - successful_count,
                    'success_rate': successful_count / len(employee_ids)
                }
            })
            
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# 에러 핸들러
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500


# 앱 초기화
def create_app():
    """Flask 앱 생성 및 초기화"""
    try:
        initialize_supervisor()
        logger.info("Supervisor Flask backend initialized successfully")
        return app
    except Exception as e:
        logger.error(f"Failed to create app: {e}")
        raise


if __name__ == '__main__':
    try:
        # 환경변수 설정
        port = int(os.getenv('SUPERVISOR_PORT', '5006'))
        debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        
        # 앱 초기화
        app = create_app()
        
        print(f"🚀 Supervisor Flask 서버 시작")
        print(f"📡 서버 주소: http://localhost:{port}")
        print(f"🔧 디버그 모드: {debug}")
        print("\n사용 가능한 엔드포인트:")
        print("  GET  /health - 서버 상태 확인")
        print("  POST /analyze_employee - 직원 분석")
        print("  GET  /get_workflow_status/<session_id> - 워크플로우 상태")
        print("  GET  /list_active_sessions - 활성 세션 목록")
        print("  GET  /worker_health_check - 워커 상태 확인")
        print("  POST /batch_analyze - 배치 분석")
        print("  GET  /system_info - 시스템 정보")
        
        # 서버 실행
        app.run(host='0.0.0.0', port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(traceback.format_exc())
        exit(1)
