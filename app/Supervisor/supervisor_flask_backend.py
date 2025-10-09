"""
Supervisor Flask 백엔드
슈퍼바이저 에이전트의 REST API 서버
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import asyncio
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
import os
import json
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import requests
import time

from langchain_openai import ChatOpenAI
import openai

# 환경변수 로드
load_dotenv()

from .langgraph_workflow import SupervisorWorkflow
from .worker_integrator import DEFAULT_WORKER_CONFIGS
from .agent_state import AgentState
from .hierarchical_result_manager import hierarchical_result_manager

# Flask 앱 생성
app = Flask(__name__)

# CORS 설정 (React 연동)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Origin"],
        "supports_credentials": True
    }
})

# Flask 설정
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300MB 파일 업로드 제한

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 전역 변수
supervisor_workflow: Optional[SupervisorWorkflow] = None
active_sessions: Dict[str, Dict[str, Any]] = {}
openai_client: Optional[openai.OpenAI] = None


def initialize_supervisor():
    """슈퍼바이저 워크플로우 초기화"""
    global supervisor_workflow, openai_client
    
    try:
        # 환경변수에서 설정 로드
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            logger.warning("Supervisor 서버는 OpenAI API 키 없이 시작됩니다. LLM 기능은 비활성화됩니다.")
            llm = None
            openai_client = None
        else:
            # OpenAI 클라이언트 초기화
            openai_client = openai.OpenAI(api_key=openai_api_key)
            
            # LLM 초기화 (gpt-5 모델 사용)
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model="gpt-5",
                    temperature=0.1,
                    api_key=openai_api_key
                )
                logger.info("✅ OpenAI LLM (gpt-5) 및 클라이언트 초기화 완료")
            except Exception as llm_error:
                logger.warning(f"⚠️ OpenAI LLM 초기화 실패: {llm_error}")
                logger.warning("LLM 없이 Supervisor 서버를 시작합니다.")
                llm = None
        
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
        
        # 저장 경로 확인 로그
        logger.info(f"📁 Hierarchical Result Manager 저장 경로: {hierarchical_result_manager.base_output_dir}")
        logger.info(f"📁 절대 경로: {hierarchical_result_manager.base_output_dir.resolve()}")
        
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


def analyze_single_agent_sync(agent_name, employee_id, request_data):
    """개별 에이전트 분석 함수 (동기 버전)"""
    try:
        analysis_type = request_data.get('analysis_type', 'batch')
        
        if agent_name == 'structura':
            # Structura 분석 - 개별 직원 예측 엔드포인트 사용 (배치 분석 시에도)
            url = f"{os.getenv('STRUCTURA_URL', 'http://localhost:5001')}/api/predict"
            response = requests.post(url, json={'employee_id': employee_id, 'analysis_type': analysis_type, **request_data})
            return response.json() if response.ok else {'error': f'Structura API error: {response.status_code}'}
            
        elif agent_name == 'cognita':
            # Cognita 분석 - employee_id로 관계 분석 (post 데이터 불필요)
            url = f"{os.getenv('COGNITA_URL', 'http://localhost:5002')}/api/analyze/employee/{employee_id}"
            response = requests.get(url)
            return response.json() if response.ok else {'error': f'Cognita API error: {response.status_code}'}
            
        elif agent_name == 'chronos':
            # Chronos 분석 - 개별 직원 예측 엔드포인트 사용 (배치 분석 시에도)
            url = f"{os.getenv('CHRONOS_URL', 'http://localhost:5003')}/api/predict"
            response = requests.post(url, json={'employee_id': employee_id, 'analysis_type': analysis_type})
            return response.json() if response.ok else {'error': f'Chronos API error: {response.status_code}'}
            
        elif agent_name == 'sentio':
            # Sentio 분석 - employee_id로 감정 분석 (post 데이터 불필요)
            url = f"{os.getenv('SENTIO_URL', 'http://localhost:5004')}/analyze_sentiment"
            response = requests.post(url, json={'employee_id': employee_id, 'analysis_type': 'batch'})
            return response.json() if response.ok else {'error': f'Sentio API error: {response.status_code}'}
            
        elif agent_name == 'agora':
            # Agora 분석 - 실제 직원 데이터를 전달하여 정확한 시장 분석 수행
            url = f"{os.getenv('AGORA_URL', 'http://localhost:5005')}/api/agora/comprehensive-analysis"
            
            employee_data = None
            if 'employees' in request_data:
                for emp in request_data['employees']:
                    emp_id = emp.get('EmployeeNumber') or emp.get('employee_id') or emp.get('id')
                    if str(emp_id) == str(employee_id):
                        employee_data = emp
                        break
            
            if employee_data:
                logger.info(f"📊 Agora에게 직원 {employee_id}의 실제 데이터 전달: {list(employee_data.keys())}")
                response = requests.post(url, json={
                    'employee_id': employee_id, 
                    'analysis_type': analysis_type,
                    **employee_data  # 실제 직원 데이터 포함
                })
            else:
                logger.warning(f"⚠️ 직원 {employee_id}의 데이터를 찾을 수 없습니다. employee_id만 전달합니다.")
                response = requests.post(url, json={'employee_id': employee_id, 'analysis_type': analysis_type})
            
            return response.json() if response.ok else {'error': f'Agora API error: {response.status_code}'}
            
        else:
            return {'error': f'Unknown agent: {agent_name}'}
            
    except Exception as e:
        logger.error(f"Error in analyze_single_agent_sync for {agent_name}: {e}")
        return {'error': str(e)}


@app.route('/batch_progress/<batch_id>', methods=['GET'])
def get_batch_progress(batch_id):
    """배치 분석 진행률 조회"""
    try:
        if not hasattr(app, 'batch_progress') or batch_id not in app.batch_progress:
            return jsonify({
                'success': False,
                'error': 'Batch session not found'
            }), 404
        
        progress_data = app.batch_progress[batch_id]
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'status': progress_data['status'],
            'overall_progress': progress_data.get('overall_progress', 0),
            'current_agent': progress_data.get('current_agent'),
            'agent_progress': progress_data.get('agent_progress', {}),
            'total_employees': progress_data['total_employees'],
            'start_time': progress_data['start_time'],
            'end_time': progress_data.get('end_time'),
            'completed': progress_data['status'] in ['completed', 'failed']
        })
        
    except Exception as e:
        logger.error(f"Batch progress error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
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
            logger.error("No JSON data provided in batch_analyze request")
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        logger.info(f"Received batch_analyze request with keys: {list(data.keys())}")
        
        employee_ids = data.get('employee_ids', [])
        logger.info(f"Employee IDs received: {employee_ids}")
        logger.info(f"Employee IDs type: {type(employee_ids)}")
        
        if not employee_ids or not isinstance(employee_ids, list):
            logger.error(f"Invalid employee_ids: {employee_ids} (type: {type(employee_ids)})")
            return jsonify({
                'success': False,
                'error': 'employee_ids list is required'
            }), 400
        
        # 분석 타입 추출 (기본값: batch)
        analysis_type = data.get('analysis_type', 'batch')
        logger.info(f"Analysis type: {analysis_type}")
        
        max_batch_size = int(os.getenv('MAX_BATCH_SIZE', '2000'))  # 대용량 배치 분석을 위해 2000으로 증가
        if len(employee_ids) > max_batch_size:
            logger.warning(f"Large batch size detected: {len(employee_ids)} employees (max: {max_batch_size})")
            return jsonify({
                'success': False,
                'error': f'Batch size exceeds maximum ({max_batch_size}). 현재 요청: {len(employee_ids)}명'
            }), 400
        
        logger.info(f"🚀 Starting batch analysis for {len(employee_ids)} employees")
        logger.info(f"📊 Analysis configuration: {data.get('integration_config', {})}")
        logger.info(f"🔧 Agent configuration: {data.get('agentConfig', {})}")
        logger.info(f"📁 Agent files: {data.get('agent_files', {})}")
        
        # 배치 진행률 추적을 위한 세션 생성
        import uuid
        batch_id = str(uuid.uuid4())
        
        # 전역 배치 상태 저장소 초기화
        if not hasattr(app, 'batch_progress'):
            app.batch_progress = {}
        
        # 배치 진행률 초기화 (에이전트별 진행률 추적)
        app.batch_progress[batch_id] = {
            'total_employees': len(employee_ids),
            'current_agent': 'structura',
            'agent_progress': {
                'structura': 0,
                'cognita': 0,
                'chronos': 0,
                'sentio': 0,
                'agora': 0
            },
            'overall_progress': 0,
            'status': 'processing',
            'start_time': datetime.now().isoformat(),
            'results': {}
        }
        
        # 에이전트별 순차 배치 분석
        agents = ['structura', 'cognita', 'chronos', 'sentio', 'agora']
        agent_results = {}
        
        for agent_idx, agent_name in enumerate(agents):
            logger.info(f"🚀 Starting {agent_name} analysis for {len(employee_ids)} employees")
            app.batch_progress[batch_id]['current_agent'] = agent_name
            
            # 각 에이전트별로 모든 직원 처리
            agent_results[agent_name] = []
            
            for emp_idx, employee_id in enumerate(employee_ids):
                logger.info(f"🔮 {agent_name}: Processing employee {emp_idx+1}/{len(employee_ids)}: {employee_id}")
                
                # 진행률 업데이트 (처리 시작 시점)
                agent_progress = (emp_idx / len(employee_ids)) * 100
                app.batch_progress[batch_id]['agent_progress'][agent_name] = round(agent_progress, 1)
                
                # 전체 진행률 업데이트 (처리 시작 시점)
                overall_progress = ((agent_idx * len(employee_ids) + emp_idx) / (len(agents) * len(employee_ids))) * 100
                app.batch_progress[batch_id]['overall_progress'] = round(overall_progress, 1)
                
                logger.info(f"📊 진행률 업데이트: {agent_name} {agent_progress:.1f}%, 전체 {overall_progress:.1f}%")
                
                try:
                    # 개별 에이전트 분석 (employee_id만 전달)
                    result = analyze_single_agent_sync(agent_name, employee_id, data)
                    agent_results[agent_name].append({
                        'employee_id': employee_id,
                        'success': True,
                        'result': result
                    })
                    logger.info(f"✅ {agent_name}: Employee {employee_id} 분석 완료")
                except Exception as e:
                    logger.error(f"❌ {agent_name} error for employee {employee_id}: {e}")
                    agent_results[agent_name].append({
                        'employee_id': employee_id,
                        'success': False,
                        'error': str(e)
                    })
                
                # 진행률 업데이트 (처리 완료 시점)
                agent_progress = ((emp_idx + 1) / len(employee_ids)) * 100
                app.batch_progress[batch_id]['agent_progress'][agent_name] = round(agent_progress, 1)
                
                # 전체 진행률 업데이트 (처리 완료 시점)
                overall_progress = ((agent_idx * len(employee_ids) + emp_idx + 1) / (len(agents) * len(employee_ids))) * 100
                app.batch_progress[batch_id]['overall_progress'] = round(overall_progress, 1)
                
                logger.info(f"📈 진행률 완료: {agent_name} {agent_progress:.1f}%, 전체 {overall_progress:.1f}%")
            
            logger.info(f"✅ {agent_name} analysis completed for all employees")
        
        # 결과 통합 (에이전트별 결과를 직원별로 재구성)
        logger.info(f"🔄 결과 통합 시작: {len(employee_ids)}명의 직원 결과 처리")
        batch_results = []
        successful_count = 0
        
        for i, employee_id in enumerate(employee_ids):
            employee_result = {
                'employee_id': employee_id,
                'success': True,
                'agent_results': {}
            }
            
            # 각 에이전트 결과 수집
            agent_success_count = 0
            for agent_name in agents:
                agent_result = next((r for r in agent_results[agent_name] if r['employee_id'] == employee_id), None)
                if agent_result:
                    employee_result['agent_results'][agent_name] = agent_result
                    if agent_result['success']:
                        agent_success_count += 1
                    else:
                        employee_result['success'] = False
                        logger.warning(f"⚠️ 직원 {employee_id}: {agent_name} 분석 실패")
            
            batch_results.append(employee_result)
            if employee_result['success']:
                successful_count += 1
            
            # 진행률 로그 (100명마다)
            if (i + 1) % 100 == 0 or (i + 1) == len(employee_ids):
                logger.info(f"📊 결과 통합 진행률: {i + 1}/{len(employee_ids)} ({((i + 1)/len(employee_ids)*100):.1f}%)")
                logger.info(f"   - 성공한 직원: {successful_count}명")
                logger.info(f"   - 평균 에이전트 성공률: {agent_success_count}/{len(agents)}")
        
        # 배치 완료 상태 업데이트
        app.batch_progress[batch_id]['status'] = 'completed'
        app.batch_progress[batch_id]['processed_employees'] = len(employee_ids)
        app.batch_progress[batch_id]['end_time'] = datetime.now().isoformat()
        
        logger.info(f"🎉 배치 분석 완료!")
        logger.info(f"   📊 총 직원 수: {len(employee_ids)}명")
        logger.info(f"   ✅ 성공한 분석: {successful_count}명")
        logger.info(f"   ❌ 실패한 분석: {len(employee_ids) - successful_count}명")
        logger.info(f"   📈 성공률: {(successful_count/len(employee_ids)*100):.1f}%")
        logger.info(f"   🆔 배치 ID: {batch_id}")
        
        # 에이전트별 성공률 로그
        for agent_name in agents:
            agent_success = sum(1 for r in agent_results[agent_name] if r.get('success', False))
            logger.info(f"   🤖 {agent_name}: {agent_success}/{len(employee_ids)} ({(agent_success/len(employee_ids)*100):.1f}%)")
            
        return jsonify({
            'success': True,
            'batch_id': batch_id,  # 진행률 추적을 위한 batch_id 반환
            'batch_results': batch_results,
            'summary': {
                'total_employees': len(employee_ids),
                'successful_analyses': successful_count,
                'failed_analyses': len(employee_ids) - successful_count,
                'success_rate': successful_count / len(employee_ids)
            }
        })
            
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/post_analysis', methods=['POST'])
def post_analysis():
    """사후 분석 전용 엔드포인트 (워크플로우 문제 회피)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        employee_ids = data.get('employee_ids', [])
        agent_config = data.get('agent_config', {})
        
        if not employee_ids:
            return jsonify({
                'success': False,
                'error': 'employee_ids list is required'
            }), 400
        
        logger.info(f"Starting post-analysis for {len(employee_ids)} employees")
        
        # 각 에이전트별로 직접 API 호출 (워크플로우 우회)
        import requests
        
        results = {}
        successful_analyses = 0
        
        for employee_id in employee_ids:  # 전체 직원 분석
            employee_result = {
                'employee_id': employee_id,
                'agent_results': {}
            }
            
            # 각 에이전트별 분석
            for agent_name, is_enabled in agent_config.items():
                if not is_enabled:
                    continue
                    
                agent_type = agent_name.replace('use_', '')
                
                try:
                    if agent_type == 'structura':
                        # Structura API 호출
                        response = requests.get(f'http://localhost:5001/api/predict/{employee_id}', timeout=10)
                        if response.status_code == 200:
                            employee_result['agent_results']['structura'] = response.json()
                    
                    elif agent_type == 'cognita':
                        # Cognita API 호출
                        response = requests.get(f'http://localhost:5002/api/analyze/employee/{employee_id}', timeout=10)
                        if response.status_code == 200:
                            employee_result['agent_results']['cognita'] = response.json()
                    
                    elif agent_type == 'chronos':
                        # Chronos API 호출
                        response = requests.post(f'http://localhost:5003/api/predict', 
                                               json={'employee_id': employee_id}, timeout=10)
                        if response.status_code == 200:
                            employee_result['agent_results']['chronos'] = response.json()
                    
                    elif agent_type == 'sentio':
                        # Sentio API 호출
                        response = requests.post(f'http://localhost:5004/analyze_sentiment', 
                                               json={'employee_id': employee_id}, timeout=10)
                        if response.status_code == 200:
                            employee_result['agent_results']['sentio'] = response.json()
                    
                    elif agent_type == 'agora':
                        # Agora API 호출
                        response = requests.post(f'http://localhost:5005/api/agora/comprehensive-analysis', 
                                               json={'employee_id': employee_id}, timeout=10)
                        if response.status_code == 200:
                            employee_result['agent_results']['agora'] = response.json()
                
                except Exception as e:
                    logger.warning(f"Agent {agent_type} failed for employee {employee_id}: {e}")
                    employee_result['agent_results'][agent_type] = {'error': str(e)}
            
            results[employee_id] = employee_result
            if len(employee_result['agent_results']) > 0:
                successful_analyses += 1
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total_employees': len(employee_ids),
                'successful_analyses': successful_analyses,
                'processed_employees': len(results)
            }
        })
        
    except Exception as e:
        logger.error(f"Post-analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chat', methods=['POST'])
def chat_with_llm():
    """
    LLM과 채팅하는 API 엔드포인트 (GPT-5-nano-2025-08-07 사용)
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "메시지가 필요합니다."}), 400
        
        user_message = data['message']
        context = data.get('context', {})  # 분석 결과 컨텍스트
        
        if not openai_client:
            return jsonify({"error": "OpenAI API가 초기화되지 않았습니다."}), 500
        
        # 사용자 메시지 유형 판단
        is_simple_greeting = is_greeting_or_simple_question(user_message)
        
        # 시스템 프롬프트 생성
        system_prompt = create_system_prompt(context, is_simple_greeting)
        
        # OpenAI API 호출 (GPT-5-nano-2025-08-07 사용)
        try:
            # GPT-5-nano 모델 사용 시도
            response = openai_client.responses.create(
                model="gpt-5-nano-2025-08-07",
                input=f"{system_prompt}\n\nUser: {user_message}",
                reasoning={"effort": "medium"},
                text={"verbosity": "medium"}
            )
            ai_response = response.output_text
            model_used = "gpt-5-nano-2025-08-07"
            tokens_used = len(ai_response.split())  # 대략적인 토큰 수
            
        except Exception as e:
            logger.warning(f"GPT-5-nano 호출 실패, GPT-4o-mini로 fallback: {e}")
            # Fallback to GPT-4o-mini
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            ai_response = response.choices[0].message.content
            model_used = "gpt-4o-mini"
            tokens_used = response.usage.total_tokens if response.usage else 0
        
        return jsonify({
            "response": ai_response,
            "timestamp": datetime.now().isoformat(),
            "model": model_used,
            "tokens_used": tokens_used
        })
        
    except Exception as e:
        logger.error(f"채팅 API 오류: {str(e)}")
        return jsonify({"error": f"채팅 처리 중 오류가 발생했습니다: {str(e)}"}), 500


def is_greeting_or_simple_question(message: str) -> bool:
    """사용자 메시지가 단순한 인사말이나 일반적인 질문인지 판단"""
    message_lower = message.lower().strip()
    
    # 인사말 패턴
    greetings = [
        '안녕', '안녕하세요', '안녕하십니까', '반갑습니다', '처음 뵙겠습니다',
        'hello', 'hi', '하이', '헬로', '좋은 아침', '좋은 오후', '좋은 저녁'
    ]
    
    # 간단한 질문 패턴
    simple_questions = [
        '뭐해', '뭐 하세요', '어떻게 지내', '잘 지내', '어떠세요', '괜찮아',
        '도움', '도와줘', '도와주세요', '뭐가 가능해', '뭘 할 수 있어',
        '기능', '사용법', '어떻게 사용'
    ]
    
    # 메시지가 짧고 (20자 이하) 인사말이나 간단한 질문 패턴에 해당하는지 확인
    if len(message_lower) <= 20:
        for greeting in greetings:
            if greeting in message_lower:
                return True
        for question in simple_questions:
            if question in message_lower:
                return True
    
    return False


def create_system_prompt(context: Dict[str, Any], is_simple_greeting: bool = False) -> str:
    """분석 결과를 바탕으로 시스템 프롬프트 생성"""
    
    if is_simple_greeting:
        # 간단한 인사말이나 일반적인 질문에 대한 프롬프트
        base_prompt = """당신은 Retain Sentinel 360 AI 어시스턴트입니다. 
HR 데이터 분석과 이직 예측 분석을 도와드리는 친근한 AI 어시스턴트입니다.

사용자가 간단한 인사말이나 일반적인 질문을 했습니다. 
다음과 같이 응답해주세요:
- 간단하고 친근하게 인사를 나누세요 (2-3문장 정도)
- 어떤 도움을 드릴 수 있는지 간략하게 안내해주세요
- 너무 길거나 전문적인 설명은 피해주세요
- 따뜻하고 자연스러운 톤으로 대화해주세요"""
    else:
        # 전문적인 질문이나 분석 관련 질문에 대한 프롬프트
        base_prompt = """당신은 Retain Sentinel 360 AI 어시스턴트입니다. 
HR 데이터 분석과 이직 예측 분석을 도와드리는 전문 AI 어시스턴트입니다.

응답 가이드라인:
- 사용자의 질문에 맞게 전문적이고 실용적인 조언을 제공하세요
- 구체적인 데이터와 수치를 활용하여 설명하세요
- 실행 가능한 조치 방안을 제시해주세요
- 이해하기 쉽게 설명하되, 전문성을 유지해주세요

주요 기능:
1. 이직 위험도 분석 결과 해석 및 조언
2. HR 관리 관련 실용적인 가이드 제공
3. 데이터 기반 인사이트 및 권장사항 제시
4. 구체적이고 실행 가능한 개선 방안 제안"""

    # 분석 결과 컨텍스트 추가
    if context:
        context_info = "\n\n현재 분석 결과 컨텍스트:\n"
        
        if 'totalEmployees' in context:
            context_info += f"- 전체 직원 수: {context['totalEmployees']}명\n"
        
        if 'highRiskCount' in context:
            context_info += f"- 고위험군: {context['highRiskCount']}명\n"
        
        if 'mediumRiskCount' in context:
            context_info += f"- 중위험군: {context['mediumRiskCount']}명\n"
        
        if 'lowRiskCount' in context:
            context_info += f"- 저위험군: {context['lowRiskCount']}명\n"
        
        if 'accuracy' in context:
            context_info += f"- 모델 정확도: {context['accuracy']}%\n"
        
        if 'departmentStats' in context:
            context_info += "- 부서별 현황 데이터 보유\n"
        
        if 'keyInsights' in context:
            context_info += f"- 주요 인사이트: {len(context['keyInsights'])}개\n"
        
        base_prompt += context_info
    
    return base_prompt


# 에러 핸들러
# ------------------------------------------------------
# 파일 업로드 및 관리 기능
# ------------------------------------------------------

# 에이전트별 파일 업로드 기능 (새로운 체계적 관리)
@app.route('/api/upload/agent', methods=['POST'])
def upload_agent_file():
    """에이전트별 파일 업로드 (batch/post 분석용)"""
    try:
        # 파일 확인
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "파일이 업로드되지 않았습니다."
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "파일이 선택되지 않았습니다."
            }), 400
        
        # 파라미터 검증
        agent_type = request.form.get('agent_type')  # structura, chronos, sentio, agora
        analysis_type = request.form.get('analysis_type')  # batch, post
        
        if not agent_type or not analysis_type:
            return jsonify({
                "success": False,
                "error": "agent_type과 analysis_type이 필요합니다."
            }), 400
        
        if agent_type not in ['structura', 'chronos', 'sentio', 'agora']:
            return jsonify({
                "success": False,
                "error": "유효하지 않은 agent_type입니다. (structura, chronos, sentio, agora)"
            }), 400
        
        if analysis_type not in ['batch', 'post']:
            return jsonify({
                "success": False,
                "error": "유효하지 않은 analysis_type입니다. (batch, post)"
            }), 400
        
        # 파일 확장자 확인
        allowed_extensions = ['.csv', '.json']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                "success": False,
                "error": "CSV 또는 JSON 파일만 업로드 가능합니다."
            }), 400
        
        # 파일 저장 경로 생성
        filename = secure_filename(file.filename)
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads', agent_type, analysis_type)
        os.makedirs(upload_dir, exist_ok=True)
        
        # 타임스탬프를 포함한 파일명으로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(filename)[0]
        new_filename = f"{base_name}_{timestamp}{file_ext}"
        file_path = os.path.join(upload_dir, new_filename)
        
        # 파일 저장
        file.save(file_path)
        
        # 파일 정보 검증 및 메타데이터 생성
        try:
            file_info = {
                "original_filename": filename,
                "saved_filename": new_filename,
                "agent_type": agent_type,
                "analysis_type": analysis_type,
                "file_path": file_path,
                "relative_path": os.path.join('uploads', agent_type, analysis_type, new_filename),
                "upload_time": datetime.now().isoformat(),
                "file_size": os.path.getsize(file_path),
                "file_extension": file_ext
            }
            
            # CSV 파일인 경우 추가 정보
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
                file_info.update({
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist()
                })
                
                # Structura의 경우 Attrition 컬럼 확인
                if agent_type == 'structura' and 'Attrition' not in df.columns:
                    logger.warning(f"Structura 파일에 Attrition 컬럼이 없습니다: {new_filename}")
                    file_info["warning"] = "Attrition 컬럼이 없습니다. 예측 성능에 영향을 줄 수 있습니다."
            
            # JSON 파일인 경우
            elif file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                file_info.update({
                    "json_keys": list(json_data.keys()) if isinstance(json_data, dict) else None,
                    "json_type": type(json_data).__name__
                })
            
        except Exception as e:
            # 파일 삭제
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({
                "success": False,
                "error": f"파일 형식이 올바르지 않습니다: {str(e)}"
            }), 400
        
        # 메타데이터 파일 저장
        metadata_path = os.path.join(upload_dir, f"{base_name}_{timestamp}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(file_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Agent file uploaded successfully: {agent_type}/{analysis_type}/{new_filename}")
        
        return jsonify({
            "success": True,
            "message": f"{agent_type} {analysis_type} 파일이 성공적으로 업로드되었습니다.",
            "file_info": file_info
        })
        
    except Exception as e:
        logger.error(f"Agent file upload error: {e}")
        return jsonify({
            "success": False,
            "error": f"파일 업로드 중 오류가 발생했습니다: {str(e)}"
        }), 500


@app.route('/api/files/<agent_type>/<analysis_type>', methods=['GET'])
def list_agent_files(agent_type, analysis_type):
    """특정 에이전트/분석 타입의 파일 목록 조회"""
    try:
        if agent_type not in ['structura', 'chronos', 'sentio', 'agora']:
            return jsonify({
                "success": False,
                "error": "유효하지 않은 agent_type입니다."
            }), 400
        
        if analysis_type not in ['batch', 'post']:
            return jsonify({
                "success": False,
                "error": "유효하지 않은 analysis_type입니다."
            }), 400
        
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads', agent_type, analysis_type)
        
        if not os.path.exists(upload_dir):
            return jsonify({
                "success": True,
                "files": [],
                "message": f"{agent_type}/{analysis_type} 디렉토리가 없습니다."
            })
        
        files = []
        for filename in os.listdir(upload_dir):
            if filename.endswith('.json'):  # 메타데이터 파일
                metadata_path = os.path.join(upload_dir, filename)
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        file_info = json.load(f)
                    files.append(file_info)
                except Exception as e:
                    logger.warning(f"메타데이터 파일 읽기 실패: {filename}, {str(e)}")
        
        # 업로드 시간 순으로 정렬 (최신순)
        files.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
        
        return jsonify({
            "success": True,
            "files": files,
            "agent_type": agent_type,
            "analysis_type": analysis_type,
            "total_files": len(files)
        })
        
    except Exception as e:
        logger.error(f"List agent files error: {e}")
        return jsonify({
            "success": False,
            "error": f"파일 목록 조회 중 오류가 발생했습니다: {str(e)}"
        }), 500


@app.route('/api/files/<agent_type>/<analysis_type>/<filename>', methods=['DELETE'])
def delete_agent_file(agent_type, analysis_type, filename):
    """에이전트 파일 삭제"""
    try:
        if agent_type not in ['structura', 'chronos', 'sentio', 'agora']:
            return jsonify({
                "success": False,
                "error": "유효하지 않은 agent_type입니다."
            }), 400
        
        if analysis_type not in ['batch', 'post']:
            return jsonify({
                "success": False,
                "error": "유효하지 않은 analysis_type입니다."
            }), 400
        
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads', agent_type, analysis_type)
        file_path = os.path.join(upload_dir, filename)
        
        # 메타데이터 파일 경로
        base_name = os.path.splitext(filename)[0]
        metadata_path = os.path.join(upload_dir, f"{base_name}.json")
        
        deleted_files = []
        
        # 실제 파일 삭제
        if os.path.exists(file_path):
            os.remove(file_path)
            deleted_files.append(filename)
        
        # 메타데이터 파일 삭제
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            deleted_files.append(f"{base_name}.json")
        
        if not deleted_files:
            return jsonify({
                "success": False,
                "error": "파일을 찾을 수 없습니다."
            }), 404
        
        logger.info(f"Agent files deleted: {agent_type}/{analysis_type}/{deleted_files}")
        
        return jsonify({
            "success": True,
            "message": f"파일이 성공적으로 삭제되었습니다.",
            "deleted_files": deleted_files
        })
        
    except Exception as e:
        logger.error(f"Delete agent file error: {e}")
        return jsonify({
            "success": False,
            "error": f"파일 삭제 중 오류가 발생했습니다: {str(e)}"
        }), 500

@app.route('/upload_file', methods=['POST'])
def upload_file():
    """파일 업로드"""
    try:
        # 파일 확인
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "파일이 업로드되지 않았습니다."
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "파일이 선택되지 않았습니다."
            }), 400
        
        # 파일 확장자 확인
        if not file.filename.lower().endswith('.csv'):
            return jsonify({
                "success": False,
                "error": "CSV 파일만 업로드 가능합니다."
            }), 400
        
        # 파일 저장 - 에이전트별/분석타입별 폴더 지원
        filename = secure_filename(file.filename)
        
        # 에이전트 타입과 분석 타입 확인 (폼 데이터에서)
        agent_type = request.form.get('agent_type', 'supervisor')
        analysis_type = request.form.get('analysis_type', 'general')
        
        # 에이전트별 폴더 구조 생성
        if agent_type in ['structura', 'chronos', 'sentio', 'agora'] and analysis_type in ['batch', 'post']:
            upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads', agent_type, analysis_type)
        else:
            upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'supervisor')
        
        os.makedirs(upload_dir, exist_ok=True)
        
        # 타임스탬프를 포함한 파일명으로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(filename)[0]
        new_filename = f"{base_name}_{timestamp}.csv"
        file_path = os.path.join(upload_dir, new_filename)
        
        file.save(file_path)
        
        # 파일 정보 검증
        try:
            df = pd.read_csv(file_path)
            file_info = {
                "original_filename": filename,
                "saved_filename": new_filename,
                "file_path": file_path,
                "relative_path": os.path.join('uploads', agent_type, analysis_type, new_filename) if agent_type in ['structura', 'chronos', 'sentio', 'agora'] else os.path.join('uploads', 'supervisor', new_filename),
                "agent_type": agent_type,
                "analysis_type": analysis_type,
                "upload_time": datetime.now().isoformat(),
                "file_size": os.path.getsize(file_path),
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist()
            }
        except Exception as e:
            # 파일 삭제
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({
                "success": False,
                "error": f"CSV 파일 형식이 올바르지 않습니다: {str(e)}"
            }), 400
        
        logger.info(f"File uploaded successfully: {new_filename}")
        
        return jsonify({
            "success": True,
            "message": "파일이 성공적으로 업로드되었습니다.",
            "file_info": file_info
        })
        
    except Exception as e:
        logger.error(f"File upload error: {e}")
        return jsonify({
            "success": False,
            "error": f"파일 업로드 중 오류가 발생했습니다: {str(e)}"
        }), 500


@app.route('/list_uploaded_files', methods=['GET'])
def list_uploaded_files():
    """업로드된 파일 목록 조회"""
    try:
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'supervisor')
        
        if not os.path.exists(upload_dir):
            return jsonify({
                "success": True,
                "files": [],
                "message": "업로드된 파일이 없습니다."
            })
        
        files = []
        for filename in os.listdir(upload_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(upload_dir, filename)
                stat = os.stat(file_path)
                
                # 파일 정보 수집
                try:
                    df = pd.read_csv(file_path)
                    file_info = {
                        "filename": filename,
                        "file_path": file_path,
                        "size": stat.st_size,
                        "upload_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "rows": len(df),
                        "columns": len(df.columns),
                        "column_names": df.columns.tolist()[:10]  # 처음 10개 컬럼만
                    }
                except Exception as e:
                    file_info = {
                        "filename": filename,
                        "file_path": file_path,
                        "size": stat.st_size,
                        "upload_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "error": f"파일 읽기 오류: {str(e)}"
                    }
                
                files.append(file_info)
        
        # 업로드 시간 기준 내림차순 정렬
        files.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
        
        return jsonify({
            "success": True,
            "files": files,
            "total_files": len(files)
        })
        
    except Exception as e:
        logger.error(f"List files error: {e}")
        return jsonify({
            "success": False,
            "error": f"파일 목록 조회 중 오류가 발생했습니다: {str(e)}"
        }), 500


@app.route('/download_file/<filename>', methods=['GET'])
def download_file(filename):
    """파일 다운로드"""
    try:
        # 파일명 보안 검증
        filename = secure_filename(filename)
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'supervisor')
        file_path = os.path.join(upload_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                "success": False,
                "error": "파일을 찾을 수 없습니다."
            }), 404
        
        return send_file(file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        logger.error(f"File download error: {e}")
        return jsonify({
            "success": False,
            "error": f"파일 다운로드 중 오류가 발생했습니다: {str(e)}"
        }), 500


@app.route('/delete_file/<filename>', methods=['DELETE'])
def delete_file(filename):
    """파일 삭제"""
    try:
        # 파일명 보안 검증
        filename = secure_filename(filename)
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'supervisor')
        file_path = os.path.join(upload_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                "success": False,
                "error": "파일을 찾을 수 없습니다."
            }), 404
        
        os.remove(file_path)
        logger.info(f"File deleted successfully: {filename}")
        
        return jsonify({
            "success": True,
            "message": f"파일 '{filename}'이 성공적으로 삭제되었습니다."
        })
        
    except Exception as e:
        logger.error(f"File delete error: {e}")
        return jsonify({
            "success": False,
            "error": f"파일 삭제 중 오류가 발생했습니다: {str(e)}"
        }), 500


@app.route('/get_file_info/<filename>', methods=['GET'])
def get_file_info(filename):
    """파일 상세 정보 조회"""
    try:
        # 파일명 보안 검증
        filename = secure_filename(filename)
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'supervisor')
        file_path = os.path.join(upload_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                "success": False,
                "error": "파일을 찾을 수 없습니다."
            }), 404
        
        # 파일 기본 정보
        stat = os.stat(file_path)
        file_info = {
            "filename": filename,
            "file_path": file_path,
            "size": stat.st_size,
            "upload_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
        
        # CSV 파일 상세 정보
        try:
            df = pd.read_csv(file_path)
            file_info.update({
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "sample_data": df.head(5).to_dict('records'),
                "missing_values": df.isnull().sum().to_dict()
            })
        except Exception as e:
            file_info["csv_error"] = f"CSV 파일 분석 오류: {str(e)}"
        
        return jsonify({
            "success": True,
            "file_info": file_info
        })
        
    except Exception as e:
        logger.error(f"Get file info error: {e}")
        return jsonify({
            "success": False,
            "error": f"파일 정보 조회 중 오류가 발생했습니다: {str(e)}"
        }), 500


# ------------------------------------------------------
# 워커 에이전트 API 통합 엔드포인트
# ------------------------------------------------------

@app.route('/api/workers/health_check_all', methods=['GET'])
def check_all_workers_health():
    """모든 워커 에이전트의 상태를 확인"""
    try:
        if not supervisor_workflow:
            return jsonify({
                'success': False,
                'error': 'Supervisor workflow not initialized'
            }), 500
        
        # 워커 상태 확인
        health_status = asyncio.run(supervisor_workflow.worker_integrator.health_check_all())
        
        return jsonify({
            'success': True,
            'worker_status': {
                worker.value: {
                    'healthy': is_healthy,
                    'status': 'online' if is_healthy else 'offline'
                }
                for worker, is_healthy in health_status.items()
            },
            'summary': {
                'total_workers': len(health_status),
                'healthy_workers': sum(health_status.values()),
                'health_rate': sum(health_status.values()) / len(health_status)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Worker health check error: {e}")
        return jsonify({
            'success': False,
            'error': f'Health check failed: {str(e)}'
        }), 500


@app.route('/api/workers/structura/predict', methods=['POST'])
def structura_predict():
    """Structura 예측 API 프록시"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Structura API 호출
        import requests
        response = requests.post(
            'http://localhost:5001/api/predict',
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'structura'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Structura API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Structura predict error: {e}")
        return jsonify({
            'success': False,
            'error': f'Structura prediction failed: {str(e)}'
        }), 500


@app.route('/api/workers/cognita/analyze/<employee_id>', methods=['GET'])
def cognita_analyze_employee(employee_id):
    """Cognita 직원 분석 API 프록시"""
    try:
        # Cognita API 호출
        import requests
        response = requests.get(
            f'http://localhost:5002/api/analyze/employee/{employee_id}',
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'cognita',
                'employee_id': employee_id
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Cognita API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Cognita analyze error: {e}")
        return jsonify({
            'success': False,
            'error': f'Cognita analysis failed: {str(e)}'
        }), 500


@app.route('/api/workers/cognita/departments', methods=['GET'])
def cognita_get_departments():
    """Cognita 부서 목록 API 프록시"""
    try:
        # Cognita API 호출
        import requests
        response = requests.get(
            'http://localhost:5002/api/departments',
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'cognita'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Cognita API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Cognita departments error: {e}")
        return jsonify({
            'success': False,
            'error': f'Cognita departments failed: {str(e)}'
        }), 500


@app.route('/api/workers/cognita/employees', methods=['GET'])
def cognita_get_employees():
    """Cognita 직원 목록 API 프록시"""
    try:
        # 쿼리 파라미터 전달
        limit = request.args.get('limit', '10')
        offset = request.args.get('offset', '0')
        
        # Cognita API 호출
        import requests
        response = requests.get(
            f'http://localhost:5002/api/employees?limit={limit}&offset={offset}',
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'cognita'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Cognita API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Cognita employees error: {e}")
        return jsonify({
            'success': False,
            'error': f'Cognita employees failed: {str(e)}'
        }), 500


@app.route('/api/workers/cognita/analyze/department', methods=['POST'])
def cognita_analyze_department():
    """Cognita 부서 분석 API 프록시"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Cognita API 호출
        import requests
        response = requests.post(
            'http://localhost:5002/api/analyze/department',
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'cognita'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Cognita API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Cognita department analysis error: {e}")
        return jsonify({
            'success': False,
            'error': f'Cognita department analysis failed: {str(e)}'
        }), 500


@app.route('/api/cognita/setup/neo4j', methods=['POST'])
def cognita_setup_neo4j():
    """Cognita Neo4j 설정 API 프록시"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Cognita API 호출
        import requests
        response = requests.post(
            'http://localhost:5002/api/setup/neo4j',
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'cognita'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Cognita API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Cognita Neo4j setup error: {e}")
        return jsonify({
            'success': False,
            'error': f'Cognita Neo4j setup failed: {str(e)}'
        }), 500


@app.route('/api/workers/chronos/predict', methods=['POST'])
def chronos_predict():
    """Chronos 예측 API 프록시"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Chronos API 호출
        import requests
        response = requests.post(
            'http://localhost:5003/api/predict',
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'chronos'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Chronos API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Chronos predict error: {e}")
        return jsonify({
            'success': False,
            'error': f'Chronos prediction failed: {str(e)}'
        }), 500


@app.route('/api/workers/sentio/analyze_sentiment', methods=['POST'])
def sentio_analyze_sentiment():
    """Sentio 감정 분석 API 프록시"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Sentio API 호출
        import requests
        response = requests.post(
            'http://localhost:5004/analyze_sentiment',
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'sentio'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Sentio API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Sentio analyze error: {e}")
        return jsonify({
            'success': False,
            'error': f'Sentio analysis failed: {str(e)}'
        }), 500


@app.route('/api/workers/agora/comprehensive_analysis', methods=['POST'])
def agora_comprehensive_analysis():
    """Agora 종합 시장 분석 API 프록시"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Agora API 호출
        import requests
        response = requests.post(
            'http://localhost:5005/api/agora/comprehensive-analysis',
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'agora'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Agora API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Agora analysis error: {e}")
        return jsonify({
            'success': False,
            'error': f'Agora analysis failed: {str(e)}'
        }), 500


@app.route('/api/workers/integration/predict_employee', methods=['POST'])
def integration_predict_employee():
    """Integration 개별 직원 예측 API 프록시"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Integration API 호출
        import requests
        response = requests.post(
            'http://localhost:5007/predict_employee',
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'integration'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Integration API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Integration predict error: {e}")
        return jsonify({
            'success': False,
            'error': f'Integration prediction failed: {str(e)}'
        }), 500


@app.route('/api/workers/integration/generate_report', methods=['POST'])
def integration_generate_report():
    """Integration 레포트 생성 API 프록시"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Integration API 호출
        import requests
        response = requests.post(
            'http://localhost:5007/generate_report',
            json=data,
            timeout=60  # 레포트 생성은 시간이 더 걸릴 수 있음
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'integration'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Integration API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Integration report error: {e}")
        return jsonify({
            'success': False,
            'error': f'Integration report generation failed: {str(e)}'
        }), 500


@app.route('/api/workers/integration/generate_batch_analysis_report', methods=['POST'])
def integration_generate_batch_analysis_report():
    """Integration 배치 분석 보고서 생성 API 프록시"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Integration API 호출
        import requests
        response = requests.post(
            'http://localhost:5007/generate_batch_analysis_report',
            json=data,
            timeout=120  # 배치 보고서 생성은 시간이 더 걸릴 수 있음
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'integration'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Integration API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Integration batch analysis report error: {e}")
        return jsonify({
            'success': False,
            'error': f'Integration batch analysis report generation failed: {str(e)}'
        }), 500


@app.route('/api/workers/integration/load_data', methods=['POST'])
def integration_load_data():
    """Integration 데이터 로드 API 프록시"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Integration API 호출
        import requests
        response = requests.post(
            'http://localhost:5007/load_data',
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'integration'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Integration API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Integration load data error: {e}")
        return jsonify({
            'success': False,
            'error': f'Integration data loading failed: {str(e)}'
        }), 500


@app.route('/api/workers/integration/calculate_thresholds', methods=['POST'])
def integration_calculate_thresholds():
    """Integration 임계값 계산 API 프록시"""
    try:
        data = request.get_json() or {}
        
        # Integration API 호출
        import requests
        response = requests.post(
            'http://localhost:5007/calculate_thresholds',
            json=data,
            timeout=60  # 임계값 계산은 시간이 걸릴 수 있음
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'integration'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Integration API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Integration calculate thresholds error: {e}")
        return jsonify({
            'success': False,
            'error': f'Integration threshold calculation failed: {str(e)}'
        }), 500


@app.route('/api/workers/integration/optimize_weights', methods=['POST'])
def integration_optimize_weights():
    """Integration 가중치 최적화 API 프록시"""
    try:
        data = request.get_json() or {}
        
        # Integration API 호출
        import requests
        response = requests.post(
            'http://localhost:5007/optimize_weights',
            json=data,
            timeout=120  # 가중치 최적화는 시간이 많이 걸릴 수 있음
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'integration'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Integration API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Integration optimize weights error: {e}")
        return jsonify({
            'success': False,
            'error': f'Integration weight optimization failed: {str(e)}'
        }), 500


@app.route('/api/workers/integration/get_results', methods=['GET'])
def integration_get_results():
    """Integration 결과 조회 API 프록시"""
    try:
        # Integration API 호출
        import requests
        response = requests.get(
            'http://localhost:5007/get_results',
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'integration'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Integration API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Integration get results error: {e}")
        return jsonify({
            'success': False,
            'error': f'Integration results retrieval failed: {str(e)}'
        }), 500


@app.route('/api/workers/integration/compare_methods', methods=['POST'])
def integration_compare_methods():
    """Integration 최적화 방법 비교 API 프록시"""
    try:
        data = request.get_json() or {}
        
        # Integration API 호출
        import requests
        response = requests.post(
            'http://localhost:5007/compare_methods',
            json=data,
            timeout=180  # 방법 비교는 시간이 많이 걸릴 수 있음
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'integration'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Integration API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Integration compare methods error: {e}")
        return jsonify({
            'success': False,
            'error': f'Integration method comparison failed: {str(e)}'
        }), 500


@app.route('/api/workers/integration/generate_batch_reports', methods=['POST'])
def integration_generate_batch_reports():
    """Integration 일괄 레포트 생성 API 프록시"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Integration API 호출
        import requests
        response = requests.post(
            'http://localhost:5007/generate_batch_reports',
            json=data,
            timeout=300  # 일괄 레포트 생성은 시간이 많이 걸릴 수 있음
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'integration'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Integration API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Integration batch reports error: {e}")
        return jsonify({
            'success': False,
            'error': f'Integration batch report generation failed: {str(e)}'
        }), 500


@app.route('/api/workers/integration/upload_employee_data', methods=['POST'])
def integration_upload_employee_data():
    """Integration 직원 데이터 업로드 API 프록시"""
    try:
        # 파일 확인
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "파일이 업로드되지 않았습니다."
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "파일이 선택되지 않았습니다."
            }), 400
        
        # Integration API에 파일 전송
        import requests
        files = {'file': (file.filename, file.read(), 'text/csv')}
        response = requests.post(
            'http://localhost:5007/upload/employee_data',
            files=files,
            timeout=60
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'integration'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Integration API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Integration upload employee data error: {e}")
        return jsonify({
            'success': False,
            'error': f'Integration employee data upload failed: {str(e)}'
        }), 500


@app.route('/api/workers/integration/get_employee_list', methods=['GET'])
def integration_get_employee_list():
    """Integration 직원 목록 조회 API 프록시"""
    try:
        # Integration API 호출
        import requests
        response = requests.get(
            'http://localhost:5007/get_employee_list',
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'data': response.json(),
                'source': 'integration'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Integration API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        logger.error(f"Integration get employee list error: {e}")
        return jsonify({
            'success': False,
            'error': f'Integration employee list retrieval failed: {str(e)}'
        }), 500


@app.route('/api/workers/upload_data', methods=['POST'])
def upload_data_to_workers():
    """모든 워커 에이전트에 데이터 업로드"""
    try:
        # 파일 확인
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "파일이 업로드되지 않았습니다."
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "파일이 선택되지 않았습니다."
            }), 400
        
        # 파일 확장자 확인
        if not file.filename.lower().endswith('.csv'):
            return jsonify({
                "success": False,
                "error": "CSV 파일만 업로드 가능합니다."
            }), 400
        
        upload_results = {}
        
        # 각 워커에 파일 업로드 시도
        worker_endpoints = {
            'structura': 'http://localhost:5001/api/upload/data',
            'chronos': 'http://localhost:5003/api/upload/timeseries',
            'sentio': 'http://localhost:5004/upload/text_data'
        }
        
        import requests
        
        for worker_name, endpoint in worker_endpoints.items():
            try:
                # 파일 포인터를 처음으로 되돌리기
                file.seek(0)
                
                files = {'file': (file.filename, file.read(), 'text/csv')}
                response = requests.post(endpoint, files=files, timeout=30)
                
                upload_results[worker_name] = {
                    'success': response.status_code == 200,
                    'status_code': response.status_code,
                    'response': response.json() if response.status_code == 200 else response.text
                }
                
            except Exception as e:
                upload_results[worker_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # 성공한 업로드 수 계산
        successful_uploads = sum(1 for result in upload_results.values() if result['success'])
        
        return jsonify({
            'success': successful_uploads > 0,
            'message': f'{successful_uploads}/{len(worker_endpoints)} 워커에 성공적으로 업로드되었습니다.',
            'upload_results': upload_results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Worker upload error: {e}")
        return jsonify({
            'success': False,
            'error': f'Worker upload failed: {str(e)}'
        }), 500


# ------------------------------------------------------
# 배치 처리 기능
# ------------------------------------------------------

@app.route('/batch_process', methods=['POST'])
def batch_process():
    """대량 직원 데이터 배치 처리"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # 배치 처리 옵션
        employees_data = data.get('employees', [])
        process_options = data.get('options', {})
        
        if not employees_data:
            return jsonify({
                'success': False,
                'error': 'No employee data provided'
            }), 400
        
        # 배치 처리 세션 ID 생성
        import uuid
        batch_id = str(uuid.uuid4())
        
        # 배치 처리 상태 초기화
        batch_status = {
            'batch_id': batch_id,
            'total_employees': len(employees_data),
            'processed_employees': 0,
            'status': 'processing',
            'start_time': datetime.now().isoformat(),
            'results': [],
            'errors': []
        }
        
        # 세션에 배치 상태 저장 (간단한 메모리 저장)
        if not hasattr(app, 'batch_sessions'):
            app.batch_sessions = {}
        app.batch_sessions[batch_id] = batch_status
        
        # 비동기 배치 처리 시작
        import threading
        thread = threading.Thread(
            target=process_batch_async, 
            args=(batch_id, employees_data, process_options)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'message': f'배치 처리가 시작되었습니다. 총 {len(employees_data)}명의 직원 데이터를 처리합니다.',
            'status_url': f'/batch_status/{batch_id}'
        })
        
    except Exception as e:
        logger.error(f"Batch process error: {e}")
        return jsonify({
            'success': False,
            'error': f'Batch processing failed: {str(e)}'
        }), 500


@app.route('/batch_status/<batch_id>', methods=['GET'])
def get_batch_status(batch_id):
    """배치 처리 상태 조회"""
    try:
        if not hasattr(app, 'batch_sessions') or batch_id not in app.batch_sessions:
            return jsonify({
                'success': False,
                'error': 'Batch session not found'
            }), 404
        
        batch_status = app.batch_sessions[batch_id]
        
        # 진행률 계산
        progress = (batch_status['processed_employees'] / batch_status['total_employees']) * 100
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'status': batch_status['status'],
            'progress': round(progress, 2),
            'total_employees': batch_status['total_employees'],
            'processed_employees': batch_status['processed_employees'],
            'start_time': batch_status['start_time'],
            'results_count': len(batch_status['results']),
            'errors_count': len(batch_status['errors']),
            'completed': batch_status['status'] in ['completed', 'failed']
        })
        
    except Exception as e:
        logger.error(f"Batch status error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get batch status: {str(e)}'
        }), 500


@app.route('/batch_upload_csv', methods=['POST'])
def batch_upload_csv():
    """CSV 파일을 업로드하여 배치 처리"""
    try:
        # 파일 확인
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "파일이 업로드되지 않았습니다."
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "파일이 선택되지 않았습니다."
            }), 400
        
        # 파일 확장자 확인
        if not file.filename.lower().endswith('.csv'):
            return jsonify({
                "success": False,
                "error": "CSV 파일만 업로드 가능합니다."
            }), 400
        
        # CSV 파일 읽기
        import pandas as pd
        import io
        
        # 파일 내용을 메모리에서 직접 읽기
        file_content = file.read().decode('utf-8')
        csv_data = pd.read_csv(io.StringIO(file_content))
        
        # DataFrame을 딕셔너리 리스트로 변환
        employees_data = csv_data.to_dict('records')
        
        # 필수 컬럼 확인
        required_columns = ['employee_id']
        missing_columns = [col for col in required_columns if col not in csv_data.columns]
        
        if missing_columns:
            return jsonify({
                "success": False,
                "error": f"필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}",
                "required_columns": required_columns,
                "found_columns": list(csv_data.columns)
            }), 400
        
        # 배치 처리 옵션
        process_options = {
            'source': 'csv_upload',
            'filename': file.filename,
            'include_reports': request.form.get('include_reports', 'true').lower() == 'true',
            'use_llm': request.form.get('use_llm', 'false').lower() == 'true'
        }
        
        # 배치 처리 시작
        import uuid
        batch_id = str(uuid.uuid4())
        
        batch_status = {
            'batch_id': batch_id,
            'total_employees': len(employees_data),
            'processed_employees': 0,
            'status': 'processing',
            'start_time': datetime.now().isoformat(),
            'source': 'csv_upload',
            'filename': file.filename,
            'results': [],
            'errors': []
        }
        
        if not hasattr(app, 'batch_sessions'):
            app.batch_sessions = {}
        app.batch_sessions[batch_id] = batch_status
        
        # 비동기 배치 처리 시작
        import threading
        thread = threading.Thread(
            target=process_batch_async, 
            args=(batch_id, employees_data, process_options)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'message': f'CSV 파일 "{file.filename}"에서 {len(employees_data)}명의 직원 데이터 배치 처리가 시작되었습니다.',
            'total_employees': len(employees_data),
            'columns': list(csv_data.columns),
            'status_url': f'/batch_status/{batch_id}'
        })
        
    except Exception as e:
        logger.error(f"CSV batch upload error: {e}")
        return jsonify({
            'success': False,
            'error': f'CSV 배치 업로드 실패: {str(e)}'
        }), 500


@app.route('/batch_results/<batch_id>', methods=['GET'])
def get_batch_results(batch_id):
    """배치 처리 결과 조회"""
    try:
        if not hasattr(app, 'batch_sessions') or batch_id not in app.batch_sessions:
            return jsonify({
                'success': False,
                'error': 'Batch session not found'
            }), 404
        
        batch_status = app.batch_sessions[batch_id]
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'status': batch_status['status'],
            'total_employees': batch_status['total_employees'],
            'processed_employees': batch_status['processed_employees'],
            'results': batch_status['results'],
            'errors': batch_status['errors'],
            'start_time': batch_status['start_time'],
            'end_time': batch_status.get('end_time'),
            'export_url': f'/batch_export_csv/{batch_id}'
        })
        
    except Exception as e:
        logger.error(f"Batch results error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get batch results: {str(e)}'
        }), 500


@app.route('/api/statistics/group', methods=['GET'])
def get_group_statistics():
    """단체 통계 조회 API"""
    try:
        group_by = request.args.get('group_by', 'department')
        department_filter = request.args.get('department')
        
        # 모든 부서의 인덱스 파일에서 통계 수집
        departments = ['Human_Resources', 'Research_&_Development', 'Sales', 'Manufacturing', 'Information_Technology']
        grouped_stats = {}
        
        for dept in departments:
            try:
                dept_stats = hierarchical_result_manager.get_department_statistics(dept)
                if not dept_stats:
                    continue
                
                # 부서별 인덱스 파일 읽기
                dept_clean = hierarchical_result_manager._sanitize_folder_name(dept)
                index_file = hierarchical_result_manager.base_output_dir / dept_clean / "department_index.json"
                
                if index_file.exists():
                    with open(index_file, 'r', encoding='utf-8') as f:
                        index_data = json.load(f)
                    
                    # 그룹화 기준에 따라 통계 생성
                    if group_by == 'department':
                        if department_filter is None or dept == department_filter:
                            grouped_stats[dept] = {
                                'total_employees': dept_stats.get('total_employees', 0),
                                'high_risk': 0,  # 실제 위험도 계산 필요
                                'medium_risk': 0,
                                'low_risk': dept_stats.get('total_employees', 0),
                                'avg_risk_score': 0.3,  # 기본값
                                'common_risk_factors': {}
                            }
                    
                    elif group_by == 'job_role':
                        job_roles = index_data.get('job_roles', {})
                        for role, employees in job_roles.items():
                            if department_filter is None or dept == department_filter:
                                grouped_stats[role] = {
                                    'total_employees': len(employees),
                                    'high_risk': 0,
                                    'medium_risk': 0,
                                    'low_risk': len(employees),
                                    'avg_risk_score': 0.3,
                                    'common_risk_factors': {}
                                }
                    
                    elif group_by == 'job_level':
                        positions = index_data.get('positions', {})
                        for pos, employees in positions.items():
                            if department_filter is None or dept == department_filter:
                                grouped_stats[pos] = {
                                    'total_employees': len(employees),
                                    'high_risk': 0,
                                    'medium_risk': 0,
                                    'low_risk': len(employees),
                                    'avg_risk_score': 0.3,
                                    'common_risk_factors': {}
                                }
                        
            except Exception as e:
                logger.error(f"부서 {dept} 통계 처리 실패: {e}")
                continue
        
        return jsonify({
            'success': True,
            'group_by': group_by,
            'department_filter': department_filter,
            'statistics': grouped_stats,
            'generated_at': datetime.now().isoformat(),
            'data_source': 'department_index'
        })
        
    except Exception as e:
        logger.error(f"Group statistics error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get group statistics: {str(e)}'
        }), 500


@app.route('/batch_export_csv/<batch_id>', methods=['GET'])
def batch_export_csv(batch_id):
    """배치 처리 결과를 CSV로 내보내기"""
    try:
        if not hasattr(app, 'batch_sessions') or batch_id not in app.batch_sessions:
            return jsonify({
                'success': False,
                'error': 'Batch session not found'
            }), 404
        
        batch_status = app.batch_sessions[batch_id]
        
        # 결과를 CSV 형태로 변환
        import pandas as pd
        import io
        
        csv_rows = []
        
        # 성공한 결과들
        for result in batch_status['results']:
            employee_id = result.get('employee_id')
            analysis = result.get('analysis', {})
            
            row = {
                'employee_id': employee_id,
                'processed_at': result.get('processed_at'),
                'status': 'success'
            }
            
            # 워커 분석 결과
            worker_analyses = analysis.get('worker_analyses', {})
            for worker, worker_result in worker_analyses.items():
                if 'error' in worker_result:
                    row[f'{worker}_status'] = 'error'
                    row[f'{worker}_error'] = worker_result['error']
                else:
                    row[f'{worker}_status'] = 'success'
                    # 주요 점수 추출
                    if worker == 'structura' and 'probability' in worker_result:
                        row[f'{worker}_score'] = worker_result['probability']
                    elif worker == 'cognita' and 'risk_score' in worker_result:
                        row[f'{worker}_score'] = worker_result['risk_score']
                    elif worker == 'chronos' and 'prediction_score' in worker_result:
                        row[f'{worker}_score'] = worker_result['prediction_score']
                    elif worker == 'sentio':
                        # psychological_risk_score를 우선적으로 사용 (JD-R 모델 기반)
                        if 'psychological_risk_score' in worker_result:
                            row[f'{worker}_score'] = worker_result['psychological_risk_score']
                        elif 'risk_score' in worker_result:
                            row[f'{worker}_score'] = worker_result['risk_score']
                        elif 'sentiment_score' in worker_result:
                            row[f'{worker}_score'] = 1.0 - worker_result['sentiment_score']
                    elif worker == 'agora' and 'overall_risk_score' in worker_result:
                        row[f'{worker}_score'] = worker_result['overall_risk_score']
            
            # Integration 결과
            if 'integration_prediction' in analysis:
                integration_pred = analysis['integration_prediction']
                if 'prediction' in integration_pred:
                    row['integration_prediction'] = integration_pred['prediction']
                if 'probability' in integration_pred:
                    row['integration_probability'] = integration_pred['probability']
            
            csv_rows.append(row)
        
        # 오류 결과들
        for error in batch_status['errors']:
            row = {
                'employee_id': error.get('employee_id'),
                'processed_at': error.get('processed_at'),
                'status': 'error',
                'error_message': error.get('error')
            }
            csv_rows.append(row)
        
        if not csv_rows:
            return jsonify({
                'success': False,
                'error': 'No results to export'
            }), 400
        
        # DataFrame 생성 및 CSV 변환
        df = pd.DataFrame(csv_rows)
        
        # CSV 문자열 생성
        output = io.StringIO()
        df.to_csv(output, index=False, encoding='utf-8')
        csv_content = output.getvalue()
        
        # 파일명 생성
        filename = f"batch_results_{batch_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # CSV 파일 응답
        from flask import make_response
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except Exception as e:
        logger.error(f"Batch CSV export error: {e}")
        return jsonify({
            'success': False,
            'error': f'CSV 내보내기 실패: {str(e)}'
        }), 500


def process_batch_async(batch_id, employees_data, options):
    """비동기 배치 처리 함수"""
    try:
        batch_status = app.batch_sessions[batch_id]
        
        for i, employee_data in enumerate(employees_data):
            try:
                # 개별 직원 분석 수행
                employee_id = employee_data.get('employee_id', f'employee_{i+1}')
                
                # 모든 워커 에이전트를 통한 분석
                analysis_result = analyze_employee_comprehensive(employee_data)
                
                # 결과 저장
                batch_status['results'].append({
                    'employee_id': employee_id,
                    'analysis': analysis_result,
                    'processed_at': datetime.now().isoformat()
                })
                
                # 진행 상황 업데이트
                batch_status['processed_employees'] = i + 1
                
                logger.info(f"Batch {batch_id}: Processed employee {i+1}/{len(employees_data)}")
                
            except Exception as e:
                # 개별 직원 처리 오류
                error_info = {
                    'employee_id': employee_data.get('employee_id', f'employee_{i+1}'),
                    'error': str(e),
                    'processed_at': datetime.now().isoformat()
                }
                batch_status['errors'].append(error_info)
                logger.error(f"Batch {batch_id}: Error processing employee {i+1}: {e}")
        
        # 배치 처리 완료
        batch_status['status'] = 'completed'
        batch_status['end_time'] = datetime.now().isoformat()
        
        logger.info(f"Batch {batch_id}: Processing completed. {len(batch_status['results'])} successful, {len(batch_status['errors'])} errors")
        
    except Exception as e:
        # 전체 배치 처리 오류
        batch_status['status'] = 'failed'
        batch_status['end_time'] = datetime.now().isoformat()
        batch_status['errors'].append({
            'error': f'Batch processing failed: {str(e)}',
            'processed_at': datetime.now().isoformat()
        })
        logger.error(f"Batch {batch_id}: Critical error: {e}")


def analyze_employee_comprehensive(employee_data):
    """개별 직원에 대한 종합 분석"""
    try:
        results = {}
        
        # 각 워커 에이전트를 통한 분석
        worker_results = {}
        
        # Structura 분석 (정형 데이터)
        try:
            structura_result = call_worker_api('structura', 'predict', employee_data)
            worker_results['structura'] = structura_result
        except Exception as e:
            worker_results['structura'] = {'error': str(e)}
        
        # Cognita 분석 (관계 분석)
        try:
            employee_id = employee_data.get('employee_id')
            if employee_id:
                cognita_result = call_worker_api('cognita', f'analyze/{employee_id}', {})
                worker_results['cognita'] = cognita_result
        except Exception as e:
            worker_results['cognita'] = {'error': str(e)}
        
        # Chronos 분석 (시계열)
        try:
            chronos_result = call_worker_api('chronos', 'predict', employee_data)
            worker_results['chronos'] = chronos_result
        except Exception as e:
            worker_results['chronos'] = {'error': str(e)}
        
        # Sentio 분석 (감정 분석)
        try:
            sentio_data = {
                'text': employee_data.get('feedback_text', ''),
                'employee_id': employee_data.get('employee_id')
            }
            sentio_result = call_worker_api('sentio', 'analyze_sentiment', sentio_data)
            worker_results['sentio'] = sentio_result
        except Exception as e:
            worker_results['sentio'] = {'error': str(e)}
        
        # Agora 분석 (시장 분석)
        try:
            agora_result = call_worker_api('agora', 'comprehensive_analysis', employee_data)
            worker_results['agora'] = agora_result
        except Exception as e:
            worker_results['agora'] = {'error': str(e)}
        
        results['worker_analyses'] = worker_results
        
        # Integration을 통한 결과 통합
        try:
            # 워커 결과를 점수로 변환
            agent_scores = extract_scores_from_worker_results(worker_results)
            
            # Integration 예측
            integration_data = {'scores': agent_scores}
            integration_result = call_integration_api('predict_employee', integration_data)
            results['integration_prediction'] = integration_result
            
            # Integration 리포트 생성
            report_data = {
                'employee_id': employee_data.get('employee_id'),
                'agent_scores': agent_scores,
                'format': 'json',
                'use_llm': True
            }
            report_result = call_integration_api('generate_report', report_data)
            results['integration_report'] = report_result
            
        except Exception as e:
            results['integration_error'] = str(e)
        
        # 계층적 결과 관리자를 통한 결과 저장
        try:
            employee_id = employee_data.get('employee_id', 'unknown')
            
            # 에러가 있는 워커 결과 필터링 (저장용)
            clean_worker_results = {}
            for worker, result in worker_results.items():
                if isinstance(result, dict) and 'error' not in result:
                    clean_worker_results[worker] = result
            
            # 결과 저장 (에러가 없는 워커 결과만) - 부서 정보 추출 및 전달
            if clean_worker_results:
                # 부서 정보 추출 (개선된 로직)
                department = employee_data.get('Department', 'Unknown')
                job_role = employee_data.get('JobRole', 'Unknown')
                job_level = employee_data.get('JobLevel', employee_data.get('Position', 'Unknown'))
                
                # 다른 필드명으로도 시도
                if department == 'Unknown':
                    department = employee_data.get('department', employee_data.get('dept', 'Unknown'))
                if job_role == 'Unknown':
                    job_role = employee_data.get('job_role', employee_data.get('role', 'Unknown'))
                if job_level == 'Unknown':
                    job_level = employee_data.get('job_level', employee_data.get('level', 'Unknown'))
                
                print(f"📋 Supervisor (단일) - 직원 {employee_id}: {department}/{job_role}/{job_level}")
                
                saved_path = hierarchical_result_manager.save_employee_result(
                    employee_id=employee_id,
                    employee_data=employee_data,
                    worker_results=clean_worker_results,
                    department=department,
                    job_role=job_role,
                    position=job_level
                )
                results['saved_path'] = saved_path
                logger.info(f"직원 {employee_id} 분석 결과가 {saved_path}에 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"결과 저장 실패 (직원 {employee_data.get('employee_id')}): {e}")
            results['save_error'] = str(e)
        
        return results
        
    except Exception as e:
        return {'error': f'Comprehensive analysis failed: {str(e)}'}


@app.route('/api/batch-analysis/save-results', methods=['POST'])
def save_batch_analysis_results():
    """배치 분석 결과 저장 API"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': '요청 데이터가 필요합니다.'
            }), 400
        
        # 배치 결과 데이터 추출
        batch_results = data.get('batchResults', {})
        analysis_summary = data.get('analysisSummary', {})
        
        if not batch_results:
            return jsonify({
                'success': False,
                'error': '배치 결과 데이터가 없습니다.'
            }), 400
        
        # 결과 저장 로직
        batch_id = f"batch_{int(time.time())}"
        
        # 배치 세션에 결과 저장
        if not hasattr(app, 'batch_sessions'):
            app.batch_sessions = {}
        
        app.batch_sessions[batch_id] = {
            'batch_id': batch_id,
            'status': 'completed',
            'results': [],
            'errors': [],
            'start_time': datetime.now().isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_employees': analysis_summary.get('totalEmployees', 0),
            'processed_employees': analysis_summary.get('totalEmployees', 0)
        }
        
        # 각 에이전트별 결과 처리
        for agent_name, agent_results in batch_results.items():
            if isinstance(agent_results, list):
                for result in agent_results:
                    if isinstance(result, dict) and result.get('employee_id'):
                        app.batch_sessions[batch_id]['results'].append({
                            'employee_id': result['employee_id'],
                            'agent': agent_name,
                            'analysis': result,
                            'processed_at': datetime.now().isoformat()
                        })
        
        logger.info(f"배치 분석 결과 저장 완료: {batch_id}")
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'message': '배치 분석 결과가 성공적으로 저장되었습니다.'
        })
        
    except Exception as e:
        logger.error(f"배치 분석 결과 저장 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'배치 분석 결과 저장 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/batch-analysis/save-hierarchical-results', methods=['POST'])
def save_hierarchical_batch_results():
    """배치 분석 결과를 계층적 구조로 저장 API (Department > JobRole > JobLevel > 직원별)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': '요청 데이터가 필요합니다.'
            }), 400
        
        hierarchical_results = data.get('hierarchical_results', {})
        analysis_summary = data.get('analysis_summary', {})
        analysis_timestamp = data.get('analysis_timestamp', datetime.now().isoformat())
        
        if not hierarchical_results:
            return jsonify({
                'success': False,
                'error': '계층적 결과 데이터가 없습니다.'
            }), 400
        
        logger.info(f"계층적 배치 분석 결과 저장 시작: {len(hierarchical_results)}개 부서")
        
        saved_paths = []
        total_employees_saved = 0
        
        # 각 부서별로 처리
        for dept_name, job_roles in hierarchical_results.items():
            logger.info(f"부서 '{dept_name}' 처리 중...")
            
            for role_name, job_levels in job_roles.items():
                for level_name, employees in job_levels.items():
                    for employee_id, employee_result in employees.items():
                        try:
                            # employee_result가 None이거나 dict가 아닌 경우 처리
                            if not employee_result or not isinstance(employee_result, dict):
                                logger.warning(f"직원 {employee_id}의 결과 데이터가 유효하지 않음: {type(employee_result)}")
                                continue
                            
                            # hierarchical_result_manager를 통해 저장
                            employee_data = employee_result.get('employee_data', {})
                            agent_results = employee_result.get('agent_results', {})
                            
                            # 에러가 있는 에이전트 결과 필터링
                            clean_agent_results = {}
                            for agent_name, agent_result in agent_results.items():
                                if isinstance(agent_result, dict) and 'error' not in agent_result:
                                    clean_agent_results[agent_name] = agent_result
                            
                            if clean_agent_results:
                                # 프론트엔드에서 이미 계층 구조를 만들어서 보냈으므로 그대로 사용
                                # 변수 섀도잉 방지를 위해 다른 이름 사용
                                department = dept_name
                                job_role = role_name
                                job_level = level_name
                                
                                # 'Unknown' 값은 건너뜀
                                if department == 'Unknown' or department == '미분류':
                                    logger.warning(f"직원 {employee_id}의 부서가 'Unknown' 또는 '미분류'입니다. 건너뜁니다.")
                                    continue
                                
                                print(f"📋 Supervisor - 직원 {employee_id}: {department}/{job_role}/{job_level}")
                                
                                # 계층적 구조로 저장 (Department/JobRole/JobLevel/employee_ID) - 매개변수 전달
                                saved_path = hierarchical_result_manager.save_employee_result(
                                    employee_id=employee_id,
                                    employee_data=employee_data,
                                    worker_results=clean_agent_results,
                                    department=department,
                                    job_role=job_role,
                                    position=job_level
                                )
                                saved_paths.append(saved_path)
                                total_employees_saved += 1
                                
                                logger.debug(f"직원 {employee_id} 저장 완료: {saved_path}")
                            
                        except Exception as e:
                            logger.error(f"직원 {employee_id} 저장 실패: {e}")
                            continue
        
        # 전체 통계 업데이트
        try:
            # 부서별 통계 업데이트
            for department in hierarchical_results.keys():
                hierarchical_result_manager.update_department_statistics(department)
        except Exception as e:
            logger.warning(f"부서별 통계 업데이트 실패: {e}")
        
        # batch_analysis 폴더에 전체 결과 요약 JSON 파일 저장
        try:
            batch_analysis_dir = hierarchical_result_manager.base_output_dir / 'batch_analysis'
            batch_analysis_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
            
            # 1. department_summary JSON 생성
            department_summary = {
                'analysis_timestamp': analysis_timestamp,
                'total_employees': total_employees_saved,
                'total_departments': len(hierarchical_results),
                'department_results': {}
            }
            
            # 2. individual_results JSON 생성 (플랫 구조)
            individual_results = {
                'analysis_timestamp': analysis_timestamp,
                'total_employees': total_employees_saved,
                'individual_results': []
            }
            
            # 모든 직원 데이터를 수집
            for dept_name, job_roles in hierarchical_results.items():
                dept_employees = []
                dept_risk_distribution = {'안전군': 0, '주의군': 0, '고위험군': 0}
                
                for role_name, job_levels in job_roles.items():
                    for level_name, employees in job_levels.items():
                        for employee_id, employee_result in employees.items():
                            if not employee_result or not isinstance(employee_result, dict):
                                continue
                            
                            employee_data = employee_result.get('employee_data', {})
                            agent_results = employee_result.get('agent_results', {})
                            
                            # 위험 점수 계산
                            risk_scores = []
                            if 'structura' in agent_results:
                                risk_scores.append(agent_results['structura'].get('attrition_probability', 0))
                            if 'chronos' in agent_results:
                                risk_scores.append(agent_results['chronos'].get('risk_score', 0))
                            if 'cognita' in agent_results:
                                risk_scores.append(agent_results['cognita'].get('overall_risk_score', 0))
                            if 'sentio' in agent_results:
                                risk_scores.append(agent_results['sentio'].get('risk_score', 0))
                            if 'agora' in agent_results:
                                risk_scores.append(agent_results['agora'].get('agora_score', agent_results['agora'].get('market_risk_score', 0)))
                            
                            avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0
                            
                            # 위험도 레벨 결정
                            if avg_risk_score >= 0.7:
                                risk_level = 'high'
                                risk_level_kr = '고위험군'
                            elif avg_risk_score >= 0.3:
                                risk_level = 'medium'
                                risk_level_kr = '주의군'
                            else:
                                risk_level = 'low'
                                risk_level_kr = '안전군'
                            
                            dept_risk_distribution[risk_level_kr] += 1
                            
                            # 직원 결과 생성
                            employee_summary = {
                                'employee_id': employee_id,
                                'department': dept_name,
                                'risk_score': avg_risk_score,
                                'risk_level': risk_level,
                                'analysis_timestamp': analysis_timestamp,
                                'agent_results': {
                                    'structura': agent_results.get('structura', {}),
                                    'chronos': agent_results.get('chronos', {}),
                                    'cognita': agent_results.get('cognita', {}),
                                    'sentio': agent_results.get('sentio', {}),
                                    'agora': agent_results.get('agora', {})
                                }
                            }
                            
                            dept_employees.append(employee_summary)
                            individual_results['individual_results'].append(employee_summary)
                
                # 부서별 요약 추가
                department_summary['department_results'][dept_name] = {
                    'department': dept_name,
                    'original_name': dept_name,
                    'total_employees': len(dept_employees),
                    'risk_distribution': dept_risk_distribution,
                    'employees': dept_employees
                }
            
            # 파일 저장
            dept_summary_file = batch_analysis_dir / f'department_summary_{timestamp_str}.json'
            with open(dept_summary_file, 'w', encoding='utf-8') as f:
                json.dump(department_summary, f, ensure_ascii=False, indent=2)
            
            individual_results_file = batch_analysis_dir / f'individual_results_{timestamp_str}.json'
            with open(individual_results_file, 'w', encoding='utf-8') as f:
                json.dump(individual_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📊 배치 분석 요약 파일 저장 완료:")
            logger.info(f"  - {dept_summary_file}")
            logger.info(f"  - {individual_results_file}")
            
        except Exception as e:
            logger.error(f"배치 분석 요약 파일 저장 실패: {e}")
            logger.error(traceback.format_exc())
        
        logger.info(f"계층적 배치 분석 결과 저장 완료: {total_employees_saved}명")
        
        return jsonify({
            'success': True,
            'message': f'계층적 구조로 배치 분석 결과가 저장되었습니다.',
            'statistics': {
                'total_departments': len(hierarchical_results),
                'total_employees_saved': total_employees_saved,
                'saved_paths_count': len(saved_paths),
                'structure': 'Department > JobRole > JobLevel > Employee'
            },
            'analysis_timestamp': analysis_timestamp,
            'batch_summary_files': {
                'department_summary': f'department_summary_{timestamp_str}.json',
                'individual_results': f'individual_results_{timestamp_str}.json'
            }
        })
        
    except Exception as e:
        logger.error(f"계층적 배치 분석 결과 저장 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'계층적 배치 분석 결과 저장 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


def call_worker_api(worker_name, endpoint, data):
    """워커 API 호출 헬퍼 함수"""
    import requests
    
    worker_ports = {
        'structura': 5001,
        'cognita': 5002,
        'chronos': 5003,
        'sentio': 5004,
        'agora': 5005
    }
    
    port = worker_ports.get(worker_name)
    if not port:
        raise ValueError(f"Unknown worker: {worker_name}")
    
    if endpoint.startswith('analyze/'):
        # GET 요청
        url = f"http://localhost:{port}/api/{endpoint}"
        response = requests.get(url, timeout=30)
    else:
        # POST 요청
        url = f"http://localhost:{port}/api/{endpoint}"
        response = requests.post(url, json=data, timeout=30)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Worker API error: {response.status_code} - {response.text}")


def call_integration_api(endpoint, data):
    """Integration API 호출 헬퍼 함수"""
    import requests
    
    url = f"http://localhost:5007/{endpoint}"
    response = requests.post(url, json=data, timeout=60)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Integration API error: {response.status_code} - {response.text}")


def extract_scores_from_worker_results(worker_results):
    """워커 결과에서 점수 추출"""
    scores = {}
    
    # Structura 점수 추출
    if 'structura' in worker_results and 'error' not in worker_results['structura']:
        structura_data = worker_results['structura']
        if 'probability' in structura_data:
            scores['structura_score'] = structura_data['probability']
        elif 'prediction_probability' in structura_data:
            scores['structura_score'] = structura_data['prediction_probability']
    
    # Cognita 점수 추출
    if 'cognita' in worker_results and 'error' not in worker_results['cognita']:
        cognita_data = worker_results['cognita']
        if 'risk_score' in cognita_data:
            scores['cognita_score'] = cognita_data['risk_score']
    
    # Chronos 점수 추출
    if 'chronos' in worker_results and 'error' not in worker_results['chronos']:
        chronos_data = worker_results['chronos']
        if 'prediction_score' in chronos_data:
            scores['chronos_score'] = chronos_data['prediction_score']
    
    # Sentio 점수 추출
    if 'sentio' in worker_results and 'error' not in worker_results['sentio']:
        sentio_data = worker_results['sentio']
        # psychological_risk_score를 우선적으로 사용 (JD-R 모델 기반)
        if 'psychological_risk_score' in sentio_data:
            scores['sentio_score'] = sentio_data['psychological_risk_score']
        elif 'risk_score' in sentio_data:
            scores['sentio_score'] = sentio_data['risk_score']
        # sentiment_score도 대안으로 사용 가능
        elif 'sentiment_score' in sentio_data:
            scores['sentio_score'] = 1.0 - sentio_data['sentiment_score']  # 감정 점수를 위험 점수로 변환
    
    # Agora 점수 추출
    if 'agora' in worker_results and 'error' not in worker_results['agora']:
        agora_data = worker_results['agora']
        if 'overall_risk_score' in agora_data:
            scores['agora_score'] = agora_data['overall_risk_score']
    
    return scores


# ------------------------------------------------------
# 에러 핸들러
# ------------------------------------------------------

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404


@app.route('/api/results/<path:file_path>')
def serve_results_file(file_path):
    """결과 파일 서빙 (404 오류 방지용)"""
    try:
        # 실제 파일 경로 구성
        # app/results 경로 사용 (hierarchical_result_manager와 동일)
        # 현재 파일: app/Supervisor/supervisor_flask_backend.py
        app_dir = Path(__file__).parent.parent  # app 폴더
        results_dir = app_dir / 'results'
        full_path = results_dir / file_path
        
        # 보안을 위해 results 디렉토리 내부인지 확인
        if not str(full_path.resolve()).startswith(str(results_dir.resolve())):
            return jsonify({
                'success': False,
                'error': 'Invalid file path'
            }), 403
        
        # 파일이 존재하는지 확인
        if full_path.exists() and full_path.is_file():
            return send_file(str(full_path))
        else:
            # 파일이 없으면 빈 JSON 반환 (404 대신)
            return jsonify({
                'success': False,
                'error': 'File not found',
                'message': f'파일을 찾을 수 없습니다: {file_path}'
            }), 404
            
    except Exception as e:
        logger.error(f"Error serving results file {file_path}: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@app.route('/api/search/employees', methods=['GET'])
def search_employees():
    """직원 검색 API"""
    try:
        # 쿼리 파라미터 추출
        query = request.args.get('query', '').strip()
        department = request.args.get('department', '').strip()
        job_role = request.args.get('job_role', '').strip()
        job_level = request.args.get('job_level', '').strip()
        risk_level = request.args.get('risk_level', '').strip()
        limit = int(request.args.get('limit', 50))
        
        # app/results 경로 사용 (hierarchical_result_manager와 동일)
        # 현재 파일: app/Supervisor/supervisor_flask_backend.py
        app_dir = Path(__file__).parent.parent  # app 폴더
        results_dir = app_dir / 'results'
        employees = []
        
        # 모든 부서 디렉토리 탐색
        for dept_dir in results_dir.iterdir():
            if not dept_dir.is_dir() or dept_dir.name in ['global_reports', 'models', 'temp']:
                continue
                
            dept_name = dept_dir.name
            
            # 부서 필터링
            if department and department.lower() not in dept_name.lower():
                continue
            
            # 부서 인덱스 파일 읽기
            dept_index_file = dept_dir / 'department_index.json'
            if not dept_index_file.exists():
                continue
                
            with open(dept_index_file, 'r', encoding='utf-8') as f:
                dept_index = json.load(f)
            
            # 직무별 탐색
            for role_name, role_data in dept_index.get('job_roles', {}).items():
                # 직무 필터링
                if job_role and job_role.lower() not in role_name.lower():
                    continue
                
                # 직급별 탐색
                for level, employee_ids in role_data.items():
                    # 직급 필터링
                    if job_level and job_level != level:
                        continue
                    
                    # 직원별 탐색
                    for emp_id in employee_ids:
                        # 쿼리 필터링
                        if query and query.lower() not in emp_id.lower():
                            continue
                        
                        # 직원 정보 수집
                        emp_dir = dept_dir / role_name / level / f'employee_{emp_id}'
                        if emp_dir.exists():
                            # 직원 기본 정보 로드
                            emp_info_file = emp_dir / 'employee_info.json'
                            comprehensive_file = emp_dir / 'comprehensive_report.json'
                            
                            employee_data = {
                                'employee_id': emp_id,
                                'department': dept_name,
                                'job_role': role_name,
                                'job_level': level,
                                'file_path': str(emp_dir.relative_to(results_dir))
                            }
                            
                            # 종합 보고서에서 위험도 정보 추출
                            if comprehensive_file.exists():
                                try:
                                    with open(comprehensive_file, 'r', encoding='utf-8') as f:
                                        comp_data = json.load(f)
                                    
                                    risk_score = comp_data.get('risk_assessment', {}).get('overall_risk_score', 0)
                                    employee_data['risk_score'] = risk_score
                                    
                                    # 위험도 레벨 계산
                                    if risk_score >= 0.7:
                                        employee_data['risk_level'] = 'HIGH'
                                    elif risk_score >= 0.3:
                                        employee_data['risk_level'] = 'MEDIUM'
                                    else:
                                        employee_data['risk_level'] = 'LOW'
                                        
                                except Exception:
                                    employee_data['risk_score'] = 0
                                    employee_data['risk_level'] = 'UNKNOWN'
                            
                            # 위험도 레벨 필터링
                            if risk_level and risk_level.upper() != employee_data.get('risk_level', 'UNKNOWN'):
                                continue
                            
                            employees.append(employee_data)
                            
                            # 제한 확인
                            if len(employees) >= limit:
                                break
                    
                    if len(employees) >= limit:
                        break
                if len(employees) >= limit:
                    break
            if len(employees) >= limit:
                break
        
        return jsonify({
            'success': True,
            'employees': employees,
            'total': len(employees),
            'query_params': {
                'query': query,
                'department': department,
                'job_role': job_role,
                'job_level': job_level,
                'risk_level': risk_level,
                'limit': limit
            }
        })
        
    except Exception as e:
        logger.error(f"Employee search error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/search/departments', methods=['GET'])
def search_departments():
    """부서 목록 조회 API"""
    try:
        # app/results 경로 사용 (hierarchical_result_manager와 동일)
        # 현재 파일: app/Supervisor/supervisor_flask_backend.py
        app_dir = Path(__file__).parent.parent  # app 폴더
        results_dir = app_dir / 'results'
        departments = []
        
        for dept_dir in results_dir.iterdir():
            if not dept_dir.is_dir() or dept_dir.name in ['global_reports', 'models', 'temp']:
                continue
            
            dept_name = dept_dir.name
            dept_index_file = dept_dir / 'department_index.json'
            
            dept_info = {
                'name': dept_name,
                'display_name': dept_name.replace('_', ' '),
                'total_employees': 0,
                'job_roles': []
            }
            
            if dept_index_file.exists():
                try:
                    with open(dept_index_file, 'r', encoding='utf-8') as f:
                        dept_index = json.load(f)
                    
                    # 직무별 직원 수 계산
                    for role_name, role_data in dept_index.get('job_roles', {}).items():
                        role_employees = sum(len(emp_list) for emp_list in role_data.values())
                        dept_info['total_employees'] += role_employees
                        dept_info['job_roles'].append({
                            'name': role_name,
                            'employee_count': role_employees,
                            'job_levels': list(role_data.keys())
                        })
                        
                except Exception as e:
                    logger.warning(f"Error reading department index {dept_name}: {e}")
            
            departments.append(dept_info)
        
        return jsonify({
            'success': True,
            'departments': departments,
            'total': len(departments)
        })
        
    except Exception as e:
        logger.error(f"Department search error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/employee/<employee_id>/details', methods=['GET'])
def get_employee_details(employee_id):
    """특정 직원의 상세 분석 결과 조회"""
    try:
        include_xai = request.args.get('include_xai', 'false').lower() == 'true'
        
        # app/results 경로 사용 (hierarchical_result_manager와 동일)
        # 현재 파일: app/Supervisor/supervisor_flask_backend.py
        app_dir = Path(__file__).parent.parent  # app 폴더
        results_dir = app_dir / 'results'
        
        # 직원 디렉토리 찾기
        employee_dir = None
        for dept_dir in results_dir.iterdir():
            if not dept_dir.is_dir() or dept_dir.name in ['global_reports', 'models', 'temp']:
                continue
            
            for role_dir in dept_dir.iterdir():
                if not role_dir.is_dir():
                    continue
                    
                for level_dir in role_dir.iterdir():
                    if not level_dir.is_dir():
                        continue
                    
                    emp_dir = level_dir / f'employee_{employee_id}'
                    if emp_dir.exists():
                        employee_dir = emp_dir
                        break
                if employee_dir:
                    break
            if employee_dir:
                break
        
        if not employee_dir:
            return jsonify({
                'success': False,
                'error': f'직원 {employee_id}를 찾을 수 없습니다.'
            }), 404
        
        # 결과 파일들 로드
        result_data = {
            'employee_id': employee_id,
            'directory_path': str(employee_dir.relative_to(results_dir))
        }
        
        # 각 결과 파일 로드
        result_files = [
            'employee_info.json',
            'comprehensive_report.json',
            'structura_result.json',
            'chronos_result.json',
            'cognita_result.json',
            'sentio_result.json'
        ]
        
        for file_name in result_files:
            file_path = employee_dir / file_name
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        result_data[file_name.replace('.json', '')] = json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading {file_name}: {e}")
                    result_data[file_name.replace('.json', '')] = None
        
        # 시각화 파일 목록
        viz_dir = employee_dir / 'visualizations'
        if viz_dir.exists():
            viz_files = [f.name for f in viz_dir.iterdir() if f.is_file()]
            result_data['visualizations'] = viz_files
        else:
            result_data['visualizations'] = []
        
        return jsonify({
            'success': True,
            'employee_data': result_data
        })
        
    except Exception as e:
        logger.error(f"Employee details error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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
        print("  📊 분석 기능:")
        print("    GET  /health - 서버 상태 확인")
        print("    POST /analyze_employee - 직원 분석")
        print("    GET  /get_workflow_status/<session_id> - 워크플로우 상태")
        print("    GET  /list_active_sessions - 활성 세션 목록")
        print("    GET  /worker_health_check - 워커 상태 확인")
        print("    POST /batch_analyze - 배치 분석")
        print("    GET  /system_info - 시스템 정보")
        print("  📁 파일 관리:")
        print("    POST /upload_file - 파일 업로드")
        print("    GET  /list_uploaded_files - 업로드된 파일 목록")
        print("    GET  /download_file/<filename> - 파일 다운로드")
        print("    DELETE /delete_file/<filename> - 파일 삭제")
        print("    GET  /get_file_info/<filename> - 파일 상세 정보")
        print("  🔗 워커 에이전트 API:")
        print("    GET  /api/workers/health_check_all - 모든 워커 상태 확인")
        print("    POST /api/workers/structura/predict - Structura 예측")
        print("    GET  /api/workers/cognita/analyze/<employee_id> - Cognita 분석")
        print("    POST /api/workers/chronos/predict - Chronos 예측")
        print("    POST /api/workers/sentio/analyze_sentiment - Sentio 감정 분석")
        print("    POST /api/workers/agora/comprehensive_analysis - Agora 시장 분석")
        print("    POST /api/workers/upload_data - 모든 워커에 데이터 업로드")
        print("  🔧 Integration 에이전트 API:")
        print("    POST /api/workers/integration/load_data - 데이터 로드")
        print("    POST /api/workers/integration/calculate_thresholds - 임계값 계산")
        print("    POST /api/workers/integration/optimize_weights - 가중치 최적화")
        print("    POST /api/workers/integration/predict_employee - 개별 직원 예측")
        print("    GET  /api/workers/integration/get_results - 결과 조회")
        print("    POST /api/workers/integration/compare_methods - 최적화 방법 비교")
        print("    POST /api/workers/integration/generate_report - 개별 레포트 생성")
        print("    POST /api/workers/integration/generate_batch_reports - 일괄 레포트 생성")
        print("    POST /api/workers/integration/upload_employee_data - 직원 데이터 업로드")
        print("    GET  /api/workers/integration/get_employee_list - 직원 목록 조회")
        print("  🚀 배치 처리 기능:")
        print("    POST /batch_process - 대량 직원 데이터 배치 처리")
        print("    POST /batch_upload_csv - CSV 파일 업로드 배치 처리")
        print("    GET  /batch_status/<batch_id> - 배치 처리 상태 조회")
        print("    GET  /batch_results/<batch_id> - 배치 처리 결과 조회")
        print("    GET  /batch_export_csv/<batch_id> - 배치 처리 결과 CSV 내보내기")
        print("  📊 통계 분석 기능:")
        print("    GET  /api/statistics/group - 단체 통계 조회 (부서별/직무별/직급별)")
        
        # 서버 실행
        app.run(host='0.0.0.0', port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(traceback.format_exc())
        exit(1)
