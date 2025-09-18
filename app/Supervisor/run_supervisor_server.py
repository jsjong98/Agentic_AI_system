#!/usr/bin/env python3
"""
Supervisor 서버 실행 스크립트
"""

import os
import sys
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.Supervisor.supervisor_flask_backend import create_app

def setup_environment():
    """환경 변수 설정"""
    
    # 기본 환경 변수 설정
    default_env = {
        'SUPERVISOR_PORT': '5006',
        'FLASK_DEBUG': 'False',
        'MAX_RETRY_COUNT': '3',
        'TIMEOUT_MINUTES': '30',
        'MAX_BATCH_SIZE': '10',
        
        # 워커 서버 URL (기본값)
        'STRUCTURA_URL': 'http://localhost:5001',
        'COGNITA_URL': 'http://localhost:5002',
        'CHRONOS_URL': 'http://localhost:5003',
        'SENTIO_URL': 'http://localhost:5004',
        'AGORA_URL': 'http://localhost:5005',
    }
    
    for key, value in default_env.items():
        if key not in os.environ:
            os.environ[key] = value
    
    # OpenAI API 키 확인
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   LLM 기능이 제한될 수 있습니다.")
        print("   .env 파일에 OPENAI_API_KEY=your_key_here 를 추가하세요.")
        print()

def check_worker_availability():
    """워커 서버 가용성 확인"""
    import requests
    from urllib.parse import urljoin
    
    workers = {
        'Structura': os.getenv('STRUCTURA_URL'),
        'Cognita': os.getenv('COGNITA_URL'),
        'Chronos': os.getenv('CHRONOS_URL'),
        'Sentio': os.getenv('SENTIO_URL'),
        'Agora': os.getenv('AGORA_URL'),
    }
    
    print("🔍 워커 서버 상태 확인:")
    available_workers = []
    
    for worker_name, base_url in workers.items():
        try:
            # 먼저 /health를 시도하고, 실패하면 /api/health를 시도
            health_urls = ['/health', '/api/health']
            success = False
            
            for health_path in health_urls:
                try:
                    health_url = urljoin(base_url, health_path)
                    response = requests.get(health_url, timeout=5)
                    
                    if response.status_code == 200:
                        print(f"  ✅ {worker_name}: {base_url} - 온라인")
                        available_workers.append(worker_name)
                        success = True
                        break
                except requests.exceptions.RequestException:
                    continue
            
            if not success:
                print(f"  ❌ {worker_name}: {base_url} - 오프라인")
                
        except Exception as e:
            print(f"  ❌ {worker_name}: {base_url} - 연결 실패 ({str(e)[:50]}...)")
    
    print(f"\n📊 사용 가능한 워커: {len(available_workers)}/5")
    
    if len(available_workers) == 0:
        print("⚠️  경고: 사용 가능한 워커가 없습니다.")
        print("   각 워커 서버가 실행 중인지 확인하세요:")
        for worker_name, base_url in workers.items():
            print(f"   - {worker_name}: {base_url}")
        print()
    
    return available_workers

def main():
    """메인 실행 함수"""
    
    print("🚀 Supervisor 에이전트 서버 시작")
    print("=" * 50)
    
    # 환경 설정
    setup_environment()
    
    # 워커 상태 확인
    available_workers = check_worker_availability()
    
    # 로깅 설정
    log_level = logging.DEBUG if os.getenv('FLASK_DEBUG', 'False').lower() == 'true' else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Flask 앱 생성
        app = create_app()
        
        # 서버 설정
        port = int(os.getenv('SUPERVISOR_PORT', '5006'))
        debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        
        print(f"\n🌐 서버 정보:")
        print(f"  📡 주소: http://localhost:{port}")
        print(f"  🔧 디버그: {debug}")
        print(f"  🤖 사용 가능한 워커: {len(available_workers)}/5")
        
        print(f"\n📋 주요 엔드포인트:")
        print(f"  GET  http://localhost:{port}/health")
        print(f"  POST http://localhost:{port}/analyze_employee")
        print(f"  GET  http://localhost:{port}/worker_health_check")
        print(f"  POST http://localhost:{port}/batch_analyze")
        print(f"  GET  http://localhost:{port}/system_info")
        
        print(f"\n🔥 서버 시작 중...")
        print("=" * 50)
        
        # 서버 실행
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug,
            use_reloader=False  # 프로덕션에서는 reloader 비활성화
        )
        
    except KeyboardInterrupt:
        print("\n\n⏹️  서버가 사용자에 의해 중단되었습니다.")
        
    except Exception as e:
        print(f"\n❌ 서버 시작 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
