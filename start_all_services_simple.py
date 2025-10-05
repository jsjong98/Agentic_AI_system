#!/usr/bin/env python3
"""
현재 활성화된 conda 환경에서 모든 에이전트 서버를 실행하는 스크립트
(이미 conda activate nlp가 실행된 상태에서 사용)
"""

import subprocess
import sys
import time
import os
import signal
import atexit
from pathlib import Path
import threading

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent
APP_DIR = PROJECT_ROOT / "app"

# 각 에이전트 서버 정보
AGENTS = [
    {
        "name": "Structura",
        "script": APP_DIR / "Structura" / "run_structura_server.py",
        "port": 5001,
        "description": "정형 데이터 ML 분석"
    },
    {
        "name": "Cognita", 
        "script": APP_DIR / "Cognita" / "run_cognita_server.py",
        "port": 5002,
        "description": "네트워크 관계 분석"
    },
    {
        "name": "Chronos",
        "script": APP_DIR / "Chronos" / "run_chronos_server.py", 
        "port": 5003,
        "description": "시계열 딥러닝 분석"
    },
    {
        "name": "Sentio",
        "script": APP_DIR / "Sentio" / "run_sentio_server.py",
        "port": 5004, 
        "description": "텍스트 감정 분석"
    },
    {
        "name": "Agora",
        "script": APP_DIR / "Agora" / "run_agora_server.py",
        "port": 5005,
        "description": "시장 분석 + LLM"
    },
    {
        "name": "Supervisor",
        "script": APP_DIR / "Supervisor" / "run_supervisor_server.py",
        "port": 5006,
        "description": "LangGraph 워크플로우 관리"
    },
    {
        "name": "Integration",
        "script": APP_DIR / "Integration" / "run_integration_server.py", 
        "port": 5007,
        "description": "결과 통합 및 최적화"
    }
]

# 실행 중인 프로세스들
running_processes = []

def cleanup_processes():
    """모든 프로세스 정리"""
    print("\n🛑 모든 서비스를 종료합니다...")
    for process, agent in running_processes:
        try:
            if process.poll() is None:  # 프로세스가 아직 실행 중인 경우
                print(f"   ⏹️  {agent['name']} 서버 종료 중...")
                process.terminate()
                process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"   ⚠️  {agent['name']} 서버 강제 종료...")
            process.kill()
        except Exception as e:
            print(f"   ❌ {agent['name']} 종료 중 오류: {e}")
    
    print("✅ 모든 서비스가 종료되었습니다.")

def start_agent_background(agent):
    """에이전트를 현재 Python 환경에서 백그라운드로 시작"""
    script_path = agent["script"]
    
    if not script_path.exists():
        print(f"❌ {agent['name']} 스크립트를 찾을 수 없습니다: {script_path}")
        return None
    
    print(f"🚀 {agent['name']} 서버 시작 중... (포트 {agent['port']})")
    print(f"   🐍 Python: {sys.executable}")
    print(f"   📁 스크립트: {script_path}")
    
    try:
        # 로그 파일 설정
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{agent['name'].lower()}_server.log"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            # 환경 변수 설정 (UTF-8 인코딩)
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            if sys.platform == "win32":
                env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
            
            # 현재 Python 환경에서 직접 실행 (Windows Git Bash 호환)
            python_exe = sys.executable
            if sys.platform == "win32" and "git" in os.environ.get('PATH', '').lower():
                # Git Bash에서는 전체 경로 사용
                python_exe = python_exe.replace('\\', '/')
            
            process = subprocess.Popen([
                python_exe, str(script_path)
            ], 
            cwd=str(script_path.parent),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
        
        print(f"✅ {agent['name']} 서버 시작됨 (PID: {process.pid})")
        print(f"   📝 로그: {log_file}")
        return process
        
    except Exception as e:
        print(f"❌ {agent['name']} 서버 시작 실패: {e}")
        return None

def check_service_health(port, service_name):
    """서비스 상태 확인"""
    try:
        import requests
        # 서비스별 health check 엔드포인트
        if service_name.lower() == "chronos":
            health_endpoints = ["/api/status"]
        else:
            health_endpoints = ["/health", "/api/health"]
        
        for endpoint in health_endpoints:
            try:
                response = requests.get(f"http://localhost:{port}{endpoint}", timeout=5)
                if response.status_code == 200:
                    return True
            except:
                continue
        return False
    except:
        pass
    return False

def monitor_services():
    """서비스 상태 모니터링"""
    while True:
        time.sleep(15)  # 15초마다 체크
        
        print("\n📊 서비스 상태 체크:")
        all_healthy = True
        
        for process, agent in running_processes:
            if process.poll() is None:  # 프로세스 실행 중
                if check_service_health(agent['port'], agent['name']):
                    print(f"   ✅ {agent['name']:<12} → 정상 (포트 {agent['port']})")
                else:
                    print(f"   ⚠️  {agent['name']:<12} → 응답 없음 (포트 {agent['port']})")
                    all_healthy = False
            else:
                print(f"   ❌ {agent['name']:<12} → 프로세스 종료됨")
                log_file = PROJECT_ROOT / "logs" / f"{agent['name'].lower()}_server.log"
                if log_file.exists():
                    print(f"      📝 로그 확인: {log_file}")
                    # 로그의 마지막 몇 줄 표시
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            if lines:
                                print(f"      🔍 마지막 오류: {lines[-1].strip()}")
                    except:
                        pass
                all_healthy = False
        
        if all_healthy:
            print("🎉 모든 서비스가 정상 작동 중입니다!")
        else:
            print("⚠️  일부 서비스에 문제가 있습니다.")

def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("🤖 Agentic AI System - 통합 서비스 관리자 (현재 환경)")
    print("=" * 70)
    print()
    
    # 현재 Python 환경 정보 표시
    print(f"🐍 Python 실행 파일: {sys.executable}")
    print(f"🌍 Python 버전: {sys.version}")
    
    # conda 환경 확인
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    print(f"🔧 Conda 환경: {conda_env}")
    
    if conda_env != 'nlp':
        print("⚠️  경고: 현재 conda 환경이 'nlp'가 아닙니다.")
        print("   conda activate nlp 명령으로 환경을 활성화하세요.")
        response = input("계속 진행하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            return
    
    print()
    
    # 종료 시 정리 함수 등록
    atexit.register(cleanup_processes)
    signal.signal(signal.SIGINT, lambda s, f: cleanup_processes() or sys.exit(0))
    
    # 각 에이전트 시작
    for agent in AGENTS:
        process = start_agent_background(agent)
        if process:
            running_processes.append((process, agent))
            time.sleep(3)  # 서버 시작 간격을 늘림
        else:
            print(f"❌ {agent['name']} 시작 실패")
    
    if not running_processes:
        print("❌ 시작된 서비스가 없습니다.")
        return
    
    print()
    print("=" * 70)
    print("🎉 모든 에이전트 서버 시작 완료!")
    print("=" * 70)
    print()
    print("📡 실행 중인 서버들:")
    for _, agent in running_processes:
        print(f"  • {agent['name']:<12} → http://localhost:{agent['port']}")
    
    print()
    print("🌐 React 대시보드:")
    print("  cd Dashboard && npm start")
    print()
    print("💡 메인 API 엔드포인트:")
    print("  Supervisor → http://localhost:5006 (모든 기능 통합)")
    print("  - 파일 업로드/관리")
    print("  - 모든 워커 에이전트 API 프록시")
    print("  - Integration 에이전트 API 프록시")
    print()
    print("⚠️  제어 명령:")
    print("  - Ctrl+C: 모든 서비스 종료")
    print("  - 서비스 상태는 자동으로 모니터링됩니다")
    print()
    
    # 잠시 대기 후 첫 번째 상태 체크
    print("⏳ 서버들이 시작될 때까지 잠시 대기 중...")
    time.sleep(10)
    
    try:
        # 모니터링 스레드 시작
        monitor_thread = threading.Thread(target=monitor_services, daemon=True)
        monitor_thread.start()
        
        # 메인 스레드는 대기
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  서비스 종료 요청을 받았습니다...")
        cleanup_processes()

if __name__ == "__main__":
    main()