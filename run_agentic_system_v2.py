#!/usr/bin/env python3
"""
개선된 Agentic AI System 실행기
- 모든 서비스를 하나의 터미널에서 관리
- 이전 run_agentic_system.py와 유사한 사용법
- 마이크로서비스 아키텍처 유지
"""

import subprocess
import sys
import time
import os
import threading
import queue
from pathlib import Path
import signal
import atexit

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent
APP_DIR = PROJECT_ROOT / "app"

class ServiceManager:
    def __init__(self):
        self.services = [
            {
                "name": "Structura",
                "script": APP_DIR / "Structura" / "run_structura_server.py",
                "port": 5001,
                "process": None
            },
            {
                "name": "Cognita", 
                "script": APP_DIR / "Cognita" / "run_cognita_server.py",
                "port": 5002,
                "process": None
            },
            {
                "name": "Chronos",
                "script": APP_DIR / "Chronos" / "run_chronos_server.py", 
                "port": 5003,
                "process": None
            },
            {
                "name": "Sentio",
                "script": APP_DIR / "Sentio" / "run_sentio_server.py",
                "port": 5004,
                "process": None
            },
            {
                "name": "Agora",
                "script": APP_DIR / "Agora" / "run_agora_server.py",
                "port": 5005,
                "process": None
            },
            {
                "name": "Supervisor",
                "script": APP_DIR / "Supervisor" / "run_supervisor_server.py",
                "port": 5006,
                "process": None
            },
            {
                "name": "Integration",
                "script": APP_DIR / "Integration" / "run_integration_server.py", 
                "port": 5007,
                "process": None
            },
        ]
        
        # 종료 시 정리 함수 등록
        atexit.register(self.cleanup)
        
    def start_service(self, service):
        """개별 서비스 시작"""
        if not service["script"].exists():
            print(f"❌ {service['name']} 스크립트를 찾을 수 없습니다: {service['script']}")
            return False
            
        print(f"🚀 {service['name']} 시작 중... (포트 {service['port']})")
        
        try:
            # conda 환경에서 백그라운드 실행
            if sys.platform == "win32":
                # Windows에서 conda 환경 활성화
                conda_cmd = f"conda activate nlp && python {service['script']}"
                service["process"] = subprocess.Popen(
                    conda_cmd,
                    cwd=str(service["script"].parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    shell=True
                )
            else:
                # Linux/Mac에서 conda 환경 활성화
                conda_cmd = f"source activate nlp && python {service['script']}"
                service["process"] = subprocess.Popen(
                    conda_cmd,
                    cwd=str(service["script"].parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    shell=True
                )
            
            print(f"✅ {service['name']} 시작됨 (PID: {service['process'].pid})")
            return True
            
        except Exception as e:
            print(f"❌ {service['name']} 시작 실패: {e}")
            return False
    
    def start_all_services(self):
        """모든 서비스 시작"""
        print("=" * 70)
        print("🤖 Agentic AI System 시작")
        print("=" * 70)
        print()
        
        # 모든 에이전트 서버들 시작
        for service in self.services:
            if self.start_service(service):
                time.sleep(2)  # 서비스 간 시작 간격
        
        print()
        print("⏳ 모든 서버가 준비될 때까지 대기 중...")
        time.sleep(10)  # 서버들이 완전히 시작될 때까지 대기
        
        print()
        print("=" * 70)
        print("🎉 Agentic AI System 시작 완료!")
        print("=" * 70)
        print()
        print("📡 실행 중인 서비스들:")
        for service in self.services:
            if service["process"] and service["process"].poll() is None:
                print(f"  • {service['name']:<12} → http://localhost:{service['port']}")
        
        print()
        print("🌐 웹 인터페이스:")
        print("💡 메인 API 엔드포인트:")
        print("  Supervisor → http://localhost:5006 (모든 기능 통합)")
        print("  • React Dashboard → http://localhost:3000 (별도 실행 필요)")
        print()
        print("⚠️  종료하려면 Ctrl+C를 누르세요")
        print()
    
    def monitor_services(self):
        """서비스 상태 모니터링"""
        while True:
            time.sleep(30)  # 30초마다 체크
            
            failed_services = []
            for service in self.services:
                if service["process"] and service["process"].poll() is not None:
                    failed_services.append(service["name"])
            
            if failed_services:
                print(f"⚠️  다음 서비스들이 종료되었습니다: {', '.join(failed_services)}")
    
    def cleanup(self):
        """모든 서비스 정리"""
        print("\n🛑 모든 서비스를 종료합니다...")
        
        for service in self.services:
            if service["process"] and service["process"].poll() is None:
                try:
                    print(f"   ⏹️  {service['name']} 종료 중...")
                    service["process"].terminate()
                    service["process"].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"   ⚠️  {service['name']} 강제 종료...")
                    service["process"].kill()
                except Exception as e:
                    print(f"   ❌ {service['name']} 종료 중 오류: {e}")
        
        print("✅ 모든 서비스가 종료되었습니다.")
    
    def run(self):
        """메인 실행 함수"""
        try:
            # 모든 서비스 시작
            self.start_all_services()
            
            # 모니터링 스레드 시작
            monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
            monitor_thread.start()
            
            # 메인 스레드는 대기
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n👋 사용자가 종료를 요청했습니다.")
            sys.exit(0)

def main():
    """메인 함수"""
    manager = ServiceManager()
    manager.run()

if __name__ == "__main__":
    main()
