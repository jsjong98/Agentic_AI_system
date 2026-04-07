#!/usr/bin/env python3
# ============================================================================
# Chronos 서버 실행 스크립트
# ============================================================================

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """
    필요한 의존성 패키지 확인
    """
    # 패키지명과 실제 import명이 다른 경우를 매핑
    package_mapping = {
        'flask': 'flask',
        'flask-cors': 'flask_cors',
        'torch': 'torch',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',  # 실제 import명은 sklearn
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'joblib': 'joblib'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_mapping.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ 다음 패키지들이 설치되지 않았습니다:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n설치 명령어:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 모든 의존성 패키지가 설치되어 있습니다.")
    return True

def check_data_files():
    """
    데이터 파일 존재 확인
    """
    data_files = [
        'data/IBM_HR_timeseries.csv',
        'data/IBM_HR.csv'
    ]
    
    missing_files = []
    for file_path in data_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️ 다음 데이터 파일들을 찾을 수 없습니다:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n데이터 파일이 없어도 서버는 실행되지만, 모델 학습은 불가능합니다.")
        return False
    
    print("✅ 모든 데이터 파일이 존재합니다.")
    return True

def setup_environment():
    """
    환경 설정
    """
    # 현재 디렉토리를 프로젝트 루트로 변경
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    os.chdir(project_root)
    
    # Python 경로에 현재 디렉토리 추가
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    print(f"📁 작업 디렉토리: {os.getcwd()}")

def run_server():
    """
    Chronos 서버 실행
    """
    print("🚀 Chronos 서버를 시작합니다...")
    print("=" * 50)
    
    # 환경 설정
    setup_environment()
    
    # 의존성 확인 (경고만 표시하고 계속 진행)
    if not check_dependencies():
        print("\n⚠️  일부 의존성 패키지가 누락되었지만 서버를 시작합니다.")
        print("   일부 기능이 제한될 수 있습니다.")
        time.sleep(2)  # 경고 메시지를 읽을 시간 제공
    
    # 데이터 파일 확인
    check_data_files()
    
    try:
        # 서버 실행
        from chronos_flask_backend import app
        
        print("\n🌐 서버 정보:")
        print("   - URL: http://localhost:5003")
        print("   - API 문서: http://localhost:5003")
        print("   - 종료: Ctrl+C")
        print("=" * 50)
        
        # debug=False로 설정하여 reloader 문제 방지
        port = int(os.environ.get('PORT', 5003))
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except KeyboardInterrupt:
        print("\n👋 서버가 종료되었습니다.")
        return True
    except Exception as e:
        print(f"\n❌ 서버 실행 중 오류 발생: {str(e)}")
        return False

if __name__ == '__main__':
    success = run_server()
    sys.exit(0 if success else 1)
