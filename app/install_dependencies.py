#!/usr/bin/env python3
"""
Agentic AI System 의존성 설치 및 검증 스크립트
모든 필요한 라이브러리를 설치하고 import 테스트를 수행합니다.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """requirements_agentic.txt 파일을 사용하여 의존성 설치"""
    
    requirements_file = Path(__file__).parent / "requirements_agentic.txt"
    
    if not requirements_file.exists():
        print("❌ requirements_agentic.txt 파일을 찾을 수 없습니다.")
        return False
    
    print("🚀 Agentic AI System 의존성 설치를 시작합니다...")
    print(f"📁 Requirements 파일: {requirements_file}")
    
    try:
        # pip install 실행
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True, capture_output=True, text=True)
        
        print("✅ 모든 의존성이 성공적으로 설치되었습니다!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 설치 중 오류 발생: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def test_imports():
    """주요 라이브러리 import 테스트"""
    
    print("\n🧪 라이브러리 import 테스트를 시작합니다...")
    
    test_modules = {
        # 기본 데이터 처리
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'scipy': 'scipy',
        
        # 머신러닝
        'scikit-learn': 'sklearn',
        'xgboost': 'xgboost',
        'torch': 'torch',
        
        # 하이퍼파라미터 최적화
        'optuna': 'optuna',
        
        # XAI
        'shap': 'shap',
        'lime': 'lime',
        
        # 불균형 데이터
        'imbalanced-learn': 'imblearn',
        
        # 그래프 DB
        'neo4j': 'neo4j',
        
        # Flask
        'flask': 'flask',
        'flask-cors': 'flask_cors',
        
        # LLM & LangChain
        'openai': 'openai',
        'langchain': 'langchain',
        'langchain-openai': 'langchain_openai',
        'langchain-core': 'langchain_core',
        'langgraph': 'langgraph',
        
        # 시각화
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        
        # 유틸리티
        'python-dotenv': 'dotenv',
        'requests': 'requests',
        'joblib': 'joblib',
        'tqdm': 'tqdm',
        'pydantic': 'pydantic',
        
        # 비동기
        'aiohttp': 'aiohttp'
    }
    
    success_count = 0
    total_count = len(test_modules)
    
    for package_name, import_name in test_modules.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name} ({import_name})")
            success_count += 1
        except ImportError as e:
            print(f"❌ {package_name} ({import_name}): {e}")
    
    print(f"\n📊 Import 테스트 결과: {success_count}/{total_count} 성공")
    
    if success_count == total_count:
        print("🎉 모든 라이브러리가 정상적으로 설치되었습니다!")
        return True
    else:
        print("⚠️ 일부 라이브러리 설치에 문제가 있습니다.")
        return False

def check_gpu_support():
    """GPU 지원 확인 (PyTorch)"""
    
    print("\n🔍 GPU 지원 확인...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            
            print(f"✅ CUDA 사용 가능!")
            print(f"   GPU 개수: {gpu_count}")
            print(f"   현재 GPU: {gpu_name}")
            print(f"   CUDA 버전: {torch.version.cuda}")
        else:
            print("⚠️ CUDA 사용 불가 - CPU 모드로 실행됩니다.")
            
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다.")

def main():
    """메인 실행 함수"""
    
    print("=" * 60)
    print("🤖 Agentic AI System 의존성 설치 및 검증")
    print("=" * 60)
    
    # 1. 의존성 설치
    if not install_requirements():
        print("\n❌ 설치 실패로 인해 테스트를 중단합니다.")
        sys.exit(1)
    
    # 2. Import 테스트
    if not test_imports():
        print("\n⚠️ 일부 라이브러리에 문제가 있지만 계속 진행합니다.")
    
    # 3. GPU 지원 확인
    check_gpu_support()
    
    print("\n" + "=" * 60)
    print("🎯 설치 및 검증 완료!")
    print("=" * 60)
    print("\n다음 단계:")
    print("1. 환경변수 설정: OPENAI_API_KEY")
    print("2. Neo4j 데이터베이스 연결 설정")
    print("3. python run_agentic_system.py 실행")

if __name__ == "__main__":
    main()
