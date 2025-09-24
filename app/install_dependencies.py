#!/usr/bin/env python3
"""
Agentic AI System 의존성 설치 스크립트
PyTorch CUDA와 일반 패키지를 순차적으로 설치합니다.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """명령어 실행 및 결과 출력"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"실행 명령어: {command}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ {description} 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 실패!")
        print(f"오류 코드: {e.returncode}")
        return False

def main():
    """메인 설치 프로세스"""
    print("🤖 Agentic AI System 의존성 설치를 시작합니다...")
    
    # 현재 디렉토리 확인
    current_dir = os.getcwd()
    print(f"📁 현재 디렉토리: {current_dir}")
    
    # 모든 패키지 일괄 설치 (PyTorch CUDA 포함)
    requirements_command = "pip install -r requirements_agentic.txt"
    if not run_command(requirements_command, "모든 패키지 설치 (PyTorch CUDA 11.8 포함)"):
        print("\n⚠️  CUDA 버전 설치에 실패했습니다. CPU 버전으로 시도합니다...")
        # CPU 버전으로 fallback
        pytorch_cpu_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        if not run_command(pytorch_cpu_command, "PyTorch CPU 버전 설치"):
            print("❌ PyTorch 설치에 완전히 실패했습니다.")
            return False
        
        # CPU 버전 설치 후 나머지 패키지 설치
        other_packages_command = "pip install numpy pandas scipy scikit-learn xgboost optuna shap lime imbalanced-learn neo4j flask flask-cors werkzeug openai langchain langchain-openai langchain-core langgraph matplotlib seaborn plotly python-dotenv requests joblib tqdm pydantic aiohttp"
        if not run_command(other_packages_command, "나머지 패키지 설치"):
            print("❌ 나머지 패키지 설치에 실패했습니다.")
            return False
    
    # 3. 설치 확인
    print(f"\n{'='*60}")
    print("🔍 설치된 패키지 확인")
    print(f"{'='*60}")
    
    # 주요 패키지들 확인
    packages_to_check = [
        "torch", "torchvision", "torchaudio",
        "numpy", "pandas", "scikit-learn",
        "flask", "openai", "langchain"
    ]
    
    failed_packages = []
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"✅ {package}: 설치됨")
        except ImportError:
            print(f"❌ {package}: 설치되지 않음")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  다음 패키지들이 설치되지 않았습니다: {', '.join(failed_packages)}")
        print("수동으로 설치를 시도해보세요:")
        for package in failed_packages:
            print(f"  pip install {package}")
        return False
    
    print(f"\n{'='*60}")
    print("🎉 모든 의존성 설치가 완료되었습니다!")
    print(f"{'='*60}")
    print("이제 Agentic AI System을 실행할 수 있습니다.")
    print("실행 명령어: python main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)