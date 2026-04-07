"""
Integration 서버 실행 스크립트
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_requirements():
    """필요한 패키지 확인"""
    required_packages = [
        'flask',
        'flask-cors',
        'pandas',
        'numpy',
        'scikit-learn',
        'scipy'
    ]
    
    optional_packages = [
        'scikit-optimize'  # Bayesian Optimization용
    ]
    
    print("필요한 패키지 확인 중...")
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            # scikit-learn은 sklearn으로 import됨
            import_name = 'sklearn' if package == 'scikit-learn' else package.replace('-', '_')
            __import__(import_name)
            print(f"✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"누락: {package} (필수)")
    
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} (선택사항)")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠️ {package} (선택사항 - Bayesian Optimization 사용 불가)")
    
    if missing_required:
        print(f"\n필수 패키지가 누락되었습니다: {', '.join(missing_required)}")
        print("다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n⚠️ 선택적 패키지가 누락되었습니다: {', '.join(missing_optional)}")
        print("Bayesian Optimization을 사용하려면 다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_optional)}")
    
    return True


def check_data_files():
    """데이터 파일 존재 확인"""
    print("\n📁 데이터 파일 확인 중...")
    
    # 프로젝트 루트에서 data 폴더 찾기
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_dir = project_root / 'data'
    
    required_files = [
        'Total_score.csv'
    ]
    
    optional_files = [
        'Total_score_with_predictions.csv',
        'IBM_HR.csv'
    ]
    
    print(f"데이터 디렉토리: {data_dir}")
    
    if not data_dir.exists():
        print(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}")
        return False
    
    missing_required = []
    
    for file_name in required_files:
        file_path = data_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            missing_required.append(file_name)
            print(f"누락: {file_name} (필수)")
    
    for file_name in optional_files:
        file_path = data_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} (선택사항)")
        else:
            print(f"⚠️ {file_name} (선택사항)")
    
    if missing_required:
        print(f"\n필수 데이터 파일이 누락되었습니다: {', '.join(missing_required)}")
        print(f"다음 위치에 파일을 배치하세요: {data_dir}")
        return False
    
    return True


def create_requirements_file():
    """requirements.txt 파일 생성"""
    requirements_content = """flask==2.3.3
flask-cors==4.0.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.1
scikit-optimize==0.9.0
matplotlib==3.7.2
seaborn==0.12.2
"""
    
    requirements_path = Path(__file__).parent / 'requirements.txt'
    
    with open(requirements_path, 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    
    print(f"📝 requirements.txt 파일이 생성되었습니다: {requirements_path}")
    return requirements_path


def run_server():
    """서버 실행"""
    print("\n🚀 Integration 서버 시작 중...")
    
    # 현재 디렉토리를 Integration 폴더로 변경
    os.chdir(Path(__file__).parent)
    
    try:
        # Flask 서버 실행
        from integration_flask_backend import app
        
        print("🌐 서버가 다음 주소에서 실행됩니다:")
        print("   http://localhost:5007")
        print("\n📋 사용 가능한 엔드포인트:")
        print("   GET  /health - 서버 상태 확인")
        print("   POST /load_data - 데이터 로드")
        print("   POST /calculate_thresholds - 임계값 계산")
        print("   POST /optimize_weights - 가중치 최적화")
        print("   POST /predict_employee - 개별 직원 예측")
        print("   GET  /get_results - 현재 결과 조회")
        print("   POST /compare_methods - 최적화 방법 비교")
        print("   POST /export_results - 결과 내보내기")
        print("   POST /load_employee_data - 직원 기본 데이터 로드")
        print("   GET  /get_employee_list - 직원 목록 조회")
        print("   POST /generate_report - 개별 직원 레포트 생성")
        print("   POST /generate_batch_reports - 일괄 레포트 생성")
        print("\n테스트 방법:")
        print("   python test_integration_api.py")
        print("\n⏹️ 서버 중지: Ctrl+C")
        print("=" * 60)
        
        port = int(os.environ.get('PORT', 5007))
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except ImportError as e:
        print(f"모듈 import 오류: {e}")
        print("필요한 패키지가 설치되었는지 확인하세요.")
        return False
    except Exception as e:
        print(f"서버 실행 오류: {e}")
        return False


def seed_results_if_empty():
    """seed_results/ → results/ 초기 데이터 복사 (Volume이 비어있을 때만)"""
    integration_dir = Path(__file__).parent
    seed_dir   = integration_dir / 'seed_results'
    results_dir = integration_dir.parent.parent / 'app' / 'results'

    # results_dir이 /app/results 인 경우 (Railway 컨테이너 환경)
    # __file__ = /app/run_integration_server.py → parent = /app
    # 그러므로 results_dir = /app/../../app/results 가 아닌 /app/results 로 설정
    container_results = Path('/app/results')
    if container_results.exists() or not results_dir.exists():
        results_dir = container_results

    if not seed_dir.exists():
        print("⚠️  seed_results/ 디렉토리 없음, 초기화 건너뜀")
        return

    # results/ 안에 이미 직원 폴더가 있으면 건너뜀
    existing = [p for p in results_dir.iterdir() if p.is_dir()] if results_dir.exists() else []
    if existing:
        print(f"✅ results/ 에 기존 데이터 있음 ({len(existing)}개 폴더), seed 건너뜀")
        return

    print(f"📦 seed_results/ → {results_dir} 초기 데이터 복사 중...")
    results_dir.mkdir(parents=True, exist_ok=True)
    for item in seed_dir.iterdir():
        dest = results_dir / item.name
        if item.is_dir():
            shutil.copytree(str(item), str(dest), dirs_exist_ok=True)
        else:
            shutil.copy2(str(item), str(dest))
    count = sum(1 for _ in results_dir.rglob('*.json'))
    print(f"✅ seed 완료: {count}개 JSON 파일 복사됨")


def main():
    """메인 함수"""
    print("Integration 서버 설정 및 실행")
    print("=" * 50)

    # 0. 초기 seed 데이터 복사 (Volume 마운트 후 비어있을 때)
    seed_results_if_empty()

    # 1. 패키지 확인
    if not check_requirements():
        print("\n필수 패키지가 누락되어 서버를 시작할 수 없습니다.")
        
        # requirements.txt 생성
        req_file = create_requirements_file()
        print(f"\n다음 명령어로 필요한 패키지를 설치하세요:")
        print(f"pip install -r {req_file}")
        return
    
    # 2. 데이터 파일 확인 (Railway 환경에서는 API로 데이터를 수신하므로 non-fatal)
    if not check_data_files():
        print("\n⚠️ 데이터 파일이 없습니다. API를 통해 데이터를 수신합니다.")

    print("\n✅ 모든 사전 조건이 충족되었습니다.")
    
    # 3. 서버 실행
    run_server()


if __name__ == "__main__":
    main()
