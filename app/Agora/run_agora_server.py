#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agora HR Market Analysis API Server
외부 노동 시장 분석 서비스 실행 스크립트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

def main():
    """메인 실행 함수"""
    print("🚀 Agora HR Market Analysis API 서버 시작")
    print("=" * 60)
    print("📋 서비스 정보:")
    print("   - 서비스명: Agora Market Analysis")
    print("   - 포트: 5004")
    print("   - 기능: 외부 노동 시장 분석 및 이직 위험도 평가")
    print("=" * 60)
    
    # 환경 변수 확인
    print("🔧 환경 설정 확인:")
    
    # OpenAI API 키 확인
    openai_key = os.environ.get('OPENAI_API_KEY')
    if openai_key:
        print("   ✅ OpenAI API 키: 설정됨")
    else:
        print("   ⚠️  OpenAI API 키: 미설정 (LLM 기능 제한)")
    
    # 데이터 파일 확인
    hr_data_path = project_root / "data" / "IBM_HR.csv"
    if hr_data_path.exists():
        print(f"   ✅ HR 데이터: {hr_data_path}")
    else:
        print(f"   ⚠️  HR 데이터: {hr_data_path} (파일 없음)")
    
    print("=" * 60)
    print("🌐 API 엔드포인트:")
    print("   - 홈페이지: http://localhost:5004/")
    print("   - 헬스체크: http://localhost:5004/health")
    print("   - 개별 분석: POST http://localhost:5004/analyze/market")
    print("   - 직무 분석: POST http://localhost:5004/analyze/job_market")
    print("   - 배치 분석: POST http://localhost:5004/analyze/batch")
    print("   - 시장 보고서: GET http://localhost:5004/market/report/<job_role>")
    print("   - 시장 트렌드: GET http://localhost:5004/market/trends")
    print("=" * 60)
    print("💡 사용 예시:")
    print("   1. 개별 직원 시장 분석:")
    print("      curl -X POST http://localhost:5004/analyze/market \\")
    print("           -H 'Content-Type: application/json' \\")
    print("           -d '{\"EmployeeNumber\": 1001, \"JobRole\": \"Sales Executive\", \"MonthlyIncome\": 5000}'")
    print()
    print("   2. 직무별 시장 현황:")
    print("      curl http://localhost:5004/market/report/Sales%20Executive")
    print("=" * 60)
    
    try:
        # Flask 앱 import 및 실행
        from agora_flask_backend import app
        
        print("✅ Agora 서버 시작 중...")
        app.run(
            host='0.0.0.0',
            port=5004,
            debug=True,
            threaded=True
        )
        
    except ImportError as e:
        print(f"❌ 모듈 import 실패: {e}")
        print("   다음을 확인해주세요:")
        print("   1. 필요한 패키지가 설치되었는지 확인")
        print("   2. pip install -r requirements.txt")
        print("   3. Python 경로 설정이 올바른지 확인")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
