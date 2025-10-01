#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structura HR 예측 Flask 백엔드 서버 실행 스크립트
xAI 기능 포함 (SHAP, LIME, Feature Importance)
"""

import os
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Structura Flask 서버 실행 메인 함수"""
    
    print("=" * 60)
    print("🚀 Structura HR 예측 Flask 백엔드 서버 시작")
    print("=" * 60)
    
    # 환경 변수 설정 (선택사항)
    os.environ.setdefault("FLASK_ENV", "development")
    
    # 서버 설정
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5001))  # Structura 전용 포트
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print(f"📡 서버 주소: http://{host}:{port}")
    print(f"🔗 React 연동: http://localhost:3000에서 접근 가능")
    print(f"🤖 xAI 기능: SHAP, LIME, Feature Importance")
    print(f"🔄 디버그 모드: {'활성화' if debug else '비활성화'}")
    print()
    print("주요 엔드포인트:")
    print(f"  • 헬스체크: http://{host}:{port}/api/health")
    print(f"  • 모델 훈련: http://{host}:{port}/api/train")
    print(f"  • 이직 예측: http://{host}:{port}/api/predict")
    print(f"  • 예측 설명: http://{host}:{port}/api/explain")
    print(f"  • 피처 중요도: http://{host}:{port}/api/feature-importance")
    print(f"  • 모델 정보: http://{host}:{port}/api/model/info")
    print()
    print("React 연동 예시:")
    print("  // 헬스체크")
    print("  fetch('http://localhost:5001/api/health')")
    print("    .then(response => response.json())")
    print("    .then(data => console.log(data));")
    print()
    print("  // 이직 예측")
    print("  fetch('http://localhost:5001/api/predict', {")
    print("    method: 'POST',")
    print("    headers: { 'Content-Type': 'application/json' },")
    print("    body: JSON.stringify({")
    print("      Age: 35,")
    print("      JobSatisfaction: 3,")
    print("      WorkLifeBalance: 2,")
    print("      // ... 기타 피처들")
    print("    })")
    print("  })")
    print()
    print("  // 예측 설명 (xAI)")
    print("  fetch('http://localhost:5001/api/explain', {")
    print("    method: 'POST',")
    print("    headers: { 'Content-Type': 'application/json' },")
    print("    body: JSON.stringify(employeeData)")
    print("  })")
    print()
    print("서버를 중지하려면 Ctrl+C를 누르세요.")
    print("=" * 60)
    
    try:
        # Flask 앱 import 및 실행
        from structura_flask_backend import run_server
        run_server(host=host, port=port, debug=debug)
        
    except KeyboardInterrupt:
        print("\n서버가 중지되었습니다.")
    except ImportError as e:
        print(f"\n모듈 import 실패: {e}")
        print("다음을 확인하세요:")
        print("1. pip install -r requirements.txt")
        print("2. structura_flask_backend.py 파일 존재 여부")
        print("3. 필요한 xAI 라이브러리 설치:")
        print("   pip install shap lime")
        sys.exit(1)
    except Exception as e:
        print(f"\n서버 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
