#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cognita 관계형 위험도 분석 Flask 백엔드 서버 실행 스크립트
Neo4j 기반 관계형 데이터 분석
"""

import os
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Cognita Flask 서버 실행 메인 함수"""
    
    print("=" * 60)
    print("🚀 Cognita 관계형 위험도 분석 Flask 백엔드 서버 시작")
    print("=" * 60)
    
    # 환경 변수 설정 (최적화된 연결 정보)
    os.environ.setdefault("NEO4J_URI", "bolt://34.227.31.16:7687")
    os.environ.setdefault("NEO4J_USERNAME", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "cover-site-establishment")
    
    # 서버 설정
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5002))  # Cognita 전용 포트
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    print(f"📡 서버 주소: http://{host}:{port}")
    print(f"🔗 React 연동: http://localhost:3000에서 접근 가능")
    print(f"🕸️  Neo4j 연결: {os.getenv('NEO4J_URI')}")
    print(f"🔄 디버그 모드: {'활성화' if debug else '비활성화'}")
    print()
    print("주요 엔드포인트:")
    print(f"  • 헬스체크: http://{host}:{port}/api/health")
    print(f"  • 직원 목록: http://{host}:{port}/api/employees")
    print(f"  • 부서 목록: http://{host}:{port}/api/departments")
    print(f"  • 직원 분석: http://{host}:{port}/api/analyze/employee/<employee_id>")
    print(f"  • 부서 분석: http://{host}:{port}/api/analyze/department")
    print()
    print("React 연동 예시:")
    print("  // 헬스체크")
    print("  fetch('http://localhost:5000/api/health')")
    print("    .then(response => response.json())")
    print("    .then(data => console.log(data));")
    print()
    print("  // 직원 분석")
    print("  fetch('http://localhost:5000/api/analyze/employee/1')")
    print("    .then(response => response.json())")
    print("    .then(data => console.log(data));")
    print()
    print("  // 부서 분석")
    print("  fetch('http://localhost:5000/api/analyze/department', {")
    print("    method: 'POST',")
    print("    headers: { 'Content-Type': 'application/json' },")
    print("    body: JSON.stringify({")
    print("      department_name: 'Sales',")
    print("      sample_size: 20")
    print("    })")
    print("  })")
    print()
    print("서버를 중지하려면 Ctrl+C를 누르세요.")
    print("=" * 60)
    
    try:
        # Flask 앱 import 및 실행
        from cognita_flask_backend import run_server
        run_server(host=host, port=port, debug=debug)
        
    except KeyboardInterrupt:
        print("\n서버가 중지되었습니다.")
    except ImportError as e:
        print(f"\n모듈 import 실패: {e}")
        print("다음을 확인하세요:")
        print("1. pip install -r requirements.txt")
        print("2. cognita_flask_backend.py 파일 존재 여부")
        print("3. Neo4j 연결 정보 확인")
        sys.exit(1)
    except Exception as e:
        print(f"\n서버 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
