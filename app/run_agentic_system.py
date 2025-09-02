#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agentic AI System 통합 실행 스크립트
모든 워커 에이전트를 동시에 실행하는 마스터 서버
"""

import os
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Agentic AI System 실행 메인 함수"""
    
    print("=" * 70)
    print("🤖 Agentic AI System 시작")
    print("=" * 70)
    
    # 환경 변수 설정
    os.environ.setdefault("NEO4J_URI", "bolt://YOUR_NEO4J_HOST:7687")
    os.environ.setdefault("NEO4J_USERNAME", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "YOUR_NEO4J_PASSWORD")
    
    # 서버 설정
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    print(f"📡 마스터 서버: http://{host}:{port}")
    print(f"🔗 React 연동: http://localhost:3000에서 접근 가능")
    print(f"🔄 디버그 모드: {'활성화' if debug else '비활성화'}")
    print()
    
    print("🏗️ 에이전틱 AI 아키텍처:")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    Supervisor 에이전트                      │")
    print("│                      (향후 구현)                           │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  워커1     │  워커2     │  워커3     │  워커4     │  워커5   │")
    print("│  정형      │  관계형    │  시계열    │  자연어    │  외부    │")
    print("│  데이터    │  데이터    │  데이터    │  데이터    │  시장    │")
    print("│  분석      │  분석      │  분석      │  분석      │  분석    │")
    print("│  ✅       │  ✅       │  ⏳       │  ⏳       │  ⏳     │")
    print("└─────────────────────────────────────────────────────────────┘")
    print("                           │")
    print("                    최종 종합 에이전트")
    print("                      (향후 구현)")
    print()
    
    print("현재 구현된 워커 에이전트:")
    print("  🤖 워커 에이전트 1: Structura (정형 데이터 분석)")
    print("     - XGBoost 기반 이직 예측")
    print("     - SHAP, LIME을 통한 설명 가능한 AI")
    print("     - 개별 직원 위험 요인 분석")
    print()
    print("  🕸️  워커 에이전트 2: Cognita (관계형 데이터 분석)")
    print("     - Neo4j 기반 사회적 네트워크 분석")
    print("     - 관계형 위험도 평가")
    print("     - 팀 역학 및 조직 구조 분석")
    print()
    
    print("향후 확장 예정:")
    print("  ⏳ 워커 에이전트 3: 시계열 데이터 분석")
    print("  ⏳ 워커 에이전트 4: 자연어 데이터 분석")
    print("  ⏳ 워커 에이전트 5: 외부 시장 분석")
    print("  ⏳ Supervisor 에이전트: 전체 워커 조정")
    print("  ⏳ 최종 종합 에이전트: 결과 통합")
    print()
    
    print("주요 API 엔드포인트:")
    print(f"  • 시스템 상태: http://{host}:{port}/api/health")
    print(f"  • 워커 상태: http://{host}:{port}/api/workers/status")
    print(f"  • 개별 분석: http://{host}:{port}/api/analyze/individual")
    print(f"  • 부서 분석: http://{host}:{port}/api/analyze/department")
    print()
    
    print("React 연동 예시:")
    print("  // 개별 직원 통합 분석")
    print("  fetch('http://localhost:8000/api/analyze/individual', {")
    print("    method: 'POST',")
    print("    headers: { 'Content-Type': 'application/json' },")
    print("    body: JSON.stringify({")
    print("      employee_id: '1',")
    print("      Age: 35,")
    print("      JobSatisfaction: 3,")
    print("      // ... 기타 데이터")
    print("      use_structura: true,")
    print("      use_cognita: true")
    print("    })")
    print("  })")
    print()
    
    print("  // 부서별 통합 분석")
    print("  fetch('http://localhost:8000/api/analyze/department', {")
    print("    method: 'POST',")
    print("    headers: { 'Content-Type': 'application/json' },")
    print("    body: JSON.stringify({")
    print("      department_name: 'Sales',")
    print("      sample_size: 20,")
    print("      use_structura: true,")
    print("      use_cognita: true")
    print("    })")
    print("  })")
    print()
    
    print("서버를 중지하려면 Ctrl+C를 누르세요.")
    print("=" * 70)
    
    try:
        # Agentic Master Server 실행
        from agentic_master_server import run_server
        run_server(host=host, port=port, debug=debug)
        
    except KeyboardInterrupt:
        print("\n🤖 Agentic AI System이 중지되었습니다.")
    except ImportError as e:
        print(f"\n모듈 import 실패: {e}")
        print("다음을 확인하세요:")
        print("1. 각 워커 에이전트 폴더의 requirements.txt 설치")
        print("   cd Structura && pip install -r requirements.txt")
        print("   cd Cognita && pip install -r requirements.txt")
        print("2. Neo4j 서버 실행 상태")
        print("3. 데이터 파일 경로 확인")
        sys.exit(1)
    except Exception as e:
        print(f"\n서버 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
