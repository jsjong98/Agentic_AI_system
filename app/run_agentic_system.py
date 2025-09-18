#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agentic AI System 통합 실행 스크립트
모든 워커 에이전트를 동시에 실행하는 마스터 서버
"""

import os
import sys
import subprocess
import threading
import time
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 단순화된 아키텍처: 마스터 서버에서 모든 기능을 통합 제공

def main():
    """Agentic AI System 실행 메인 함수"""
    
    print("=" * 70)
    print("🤖 Agentic AI System 시작")
    print("=" * 70)
    
    # 환경 변수 설정
    os.environ.setdefault("NEO4J_URI", "bolt://44.212.67.74:7687")
    os.environ.setdefault("NEO4J_USERNAME", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "legs-augmentations-cradle")
    
    # OpenAI API 키 확인 및 안내
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   Supervisor, Integration, Sentio, Agora의 LLM 기능이 제한됩니다.")
        print("   다음 명령어로 설정하세요:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print()
    
    # 서버 설정
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    print(f"📡 마스터 서버: http://{host}:{port}")
    print(f"🔗 React 연동: http://localhost:3000에서 접근 가능")
    print(f"🔄 디버그 모드: {'활성화' if debug else '비활성화'}")
    print()
    
    print("🏗️ 통합 AI 아키텍처:")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                     마스터 서버 (포트 8000)                 │")
    print("│                        ✅ 통합 구현                        │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  워커1     │  워커2     │  워커3     │  워커4     │  워커5   │")
    print("│  정형      │  관계형    │  시계열    │  텍스트    │  외부    │")
    print("│  데이터    │  데이터    │  데이터    │  감정분석  │  시장    │")
    print("│  분석      │  분석      │  분석      │  분석      │  분석    │")
    print("│  ✅       │  ✅       │  ✅       │  ✅       │  ✅     │")
    print("└─────────────────────────────────────────────────────────────┘")
    print()
    
    print("현재 구현된 워커 에이전트:")
    print("  🏢 워커 에이전트 1: Structura (정형 데이터 분석)")
    print("     - XGBoost 기반 이직 예측")
    print("     - SHAP, LIME을 통한 설명 가능한 AI")
    print("     - 개별 직원 위험 요인 분석")
    print()
    print("  🕸️ 워커 에이전트 2: Cognita (관계형 데이터 분석)")
    print("     - Neo4j 기반 사회적 네트워크 분석")
    print("     - 관계형 위험도 평가")
    print("     - 팀 역학 및 조직 구조 분석")
    print()
    print("  ⏰ 워커 에이전트 3: Chronos (시계열 데이터 분석)")
    print("     - GRU+CNN+Attention 하이브리드 딥러닝")
    print("     - 시간적 패턴 분석 및 예측")
    print("     - Attention 메커니즘 기반 해석 가능성")
    print()
    print("  📝 워커 에이전트 4: Sentio (텍스트 감정 분석)")
    print("     - NLP 기반 퇴직 위험 신호 탐지")
    print("     - 키워드 분석 및 감정 점수 계산")
    print("     - GPT-5 기반 페르소나별 텍스트 생성")
    print()
    print("  🌍 워커 에이전트 5: Agora (외부 시장 분석)")
    print("     - 시장 압력 지수 및 보상 격차 분석")
    print("     - 직무별 채용 공고 및 급여 수준 분석")
    print("     - LLM 기반 시장 상황 해석")
    print()
    
    print("통합 시스템:")
    print("  🎯 마스터 서버: 모든 에이전트 기능을 통합 제공")
    print("     - 5개 워커 에이전트 내장")
    print("     - 결과 통합 및 최적화")
    print("     - 단일 포트(8000)에서 모든 API 제공")
    print()
    
    print("주요 API 엔드포인트:")
    print(f"  📡 마스터 서버 (포트 {port}):")
    print(f"    • 시스템 상태: http://{host}:{port}/api/health")
    print(f"    • 워커 상태: http://{host}:{port}/api/workers/status")
    print(f"    • 개별 분석: http://{host}:{port}/api/analyze/individual")
    print(f"    • 부서 분석: http://{host}:{port}/api/analyze/department")
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
    print("      use_cognita: true,")
    print("      use_sentio: true,")
    print("      use_chronos: true,")
    print("      use_agora: true")
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
        print("🎯 마스터 서버 시작 중...")
        # 마스터 서버만 실행 (모든 기능을 통합)
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
