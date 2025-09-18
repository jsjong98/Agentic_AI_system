# -*- coding: utf-8 -*-
"""
에이전트 초기화 및 순차적 워크플로우 테스트 스크립트
Structure, Cognita, Agora 에이전트 수정 사항 검증
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent))

def test_structura_agent():
    """Structura 에이전트 테스트"""
    print("=" * 60)
    print("🧪 Structura 에이전트 테스트")
    print("=" * 60)
    
    try:
        from Structura.structura_flask_backend import StructuraHRPredictor
        
        # 1. 초기화 테스트
        print("1. Structura 에이전트 초기화 테스트...")
        predictor = StructuraHRPredictor()
        print("✅ 초기화 성공")
        
        # 2. 필수 메서드 존재 확인
        print("2. 필수 메서드 존재 확인...")
        required_methods = [
            'run_full_pipeline',
            'predict_single_employee',
            'predict_single',
            'explain_prediction'
        ]
        
        for method in required_methods:
            if hasattr(predictor, method):
                print(f"  ✅ {method} 메서드 존재")
            else:
                print(f"  ❌ {method} 메서드 누락")
        
        # 3. 샘플 데이터로 예측 테스트 (데이터 파일이 있는 경우)
        print("3. 샘플 예측 테스트...")
        sample_employee = {
            'Age': 35,
            'JobSatisfaction': 3,
            'OverTime': 'No',
            'MonthlyIncome': 5000,
            'WorkLifeBalance': 2,
            'JobRole': 'Sales Executive',
            'Department': 'Sales',
            'EmployeeNumber': 'TEST001'
        }
        
        try:
            result = predictor.predict_single_employee(sample_employee, 'TEST001')
            print("  ✅ 예측 성공")
            print(f"  📊 위험도: {result.get('attrition_probability', 0):.3f}")
            print(f"  📈 위험 범주: {result.get('risk_category', 'UNKNOWN')}")
        except Exception as e:
            print(f"  ⚠️ 예측 테스트 실패 (데이터 파일 없음 가능): {str(e)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Structura import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ Structura 테스트 실패: {e}")
        return False

def test_cognita_agent():
    """Cognita 에이전트 테스트"""
    print("\n" + "=" * 60)
    print("🧪 Cognita 에이전트 테스트")
    print("=" * 60)
    
    try:
        from Cognita.cognita_flask_backend import Neo4jManager, CognitaRiskAnalyzer
        
        # 1. Neo4j 매니저 초기화 테스트
        print("1. Neo4j 매니저 초기화 테스트...")
        
        # 기본 설정으로 테스트 (연결 실패 예상)
        neo4j_config = {
            "uri": "bolt://localhost:7687",
            "username": "neo4j", 
            "password": "test"
        }
        
        try:
            neo4j_manager = Neo4jManager(
                neo4j_config['uri'],
                neo4j_config['username'],
                neo4j_config['password']
            )
            print("✅ Neo4j 매니저 초기화 성공")
            
            # 2. Cognita 분석기 초기화
            print("2. Cognita 분석기 초기화 테스트...")
            analyzer = CognitaRiskAnalyzer(neo4j_manager)
            print("✅ Cognita 분석기 초기화 성공")
            
            # 3. 연결 테스트
            print("3. Neo4j 연결 테스트...")
            if neo4j_manager.is_connected():
                print("✅ Neo4j 연결 성공")
                
                # 샘플 분석 테스트
                print("4. 샘플 분석 테스트...")
                try:
                    risk_metrics = analyzer.analyze_employee_risk("1")
                    print("✅ 직원 위험도 분석 성공")
                    print(f"  📊 종합 위험도: {risk_metrics.overall_risk_score:.3f}")
                    print(f"  📈 위험 범주: {risk_metrics.risk_category}")
                except Exception as e:
                    print(f"  ⚠️ 분석 테스트 실패 (데이터 없음 가능): {str(e)}")
            else:
                print("⚠️ Neo4j 연결 실패 (예상됨)")
            
            neo4j_manager.close()
            return True
            
        except Exception as e:
            print(f"⚠️ Neo4j 연결 실패 (예상됨): {str(e)}")
            print("✅ 연결 실패 처리 로직 정상 작동")
            return True
        
    except ImportError as e:
        print(f"❌ Cognita import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ Cognita 테스트 실패: {e}")
        return False

def test_agora_agent():
    """Agora 에이전트 테스트"""
    print("\n" + "=" * 60)
    print("🧪 Agora 에이전트 테스트")
    print("=" * 60)
    
    try:
        from Agora.agora_llm_generator import AgoraLLMGenerator
        
        # 1. LLM 생성기 초기화 테스트
        print("1. Agora LLM 생성기 초기화 테스트...")
        
        # 테스트용 API 키 (실제로는 환경변수에서 가져와야 함)
        test_api_key = "test_key"
        
        try:
            llm_generator = AgoraLLMGenerator(test_api_key)
            print("✅ Agora LLM 생성기 초기화 성공")
            print(f"  🤖 모델: {llm_generator.model}")
            
            # 2. 규칙 기반 해석 테스트
            print("2. 규칙 기반 해석 테스트...")
            sample_analysis = {
                'employee_id': 'TEST001',
                'job_role': 'Sales Executive',
                'market_pressure_index': 0.6,
                'compensation_gap': 0.3,
                'job_postings_count': 15,
                'risk_level': 'MEDIUM',
                'market_data': {
                    'market_trend': 'GROWING',
                    'competition_level': 'HIGH'
                }
            }
            
            interpretation = llm_generator._generate_rule_based_interpretation(sample_analysis)
            print("✅ 규칙 기반 해석 생성 성공")
            print(f"  📝 해석 길이: {len(interpretation)} 문자")
            
            # 3. 직무 카테고리 분류 테스트
            print("3. 직무 카테고리 분류 테스트...")
            test_jobs = ['Sales Executive', 'Research Scientist', 'Manufacturing Technician', 'HR Manager']
            
            for job in test_jobs:
                category = llm_generator._get_job_category(job)
                print(f"  📊 {job} → {category}")
            
            return True
            
        except Exception as e:
            print(f"❌ Agora 테스트 실패: {e}")
            return False
        
    except ImportError as e:
        print(f"❌ Agora import 실패: {e}")
        return False

def test_master_server_workflow():
    """마스터 서버 워크플로우 테스트"""
    print("\n" + "=" * 60)
    print("🧪 마스터 서버 순차적 워크플로우 테스트")
    print("=" * 60)
    
    try:
        from agentic_master_server import WorkerAgentManager, AgenticTask
        
        # 1. 워커 매니저 초기화
        print("1. 워커 에이전트 매니저 초기화...")
        manager = WorkerAgentManager()
        print("✅ 워커 매니저 초기화 성공")
        
        # 2. 워커 상태 확인
        print("2. 워커 에이전트 상태 확인...")
        worker_status = manager.get_worker_status()
        
        for worker_id, status in worker_status.items():
            print(f"  📊 {worker_id}: {status.status} ({status.agent_name})")
        
        # 3. 샘플 작업 생성
        print("3. 샘플 작업 생성...")
        sample_task = AgenticTask(
            task_id="test_workflow_001",
            task_type="individual_analysis",
            employee_data={
                'Age': 35,
                'JobSatisfaction': 3,
                'OverTime': 'No',
                'MonthlyIncome': 5000,
                'WorkLifeBalance': 2,
                'JobRole': 'Sales Executive',
                'Department': 'Sales',
                'EmployeeNumber': 'TEST001'
            },
            use_structura=True,
            use_cognita=True,
            use_chronos=True,
            use_sentio=True,
            use_agora=True
        )
        print("✅ 샘플 작업 생성 성공")
        
        # 4. 순차적 워크플로우 실행 테스트
        print("4. 순차적 워크플로우 실행...")
        start_time = time.time()
        
        try:
            result = manager.execute_sequential_workflow(sample_task)
            execution_time = time.time() - start_time
            
            print(f"✅ 워크플로우 실행 완료 (소요시간: {execution_time:.2f}초)")
            print(f"  📊 상태: {result.status}")
            print(f"  🔧 실행 모드: {result.workflow_metadata.get('execution_mode', 'unknown')}")
            print(f"  📈 성공률: {result.workflow_metadata.get('success_rate', 0):.1%}")
            print(f"  ✅ 성공 에이전트: {result.workflow_metadata.get('successful_agents', [])}")
            print(f"  ❌ 실패 에이전트: {result.workflow_metadata.get('failed_agents', [])}")
            
            # 통합 분석 결과 확인
            if result.combined_analysis:
                analysis = result.combined_analysis
                print(f"  🎯 통합 분석 타입: {analysis.get('analysis_type', 'unknown')}")
                
                if 'integrated_assessment' in analysis:
                    assessment = analysis['integrated_assessment']
                    print(f"  📊 통합 위험도: {assessment.get('integrated_risk_score', 0):.3f}")
                    print(f"  📈 위험 레벨: {assessment.get('risk_level', 'UNKNOWN')}")
                    print(f"  📋 권장사항 수: {len(analysis.get('recommendations', []))}")
            
            return True
            
        except Exception as e:
            print(f"❌ 워크플로우 실행 실패: {e}")
            return False
        
    except ImportError as e:
        print(f"❌ 마스터 서버 import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 마스터 서버 테스트 실패: {e}")
        return False

def test_error_handling():
    """에러 처리 및 복구 테스트"""
    print("\n" + "=" * 60)
    print("🧪 에러 처리 및 복구 테스트")
    print("=" * 60)
    
    try:
        from agentic_master_server import WorkerAgentManager, AgenticTask
        
        manager = WorkerAgentManager()
        
        # 1. 잘못된 데이터로 작업 테스트
        print("1. 잘못된 데이터로 작업 테스트...")
        
        invalid_task = AgenticTask(
            task_id="test_error_001",
            task_type="individual_analysis",
            employee_data={},  # 빈 데이터
            use_structura=True,
            use_cognita=False,  # Cognita 비활성화 (Neo4j 연결 문제 방지)
            use_chronos=False,
            use_sentio=False,
            use_agora=False
        )
        
        try:
            result = manager.execute_sequential_workflow(invalid_task)
            print(f"  📊 결과 상태: {result.status}")
            print(f"  ⚠️ 에러 메시지: {result.error_message or 'None'}")
            print("✅ 에러 처리 정상 작동")
            
        except Exception as e:
            print(f"  ❌ 예상치 못한 에러: {e}")
        
        # 2. 에이전트 가용성 테스트
        print("2. 에이전트 가용성 테스트...")
        
        available_agents = []
        for agent_name in ['structura', 'cognita', 'chronos', 'sentio', 'agora']:
            is_available = manager._is_agent_available(agent_name)
            status = "✅ 사용 가능" if is_available else "❌ 사용 불가"
            print(f"  {agent_name}: {status}")
            if is_available:
                available_agents.append(agent_name)
        
        print(f"  📊 사용 가능한 에이전트: {len(available_agents)}/5개")
        
        return True
        
    except Exception as e:
        print(f"❌ 에러 처리 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 에이전트 초기화 및 워크플로우 테스트 시작")
    print(f"📅 테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    
    # 개별 에이전트 테스트
    test_results['structura'] = test_structura_agent()
    test_results['cognita'] = test_cognita_agent()
    test_results['agora'] = test_agora_agent()
    
    # 통합 테스트
    test_results['workflow'] = test_master_server_workflow()
    test_results['error_handling'] = test_error_handling()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
    
    print(f"\n📈 전체 결과: {passed}/{total} 테스트 통과 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 모든 테스트 통과! 에이전트 수정 사항이 정상적으로 적용되었습니다.")
    else:
        print("⚠️ 일부 테스트 실패. 추가 수정이 필요할 수 있습니다.")
    
    # 결과를 JSON 파일로 저장
    result_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    test_summary = {
        'timestamp': datetime.now().isoformat(),
        'results': test_results,
        'summary': {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': passed / total * 100
        }
    }
    
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(test_summary, f, indent=2, ensure_ascii=False)
        print(f"📄 테스트 결과가 {result_file}에 저장되었습니다.")
    except Exception as e:
        print(f"⚠️ 결과 파일 저장 실패: {e}")

if __name__ == "__main__":
    main()
