#!/usr/bin/env python3
"""
Supervisor 워커 API 통합 기능 테스트
"""

import requests
import json
from pathlib import Path

# 서버 설정
BASE_URL = "http://localhost:5006"

def test_health_check():
    """서버 상태 확인"""
    print("🏥 Health Check 테스트...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ 서버 상태: {data.get('status')}")
            print(f"  🤖 워크플로우 초기화: {data.get('workflow_initialized')}")
            return True
        else:
            print(f"  ❌ Health check 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Health check 오류: {e}")
        return False

def test_all_workers_health():
    """모든 워커 상태 확인"""
    print("\n🔍 모든 워커 상태 확인 테스트...")
    try:
        response = requests.get(f"{BASE_URL}/api/workers/health_check_all")
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ 워커 상태 확인 성공")
            print(f"    총 워커 수: {data['summary']['total_workers']}")
            print(f"    정상 워커 수: {data['summary']['healthy_workers']}")
            print(f"    정상률: {data['summary']['health_rate']:.1%}")
            
            for worker, status in data['worker_status'].items():
                status_icon = "✅" if status['healthy'] else "❌"
                print(f"    {status_icon} {worker}: {status['status']}")
            
            return data['summary']['health_rate'] > 0
        else:
            print(f"  ❌ 워커 상태 확인 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ 워커 상태 확인 오류: {e}")
        return False

def test_structura_predict():
    """Structura 예측 테스트"""
    print("\n📊 Structura 예측 테스트...")
    try:
        # 테스트용 직원 데이터
        test_data = {
            "EmployeeNumber": "TEST001",
            "Age": 35,
            "JobSatisfaction": 3,
            "OverTime": "No",
            "MonthlyIncome": 5000,
            "WorkLifeBalance": 3,
            "Department": "Research & Development",
            "JobRole": "Research Scientist"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/workers/structura/predict",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Structura 예측 성공")
            print(f"    소스: {data.get('source')}")
            if 'data' in data:
                pred_data = data['data']
                if 'attrition_probability' in pred_data:
                    print(f"    이직 확률: {pred_data['attrition_probability']:.3f}")
                    print(f"    위험 카테고리: {pred_data.get('risk_category', 'N/A')}")
            return True
        else:
            print(f"  ❌ Structura 예측 실패: {response.status_code}")
            print(f"    응답: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ Structura 예측 오류: {e}")
        return False

def test_sentio_analyze():
    """Sentio 감정 분석 테스트"""
    print("\n💭 Sentio 감정 분석 테스트...")
    try:
        test_data = {
            "employee_id": "TEST001",
            "text_data": {
                "self_review": "I feel overwhelmed with my current workload and lack support from management.",
                "peer_feedback": "Good team player but seems stressed lately.",
                "weekly_survey": "Work-life balance is challenging."
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/workers/sentio/analyze_sentiment",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Sentio 감정 분석 성공")
            print(f"    소스: {data.get('source')}")
            if 'data' in data:
                sent_data = data['data']
                print(f"    감정 점수: {sent_data.get('sentiment_score', 0):.3f}")
                print(f"    감정 상태: {sent_data.get('emotional_state', 'N/A')}")
                print(f"    신뢰도: {sent_data.get('confidence_score', 0):.3f}")
            return True
        else:
            print(f"  ❌ Sentio 감정 분석 실패: {response.status_code}")
            print(f"    응답: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ Sentio 감정 분석 오류: {e}")
        return False

def test_agora_analysis():
    """Agora 시장 분석 테스트"""
    print("\n🏢 Agora 시장 분석 테스트...")
    try:
        test_data = {
            "JobRole": "Research Scientist",
            "Department": "Research & Development",
            "MonthlyIncome": 5000,
            "EmployeeNumber": "TEST001"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/workers/agora/comprehensive_analysis",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Agora 시장 분석 성공")
            print(f"    소스: {data.get('source')}")
            if 'data' in data and 'data' in data['data']:
                agora_data = data['data']['data']
                print(f"    시장 압력 지수: {agora_data.get('market_pressure_index', 0):.3f}")
                print(f"    보상 격차: {agora_data.get('compensation_gap', 0):.3f}")
                print(f"    위험 수준: {agora_data.get('risk_level', 'N/A')}")
            return True
        else:
            print(f"  ❌ Agora 시장 분석 실패: {response.status_code}")
            print(f"    응답: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ Agora 시장 분석 오류: {e}")
        return False

def test_cognita_analyze():
    """Cognita 관계 분석 테스트"""
    print("\n🕸️ Cognita 관계 분석 테스트...")
    try:
        employee_id = "1"  # 테스트용 직원 ID
        
        response = requests.get(
            f"{BASE_URL}/api/workers/cognita/analyze/{employee_id}",
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Cognita 관계 분석 성공")
            print(f"    소스: {data.get('source')}")
            print(f"    직원 ID: {data.get('employee_id')}")
            if 'data' in data:
                cog_data = data['data']
                if 'overall_risk_score' in cog_data:
                    print(f"    전체 위험 점수: {cog_data['overall_risk_score']:.3f}")
                    print(f"    위험 카테고리: {cog_data.get('risk_category', 'N/A')}")
            return True
        else:
            print(f"  ❌ Cognita 관계 분석 실패: {response.status_code}")
            print(f"    응답: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ Cognita 관계 분석 오류: {e}")
        return False

def test_chronos_predict():
    """Chronos 시계열 예측 테스트"""
    print("\n⏰ Chronos 시계열 예측 테스트...")
    try:
        test_data = {
            "employee_ids": [1, 2, 3]  # 테스트용 직원 ID 목록
        }
        
        response = requests.post(
            f"{BASE_URL}/api/workers/chronos/predict",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Chronos 시계열 예측 성공")
            print(f"    소스: {data.get('source')}")
            if 'data' in data and 'predictions' in data['data']:
                predictions = data['data']['predictions']
                print(f"    예측 결과 수: {len(predictions)}")
                if predictions:
                    first_pred = predictions[0]
                    print(f"    첫 번째 예측 - 직원 ID: {first_pred.get('employee_id')}")
                    print(f"    이직 확률: {first_pred.get('attrition_probability', 0):.3f}")
            return True
        else:
            print(f"  ❌ Chronos 시계열 예측 실패: {response.status_code}")
            print(f"    응답: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ Chronos 시계열 예측 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🧪 Supervisor 워커 API 통합 기능 테스트 시작")
    print("=" * 60)
    
    # 1. Health Check
    if not test_health_check():
        print("\n❌ 서버가 실행되지 않았습니다. 먼저 Supervisor 서버를 시작하세요:")
        print("  cd app/Supervisor")
        print("  python run_supervisor_server.py")
        return
    
    # 2. 모든 워커 상태 확인
    workers_healthy = test_all_workers_health()
    
    # 3. 개별 워커 API 테스트 (워커가 정상일 때만)
    test_results = []
    
    if workers_healthy:
        test_results.append(("Structura 예측", test_structura_predict()))
        test_results.append(("Sentio 감정 분석", test_sentio_analyze()))
        test_results.append(("Agora 시장 분석", test_agora_analysis()))
        test_results.append(("Cognita 관계 분석", test_cognita_analyze()))
        test_results.append(("Chronos 시계열 예측", test_chronos_predict()))
    else:
        print("\n⚠️ 일부 워커가 오프라인 상태입니다. 개별 테스트를 건너뜁니다.")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 테스트 결과 요약:")
    
    if test_results:
        passed = sum(1 for _, result in test_results if result)
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} {test_name}")
        
        print(f"\n🎯 전체 결과: {passed}/{total} 테스트 통과 ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 모든 워커 API 통합 테스트가 성공했습니다!")
        else:
            print("⚠️ 일부 테스트가 실패했습니다. 워커 서버 상태를 확인해주세요.")
    else:
        print("  ⚠️ 워커 서버들이 오프라인 상태여서 개별 테스트를 수행할 수 없었습니다.")
    
    # Integration API 테스트 추가
    test_integration_apis()


def test_integration_apis():
    """Integration API 테스트"""
    print("\n🔧 Integration API 테스트...")
    
    # 1. Integration 상태 확인
    try:
        response = requests.get(f"{BASE_URL}/api/workers/integration/get_results")
        if response.status_code == 200:
            print("  ✅ Integration 결과 조회 성공")
        else:
            print(f"  ⚠️  Integration 결과 조회: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Integration 결과 조회 오류: {e}")
    
    # 2. Integration 직원 목록 조회
    try:
        response = requests.get(f"{BASE_URL}/api/workers/integration/get_employee_list")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("  ✅ Integration 직원 목록 조회 성공")
            else:
                print(f"  ⚠️  Integration 직원 목록: {data.get('error')}")
        else:
            print(f"  ⚠️  Integration 직원 목록 조회: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Integration 직원 목록 조회 오류: {e}")
    
    # 3. Integration 데이터 로드 테스트
    try:
        test_data = {
            "file_path": "Total_score.csv"
        }
        response = requests.post(f"{BASE_URL}/api/workers/integration/load_data", json=test_data)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("  ✅ Integration 데이터 로드 성공")
            else:
                print(f"  ⚠️  Integration 데이터 로드: {data.get('error')}")
        else:
            print(f"  ⚠️  Integration 데이터 로드: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Integration 데이터 로드 오류: {e}")
    
    # 4. Integration 개별 직원 예측 테스트
    try:
        test_prediction = {
            "scores": {
                "structura_score": 0.75,
                "cognita_score": 0.65,
                "chronos_score": 0.80,
                "sentio_score": 0.70,
                "agora_score": 0.60
            }
        }
        response = requests.post(f"{BASE_URL}/api/workers/integration/predict_employee", json=test_prediction)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("  ✅ Integration 개별 예측 성공")
            else:
                print(f"  ⚠️  Integration 개별 예측: {data.get('error')}")
        else:
            print(f"  ⚠️  Integration 개별 예측: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Integration 개별 예측 오류: {e}")
    
    # 5. Integration 레포트 생성 테스트
    try:
        test_report = {
            "employee_id": "TEST001",
            "agent_scores": {
                "structura_score": 0.75,
                "cognita_score": 0.65,
                "chronos_score": 0.80,
                "sentio_score": 0.70,
                "agora_score": 0.60
            },
            "format": "json",
            "use_llm": False  # LLM 없이 테스트
        }
        response = requests.post(f"{BASE_URL}/api/workers/integration/generate_report", json=test_report)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("  ✅ Integration 레포트 생성 성공")
            else:
                print(f"  ⚠️  Integration 레포트 생성: {data.get('error')}")
        else:
            print(f"  ⚠️  Integration 레포트 생성: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Integration 레포트 생성 오류: {e}")


if __name__ == "__main__":
    main()
