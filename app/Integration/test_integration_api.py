"""
Integration API 테스트 스크립트
"""

import requests
import json
import time
import os
from typing import Dict, Any
from dotenv import load_dotenv

# 서버 설정
BASE_URL = "http://localhost:5007"
HEADERS = {"Content-Type": "application/json"}

# 환경변수 로드
load_dotenv()

def test_health_check():
    """서버 상태 확인 테스트"""
    print("=== 서버 상태 확인 테스트 ===")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 서버 상태: {result['status']}")
            print(f"   서비스: {result['service']}")
            print(f"   버전: {result['version']}")
            print(f"   시간: {result['timestamp']}")
            print(f"   LLM 활성화: {result.get('llm_enabled', False)}")
            return True
        else:
            print(f"❌ 서버 상태 확인 실패: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def test_env_check():
    """환경변수 확인 테스트"""
    print("\n=== 환경변수 확인 테스트 ===")
    
    env_api_key = os.getenv("OPENAI_API_KEY")
    if env_api_key:
        print(f"✅ .env 파일에서 API 키 발견: {env_api_key[:10]}...")
        return True
    else:
        print("⚠️  .env 파일에 OPENAI_API_KEY가 설정되지 않음")
        return False


def test_set_api_key():
    """API 키 설정 테스트"""
    print("\n=== API 키 설정 테스트 ===")
    
    try:
        # 환경변수에서 실제 키가 있으면 사용, 없으면 더미 키 사용
        env_api_key = os.getenv("OPENAI_API_KEY")
        test_api_key = env_api_key if env_api_key else "sk-test-gpt5nano-dummy-key-for-testing-purposes-only"
        
        payload = {
            "api_key": test_api_key
        }
        
        response = requests.post(f"{BASE_URL}/set_api_key", 
                               json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"✅ API 키 설정 성공")
                print(f"   메시지: {result['message']}")
                print(f"   LLM 활성화: {result['llm_enabled']}")
                return True
            else:
                print(f"❌ API 키 설정 실패: {result['error']}")
                return False
        else:
            print(f"❌ HTTP 오류: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def test_load_data():
    """데이터 로드 테스트"""
    print("\n=== 데이터 로드 테스트 ===")
    
    try:
        # 데이터 파일 경로 (상대 경로)
        payload = {
            "file_path": "Total_score.csv"
        }
        
        response = requests.post(f"{BASE_URL}/load_data", 
                               headers=HEADERS, 
                               data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 데이터 로드 성공")
            print(f"   파일 경로: {result['file_path']}")
            print(f"   전체 행 수: {result['statistics']['total_rows']:,}")
            print(f"   전체 컬럼 수: {result['statistics']['total_columns']}")
            print(f"   Score 컬럼: {result['statistics']['score_columns']}")
            print(f"   Attrition 분포: {result['statistics']['attrition_distribution']}")
            return True
        else:
            result = response.json()
            print(f"❌ 데이터 로드 실패: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def test_calculate_thresholds():
    """임계값 계산 테스트"""
    print("\n=== 임계값 계산 테스트 ===")
    
    try:
        # 빈 payload (자동으로 Score 컬럼 감지)
        payload = {}
        
        response = requests.post(f"{BASE_URL}/calculate_thresholds", 
                               headers=HEADERS, 
                               data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 임계값 계산 성공")
            print(f"   총 예측 레코드: {result['results']['total_predictions']:,}")
            
            # 요약 결과 출력
            print("\n📊 임계값 요약:")
            for item in result['results']['summary']:
                print(f"   {item['Score']}: 임계값={item['Optimal_Threshold']:.4f}, F1={item['F1_Score']:.4f}")
            
            # 최고 성능 Score
            best = result['results']['best_score']
            print(f"\n🏆 최고 성능: {best['Score']} (F1-Score: {best['F1_Score']:.4f})")
            
            # 저장된 파일
            print(f"\n📁 저장된 파일:")
            for file_type, file_path in result['files'].items():
                print(f"   {file_type}: {file_path}")
            
            return True
        else:
            result = response.json()
            print(f"❌ 임계값 계산 실패: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def test_optimize_weights(method='bayesian'):
    """가중치 최적화 테스트"""
    print(f"\n=== 가중치 최적화 테스트 ({method}) ===")
    
    try:
        payload = {
            "method": method
        }
        
        # 방법별 추가 파라미터
        if method == 'grid':
            payload['n_points_per_dim'] = 3  # 빠른 테스트를 위해
        elif method == 'bayesian':
            payload['n_calls'] = 50  # 빠른 테스트를 위해
        
        print(f"⏳ {method} 최적화 실행 중... (시간이 걸릴 수 있습니다)")
        
        response = requests.post(f"{BASE_URL}/optimize_weights", 
                               headers=HEADERS, 
                               data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 가중치 최적화 성공")
            print(f"   방법: {result['results']['method']}")
            print(f"   최적 임계값: {result['results']['optimal_threshold']:.4f}")
            print(f"   최고 F1-Score: {result['results']['best_f1_score']:.4f}")
            print(f"   총 레코드: {result['results']['total_records']:,}")
            
            # 최적 가중치 출력
            print(f"\n🎯 최적 가중치:")
            for var, weight in result['results']['optimal_weights'].items():
                print(f"   {var}: {weight:.4f}")
            
            # 성능 지표
            metrics = result['results']['performance_metrics']
            print(f"\n📈 성능 지표:")
            print(f"   정밀도: {metrics['precision']:.4f}")
            print(f"   재현율: {metrics['recall']:.4f}")
            print(f"   정확도: {metrics['accuracy']:.4f}")
            print(f"   AUC: {metrics['auc']:.4f}")
            
            # 위험도 통계
            risk_stats = result['results']['risk_statistics']
            if risk_stats.get('counts'):
                print(f"\n🚦 위험도 구간별 분포:")
                for level, count in risk_stats['counts'].items():
                    attrition_rate = risk_stats['attrition_rates'].get(level, 0)
                    print(f"   {level}: {count:,}명 (이탈률: {attrition_rate:.1%})")
            
            # 저장된 파일
            print(f"\n📁 저장된 파일:")
            for file_type, file_path in result['files'].items():
                print(f"   {file_type}: {file_path}")
            
            return True
        else:
            result = response.json()
            print(f"❌ 가중치 최적화 실패: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def test_predict_employee():
    """개별 직원 예측 테스트"""
    print("\n=== 개별 직원 예측 테스트 ===")
    
    try:
        # 테스트용 직원 데이터
        test_employees = [
            {
                "name": "직원A (고위험 예상)",
                "scores": {
                    "Structura_score": 0.95,
                    "Cognita_score": 0.6,
                    "Chronos_score": 0.8,
                    "Sentio_score": 0.7,
                    "Agora_score": 0.5
                }
            },
            {
                "name": "직원B (안전 예상)",
                "scores": {
                    "Structura_score": 0.1,
                    "Cognita_score": 0.3,
                    "Chronos_score": 0.0001,
                    "Sentio_score": 0.2,
                    "Agora_score": 0.15
                }
            },
            {
                "name": "직원C (중간 위험)",
                "scores": {
                    "Structura_score": 0.5,
                    "Cognita_score": 0.5,
                    "Chronos_score": 0.1,
                    "Sentio_score": 0.4,
                    "Agora_score": 0.3
                }
            }
        ]
        
        for employee in test_employees:
            print(f"\n👤 {employee['name']} 예측:")
            
            payload = {
                "scores": employee["scores"]
            }
            
            response = requests.post(f"{BASE_URL}/predict_employee", 
                                   headers=HEADERS, 
                                   data=json.dumps(payload))
            
            if response.status_code == 200:
                result = response.json()
                predictions = result['predictions']
                
                # 가중치 기반 예측 결과
                if 'weighted_prediction' in predictions:
                    wp = predictions['weighted_prediction']
                    print(f"   가중 점수: {wp['weighted_score']:.4f}")
                    print(f"   최종 예측: {wp['prediction_label']}")
                    print(f"   위험도: {wp['risk_level']}")
                
                # 임계값 기반 예측 결과 (요약)
                if 'threshold_predictions' in predictions:
                    tp = predictions['threshold_predictions']
                    risk_count = sum(1 for k, v in tp.items() if k.endswith('_prediction') and v == '위험')
                    total_scores = sum(1 for k in tp.keys() if k.endswith('_prediction'))
                    print(f"   개별 위험 점수: {risk_count}/{total_scores}")
                
                print("   ✅ 예측 완료")
            else:
                result = response.json()
                print(f"   ❌ 예측 실패: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def test_compare_methods():
    """최적화 방법 비교 테스트"""
    print("\n=== 최적화 방법 비교 테스트 ===")
    
    try:
        payload = {
            "methods": ["grid", "scipy"]  # 빠른 방법들만 테스트
        }
        
        print("⏳ 여러 최적화 방법 비교 중...")
        
        response = requests.post(f"{BASE_URL}/compare_methods", 
                               headers=HEADERS, 
                               data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 방법 비교 완료")
            print(f"   테스트된 방법 수: {result['total_methods_tested']}")
            print(f"   성공한 방법 수: {result['successful_methods']}")
            
            # 각 방법별 결과
            print(f"\n📊 방법별 결과:")
            for method_result in result['comparison_results']:
                if method_result['success']:
                    print(f"   {method_result['method']}: F1-Score {method_result['best_f1_score']:.4f}")
                else:
                    print(f"   {method_result['method']}: 실패 - {method_result['error']}")
            
            # 최고 성능 방법
            if result['best_method']:
                best = result['best_method']
                print(f"\n🏆 최고 성능 방법: {best['method']} (F1-Score: {best['best_f1_score']:.4f})")
            
            return True
        else:
            result = response.json()
            print(f"❌ 방법 비교 실패: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def test_get_results():
    """결과 조회 테스트"""
    print("\n=== 결과 조회 테스트 ===")
    
    try:
        response = requests.get(f"{BASE_URL}/get_results")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 결과 조회 성공")
            
            results = result['results']
            print(f"   임계값 결과 존재: {results['has_threshold_results']}")
            print(f"   가중치 최적화 결과 존재: {results['has_weight_optimization']}")
            print(f"   최종 데이터 존재: {results['has_final_data']}")
            
            return True
        else:
            result = response.json()
            print(f"❌ 결과 조회 실패: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def test_export_results():
    """결과 내보내기 테스트"""
    print("\n=== 결과 내보내기 테스트 ===")
    
    try:
        # CSV 형태로 내보내기
        payload = {
            "format": "csv",
            "include_data": True
        }
        
        response = requests.post(f"{BASE_URL}/export_results", 
                               headers=HEADERS, 
                               data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 결과 내보내기 성공")
            print(f"   형식: {result['format']}")
            print(f"   타임스탬프: {result['timestamp']}")
            
            print(f"\n📁 내보낸 파일:")
            for file_path in result['exported_files']:
                print(f"   {file_path}")
            
            return True
        else:
            result = response.json()
            print(f"❌ 결과 내보내기 실패: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def test_load_employee_data():
    """직원 데이터 로드 테스트"""
    print("\n=== 직원 데이터 로드 테스트 ===")
    
    try:
        payload = {
            "file_path": "IBM_HR.csv"
        }
        
        response = requests.post(f"{BASE_URL}/load_employee_data", 
                               json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"✅ 직원 데이터 로드 성공")
                print(f"   파일: {result['file_path']}")
                print(f"   직원 수: {result['total_employees']}")
                return True
            else:
                print(f"❌ 직원 데이터 로드 실패: {result['error']}")
                return False
        else:
            print(f"❌ HTTP 오류: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def test_get_employee_list():
    """직원 목록 조회 테스트"""
    print("\n=== 직원 목록 조회 테스트 ===")
    
    try:
        response = requests.get(f"{BASE_URL}/get_employee_list")
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"✅ 직원 목록 조회 성공")
                print(f"   전체 직원 수: {result['total_employees']}")
                print(f"   반환된 ID 수: {len(result['employee_ids'])}")
                print(f"   더 많은 데이터: {result['has_more']}")
                if result['employee_ids']:
                    print(f"   첫 번째 직원 ID: {result['employee_ids'][0]}")
                return True
            else:
                print(f"❌ 직원 목록 조회 실패: {result['error']}")
                return False
        else:
            print(f"❌ HTTP 오류: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def test_generate_report():
    """개별 직원 레포트 생성 테스트"""
    print("\n=== 개별 직원 레포트 생성 테스트 ===")
    
    try:
        # 샘플 에이전트 점수
        sample_scores = {
            "agora_score": 0.75,
            "chronos_score": 0.45,
            "cognita_score": 0.82,
            "sentio_score": 0.65,
            "structura_score": 0.38
        }
        
        payload = {
            "employee_id": "TEST_EMP_001",
            "agent_scores": sample_scores,
            "format": "text",
            "save_file": True,
            "use_llm": True
        }
        
        response = requests.post(f"{BASE_URL}/generate_report", 
                               json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"✅ 레포트 생성 성공")
                print(f"   직원 ID: {result['employee_id']}")
                print(f"   형식: {result['format']}")
                print(f"   LLM 사용: {result.get('llm_used', False)}")
                
                if 'saved_files' in result:
                    print(f"   저장된 파일: {result['saved_files']}")
                
                # 레포트 내용 일부 출력
                if result['format'] == 'text' and 'report' in result:
                    report_lines = result['report'].split('\n')
                    print(f"\n📄 레포트 미리보기 (처음 15줄):")
                    for i, line in enumerate(report_lines[:15]):
                        print(f"   {line}")
                    if len(report_lines) > 15:
                        print(f"   ... (총 {len(report_lines)}줄)")
                
                return True
            else:
                print(f"❌ 레포트 생성 실패: {result['error']}")
                return False
        else:
            print(f"❌ HTTP 오류: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def test_generate_batch_reports():
    """일괄 레포트 생성 테스트"""
    print("\n=== 일괄 레포트 생성 테스트 ===")
    
    try:
        # 여러 직원의 샘플 데이터
        employees_data = [
            {
                "employee_id": "BATCH_EMP_001",
                "agent_scores": {
                    "agora_score": 0.85,
                    "chronos_score": 0.65,
                    "cognita_score": 0.72,
                    "sentio_score": 0.55,
                    "structura_score": 0.48
                }
            },
            {
                "employee_id": "BATCH_EMP_002", 
                "agent_scores": {
                    "agora_score": 0.35,
                    "chronos_score": 0.25,
                    "cognita_score": 0.42,
                    "sentio_score": 0.75,
                    "structura_score": 0.68
                }
            },
            {
                "employee_id": "BATCH_EMP_003",
                "agent_scores": {
                    "agora_score": 0.95,
                    "chronos_score": 0.85,
                    "cognita_score": 0.92,
                    "sentio_score": 0.88,
                    "structura_score": 0.78
                }
            }
        ]
        
        payload = {
            "employees": employees_data
        }
        
        response = requests.post(f"{BASE_URL}/generate_batch_reports", 
                               json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"✅ 일괄 레포트 생성 성공")
                print(f"   메시지: {result['message']}")
                
                batch_results = result['results']
                print(f"   전체 요청: {batch_results['total']}")
                print(f"   성공: {len(batch_results['success'])}")
                print(f"   실패: {len(batch_results['failed'])}")
                print(f"   출력 디렉토리: {batch_results['output_directory']}")
                
                if batch_results['success']:
                    print(f"\n📁 성공한 레포트:")
                    for success_item in batch_results['success']:
                        print(f"     {success_item['employee_id']}: {success_item['files']}")
                
                if batch_results['failed']:
                    print(f"\n❌ 실패한 레포트:")
                    for failed_item in batch_results['failed']:
                        print(f"     {failed_item['employee_id']}: {failed_item['error']}")
                
                return True
            else:
                print(f"❌ 일괄 레포트 생성 실패: {result['error']}")
                return False
        else:
            print(f"❌ HTTP 오류: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def run_full_test():
    """전체 테스트 실행"""
    print("🧪 Integration API 전체 테스트 시작")
    print("=" * 60)
    
    test_results = []
    
    # 1. 서버 상태 확인
    test_results.append(("서버 상태 확인", test_health_check()))
    
    if not test_results[-1][1]:
        print("\n❌ 서버가 실행되지 않았습니다. 테스트를 중단합니다.")
        return
    
    # 2. 환경변수 확인
    test_results.append(("환경변수 확인", test_env_check()))
    
    # 3. API 키 설정 (선택사항)
    test_results.append(("API 키 설정", test_set_api_key()))
    
    # 4. 데이터 로드
    test_results.append(("데이터 로드", test_load_data()))
    
    if test_results[-1][1]:
        # 5. 임계값 계산
        test_results.append(("임계값 계산", test_calculate_thresholds()))
        
        if test_results[-1][1]:
            # 4. 가중치 최적화 (빠른 방법)
            test_results.append(("가중치 최적화 (scipy)", test_optimize_weights('scipy')))
            
            if test_results[-1][1]:
                # 5. 개별 직원 예측
                test_results.append(("개별 직원 예측", test_predict_employee()))
                
                # 6. 방법 비교
                test_results.append(("최적화 방법 비교", test_compare_methods()))
                
                # 7. 결과 조회
                test_results.append(("결과 조회", test_get_results()))
                
                # 8. 결과 내보내기
                test_results.append(("결과 내보내기", test_export_results()))
                
                # 9. 직원 데이터 로드 (레포트용)
                test_results.append(("직원 데이터 로드", test_load_employee_data()))
                
                # 10. 직원 목록 조회
                test_results.append(("직원 목록 조회", test_get_employee_list()))
                
                # 11. 개별 레포트 생성
                test_results.append(("개별 레포트 생성", test_generate_report()))
                
                # 12. 일괄 레포트 생성
                test_results.append(("일괄 레포트 생성", test_generate_batch_reports()))
    
    # 테스트 결과 요약
    print("\n" + "=" * 60)
    print("🏁 테스트 결과 요약")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 전체 결과: {passed}/{total} 테스트 통과 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("⚠️ 일부 테스트가 실패했습니다. 로그를 확인해주세요.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        
        if test_name == "health":
            test_health_check()
        elif test_name == "load":
            test_load_data()
        elif test_name == "threshold":
            test_calculate_thresholds()
        elif test_name == "weight":
            method = sys.argv[2] if len(sys.argv) > 2 else 'scipy'
            test_optimize_weights(method)
        elif test_name == "predict":
            test_predict_employee()
        elif test_name == "compare":
            test_compare_methods()
        elif test_name == "results":
            test_get_results()
        elif test_name == "export":
            test_export_results()
        elif test_name == "employee_data":
            test_load_employee_data()
        elif test_name == "employee_list":
            test_get_employee_list()
        elif test_name == "report":
            test_generate_report()
        elif test_name == "batch_report":
            test_generate_batch_reports()
        elif test_name == "api_key":
            test_set_api_key()
        elif test_name == "env":
            test_env_check()
        else:
            print("사용법: python test_integration_api.py [test_name]")
            print("test_name: health, load, threshold, weight, predict, compare, results, export")
            print("           employee_data, employee_list, report, batch_report, api_key, env")
    else:
        run_full_test()
