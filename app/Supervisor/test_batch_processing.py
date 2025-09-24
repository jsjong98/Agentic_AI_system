#!/usr/bin/env python3
"""
Supervisor 배치 처리 기능 테스트
"""

import requests
import json
import time
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

def create_sample_employee_data():
    """샘플 직원 데이터 생성"""
    sample_employees = [
        {
            "employee_id": "EMP001",
            "Age": 35,
            "JobSatisfaction": 2,
            "WorkLifeBalance": 1,
            "OverTime": "Yes",
            "MonthlyIncome": 3000,
            "YearsAtCompany": 5,
            "feedback_text": "I'm feeling overwhelmed with the workload and considering other opportunities.",
            "Department": "Sales",
            "JobRole": "Sales Executive"
        },
        {
            "employee_id": "EMP002", 
            "Age": 28,
            "JobSatisfaction": 4,
            "WorkLifeBalance": 3,
            "OverTime": "No",
            "MonthlyIncome": 4500,
            "YearsAtCompany": 3,
            "feedback_text": "I love working here and feel very satisfied with my role.",
            "Department": "Research & Development",
            "JobRole": "Research Scientist"
        },
        {
            "employee_id": "EMP003",
            "Age": 42,
            "JobSatisfaction": 3,
            "WorkLifeBalance": 2,
            "OverTime": "Yes",
            "MonthlyIncome": 5500,
            "YearsAtCompany": 8,
            "feedback_text": "The work is challenging but sometimes the work-life balance is difficult.",
            "Department": "Human Resources",
            "JobRole": "HR Manager"
        }
    ]
    return sample_employees

def test_batch_processing():
    """배치 처리 테스트"""
    print("\n🚀 배치 처리 테스트...")
    
    # 샘플 데이터 생성
    employees_data = create_sample_employee_data()
    
    # 배치 처리 요청
    batch_request = {
        "employees": employees_data,
        "options": {
            "include_reports": True,
            "use_llm": False  # 테스트에서는 LLM 사용 안함
        }
    }
    
    try:
        print(f"  📤 {len(employees_data)}명의 직원 데이터로 배치 처리 시작...")
        response = requests.post(f"{BASE_URL}/batch_process", json=batch_request)
        
        if response.status_code == 200:
            data = response.json()
            batch_id = data.get('batch_id')
            print(f"  ✅ 배치 처리 시작됨")
            print(f"  🆔 배치 ID: {batch_id}")
            print(f"  📝 메시지: {data.get('message')}")
            
            return batch_id
        else:
            print(f"  ❌ 배치 처리 시작 실패: {response.status_code}")
            print(f"  📄 응답: {response.text}")
            return None
            
    except Exception as e:
        print(f"  ❌ 배치 처리 오류: {e}")
        return None

def monitor_batch_progress(batch_id):
    """배치 처리 진행 상황 모니터링"""
    print(f"\n📊 배치 처리 진행 상황 모니터링 (ID: {batch_id})...")
    
    max_wait_time = 300  # 최대 5분 대기
    check_interval = 5   # 5초마다 확인
    elapsed_time = 0
    
    while elapsed_time < max_wait_time:
        try:
            response = requests.get(f"{BASE_URL}/batch_status/{batch_id}")
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                progress = data.get('progress', 0)
                processed = data.get('processed_employees', 0)
                total = data.get('total_employees', 0)
                
                print(f"  📈 진행률: {progress:.1f}% ({processed}/{total})")
                print(f"  📊 상태: {status}")
                
                if data.get('completed'):
                    print(f"  🎉 배치 처리 완료!")
                    return True
                    
            else:
                print(f"  ⚠️  상태 확인 실패: {response.status_code}")
            
            time.sleep(check_interval)
            elapsed_time += check_interval
            
        except Exception as e:
            print(f"  ❌ 상태 확인 오류: {e}")
            break
    
    print(f"  ⏰ 최대 대기 시간 초과 ({max_wait_time}초)")
    return False

def get_batch_results(batch_id):
    """배치 처리 결과 조회"""
    print(f"\n📋 배치 처리 결과 조회 (ID: {batch_id})...")
    
    try:
        response = requests.get(f"{BASE_URL}/batch_results/{batch_id}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"  📊 전체 통계:")
            print(f"    총 직원 수: {data.get('total_employees')}")
            print(f"    처리 완료: {data.get('processed_employees')}")
            print(f"    성공: {len(data.get('results', []))}")
            print(f"    오류: {len(data.get('errors', []))}")
            
            # 성공한 결과들
            results = data.get('results', [])
            if results:
                print(f"\n  ✅ 성공한 분석 결과:")
                for i, result in enumerate(results[:3]):  # 처음 3개만 표시
                    employee_id = result.get('employee_id')
                    analysis = result.get('analysis', {})
                    
                    print(f"    {i+1}. {employee_id}:")
                    
                    # 워커 분석 결과
                    worker_analyses = analysis.get('worker_analyses', {})
                    for worker, worker_result in worker_analyses.items():
                        if 'error' in worker_result:
                            print(f"      ❌ {worker}: {worker_result['error']}")
                        else:
                            print(f"      ✅ {worker}: 분석 완료")
                    
                    # Integration 결과
                    if 'integration_prediction' in analysis:
                        print(f"      🔧 Integration: 예측 완료")
                    if 'integration_report' in analysis:
                        print(f"      📄 Integration: 리포트 생성 완료")
            
            # 오류들
            errors = data.get('errors', [])
            if errors:
                print(f"\n  ❌ 오류 발생:")
                for error in errors:
                    employee_id = error.get('employee_id')
                    error_msg = error.get('error')
                    print(f"    • {employee_id}: {error_msg}")
            
            return True
            
        else:
            print(f"  ❌ 결과 조회 실패: {response.status_code}")
            print(f"  📄 응답: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ 결과 조회 오류: {e}")
        return False

def create_sample_csv():
    """샘플 CSV 파일 생성"""
    import pandas as pd
    
    sample_data = create_sample_employee_data()
    df = pd.DataFrame(sample_data)
    
    csv_filename = "sample_employees.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    print(f"  📄 샘플 CSV 파일 생성: {csv_filename}")
    return csv_filename

def test_csv_batch_processing():
    """CSV 배치 처리 테스트"""
    print("\n📄 CSV 배치 처리 테스트...")
    
    # 샘플 CSV 파일 생성
    csv_filename = create_sample_csv()
    
    try:
        # CSV 파일 업로드
        with open(csv_filename, 'rb') as f:
            files = {'file': (csv_filename, f, 'text/csv')}
            data = {
                'include_reports': 'true',
                'use_llm': 'false'
            }
            
            print(f"  📤 CSV 파일 업로드 중: {csv_filename}")
            response = requests.post(f"{BASE_URL}/batch_upload_csv", files=files, data=data)
        
        if response.status_code == 200:
            data = response.json()
            batch_id = data.get('batch_id')
            print(f"  ✅ CSV 배치 처리 시작됨")
            print(f"  🆔 배치 ID: {batch_id}")
            print(f"  📊 총 직원 수: {data.get('total_employees')}")
            print(f"  📋 컬럼: {', '.join(data.get('columns', []))}")
            
            return batch_id
        else:
            print(f"  ❌ CSV 배치 처리 시작 실패: {response.status_code}")
            print(f"  📄 응답: {response.text}")
            return None
            
    except Exception as e:
        print(f"  ❌ CSV 배치 처리 오류: {e}")
        return None
    finally:
        # 임시 CSV 파일 삭제
        try:
            import os
            os.remove(csv_filename)
            print(f"  🗑️  임시 파일 삭제: {csv_filename}")
        except:
            pass

def test_csv_export(batch_id):
    """CSV 내보내기 테스트"""
    print(f"\n📥 CSV 내보내기 테스트 (ID: {batch_id})...")
    
    try:
        response = requests.get(f"{BASE_URL}/batch_export_csv/{batch_id}")
        
        if response.status_code == 200:
            # CSV 파일 저장
            filename = f"exported_results_{batch_id[:8]}.csv"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"  ✅ CSV 내보내기 성공")
            print(f"  📄 저장된 파일: {filename}")
            
            # 파일 내용 간단히 확인
            lines = response.text.split('\n')
            print(f"  📊 총 라인 수: {len(lines)}")
            if len(lines) > 1:
                print(f"  📋 헤더: {lines[0]}")
            
            return True
        else:
            print(f"  ❌ CSV 내보내기 실패: {response.status_code}")
            print(f"  📄 응답: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ CSV 내보내기 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 Supervisor 배치 처리 기능 테스트 시작\n")
    
    # 기본 상태 확인
    if not test_health_check():
        print("❌ 서버가 실행되지 않았습니다. 테스트를 중단합니다.")
        return
    
    print("\n" + "="*60)
    print("📋 테스트 1: JSON 데이터 배치 처리")
    print("="*60)
    
    # JSON 배치 처리 시작
    batch_id = test_batch_processing()
    if batch_id:
        # 진행 상황 모니터링
        if monitor_batch_progress(batch_id):
            # 결과 조회
            get_batch_results(batch_id)
            # CSV 내보내기 테스트
            test_csv_export(batch_id)
        else:
            print("⚠️  배치 처리가 완료되지 않았지만 현재 결과를 확인합니다.")
            get_batch_results(batch_id)
    
    print("\n" + "="*60)
    print("📄 테스트 2: CSV 파일 배치 처리")
    print("="*60)
    
    # CSV 배치 처리 시작
    csv_batch_id = test_csv_batch_processing()
    if csv_batch_id:
        # 진행 상황 모니터링
        if monitor_batch_progress(csv_batch_id):
            # 결과 조회
            get_batch_results(csv_batch_id)
            # CSV 내보내기 테스트
            test_csv_export(csv_batch_id)
        else:
            print("⚠️  CSV 배치 처리가 완료되지 않았지만 현재 결과를 확인합니다.")
            get_batch_results(csv_batch_id)
    
    print("\n🎉 배치 처리 테스트 완료!")

if __name__ == "__main__":
    main()
