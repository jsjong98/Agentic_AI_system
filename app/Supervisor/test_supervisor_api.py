#!/usr/bin/env python3
"""
Supervisor API 테스트 스크립트
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List

class SupervisorAPITester:
    """Supervisor API 테스트 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:5006"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def test_health_check(self) -> bool:
        """서버 상태 확인 테스트"""
        print("🏥 Health Check 테스트...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ 서버 상태: {data.get('status', 'unknown')}")
                print(f"  📊 워크플로우 초기화: {data.get('workflow_initialized', False)}")
                
                if 'available_workers' in data:
                    workers = data['available_workers']
                    print(f"  🤖 사용 가능한 워커: {len(workers)} ({', '.join(workers)})")
                
                return True
            else:
                print(f"  ❌ Health check 실패: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"  ❌ Health check 오류: {e}")
            return False
    
    def test_worker_health_check(self) -> bool:
        """워커 상태 확인 테스트"""
        print("\n🔧 Worker Health Check 테스트...")
        
        try:
            response = self.session.get(f"{self.base_url}/worker_health_check")
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    worker_status = data.get('worker_status', {})
                    summary = data.get('summary', {})
                    
                    print(f"  📊 워커 상태 요약:")
                    print(f"    전체: {summary.get('total_workers', 0)}")
                    print(f"    온라인: {summary.get('healthy_workers', 0)}")
                    print(f"    오프라인: {summary.get('unhealthy_workers', 0)}")
                    print(f"    가용률: {summary.get('health_rate', 0):.1%}")
                    
                    print(f"  🤖 개별 워커 상태:")
                    for worker, status in worker_status.items():
                        status_icon = "✅" if status['healthy'] else "❌"
                        print(f"    {status_icon} {worker}: {status['status']}")
                    
                    return True
                else:
                    print(f"  ❌ 워커 상태 확인 실패: {data.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"  ❌ 워커 상태 확인 실패: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"  ❌ 워커 상태 확인 오류: {e}")
            return False
    
    def test_employee_analysis(self, employee_id: str = "test_employee_001") -> Dict[str, Any]:
        """직원 분석 테스트"""
        print(f"\n👤 직원 분석 테스트 (ID: {employee_id})...")
        
        try:
            payload = {
                "employee_id": employee_id,
                "session_id": f"test_session_{int(time.time())}"
            }
            
            print(f"  📤 분석 요청 전송...")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.base_url}/analyze_employee",
                json=payload,
                timeout=300  # 5분 타임아웃
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    print(f"  ✅ 분석 완료 ({elapsed_time:.1f}초)")
                    print(f"  📋 세션 ID: {data.get('session_id', 'N/A')}")
                    
                    # 실행 요약
                    if 'execution_summary' in data:
                        summary = data['execution_summary']
                        print(f"  📊 실행 요약:")
                        print(f"    성공한 워커: {summary.get('successful_workers', 0)}")
                        print(f"    실패한 워커: {summary.get('failed_workers', 0)}")
                        print(f"    오류 수: {summary.get('error_count', 0)}")
                    
                    # 최종 보고서
                    if 'final_report' in data and data['final_report']:
                        report = data['final_report']
                        print(f"  📄 최종 보고서:")
                        print(f"    위험 점수: {report.get('risk_score', 0):.1f}/100")
                        print(f"    위험 등급: {report.get('risk_grade', 'N/A')}")
                        print(f"    이탈 확률: {report.get('attrition_probability', 0):.1%}")
                        print(f"    신뢰도: {report.get('confidence_score', 0):.1%}")
                        
                        if 'summary' in report:
                            print(f"    요약: {report['summary'][:100]}...")
                    
                    return data
                else:
                    print(f"  ❌ 분석 실패: {data.get('error', 'Unknown error')}")
                    return data
            else:
                print(f"  ❌ 분석 요청 실패: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"    오류: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"    응답: {response.text[:200]}...")
                return {}
                
        except requests.exceptions.Timeout:
            print(f"  ⏰ 분석 타임아웃 (5분 초과)")
            return {}
        except Exception as e:
            print(f"  ❌ 분석 오류: {e}")
            return {}
    
    def test_workflow_status(self, session_id: str) -> bool:
        """워크플로우 상태 조회 테스트"""
        print(f"\n📊 워크플로우 상태 조회 테스트 (세션: {session_id})...")
        
        try:
            response = self.session.get(f"{self.base_url}/get_workflow_status/{session_id}")
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    status = data.get('status', {})
                    print(f"  ✅ 상태 조회 성공")
                    print(f"  📋 현재 단계: {status.get('current_step', 'N/A')}")
                    print(f"  ✅ 완료 여부: {status.get('is_completed', False)}")
                    print(f"  ⏱️ 경과 시간: {status.get('elapsed_time_minutes', 0):.1f}분")
                    
                    return True
                else:
                    print(f"  ❌ 상태 조회 실패: {data.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"  ❌ 상태 조회 실패: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"  ❌ 상태 조회 오류: {e}")
            return False
    
    def test_batch_analysis(self, employee_ids: List[str] = None) -> bool:
        """배치 분석 테스트"""
        if employee_ids is None:
            employee_ids = ["batch_test_001", "batch_test_002", "batch_test_003"]
        
        print(f"\n📦 배치 분석 테스트 ({len(employee_ids)}명)...")
        
        try:
            payload = {
                "employee_ids": employee_ids
            }
            
            print(f"  📤 배치 분석 요청 전송...")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.base_url}/batch_analyze",
                json=payload,
                timeout=600  # 10분 타임아웃
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    summary = data.get('summary', {})
                    print(f"  ✅ 배치 분석 완료 ({elapsed_time:.1f}초)")
                    print(f"  📊 결과 요약:")
                    print(f"    전체: {summary.get('total_employees', 0)}명")
                    print(f"    성공: {summary.get('successful_analyses', 0)}명")
                    print(f"    실패: {summary.get('failed_analyses', 0)}명")
                    print(f"    성공률: {summary.get('success_rate', 0):.1%}")
                    
                    return True
                else:
                    print(f"  ❌ 배치 분석 실패: {data.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"  ❌ 배치 분석 실패: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"  ⏰ 배치 분석 타임아웃 (10분 초과)")
            return False
        except Exception as e:
            print(f"  ❌ 배치 분석 오류: {e}")
            return False
    
    def test_system_info(self) -> bool:
        """시스템 정보 조회 테스트"""
        print("\n💻 시스템 정보 조회 테스트...")
        
        try:
            response = self.session.get(f"{self.base_url}/system_info")
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ 시스템 정보 조회 성공")
                print(f"  📋 서비스: {data.get('service', 'N/A')} v{data.get('version', 'N/A')}")
                print(f"  🔧 워크플로우 초기화: {data.get('workflow_initialized', False)}")
                print(f"  📊 활성 세션: {data.get('active_sessions_count', 0)}개")
                
                env = data.get('environment', {})
                print(f"  🌍 환경 설정:")
                print(f"    최대 재시도: {env.get('max_retry_count', 'N/A')}")
                print(f"    타임아웃: {env.get('timeout_minutes', 'N/A')}분")
                print(f"    OpenAI API: {'설정됨' if env.get('openai_api_key_configured') else '미설정'}")
                
                return True
            else:
                print(f"  ❌ 시스템 정보 조회 실패: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"  ❌ 시스템 정보 조회 오류: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """모든 테스트 실행"""
        print("🧪 Supervisor API 전체 테스트 시작")
        print("=" * 60)
        
        results = {}
        
        # 1. Health Check
        results['health_check'] = self.test_health_check()
        
        # 2. System Info
        results['system_info'] = self.test_system_info()
        
        # 3. Worker Health Check
        results['worker_health_check'] = self.test_worker_health_check()
        
        # 4. Employee Analysis
        analysis_result = self.test_employee_analysis()
        results['employee_analysis'] = bool(analysis_result.get('success'))
        
        # 5. Workflow Status (분석이 성공한 경우)
        if results['employee_analysis'] and analysis_result.get('session_id'):
            results['workflow_status'] = self.test_workflow_status(analysis_result['session_id'])
        else:
            results['workflow_status'] = False
        
        # 6. Batch Analysis (간단한 버전)
        results['batch_analysis'] = self.test_batch_analysis(["batch_test_001"])
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("📋 테스트 결과 요약:")
        
        passed = 0
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"  {status} {test_name}")
            if passed_test:
                passed += 1
        
        print(f"\n🎯 전체 결과: {passed}/{total} 테스트 통과 ({passed/total:.1%})")
        
        if passed == total:
            print("🎉 모든 테스트가 성공했습니다!")
        else:
            print("⚠️  일부 테스트가 실패했습니다. 로그를 확인하세요.")
        
        return results

def main():
    """메인 실행 함수"""
    
    # 명령행 인자 처리
    base_url = "http://localhost:5006"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"🎯 테스트 대상: {base_url}")
    
    # 테스터 생성 및 실행
    tester = SupervisorAPITester(base_url)
    results = tester.run_all_tests()
    
    # 종료 코드 설정
    if all(results.values()):
        sys.exit(0)  # 모든 테스트 성공
    else:
        sys.exit(1)  # 일부 테스트 실패

if __name__ == '__main__':
    main()
