#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agentic AI System 통합 테스트 스크립트
모든 워커 에이전트의 통합 동작을 테스트
"""

import requests
import json
import time
from typing import Dict, List

class AgenticSystemTester:
    """Agentic AI System 테스트 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def test_system_health(self) -> bool:
        """시스템 헬스체크 테스트"""
        print("1. 시스템 헬스체크 테스트")
        print("-" * 40)
        
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 시스템 상태: {data['status']}")
                print(f"✅ 워커 에이전트 수: {data['worker_count']}")
                
                print(f"✅ 워커 에이전트 상태:")
                for worker_id, worker_info in data['workers'].items():
                    status_icon = "✅" if worker_info['status'] == 'running' else "❌"
                    print(f"    {status_icon} {worker_info['agent_name']}: {worker_info['status']}")
                    if worker_info.get('error_message'):
                        print(f"       오류: {worker_info['error_message']}")
                
                print(f"✅ 시스템 기능:")
                capabilities = data['capabilities']
                print(f"    Structura: {'✅' if capabilities['structura_available'] else '❌'}")
                print(f"    Cognita: {'✅' if capabilities['cognita_available'] else '❌'}")
                
                return data['status'] in ['healthy', 'degraded']
            else:
                print(f"❌ 헬스체크 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 헬스체크 오류: {e}")
            return False
    
    def test_workers_status(self) -> bool:
        """워커 에이전트 상태 테스트"""
        print("\n2. 워커 에이전트 상태 테스트")
        print("-" * 40)
        
        try:
            response = self.session.get(f"{self.base_url}/api/workers/status")
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ 워커 에이전트 현황:")
                summary = data['summary']
                print(f"    전체: {summary['total_workers']}개")
                print(f"    실행 중: {summary['running']}개")
                print(f"    작업 중: {summary['busy']}개")
                print(f"    오류: {summary['error']}개")
                
                print(f"✅ 개별 워커 상태:")
                for worker_id, worker_info in data['workers'].items():
                    print(f"    📊 {worker_info['agent_name']}:")
                    print(f"       상태: {worker_info['status']}")
                    print(f"       완료 작업: {worker_info['tasks_completed']}개")
                    if worker_info['current_task']:
                        print(f"       현재 작업: {worker_info['current_task']}")
                
                return True
            else:
                print(f"❌ 워커 상태 조회 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 워커 상태 조회 오류: {e}")
            return False
    
    def test_individual_analysis(self) -> Dict:
        """개별 직원 통합 분석 테스트"""
        print("\n3. 개별 직원 통합 분석 테스트")
        print("-" * 40)
        
        # 테스트용 직원 데이터
        test_employee = {
            "employee_id": "1",
            "Age": 35,
            "BusinessTravel": "Travel_Rarely",
            "Department": "Sales",
            "DistanceFromHome": 10,
            "Education": 3,
            "EducationField": "Life Sciences",
            "EnvironmentSatisfaction": 2,
            "Gender": "Male",
            "JobInvolvement": 3,
            "JobLevel": 2,
            "JobRole": "Sales Executive",
            "JobSatisfaction": 3,
            "MaritalStatus": "Married",
            "MonthlyIncome": 5000,
            "OverTime": "Yes",
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 3,
            "StockOptionLevel": 1,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 2,
            "YearsAtCompany": 5,
            "YearsInCurrentRole": 3,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 2,
            "use_structura": True,
            "use_cognita": True
        }
        
        try:
            print(f"분석 대상: 직원 ID {test_employee['employee_id']}")
            
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/api/analyze/individual", json=test_employee)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ 통합 분석 완료 (소요시간: {end_time - start_time:.2f}초)")
                print(f"  작업 ID: {data['task_id']}")
                print(f"  실행 시간: {data['execution_time']:.2f}초")
                print(f"  상태: {data['status']}")
                
                # Structura 결과
                if data.get('structura_result'):
                    structura = data['structura_result']
                    if 'prediction' in structura:
                        pred = structura['prediction']
                        print(f"  📊 Structura 분석:")
                        print(f"     이직 확률: {pred['attrition_probability']:.1%}")
                        print(f"     위험 범주: {pred['risk_category']}")
                        print(f"     신뢰도: {pred['confidence_score']:.1%}")
                
                # Cognita 결과
                if data.get('cognita_result'):
                    cognita = data['cognita_result']
                    if 'risk_analysis' in cognita:
                        risk = cognita['risk_analysis']
                        print(f"  🕸️  Cognita 분석:")
                        print(f"     종합 위험도: {risk['overall_risk_score']:.3f}")
                        print(f"     사회적 고립: {risk['social_isolation_index']:.3f}")
                        print(f"     네트워크 중심성: {risk['network_centrality_score']:.3f}")
                
                # 통합 분석 결과
                if data.get('combined_analysis'):
                    combined = data['combined_analysis']
                    print(f"  🤖 통합 분석:")
                    
                    if 'integrated_assessment' in combined:
                        assessment = combined['integrated_assessment']
                        print(f"     통합 위험도: {assessment['integrated_risk_score']:.3f}")
                        print(f"     위험 수준: {assessment['risk_level']}")
                        print(f"     합의도: {assessment['consensus']}")
                    
                    if 'recommendations' in combined:
                        print(f"     권장 조치:")
                        for i, rec in enumerate(combined['recommendations'][:3], 1):
                            print(f"       {i}. {rec}")
                
                return data
            else:
                print(f"❌ 개별 분석 실패: {response.status_code}")
                if response.content:
                    error_data = response.json()
                    print(f"   오류: {error_data.get('error', 'Unknown error')}")
                return {}
                
        except Exception as e:
            print(f"❌ 개별 분석 오류: {e}")
            return {}
    
    def test_department_analysis(self) -> Dict:
        """부서별 통합 분석 테스트"""
        print("\n4. 부서별 통합 분석 테스트")
        print("-" * 40)
        
        test_request = {
            "department_name": "Sales",
            "sample_size": 5,  # 빠른 테스트를 위해 작은 샘플
            "use_structura": True,
            "use_cognita": True
        }
        
        try:
            print(f"분석 대상: {test_request['department_name']} 부서")
            
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/api/analyze/department", json=test_request)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ 부서 분석 완료 (소요시간: {end_time - start_time:.2f}초)")
                print(f"  작업 ID: {data['task_id']}")
                print(f"  실행 시간: {data['execution_time']:.2f}초")
                print(f"  상태: {data['status']}")
                
                # Structura 결과
                if data.get('structura_result'):
                    structura = data['structura_result']
                    print(f"  📊 Structura 분석:")
                    print(f"     메시지: {structura.get('message', 'N/A')}")
                
                # Cognita 결과
                if data.get('cognita_result'):
                    cognita = data['cognita_result']
                    if 'department_report' in cognita:
                        report = cognita['department_report']
                        if '분석_개요' in report:
                            overview = report['분석_개요']
                            print(f"  🕸️  Cognita 분석:")
                            print(f"     분석 인원: {overview['총_분석_인원']}명")
                            print(f"     고위험 비율: {overview['고위험_비율']}")
                        
                        if '위험_분포' in report:
                            dist = report['위험_분포']
                            print(f"     위험 분포: HIGH({dist['HIGH']}) / MEDIUM({dist['MEDIUM']}) / LOW({dist['LOW']})")
                
                # 통합 분석 결과
                if data.get('combined_analysis'):
                    combined = data['combined_analysis']
                    print(f"  🤖 통합 분석:")
                    print(f"     분석 유형: {combined['analysis_type']}")
                    
                    if 'recommendations' in combined:
                        print(f"     권장 조치:")
                        for i, rec in enumerate(combined['recommendations'][:2], 1):
                            print(f"       {i}. {rec}")
                
                return data
            else:
                print(f"❌ 부서 분석 실패: {response.status_code}")
                if response.content:
                    error_data = response.json()
                    print(f"   오류: {error_data.get('error', 'Unknown error')}")
                return {}
                
        except Exception as e:
            print(f"❌ 부서 분석 오류: {e}")
            return {}
    
    def test_react_integration_examples(self):
        """React 연동 예시 코드 출력"""
        print(f"\n5. React 연동 예시 코드")
        print("-" * 40)
        
        print("// React Hook for Agentic AI System")
        print("const useAgenticAI = () => {")
        print("  const [systemHealth, setSystemHealth] = useState(null);")
        print("  const [loading, setLoading] = useState(false);")
        print("  const [error, setError] = useState(null);")
        print()
        print("  const baseURL = 'http://localhost:8000/api';")
        print()
        print("  // 시스템 상태 확인")
        print("  const checkSystemHealth = async () => {")
        print("    try {")
        print("      setLoading(true);")
        print("      const response = await fetch(`${baseURL}/health`);")
        print("      const data = await response.json();")
        print("      setSystemHealth(data);")
        print("      return data;")
        print("    } catch (err) {")
        print("      setError(err.message);")
        print("    } finally {")
        print("      setLoading(false);")
        print("    }")
        print("  };")
        print()
        
        print("  // 개별 직원 통합 분석")
        print("  const analyzeEmployee = async (employeeData) => {")
        print("    try {")
        print("      setLoading(true);")
        print("      const response = await fetch(`${baseURL}/analyze/individual`, {")
        print("        method: 'POST',")
        print("        headers: { 'Content-Type': 'application/json' },")
        print("        body: JSON.stringify({")
        print("          ...employeeData,")
        print("          use_structura: true,")
        print("          use_cognita: true")
        print("        })")
        print("      });")
        print("      const result = await response.json();")
        print("      return result;")
        print("    } catch (err) {")
        print("      setError(err.message);")
        print("    } finally {")
        print("      setLoading(false);")
        print("    }")
        print("  };")
        print()
        
        print("  // 부서별 통합 분석")
        print("  const analyzeDepartment = async (departmentName, sampleSize = 20) => {")
        print("    try {")
        print("      setLoading(true);")
        print("      const response = await fetch(`${baseURL}/analyze/department`, {")
        print("        method: 'POST',")
        print("        headers: { 'Content-Type': 'application/json' },")
        print("        body: JSON.stringify({")
        print("          department_name: departmentName,")
        print("          sample_size: sampleSize,")
        print("          use_structura: true,")
        print("          use_cognita: true")
        print("        })")
        print("      });")
        print("      const result = await response.json();")
        print("      return result;")
        print("    } catch (err) {")
        print("      setError(err.message);")
        print("    } finally {")
        print("      setLoading(false);")
        print("    }")
        print("  };")
        print()
        
        print("  return {")
        print("    systemHealth,")
        print("    loading,")
        print("    error,")
        print("    checkSystemHealth,")
        print("    analyzeEmployee,")
        print("    analyzeDepartment")
        print("  };")
        print("};")
    
    def run_full_test(self):
        """전체 시스템 테스트 실행"""
        print("=" * 70)
        print("🧪 Agentic AI System 전체 테스트 시작")
        print("=" * 70)
        
        # 1. 시스템 헬스체크
        if not self.test_system_health():
            print("\n❌ 시스템 헬스체크 실패. 서버가 실행 중인지 확인하세요.")
            return False
        
        # 2. 워커 상태 확인
        if not self.test_workers_status():
            print("\n❌ 워커 상태 확인 실패")
            return False
        
        # 3. 개별 분석 테스트
        individual_result = self.test_individual_analysis()
        if not individual_result:
            print("\n❌ 개별 분석 테스트 실패")
            return False
        
        # 4. 부서 분석 테스트
        department_result = self.test_department_analysis()
        if not department_result:
            print("\n❌ 부서 분석 테스트 실패")
            return False
        
        # 5. React 연동 예시
        self.test_react_integration_examples()
        
        print("\n" + "=" * 70)
        print("✅ 모든 테스트 성공!")
        print("🤖 Agentic AI System 준비 완료!")
        print("🚀 React 프론트엔드와 연동할 준비가 되었습니다!")
        print("=" * 70)
        
        print("\n📊 테스트 결과 요약:")
        print("  ✅ 시스템 헬스체크: 통과")
        print("  ✅ 워커 에이전트 상태: 정상")
        print("  ✅ 개별 직원 통합 분석: 성공")
        print("  ✅ 부서별 통합 분석: 성공")
        print("  ✅ React 연동 준비: 완료")
        
        return True

def main():
    """메인 함수"""
    
    # 서버 URL 설정
    base_url = "http://localhost:8000"
    
    print("Agentic AI System 통합 테스트를 시작합니다...")
    print(f"마스터 서버 URL: {base_url}")
    print("워커 에이전트 통합 동작 테스트")
    print()
    
    # 테스터 생성 및 실행
    tester = AgenticSystemTester(base_url)
    
    try:
        success = tester.run_full_test()
        
        if success:
            print("\n🎉 모든 통합 테스트가 성공적으로 완료되었습니다!")
            print("🤖 Agentic AI System이 정상적으로 동작합니다!")
            print("🚀 React 프론트엔드 개발을 시작할 수 있습니다!")
        else:
            print("\n⚠️ 일부 테스트가 실패했습니다. 로그를 확인하세요.")
            
    except KeyboardInterrupt:
        print("\n테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
