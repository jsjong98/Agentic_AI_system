#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cognita 관계형 위험도 분석 Flask API 테스트 스크립트
Neo4j 기반 관계형 데이터 분석 테스트
"""

import requests
import json
import time
from typing import Dict, List

class CognitaAPITester:
    """Cognita API 테스트 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:5002"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def test_cors_preflight(self) -> bool:
        """CORS preflight 요청 테스트 (React 연동 확인)"""
        print("1. CORS Preflight 테스트 (React 연동 확인)")
        print("-" * 40)
        
        try:
            response = self.session.options(
                f"{self.base_url}/api/health",
                headers={
                    'Origin': 'http://localhost:3000',
                    'Access-Control-Request-Method': 'GET',
                    'Access-Control-Request-Headers': 'Content-Type'
                }
            )
            
            if response.status_code in [200, 204]:
                cors_headers = {
                    'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
                    'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
                    'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
                }
                
                print(f"✅ CORS 설정 확인:")
                for header, value in cors_headers.items():
                    if value:
                        print(f"  {header}: {value}")
                
                return True
            else:
                print(f"❌ CORS preflight 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ CORS 테스트 오류: {e}")
            return False
    
    def test_health_check(self) -> bool:
        """헬스체크 테스트"""
        print("\n2. 헬스체크 테스트")
        print("-" * 30)
        
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 상태: {data['status']}")
                print(f"✅ Neo4j 연결: {data['neo4j_connected']}")
                print(f"✅ 총 직원 수: {data['total_employees']:,}명")
                print(f"✅ 총 관계 수: {data['total_relationships']:,}개")
                print(f"✅ 응답 시간: {data['timestamp']}")
                
                # 성능 최적화 확인
                if data['total_employees'] >= 1470:
                    print(f"✅ 대용량 데이터 확인: 최적화된 분석 가능")
                
                return True
            else:
                print(f"❌ 헬스체크 실패: {response.status_code}")
                if response.content:
                    print(f"   오류 내용: {response.json()}")
                return False
                
        except Exception as e:
            print(f"❌ 헬스체크 오류: {e}")
            return False
    
    def test_employees_list(self, limit: int = 10) -> List[Dict]:
        """직원 목록 조회 테스트"""
        print(f"\n3. 직원 목록 조회 테스트 (상위 {limit}명)")
        print("-" * 40)
        
        try:
            response = self.session.get(f"{self.base_url}/api/employees?limit={limit}")
            
            if response.status_code == 200:
                data = response.json()
                employees = data.get('employees', [])
                pagination = data.get('pagination', {})
                
                print(f"✅ 조회된 직원 수: {len(employees)}명")
                print(f"✅ 페이지네이션: limit={pagination.get('limit')}, offset={pagination.get('offset')}")
                
                if employees:
                    print("직원 목록:")
                    for i, emp in enumerate(employees[:5], 1):
                        print(f"  {i}. {emp['employee_id']} - {emp['name']} ({emp['department']})")
                    
                    if len(employees) > 5:
                        print(f"  ... 외 {len(employees) - 5}명")
                
                return employees
            else:
                print(f"❌ 직원 목록 조회 실패: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ 직원 목록 조회 오류: {e}")
            return []
    
    def test_departments_list(self) -> List[Dict]:
        """부서 목록 조회 테스트"""
        print(f"\n4. 부서 목록 조회 테스트")
        print("-" * 30)
        
        try:
            response = self.session.get(f"{self.base_url}/api/departments")
            
            if response.status_code == 200:
                data = response.json()
                departments = data.get('departments', [])
                
                print(f"✅ 조회된 부서 수: {len(departments)}개")
                
                if departments:
                    print("부서 목록 (직원 수 기준 상위 5개):")
                    for i, dept in enumerate(departments[:5], 1):
                        print(f"  {i}. {dept['department_name']}: {dept['employee_count']}명")
                
                return departments
            else:
                print(f"❌ 부서 목록 조회 실패: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ 부서 목록 조회 오류: {e}")
            return []
    
    def test_employee_analysis(self, employee_id: str) -> Dict:
        """개별 직원 분석 테스트"""
        print(f"\n5. 직원 위험도 분석 테스트: {employee_id}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/api/analyze/employee/{employee_id}")
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                elapsed_time = end_time - start_time
                print(f"✅ 분석 완료 (소요시간: {elapsed_time:.2f}초)")
                print(f"  직원 ID: {data['employee_id']}")
                print(f"  종합 위험도: {data['overall_risk_score']:.3f} ({data['risk_category']})")
                print(f"  사회적 고립: {data['social_isolation_index']:.3f}")
                print(f"  네트워크 중심성: {data['network_centrality_score']:.3f}")
                print(f"  관리자 불안정성: {data['manager_instability_score']:.3f}")
                print(f"  팀 변동성: {data['team_volatility_index']:.3f}")
                
                if data['risk_factors']:
                    print(f"  주요 위험 요인: {', '.join(data['risk_factors'])}")
                else:
                    print(f"  주요 위험 요인: 없음")
                
                # 네트워크 통계
                if data['network_stats']:
                    connections = data['network_stats'].get('direct_connections', 0)
                    print(f"  직접 연결: {connections}명")
                
                # 성능 평가
                if elapsed_time < 1.0:
                    perf_grade = "⚡ 우수"
                elif elapsed_time < 2.0:
                    perf_grade = "✅ 양호"
                elif elapsed_time < 5.0:
                    perf_grade = "⚠️ 보통"
                else:
                    perf_grade = "🔧 개선필요"
                
                print(f"  성능 등급: {perf_grade}")
                
                # React에서 사용할 수 있는 형태 확인
                print(f"  ✅ JSON 직렬화: 성공")
                print(f"  ✅ 분석 시간: {data.get('analysis_timestamp', 'N/A')}")
                
                return data
            else:
                print(f"❌ 직원 분석 실패: {response.status_code}")
                if response.content:
                    error_data = response.json()
                    print(f"   오류: {error_data.get('error', 'Unknown error')}")
                return {}
                
        except Exception as e:
            print(f"❌ 직원 분석 오류: {e}")
            return {}
    
    def test_department_analysis(self, department_name: str, sample_size: int = 5) -> Dict:
        """부서 분석 테스트"""
        print(f"\n6. 부서 위험도 분석 테스트: {department_name}")
        print("-" * 50)
        
        try:
            payload = {
                "department_name": department_name,
                "sample_size": sample_size
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/analyze/department",
                json=payload
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ 부서 분석 완료 (소요시간: {end_time - start_time:.2f}초)")
                print(f"  부서명: {data['department_name']}")
                print(f"  전체 직원: {data['total_employees']}명")
                print(f"  분석 직원: {data['analyzed_employees']}명")
                
                # 위험 분포
                risk_dist = data['risk_distribution']
                print(f"  위험 분포: HIGH({risk_dist['HIGH']}) / MEDIUM({risk_dist['MEDIUM']}) / LOW({risk_dist['LOW']})")
                
                # 평균 점수
                avg_scores = data['average_scores']
                print(f"  평균 위험도: {avg_scores['overall_risk']:.3f}")
                
                # 고위험 직원
                if data['high_risk_employees']:
                    print(f"  고위험 직원 ({len(data['high_risk_employees'])}명):")
                    for emp in data['high_risk_employees'][:3]:
                        print(f"    - {emp['employee_id']}: {emp['overall_risk_score']:.3f}")
                
                # 주요 위험 요인
                if data['top_risk_factors']:
                    top_factors = list(data['top_risk_factors'].items())[:3]
                    print(f"  주요 위험 요인:")
                    for factor, count in top_factors:
                        print(f"    - {factor}: {count}명")
                
                # 권장 조치
                if data['recommendations']:
                    print(f"  권장 조치:")
                    for i, rec in enumerate(data['recommendations'][:2], 1):
                        print(f"    {i}. {rec}")
                
                # React 연동 확인
                print(f"  ✅ JSON 응답: 성공")
                print(f"  ✅ 분석 시간: {data.get('analysis_timestamp', 'N/A')}")
                
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
        print(f"\n7. React 연동 예시 코드")
        print("-" * 40)
        
        print("// React 컴포넌트에서 사용 예시")
        print()
        print("// 1. 헬스체크")
        print("const checkHealth = async () => {")
        print("  try {")
        print(f"    const response = await fetch('{self.base_url}/api/health');")
        print("    const data = await response.json();")
        print("    console.log('서버 상태:', data);")
        print("  } catch (error) {")
        print("    console.error('헬스체크 실패:', error);")
        print("  }")
        print("};")
        print()
        
        print("// 2. 직원 분석")
        print("const analyzeEmployee = async (employeeId) => {")
        print("  try {")
        print(f"    const response = await fetch(`{self.base_url}/api/analyze/employee/${{employeeId}}`);")
        print("    const data = await response.json();")
        print("    setEmployeeRisk(data);")
        print("  } catch (error) {")
        print("    console.error('직원 분석 실패:', error);")
        print("  }")
        print("};")
        print()
        
        print("// 3. 부서 분석")
        print("const analyzeDepartment = async (departmentName) => {")
        print("  try {")
        print(f"    const response = await fetch('{self.base_url}/api/analyze/department', {{")
        print("      method: 'POST',")
        print("      headers: { 'Content-Type': 'application/json' },")
        print("      body: JSON.stringify({")
        print("        department_name: departmentName,")
        print("        sample_size: 20")
        print("      })")
        print("    });")
        print("    const data = await response.json();")
        print("    setDepartmentRisk(data);")
        print("  } catch (error) {")
        print("    console.error('부서 분석 실패:', error);")
        print("  }")
        print("};")
    
    def run_full_test(self):
        """전체 테스트 실행"""
        print("=" * 60)
        print("🧪 Cognita 관계형 위험도 분석 API 전체 테스트 시작")
        print("=" * 60)
        
        # 1. CORS 테스트
        if not self.test_cors_preflight():
            print("\n⚠️ CORS 설정 확인 필요 (React 연동에 영향)")
        
        # 2. 헬스체크
        if not self.test_health_check():
            print("\n❌ 헬스체크 실패. 서버가 실행 중인지 확인하세요.")
            return False
        
        # 3. 직원 목록 조회
        employees = self.test_employees_list()
        if not employees:
            print("\n❌ 직원 목록 조회 실패")
            return False
        
        # 4. 부서 목록 조회
        departments = self.test_departments_list()
        if not departments:
            print("\n❌ 부서 목록 조회 실패")
            return False
        
        # 5. 개별 직원 분석 (첫 번째 직원)
        test_employee_id = employees[0]['employee_id']
        employee_result = self.test_employee_analysis(test_employee_id)
        if not employee_result:
            print(f"\n❌ 직원 {test_employee_id} 분석 실패")
            return False
        
        # 6. 부서 분석 (첫 번째 부서)
        test_department = departments[0]['department_name']
        dept_result = self.test_department_analysis(test_department, sample_size=3)
        if not dept_result:
            print(f"\n❌ 부서 {test_department} 분석 실패")
            return False
        
        # 7. React 연동 예시
        self.test_react_integration_examples()
        
        print("\n" + "=" * 60)
        print("✅ 모든 테스트 성공!")
        print("🔗 React 연동 준비 완료!")
        print("🕸️ Neo4j 기반 관계형 데이터 분석 서비스 준비!")
        print("=" * 60)
        return True

def main():
    """메인 함수"""
    
    # 서버 URL 설정
    base_url = "http://localhost:5000"
    
    print("Cognita 관계형 위험도 분석 API 테스트를 시작합니다...")
    print(f"서버 URL: {base_url}")
    print("Neo4j 기반 관계형 데이터 분석 테스트")
    print()
    
    # 테스터 생성 및 실행
    tester = CognitaAPITester(base_url)
    
    try:
        success = tester.run_full_test()
        
        if success:
            print("\n🎉 모든 API 테스트가 성공적으로 완료되었습니다!")
            print("🚀 React 프론트엔드와 연동할 준비가 되었습니다!")
            print("🕸️ Neo4j 기반 관계형 분석 서비스 제공!")
        else:
            print("\n⚠️ 일부 테스트가 실패했습니다. 로그를 확인하세요.")
            
    except KeyboardInterrupt:
        print("\n테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
