#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agentic AI System - 결과 저장 시스템 테스트
"""

import requests
import json
import time
from pathlib import Path

class ResultSystemTester:
    """결과 저장 시스템 테스트 클래스"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_individual_analysis_with_results(self):
        """개별 분석 및 결과 저장 테스트"""
        print("🧪 개별 분석 및 결과 저장 테스트")
        print("=" * 50)
        
        # 테스트 직원 데이터
        test_employee = {
            "EmployeeNumber": 1001,
            "Age": 35,
            "Department": "Research & Development",
            "JobRole": "Research Scientist",
            "YearsAtCompany": 5,
            "MonthlyIncome": 5000,
            "JobSatisfaction": 2,
            "WorkLifeBalance": 2,
            "OverTime": "Yes",
            "DistanceFromHome": 15,
            "Education": 3,
            "EnvironmentSatisfaction": 2,
            "JobInvolvement": 2,
            "NumCompaniesWorked": 3,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 3,
            "StockOptionLevel": 1,
            "TotalWorkingYears": 8,
            "TrainingTimesLastYear": 2,
            "YearsInCurrentRole": 3,
            "YearsSinceLastPromotion": 2,
            "YearsWithCurrManager": 2,
            "text_data": "업무량이 너무 많아서 번아웃이 올 것 같습니다. 야근도 자주하고 스트레스가 심해요.",
            "use_structura": True,
            "use_cognita": True,
            "use_sentio": True,
            "use_chronos": False  # 시계열 데이터가 없으므로 비활성화
        }
        
        try:
            print("📤 분석 요청 전송 중...")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.base_url}/api/analyze/individual",
                json=test_employee,
                timeout=60
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 분석 완료 (소요시간: {elapsed_time:.2f}초)")
                print(f"📋 작업 ID: {result.get('task_id')}")
                print(f"📊 상태: {result.get('status')}")
                
                # 저장된 경로 확인
                if 'saved_path' in result:
                    print(f"💾 결과 저장 경로: {result['saved_path']}")
                
                # 시각화 파일 확인
                if 'visualizations_available' in result:
                    viz_count = len(result['visualizations_available'])
                    print(f"📈 생성된 시각화: {viz_count}개")
                    for viz in result['visualizations_available'][:3]:  # 처음 3개만 표시
                        print(f"  - {Path(viz).name}")
                
                # 각 워커 결과 요약
                if result.get('structura_result'):
                    structura = result['structura_result']
                    prob = structura.get('probability', 0)
                    pred = structura.get('prediction', 0)
                    print(f"🏢 Structura: 이직 확률 {prob:.3f} (예측: {'이직' if pred else '잔류'})")
                
                if result.get('cognita_result'):
                    cognita = result['cognita_result']
                    risk_score = cognita.get('overall_risk_score', 0)
                    print(f"🕸️ Cognita: 관계형 위험도 {risk_score:.3f}")
                
                if result.get('sentio_result'):
                    sentio = result['sentio_result']
                    sentiment = sentio.get('sentiment_score', 0)
                    print(f"📝 Sentio: 감정 점수 {sentiment:.3f}")
                
                return result.get('task_id')
                
            else:
                print(f"❌ 분석 실패: {response.status_code}")
                print(f"오류 메시지: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            return None
    
    def test_result_retrieval(self, employee_id="1001"):
        """결과 조회 테스트"""
        print(f"\n🔍 직원 {employee_id} 결과 조회 테스트")
        print("=" * 50)
        
        try:
            response = self.session.get(f"{self.base_url}/api/results/employee/{employee_id}")
            
            if response.status_code == 200:
                results = response.json()
                print("✅ 결과 조회 성공")
                
                # 요약 정보
                if 'summary' in results:
                    summary = results['summary']
                    print(f"📊 분석 일시: {summary.get('timestamp', 'N/A')}")
                    print(f"🎯 위험 수준: {summary.get('risk_level', 'N/A')}")
                    print(f"📈 이직 확률: {summary.get('attrition_probability', 0):.3f}")
                
                # 각 워커 결과 확인
                workers = ['structura', 'cognita', 'sentio', 'chronos']
                for worker in workers:
                    if worker in results:
                        print(f"✅ {worker.capitalize()} 결과 있음")
                    else:
                        print(f"❌ {worker.capitalize()} 결과 없음")
                
                return True
                
            elif response.status_code == 404:
                print(f"❌ 직원 {employee_id}의 결과를 찾을 수 없습니다")
                return False
            else:
                print(f"❌ 조회 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 조회 테스트 실패: {e}")
            return False
    
    def test_visualization_list(self, employee_id="1001"):
        """시각화 목록 조회 테스트"""
        print(f"\n📈 직원 {employee_id} 시각화 목록 조회 테스트")
        print("=" * 50)
        
        try:
            response = self.session.get(f"{self.base_url}/api/results/employee/{employee_id}/visualizations")
            
            if response.status_code == 200:
                data = response.json()
                viz_count = data.get('count', 0)
                print(f"✅ 시각화 {viz_count}개 발견")
                
                for viz_file in data.get('visualizations', []):
                    file_name = Path(viz_file).name
                    file_type = "SHAP" if "shap" in file_name.lower() else "Feature Importance" if "feature" in file_name.lower() else "Attention" if "attention" in file_name.lower() else "기타"
                    print(f"  📊 {file_name} ({file_type})")
                
                return True
                
            else:
                print(f"❌ 시각화 목록 조회 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 시각화 목록 테스트 실패: {e}")
            return False
    
    def test_department_report(self, department="Research & Development"):
        """부서 보고서 테스트"""
        print(f"\n📋 부서 '{department}' 보고서 테스트")
        print("=" * 50)
        
        try:
            response = self.session.get(f"{self.base_url}/api/results/department/{department}/report")
            
            if response.status_code == 200:
                report = response.json()
                print("✅ 부서 보고서 생성 성공")
                print(f"👥 총 직원 수: {report.get('total_employees', 0)}명")
                
                risk_dist = report.get('risk_distribution', {})
                print(f"🔴 고위험: {risk_dist.get('HIGH', 0)}명")
                print(f"🟡 중위험: {risk_dist.get('MEDIUM', 0)}명")
                print(f"🟢 저위험: {risk_dist.get('LOW', 0)}명")
                
                avg_scores = report.get('average_scores', {})
                print(f"📊 평균 이직 확률: {avg_scores.get('attrition_probability', 0):.3f}")
                
                high_risk = report.get('high_risk_employees', [])
                if high_risk:
                    print(f"⚠️ 고위험 직원: {', '.join(map(str, high_risk[:5]))}")
                
                return True
                
            elif response.status_code == 404:
                print(f"❌ 부서 '{department}'의 데이터가 없습니다")
                return False
            else:
                print(f"❌ 보고서 생성 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 부서 보고서 테스트 실패: {e}")
            return False
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🚀 결과 저장 시스템 통합 테스트 시작")
        print("=" * 70)
        
        # 1. 개별 분석 및 결과 저장
        task_id = self.test_individual_analysis_with_results()
        
        if task_id:
            # 잠시 대기 (파일 저장 완료 대기)
            print("\n⏳ 결과 저장 완료 대기 중...")
            time.sleep(2)
            
            # 2. 결과 조회
            self.test_result_retrieval("1001")
            
            # 3. 시각화 목록 조회
            self.test_visualization_list("1001")
            
            # 4. 부서 보고서
            self.test_department_report("Research & Development")
        
        print("\n🎉 테스트 완료!")
        print("📁 결과 파일들을 확인하려면 'results/' 폴더를 확인하세요.")

def main():
    """메인 함수"""
    tester = ResultSystemTester()
    
    # 서버 상태 확인
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ 마스터 서버 연결 확인")
            tester.run_all_tests()
        else:
            print("❌ 마스터 서버 응답 오류")
    except requests.exceptions.RequestException:
        print("❌ 마스터 서버에 연결할 수 없습니다.")
        print("💡 먼저 'python run_agentic_system.py'로 서버를 시작하세요.")

if __name__ == "__main__":
    main()
