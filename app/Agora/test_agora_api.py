#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agora API 테스트 스크립트
외부 시장 분석 API의 모든 엔드포인트를 테스트
"""

import requests
import json
import time
from typing import Dict, Any

class AgoraAPITester:
    """Agora API 테스트 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:5004"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def test_health_check(self) -> bool:
        """헬스체크 테스트"""
        print("\n🏥 헬스체크 테스트...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                health_data = response.json()
                print("✅ 헬스체크 성공")
                print(f"   상태: {health_data['status']}")
                print(f"   컴포넌트:")
                for component, status in health_data['components'].items():
                    print(f"     - {component}: {status}")
                return True
            else:
                print(f"❌ 헬스체크 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 헬스체크 오류: {e}")
            return False
    
    def test_individual_market_analysis(self) -> bool:
        """개별 직원 시장 분석 테스트"""
        print("\n👤 개별 직원 시장 분석 테스트...")
        
        test_data = {
            "EmployeeNumber": 1001,
            "JobRole": "Sales Executive",
            "Department": "Sales",
            "MonthlyIncome": 5000,
            "YearsAtCompany": 3,
            "JobSatisfaction": 3,
            "use_llm": False  # LLM 사용 안함 (빠른 테스트)
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/analyze/market",
                json=test_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 개별 시장 분석 성공")
                print(f"   직원 ID: {result['employee_id']}")
                print(f"   직무: {result['job_role']}")
                print(f"   시장 압력 지수: {result['market_pressure_index']:.3f}")
                print(f"   보상 격차: {result['compensation_gap']:.3f}")
                print(f"   채용 공고 수: {result['job_postings_count']}개")
                print(f"   시장 경쟁력: {result['market_competitiveness']}")
                print(f"   위험 수준: {result['risk_level']}")
                return True
            else:
                print(f"❌ 개별 시장 분석 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 개별 시장 분석 오류: {e}")
            return False
    
    def test_job_market_analysis(self) -> bool:
        """직무별 시장 분석 테스트"""
        print("\n💼 직무별 시장 분석 테스트...")
        
        test_data = {
            "job_role": "Data Scientist",
            "location": "서울",
            "experience_level": "mid"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/analyze/job_market",
                json=test_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 직무별 시장 분석 성공")
                print(f"   직무: {result['job_role']}")
                print(f"   총 채용 공고: {result['total_postings']}개")
                print(f"   평균 급여: {result['average_salary']:,}원")
                print(f"   급여 범위: {result['salary_range']['min']:,} ~ {result['salary_range']['max']:,}원")
                print(f"   시장 트렌드: {result['market_trend']}")
                print(f"   경쟁 수준: {result['competition_level']}")
                print(f"   핵심 스킬: {', '.join(result['key_skills'][:3])}")
                return True
            else:
                print(f"❌ 직무별 시장 분석 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 직무별 시장 분석 오류: {e}")
            return False
    
    def test_batch_analysis(self) -> bool:
        """배치 시장 분석 테스트"""
        print("\n📊 배치 시장 분석 테스트...")
        
        test_data = {
            "employees": [
                {
                    "EmployeeNumber": 1001,
                    "JobRole": "Sales Executive",
                    "Department": "Sales",
                    "MonthlyIncome": 5000,
                    "YearsAtCompany": 3,
                    "JobSatisfaction": 3
                },
                {
                    "EmployeeNumber": 1002,
                    "JobRole": "Data Scientist",
                    "Department": "Research & Development",
                    "MonthlyIncome": 6000,
                    "YearsAtCompany": 2,
                    "JobSatisfaction": 4
                },
                {
                    "EmployeeNumber": 1003,
                    "JobRole": "Manager",
                    "Department": "Sales",
                    "MonthlyIncome": 7000,
                    "YearsAtCompany": 5,
                    "JobSatisfaction": 2
                }
            ],
            "use_llm": False
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/analyze/batch",
                json=test_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 배치 시장 분석 성공")
                print(f"   분석 대상: {result['total_analyzed']}명")
                print(f"   고위험 직원: {result['high_risk_employees']}명")
                print(f"   평균 시장 압력: {result['average_market_pressure']:.3f}")
                
                # 개별 결과 요약
                for i, emp_result in enumerate(result['results'][:3], 1):
                    print(f"   직원 {i}: {emp_result['job_role']} - {emp_result['risk_level']}")
                
                return True
            else:
                print(f"❌ 배치 시장 분석 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 배치 시장 분석 오류: {e}")
            return False
    
    def test_market_report(self) -> bool:
        """시장 보고서 조회 테스트"""
        print("\n📋 시장 보고서 조회 테스트...")
        
        job_role = "Sales Executive"
        
        try:
            response = self.session.get(f"{self.base_url}/market/report/{job_role}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 시장 보고서 조회 성공")
                print(f"   직무: {result['job_role']}")
                
                summary = result['market_summary']
                print(f"   채용 공고: {summary['total_postings']}개")
                print(f"   평균 급여: {summary['average_salary']:,}원")
                print(f"   시장 트렌드: {summary['market_trend']}")
                
                metrics = result['market_metrics']
                print(f"   시장 활성도: {metrics['market_activity_score']:.3f}")
                print(f"   수요 수준: {metrics['demand_level']}")
                
                insights = result['key_insights']
                print(f"   시장 전망: {insights['market_outlook'][:50]}...")
                print(f"   권장사항: {len(insights['recommendations'])}개")
                
                return True
            else:
                print(f"❌ 시장 보고서 조회 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 시장 보고서 조회 오류: {e}")
            return False
    
    def test_market_trends(self) -> bool:
        """시장 트렌드 조회 테스트"""
        print("\n📈 시장 트렌드 조회 테스트...")
        
        try:
            response = self.session.get(f"{self.base_url}/market/trends")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 시장 트렌드 조회 성공")
                
                stats = result['overall_statistics']
                print(f"   총 채용 공고: {stats['total_job_postings']}개")
                print(f"   평균 급여: {stats['average_salary']:,}원")
                print(f"   분석 직무: {stats['analyzed_job_roles']}개")
                
                # 트렌드 분포
                trend_dist = result['trend_distribution']
                print(f"   트렌드 분포: {dict(list(trend_dist.items())[:3])}")
                
                # 인사이트
                insights = result['market_insights']
                print(f"   주요 인사이트:")
                for insight in insights[:2]:
                    print(f"     - {insight}")
                
                return True
            else:
                print(f"❌ 시장 트렌드 조회 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 시장 트렌드 조회 오류: {e}")
            return False
    
    def test_competitive_analysis(self) -> bool:
        """경쟁력 분석 테스트"""
        print("\n🏆 경쟁력 분석 테스트...")
        
        test_data = {
            "EmployeeNumber": 1001,
            "JobRole": "Sales Executive",
            "MonthlyIncome": 5000,
            "YearsAtCompany": 3,
            "Education": 3,  # Bachelor
            "JobSatisfaction": 3
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/market/competitive_analysis",
                json=test_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 경쟁력 분석 성공")
                print(f"   직원 ID: {result['employee_id']}")
                print(f"   경쟁력 점수: {result['competitiveness_score']:.3f}")
                print(f"   경쟁력 등급: {result['competitiveness_grade']}")
                
                detailed = result['detailed_scores']
                print(f"   세부 점수:")
                print(f"     - 급여 경쟁력: {detailed['salary_competitiveness']:.3f}")
                print(f"     - 경험 점수: {detailed['experience_score']:.3f}")
                print(f"     - 교육 점수: {detailed['education_score']:.3f}")
                
                comparison = result['market_comparison']
                print(f"   시장 비교:")
                print(f"     - 현재 급여: {comparison['current_salary']:,}원")
                print(f"     - 시장 평균: {comparison['market_avg_salary']:,}원")
                print(f"     - 급여 격차: {comparison['salary_gap']:,}원")
                
                print(f"   권장사항: {len(result['recommendations'])}개")
                
                return True
            else:
                print(f"❌ 경쟁력 분석 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 경쟁력 분석 오류: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """모든 테스트 실행"""
        print("🧪 Agora API 전체 테스트 시작")
        print("=" * 60)
        
        tests = {
            "health_check": self.test_health_check,
            "individual_analysis": self.test_individual_market_analysis,
            "job_market_analysis": self.test_job_market_analysis,
            "batch_analysis": self.test_batch_analysis,
            "market_report": self.test_market_report,
            "market_trends": self.test_market_trends,
            "competitive_analysis": self.test_competitive_analysis
        }
        
        results = {}
        
        for test_name, test_func in tests.items():
            try:
                results[test_name] = test_func()
                time.sleep(1)  # API 호출 간격
            except Exception as e:
                print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
                results[test_name] = False
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("📊 테스트 결과 요약")
        print("=" * 60)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"{test_name:20s}: {status}")
        
        print(f"\n전체 결과: {passed}/{total} 테스트 통과 ({passed/total:.1%})")
        
        return results

def main():
    """메인 실행 함수"""
    print("🧪 Agora HR Market Analysis API 테스트")
    print("=" * 60)
    
    # 테스터 초기화
    tester = AgoraAPITester()
    
    # 모든 테스트 실행
    results = tester.run_all_tests()
    
    # 최종 결과
    if all(results.values()):
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("\n⚠️ 일부 테스트가 실패했습니다. 서버 상태를 확인해주세요.")

if __name__ == "__main__":
    main()