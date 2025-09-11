#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentio HR Text Analysis API 테스트 스크립트
"""

import requests
import json
import time
from typing import Dict, Any

class SentioAPITester:
    """Sentio API 테스트 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:5003"):
        """
        API 테스터 초기화
        
        Args:
            base_url: Sentio API 서버 주소
        """
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health_check(self) -> bool:
        """헬스체크 테스트"""
        print("🔍 헬스체크 테스트...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ 서버 상태: {health_data['status']}")
                print(f"   컴포넌트 상태: {health_data['components']}")
                return True
            else:
                print(f"❌ 헬스체크 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 헬스체크 오류: {e}")
            return False
    
    def test_text_analysis(self) -> bool:
        """텍스트 분석 API 테스트"""
        print("\n📝 텍스트 분석 테스트...")
        
        test_data = {
            "text": "이번 분기 업무량이 너무 많아서 정말 힘들었습니다. 야근도 자주 하게 되고 개인 시간이 거의 없어서 번아웃이 올 것 같습니다. 그래도 팀원들과 협력해서 프로젝트를 완료할 수 있었습니다.",
            "employee_id": "test_001",
            "text_type": "WEEKLY_SURVEY"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/analyze/text",
                json=test_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 텍스트 분석 성공 (JD-R 모델 적용)")
                print(f"   키워드: {result['keywords'][:5]}...")  # 처음 5개만
                print(f"   감정 점수: {result['sentiment_score']:.2f}")
                print(f"   심리적 위험 점수: {result['attrition_risk_score']:.2f}")
                print(f"   위험 수준: {result.get('risk_level', 'N/A')}")
                print(f"   이직 예측: {'예' if result.get('attrition_prediction', 0) else '아니오'}")
                
                # JD-R 지표 출력
                if 'jd_r_indicators' in result:
                    jdr = result['jd_r_indicators']
                    print(f"   직무 요구 점수: {jdr.get('job_demands_score', 0):.3f}")
                    print(f"   자원 결핍 점수: {jdr.get('job_resources_deficiency_score', 0):.3f}")
                
                print(f"   위험 요소: {result['risk_factors'][:3]}...")  # 처음 3개만
                return True
            else:
                print(f"❌ 텍스트 분석 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 텍스트 분석 오류: {e}")
            return False
    
    def test_keyword_analysis(self) -> bool:
        """키워드 분석 API 테스트"""
        print("\n🔍 키워드 분석 테스트...")
        
        test_data = {
            "min_frequency": 3
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/analyze/keywords",
                json=test_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 키워드 분석 성공")
                print(f"   분석된 컬럼: {result['columns_analyzed']}")
                print(f"   최소 빈도: {result['min_frequency']}")
                
                # 요약 정보 출력
                for col, summary in result['summary'].items():
                    print(f"   [{col}] 퇴직자 키워드: {summary['resigned_total_keywords']}개")
                    print(f"   [{col}] 재직자 키워드: {summary['stayed_total_keywords']}개")
                
                return True
            else:
                print(f"❌ 키워드 분석 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 키워드 분석 오류: {e}")
            return False
    
    def test_risk_analysis(self) -> bool:
        """퇴직 위험 분석 API 테스트"""
        print("\n⚠️ 퇴직 위험 분석 테스트...")
        
        test_data = {
            "texts": [
                {
                    "employee_id": "emp_001",
                    "text": "업무량이 너무 많아서 감당이 안 됩니다. 매일 야근하는 문화가 힘들고 개인 생활이 전혀 없습니다.",
                    "text_type": "WEEKLY_SURVEY"
                },
                {
                    "employee_id": "emp_002", 
                    "text": "팀워크가 좋고 새로운 기술을 배울 수 있어서 만족합니다. 회사에서 성장할 수 있는 기회가 많아 좋습니다.",
                    "text_type": "SELF_REVIEW"
                }
            ]
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/analyze/risk",
                json=test_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 위험 분석 성공")
                print(f"   분석 대상: {result['summary']['total_analyzed']}명")
                print(f"   평균 위험 점수: {result['summary']['average_risk_score']:.2f}")
                print(f"   고위험: {result['summary']['high_risk_count']}명")
                print(f"   중위험: {result['summary']['medium_risk_count']}명")
                print(f"   저위험: {result['summary']['low_risk_count']}명")
                
                return True
            else:
                print(f"❌ 위험 분석 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 위험 분석 오류: {e}")
            return False
    
    def test_text_generation(self) -> bool:
        """텍스트 생성 API 테스트"""
        print("\n✍️ 텍스트 생성 테스트...")
        
        # 샘플 직원 데이터
        test_data = {
            "employee_data": {
                "EmployeeNumber": 1001,
                "JobRole": "Software Engineer",
                "YearsAtCompany": 3,
                "PerformanceRating": 3,
                "JobSatisfaction": 2,
                "JobInvolvement": 2,
                "softmax_Persona_Code": "P01",
                "softmax_Persona": "번아웃 위험군",
                "Attrition": "Yes"
            },
            "text_type": "SELF_REVIEW"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/generate/text",
                json=test_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 텍스트 생성 성공")
                print(f"   직원 ID: {result['employee_id']}")
                print(f"   페르소나: {result['persona_name']} ({result['persona_code']})")
                print(f"   텍스트 타입: {result['text_type']}")
                print(f"   생성된 텍스트: {result['generated_text'][:100]}...")
                print(f"   사용된 키워드: {result['keywords_used'][:3]}...")
                return True
            else:
                print(f"❌ 텍스트 생성 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 텍스트 생성 오류: {e}")
            return False
    
    def test_personas_info(self) -> bool:
        """페르소나 정보 조회 테스트"""
        print("\n👥 페르소나 정보 조회 테스트...")
        
        try:
            response = self.session.get(f"{self.base_url}/data/personas")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 페르소나 조회 성공")
                print(f"   전체 직원 수: {result['total_employees']}명")
                print(f"   전체 퇴직자 수: {result['total_attrition']}명")
                print(f"   페르소나 종류: {len(result['personas'])}개")
                
                # 상위 3개 페르소나 정보 출력
                for persona in result['personas'][:3]:
                    print(f"   - {persona['persona_name']}: {persona['total_count']}명 (퇴직률: {persona['attrition_rate']:.1%})")
                
                return True
            else:
                print(f"❌ 페르소나 조회 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 페르소나 조회 오류: {e}")
            return False
    
    def test_comprehensive_report(self) -> bool:
        """개별 직원 종합 레포트 생성 테스트 (LLM 선택적 사용)"""
        print("\n📋 개별 직원 종합 레포트 생성 테스트...")
        
        # 가상의 한 직원의 모든 워커 분석 결과 데이터
        test_data = {
            "employee_id": "1001",
            "worker_results": {
                "structura": {
                    "attrition_probability": 0.75,
                    "prediction": "High Risk",
                    "confidence": 0.85,
                    "top_risk_factors": ["낮은 급여", "과도한 업무량", "승진 기회 부족"]
                },
                "cognita": {
                    "overall_risk_score": 0.68,
                    "risk_category": "Medium-High",
                    "network_centrality": 0.3,
                    "risk_factors": ["팀 내 갈등", "소통 부족"]
                },
                "chronos": {
                    "prediction": "Likely to Leave",
                    "probability": 0.72,
                    "trend": "Declining",
                    "risk_indicators": ["성과 하락", "참여도 감소"]
                },
                "sentio": {
                    "psychological_risk_score": 0.8,
                    "risk_level": "HIGH",
                    "jd_r_indicators": {
                        "job_demands_score": 0.9,
                        "job_resources_deficiency_score": 0.4
                    },
                    "job_demands_matches": ["업무량", "스트레스", "야근", "압박"],
                    "job_resources_deficiency_matches": ["지원부족", "피드백부족"]
                }
            },
            "use_llm": False  # LLM 사용 안함 (비용 절약)
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/analyze/comprehensive_report",
                json=test_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 개별 직원 종합 레포트 생성 성공 (LLM 미사용)")
                
                # 결과 출력
                assessment = result['comprehensive_assessment']
                print(f"   직원 ID: {result['employee_id']}")
                print(f"   {assessment['risk_indicator']} 종합 위험도: {assessment['overall_risk_level']} ({assessment['overall_risk_score']:.3f})")
                print(f"   신뢰도: {assessment['confidence_level']}")
                
                # 워커별 점수 출력
                worker_scores = result['worker_scores']
                print(f"   Structura: 퇴직확률 {worker_scores['structura']['attrition_probability']:.1%}")
                print(f"   Cognita: 위험도 {worker_scores['cognita']['overall_risk_score']:.3f}")
                print(f"   Chronos: 확률 {worker_scores['chronos']['probability']:.1%}")
                print(f"   Sentio: 심리적위험 {worker_scores['sentio']['psychological_risk_score']:.3f}")
                
                # 주요 우려사항
                concerns = result['primary_concerns'][:3]
                print(f"   주요 우려: {', '.join(concerns)}")
                
                # 규칙 기반 해석 확인
                if 'rule_based_interpretation' in result:
                    print("   📝 규칙 기반 종합 해석 생성됨")
                
                return True
            else:
                print(f"❌ 종합 레포트 생성 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 종합 레포트 테스트 오류: {e}")
            return False
    
    def test_batch_csv_analysis(self) -> bool:
        """CSV 배치 분석 테스트 (LLM 없이, 빠른 처리)"""
        print("\n📊 CSV 배치 분석 테스트...")
        
        # 가상의 대량 텍스트 데이터
        test_data = {
            "text_data_list": [
                {
                    "employee_id": "emp_001",
                    "self_review": "이번 분기 업무량이 너무 많아서 힘들었습니다. 야근도 자주 하게 되고 스트레스가 심합니다.",
                    "peer_feedback": "동료와의 협업은 좋았지만 업무 분담에 문제가 있었습니다.",
                    "weekly_survey": "번아웃이 올 것 같아 걱정됩니다. 지원이 부족합니다."
                },
                {
                    "employee_id": "emp_002",
                    "self_review": "팀원들과 잘 협력하고 있고 프로젝트도 성공적으로 완료했습니다.",
                    "peer_feedback": "항상 긍정적이고 도움이 많이 됩니다.",
                    "weekly_survey": "전반적으로 만족스럽고 성장하고 있다고 느낍니다."
                },
                {
                    "employee_id": "emp_003",
                    "self_review": "업무는 적당하지만 승진 기회가 부족한 것 같습니다.",
                    "peer_feedback": "성실하게 일하지만 소극적인 면이 있습니다.",
                    "weekly_survey": "현재 상태에서 큰 변화는 없을 것 같습니다."
                }
            ],
            "output_filename": "test_sentio_batch_analysis.csv"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/analyze/batch_csv",
                json=test_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ CSV 배치 분석 성공 (LLM 미사용)")
                
                # 처리 통계 출력
                stats = result['processing_stats']
                print(f"   처리된 직원: {stats['total_processed']}명")
                print(f"   처리 시간: {stats['processing_time_seconds']}초")
                print(f"   처리 속도: {stats['records_per_second']:.1f}명/초")
                print(f"   출력 파일: {result['output_file']}")
                
                # 분석 요약 출력
                summary = result['analysis_summary']
                print(f"   위험도 분포: {summary['risk_distribution']}")
                print(f"   평균 심리적 위험: {summary['average_scores']['psychological_risk_score']:.3f}")
                print(f"   이직 예측: 예상 {summary['prediction_distribution']['will_leave']}명, 잔류 {summary['prediction_distribution']['will_stay']}명")
                
                print(f"   💡 {result['message']}")
                
                return True
            else:
                print(f"❌ CSV 배치 분석 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ CSV 배치 분석 테스트 오류: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """모든 테스트 실행"""
        print("🧪 Sentio API 전체 테스트 시작")
        print("=" * 60)
        
        tests = {
            "health_check": self.test_health_check,
            "text_analysis": self.test_text_analysis,
            "keyword_analysis": self.test_keyword_analysis,
            "risk_analysis": self.test_risk_analysis,
            "comprehensive_report": self.test_comprehensive_report,  # 개별 직원 종합 레포트
            "batch_csv_analysis": self.test_batch_csv_analysis,  # CSV 배치 분석 (LLM 없이)
            "text_generation": self.test_text_generation,
            "personas_info": self.test_personas_info
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
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name:20s}: {status}")
        
        print(f"\n전체 결과: {passed}/{total} 테스트 통과 ({passed/total:.1%})")
        
        return results

def main():
    """메인 실행 함수"""
    print("🧪 Sentio HR Text Analysis API 테스트")
    print("=" * 60)
    
    # 테스터 초기화
    tester = SentioAPITester()
    
    # 모든 테스트 실행
    results = tester.run_all_tests()
    
    # 최종 결과
    if all(results.values()):
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("\n⚠️ 일부 테스트가 실패했습니다. 서버 상태를 확인해주세요.")

if __name__ == "__main__":
    main()
