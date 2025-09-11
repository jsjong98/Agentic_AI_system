# -*- coding: utf-8 -*-
"""
Sentio API 테스트 스크립트 (키워드 기반)
페르소나 정보 제거 후 키워드 기반으로 동작하는 API 테스트
"""

import requests
import json
import time


class SentioAPITester:
    """Sentio API 테스트 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:5003"):
        """
        테스터 초기화
        
        Args:
            base_url: Sentio API 서버 주소
        """
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health_check(self) -> bool:
        """서비스 상태 확인 테스트"""
        print("🏥 서비스 상태 확인 테스트...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 서비스 상태 확인 성공")
                print(f"   서비스: {result['service']}")
                print(f"   상태: {result['status']}")
                print(f"   구성 요소:")
                for component, status in result['components'].items():
                    status_icon = "✅" if status else "❌"
                    print(f"     - {component}: {status_icon}")
                return True
            else:
                print(f"❌ 서비스 상태 확인 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 서비스 상태 확인 오류: {e}")
            return False
    
    def test_text_analysis(self) -> bool:
        """텍스트 분석 API 테스트"""
        print("\n📝 텍스트 분석 테스트...")
        
        test_data = {
            "text": "최근 업무량이 너무 많아서 스트레스를 받고 있습니다. 워라밸이 무너져서 개인 생활이 전혀 없어요. 번아웃이 올 것 같아서 걱정됩니다.",
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
                print("✅ 텍스트 분석 성공")
                print(f"   직원 ID: {result['employee_id']}")
                print(f"   텍스트 타입: {result['text_type']}")
                print(f"   키워드: {result['keywords'][:5]}...")
                print(f"   감정 점수: {result['sentiment_score']:.2f}")
                print(f"   퇴직 위험 점수: {result['attrition_risk_score']:.2f}")
                print(f"   위험 요소: {result['risk_factors'][:3]}...")
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
            "min_frequency": 3,
            "text_columns": ["SELF_REVIEW", "PEER_FEEDBACK"]
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
                print(f"   최소 빈도: {result['min_frequency']}")
                print(f"   분석된 컬럼: {result['columns_analyzed']}")
                print(f"   차별적 키워드 발견됨")
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
                    "employee_id": "001",
                    "text": "업무량이 너무 많아서 번아웃이 올 것 같습니다. 워라밸이 완전히 무너졌어요.",
                    "text_type": "SELF_REVIEW"
                },
                {
                    "employee_id": "002", 
                    "text": "팀워크가 좋고 성장할 수 있는 기회가 많아서 만족합니다.",
                    "text_type": "WEEKLY_SURVEY"
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
                print(f"   분석된 직원 수: {result['summary']['total_analyzed']}명")
                print(f"   평균 위험 점수: {result['summary']['average_risk_score']:.2f}")
                print(f"   고위험군: {result['summary']['high_risk_count']}명")
                print(f"   중위험군: {result['summary']['medium_risk_count']}명")
                print(f"   저위험군: {result['summary']['low_risk_count']}명")
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
        
        # 키워드 기반 텍스트 생성 데이터
        test_data = {
            "keywords": [
                "번아웃", "소진", "업무량", "압박", "스트레스",
                "워라밸", "개인생활", "피로", "무기력"
            ],
            "text_type": "SELF_REVIEW",
            "employee_id": "test_001"
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
    
    def test_keywords_info(self) -> bool:
        """사용 가능한 키워드 조회 테스트"""
        print("\n🔑 사용 가능한 키워드 조회 테스트...")
        
        try:
            response = self.session.get(f"{self.base_url}/data/keywords")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 키워드 조회 성공")
                print(f"   전체 카테고리 수: {result['total_categories']}개")
                
                # 상위 3개 카테고리 정보 출력
                categories = list(result['keyword_categories'].keys())[:3]
                for category in categories:
                    keywords = result['keyword_categories'][category]
                    print(f"   - {category}: {len(keywords)}개 키워드")
                
                return True
            else:
                print(f"❌ 키워드 조회 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 키워드 조회 오류: {e}")
            return False
    
    def test_comprehensive_report(self) -> bool:
        """개별 직원 종합 레포트 생성 테스트 (LLM 선택적 사용)"""
        print("\n📊 종합 레포트 생성 테스트...")
        
        # 모든 워커 에이전트의 분석 결과를 시뮬레이션
        test_data = {
            "employee_id": "test_001",
            "use_llm": False,  # LLM 사용하지 않음 (빠른 테스트)
            "worker_results": {
                "structura": {
                    "attrition_probability": 0.75,
                    "risk_level": "high",
                    "key_factors": ["job_satisfaction", "work_life_balance"]
                },
                "cognita": {
                    "network_centrality": 0.3,
                    "influence_score": 0.4,
                    "collaboration_level": "low"
                },
                "chronos": {
                    "trend_analysis": "declining",
                    "seasonal_patterns": ["stress_peaks"],
                    "prediction_confidence": 0.8
                },
                "sentio": {
                    "sentiment_score": -0.6,
                    "risk_keywords": ["번아웃", "스트레스", "워라밸"],
                    "psychological_state": "burnout_risk"
                }
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/analyze/comprehensive_report",
                json=test_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 종합 레포트 생성 성공")
                print(f"   직원 ID: {result['employee_id']}")
                print(f"   전체 위험 점수: {result['overall_risk_score']:.2f}")
                print(f"   위험 수준: {result['risk_level']}")
                print(f"   주요 위험 요소: {result['key_risk_factors'][:3]}...")
                return True
            else:
                print(f"❌ 종합 레포트 생성 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 종합 레포트 생성 오류: {e}")
            return False
    
    def test_batch_csv_analysis(self) -> bool:
        """대량 CSV 분석 테스트 (LLM 없이)"""
        print("\n📈 대량 CSV 분석 테스트...")
        
        # 샘플 텍스트 데이터 목록
        test_data = {
            "text_data_list": [
                {
                    "employee_id": "001",
                    "text": "업무량이 너무 많아서 스트레스를 받고 있습니다. 번아웃이 올 것 같아요.",
                    "text_type": "SELF_REVIEW"
                },
                {
                    "employee_id": "002",
                    "text": "팀워크가 좋고 성장 기회가 많아서 만족합니다.",
                    "text_type": "PEER_FEEDBACK"
                },
                {
                    "employee_id": "003",
                    "text": "워라밸이 좋지 않아서 개인 생활이 힘듭니다.",
                    "text_type": "WEEKLY_SURVEY"
                }
            ],
            "output_filename": "test_batch_analysis.csv"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/analyze/batch_csv",
                json=test_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 대량 CSV 분석 성공")
                print(f"   처리된 데이터 수: {result['processing_stats']['total_processed']}개")
                print(f"   처리 시간: {result['processing_stats']['processing_time_seconds']}초")
                print(f"   초당 처리량: {result['processing_stats']['records_per_second']:.1f}개/초")
                print(f"   출력 파일: {result['output_file']}")
                return True
            else:
                print(f"❌ 대량 CSV 분석 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 대량 CSV 분석 오류: {e}")
            return False
    
    def run_all_tests(self) -> dict:
        """모든 테스트 실행"""
        print("🚀 Sentio API 전체 테스트 시작")
        print("=" * 60)
        
        tests = {
            "health_check": self.test_health_check,
            "text_analysis": self.test_text_analysis,
            "keyword_analysis": self.test_keyword_analysis,
            "risk_analysis": self.test_risk_analysis,
            "text_generation": self.test_text_generation,
            "keywords_info": self.test_keywords_info,
            "comprehensive_report": self.test_comprehensive_report,
            "batch_csv_analysis": self.test_batch_csv_analysis
        }
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests.items():
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
                time.sleep(1)  # API 호출 간격 조절
            except Exception as e:
                print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
                results[test_name] = False
        
        print("\n" + "=" * 60)
        print("📊 테스트 결과 요약")
        print("=" * 60)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        print(f"\n전체 결과: {passed}/{total} 테스트 통과 ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 모든 테스트가 성공적으로 완료되었습니다!")
        else:
            print("⚠️ 일부 테스트가 실패했습니다. 로그를 확인해주세요.")
        
        return results


def main():
    """메인 실행 함수"""
    print("Sentio API 테스트 도구 (키워드 기반)")
    print("페르소나 정보 제거 후 키워드 기반으로 동작하는 API 테스트")
    
    # 테스터 초기화
    tester = SentioAPITester()
    
    # 전체 테스트 실행
    results = tester.run_all_tests()
    
    return results


if __name__ == "__main__":
    main()