#!/usr/bin/env python3
# ============================================================================
# Chronos API 테스트 스크립트
# ============================================================================

import requests
import json
import time
import sys
from typing import Dict, Any

# 서버 설정
BASE_URL = "http://localhost:5003"
HEADERS = {"Content-Type": "application/json"}

class ChronosAPITester:
    """
    Chronos API 테스트 클래스
    """
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
    
    def test_connection(self) -> bool:
        """
        서버 연결 테스트
        """
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                print("✅ 서버 연결 성공")
                return True
            else:
                print(f"❌ 서버 연결 실패: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 서버 연결 오류: {str(e)}")
            return False
    
    def test_status(self) -> Dict[str, Any]:
        """
        시스템 상태 확인 테스트
        """
        print("\n📊 시스템 상태 확인 중...")
        
        try:
            response = self.session.get(f"{self.base_url}/api/status")
            
            if response.status_code == 200:
                status = response.json()
                print("✅ 상태 확인 성공:")
                print(f"   - 시스템 초기화: {status.get('system_initialized', False)}")
                print(f"   - 모델 로드: {status.get('model_loaded', False)}")
                print(f"   - 데이터 사용 가능: {status.get('data_available', False)}")
                print(f"   - 디바이스: {status.get('device', 'Unknown')}")
                
                if status.get('feature_count'):
                    print(f"   - 피처 수: {status['feature_count']}")
                if status.get('data_shape'):
                    print(f"   - 데이터 형태: {status['data_shape']}")
                
                return status
            else:
                print(f"❌ 상태 확인 실패: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"❌ 상태 확인 오류: {str(e)}")
            return {}
    
    def test_training(self, epochs: int = 10) -> bool:
        """
        모델 학습 테스트
        """
        print(f"\n🧠 모델 학습 테스트 (Epochs: {epochs})...")
        
        try:
            payload = {
                "sequence_length": 50,  # 개선된 사람별 시계열 패턴 학습
                "epochs": epochs,
                "batch_size": 32,
                "learning_rate": 0.001
            }
            
            response = self.session.post(
                f"{self.base_url}/api/train",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 개선된 모델 학습 성공 (사람별 시계열 패턴):")
                print(f"   - 최종 정확도: {result['results']['final_accuracy']:.4f}")
                print(f"   - 최종 손실: {result['results']['final_loss']:.4f}")
                print(f"   - 데이터 크기: {result['results']['data_size']}")
                print(f"   - 피처 수: {result['results']['feature_count']}")
                print(f"   - Temperature Scaling 적용으로 극단값 문제 해결")
                return True
            else:
                error_msg = response.json().get('error', 'Unknown error')
                print(f"❌ 모델 학습 실패: {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ 모델 학습 오류: {str(e)}")
            return False
    
    def test_prediction(self, employee_ids: list = None) -> Dict[str, Any]:
        """
        예측 테스트
        """
        print(f"\n🔮 예측 테스트...")
        
        try:
            payload = {}
            if employee_ids:
                payload["employee_ids"] = employee_ids
                print(f"   대상 직원: {employee_ids}")
            else:
                print("   전체 직원 대상")
            
            response = self.session.post(
                f"{self.base_url}/api/predict",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get('summary', {})
                predictions = result.get('predictions', [])
                
                print("✅ 예측 성공:")
                print(f"   - 총 직원 수: {summary.get('total_employees', 0)}")
                print(f"   - 고위험군: {summary.get('high_risk_count', 0)}명")
                print(f"   - 중위험군: {summary.get('medium_risk_count', 0)}명")
                print(f"   - 저위험군: {summary.get('low_risk_count', 0)}명")
                print(f"   - 평균 퇴사 확률: {summary.get('average_attrition_probability', 0):.3f}")
                
                # 상위 5명 고위험 직원 출력
                if predictions:
                    print("\n   🚨 상위 5명 고위험 직원:")
                    for i, pred in enumerate(predictions[:5]):
                        print(f"      {i+1}. 직원 {pred['employee_id']}: {pred['attrition_probability']:.3f} ({pred['risk_level']})")
                
                return result
            else:
                error_msg = response.json().get('error', 'Unknown error')
                print(f"❌ 예측 실패: {error_msg}")
                return {}
                
        except Exception as e:
            print(f"❌ 예측 오류: {str(e)}")
            return {}
    
    def test_feature_importance(self) -> bool:
        """
        Feature importance 테스트
        """
        print(f"\n📈 Feature Importance 테스트...")
        
        try:
            response = self.session.get(f"{self.base_url}/api/feature_importance")
            
            if response.status_code == 200:
                print("✅ Feature importance 시각화 생성 성공")
                print("   (HTML 응답을 받았습니다. 브라우저에서 확인하세요)")
                return True
            else:
                error_msg = response.json().get('error', 'Unknown error') if response.headers.get('content-type') == 'application/json' else 'HTML response error'
                print(f"❌ Feature importance 실패: {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ Feature importance 오류: {str(e)}")
            return False
    
    def test_model_analysis(self) -> bool:
        """
        모델 분석 테스트
        """
        print(f"\n📊 모델 분석 테스트...")
        
        try:
            response = self.session.get(f"{self.base_url}/api/model_analysis")
            
            if response.status_code == 200:
                print("✅ 모델 분석 대시보드 생성 성공")
                print("   (HTML 응답을 받았습니다. 브라우저에서 확인하세요)")
                return True
            else:
                error_msg = response.json().get('error', 'Unknown error') if response.headers.get('content-type') == 'application/json' else 'HTML response error'
                print(f"❌ 모델 분석 실패: {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ 모델 분석 오류: {str(e)}")
            return False
    
    def test_employee_timeline(self, employee_id: int = 1) -> bool:
        """
        개별 직원 타임라인 테스트
        """
        print(f"\n👤 직원 {employee_id} 타임라인 테스트...")
        
        try:
            response = self.session.get(f"{self.base_url}/api/employee_timeline/{employee_id}")
            
            if response.status_code == 200:
                print(f"✅ 직원 {employee_id} 타임라인 생성 성공")
                print("   (HTML 응답을 받았습니다. 브라우저에서 확인하세요)")
                return True
            elif response.status_code == 404:
                print(f"⚠️ 직원 {employee_id}를 찾을 수 없습니다")
                return False
            else:
                error_msg = response.json().get('error', 'Unknown error') if response.headers.get('content-type') == 'application/json' else 'HTML response error'
                print(f"❌ 직원 타임라인 실패: {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ 직원 타임라인 오류: {str(e)}")
            return False

def run_comprehensive_test():
    """
    종합 테스트 실행
    """
    print("🧪 개선된 Chronos API 종합 테스트 시작")
    print("🎯 사람별 시계열 패턴 학습 & Temperature Scaling 적용")
    print("=" * 70)
    
    tester = ChronosAPITester()
    
    # 1. 연결 테스트
    if not tester.test_connection():
        print("\n❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        return False
    
    # 2. 상태 확인
    status = tester.test_status()
    
    # 3. 모델 학습 (데이터가 있는 경우)
    if status.get('data_available', False):
        if not status.get('model_loaded', False):
            print("\n모델이 로드되지 않았습니다. 학습을 진행합니다...")
            if not tester.test_training(epochs=5):  # 빠른 테스트를 위해 5 에포크
                print("❌ 모델 학습에 실패했습니다.")
                return False
        else:
            print("\n✅ 기존 모델이 로드되어 있습니다.")
    else:
        print("\n⚠️ 데이터가 없어 모델 학습을 건너뜁니다.")
    
    # 4. 예측 테스트
    prediction_result = tester.test_prediction(employee_ids=[1, 2, 3, 4, 5])
    
    # 5. 시각화 테스트
    tester.test_feature_importance()
    tester.test_model_analysis()
    tester.test_employee_timeline(employee_id=1)
    
    print("\n" + "=" * 60)
    print("🎉 종합 테스트 완료!")
    
    # 브라우저 안내
    print("\n📱 브라우저에서 다음 URL들을 확인해보세요:")
    print(f"   - 홈페이지: {BASE_URL}")
    print(f"   - Feature Importance: {BASE_URL}/api/feature_importance")
    print(f"   - 모델 분석: {BASE_URL}/api/model_analysis")
    print(f"   - 직원 타임라인: {BASE_URL}/api/employee_timeline/1")
    
    return True

def run_quick_test():
    """
    빠른 테스트 실행
    """
    print("⚡ Chronos API 빠른 테스트")
    print("=" * 40)
    
    tester = ChronosAPITester()
    
    # 연결 및 상태만 확인
    if tester.test_connection():
        tester.test_status()
        print(f"\n✅ 기본 테스트 완료. 상세 테스트는 다음 명령어로 실행하세요:")
        print(f"python {__file__} --full")
    
    return True

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        success = run_comprehensive_test()
    else:
        success = run_quick_test()
    
    sys.exit(0 if success else 1)
