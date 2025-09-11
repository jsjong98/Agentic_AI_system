#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structura HR 예측 Flask API 테스트 스크립트
xAI 기능 테스트 포함
"""

import requests
import json
import time
from typing import Dict, List

class StructuraAPITester:
    """Structura API 테스트 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:5003"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def test_health_check(self) -> bool:
        """헬스체크 테스트"""
        print("1. 헬스체크 테스트")
        print("-" * 30)
        
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 상태: {data['status']}")
                print(f"✅ 모델 상태: {data['model_status']}")
                print(f"✅ SHAP 사용 가능: {data['xai_status']['shap_available']}")
                print(f"✅ LIME 사용 가능: {data['xai_status']['lime_available']}")
                print(f"✅ 의존성 확인:")
                for dep, available in data['dependencies'].items():
                    print(f"    {dep}: {'✅' if available else '❌'}")
                return True
            else:
                print(f"❌ 헬스체크 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 헬스체크 오류: {e}")
            return False
    
    def test_model_training(self) -> bool:
        """모델 훈련 테스트"""
        print("\n2. 모델 훈련 테스트")
        print("-" * 30)
        
        try:
            payload = {
                "optimize_hyperparameters": False,  # 빠른 테스트를 위해 False
                "use_sampling": True  # 클래스 불균형 해결 사용
            }
            
            print("모델 훈련 시작... (시간이 걸릴 수 있습니다)")
            start_time = time.time()
            
            response = self.session.post(f"{self.base_url}/api/train", json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 훈련 완료 (소요시간: {end_time - start_time:.1f}초)")
                print(f"✅ 상태: {data['status']}")
                print(f"✅ 메시지: {data['message']}")
                print(f"✅ 성능 지표:")
                for metric, value in data['metrics'].items():
                    print(f"    {metric}: {value:.4f}")
                print(f"✅ xAI 활성화:")
                for xai_type, enabled in data['xai_enabled'].items():
                    print(f"    {xai_type}: {'✅' if enabled else '❌'}")
                return True
            else:
                print(f"❌ 모델 훈련 실패: {response.status_code}")
                if response.content:
                    print(f"   오류: {response.json()}")
                return False
                
        except Exception as e:
            print(f"❌ 모델 훈련 오류: {e}")
            return False
    
    def test_prediction(self) -> Dict:
        """이직 예측 테스트 (새로운 API 형식)"""
        print("\n3. 이직 예측 테스트")
        print("-" * 30)
        
        # 테스트용 직원 데이터 (노트북 기반 변수 포함)
        test_employee = {
            "EmployeeNumber": "TEST_001",
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
            "OverTime": "No",
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 3,
            "StockOptionLevel": 1,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 2,
            "YearsAtCompany": 5,
            "YearsInCurrentRole": 3,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 2,
            "DailyRate": 1000,
            "HourlyRate": 50,
            "MonthlyRate": 15000,
            "NumCompaniesWorked": 2,
            "PercentSalaryHike": 15,
            "TotalWorkingYears": 10
        }
        
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/api/predict", json=test_employee)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 예측 완료 (소요시간: {end_time - start_time:.3f}초)")
                print(f"  직원 번호: {data.get('employee_number', 'N/A')}")
                print(f"  이직 확률: {data['attrition_probability']:.3f}")
                print(f"  위험 범주: {data['risk_category']}")
                
                # XAI 설명 요약 출력
                if 'explanation' in data and 'individual_explanation' in data['explanation']:
                    exp = data['explanation']['individual_explanation']
                    if 'top_risk_factors' in exp and len(exp['top_risk_factors']) > 0:
                        print(f"  주요 위험 요인:")
                        for factor in exp['top_risk_factors'][:3]:
                            print(f"    - {factor['feature']}: {factor['impact']:.3f}")
                
                return data
            else:
                print(f"❌ 예측 실패: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"❌ 예측 오류: {e}")
            return {}
    
    def test_explanation(self, employee_data: Dict) -> Dict:
        """예측 설명 테스트 (xAI)"""
        print("\n4. 예측 설명 테스트 (xAI)")
        print("-" * 40)
        
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/api/explain", json=employee_data)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 설명 생성 완료 (소요시간: {end_time - start_time:.3f}초)")
                
                # 피처 중요도
                if data.get('feature_importance'):
                    print(f"  📊 피처 중요도 (상위 5개):")
                    sorted_features = sorted(
                        data['feature_importance'].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                    for feature, importance in sorted_features:
                        print(f"    {feature}: {importance:.4f}")
                
                # SHAP 값
                if data.get('shap_values'):
                    print(f"  🔍 SHAP 분석 (상위 5개):")
                    sorted_shap = sorted(
                        data['shap_values'].items(), 
                        key=lambda x: abs(x[1]), 
                        reverse=True
                    )[:5]
                    for feature, shap_val in sorted_shap:
                        direction = "위험 증가" if shap_val > 0 else "위험 감소"
                        print(f"    {feature}: {shap_val:.4f} ({direction})")
                
                # LIME 설명
                if data.get('lime_explanation'):
                    print(f"  🍋 LIME 분석:")
                    lime_data = data['lime_explanation']
                    print(f"    주요 피처: {', '.join(lime_data['features'][:3])}")
                
                # 위험/보호 요인
                if data.get('top_risk_factors'):
                    print(f"  ⚠️  주요 위험 요인:")
                    for factor in data['top_risk_factors'][:3]:
                        print(f"    - {factor['feature']}: {factor['impact']:.4f}")
                
                if data.get('top_protective_factors'):
                    print(f"  🛡️  주요 보호 요인:")
                    for factor in data['top_protective_factors'][:3]:
                        print(f"    - {factor['feature']}: {factor['impact']:.4f}")
                
                return data
            else:
                print(f"❌ 설명 생성 실패: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"❌ 설명 생성 오류: {e}")
            return {}
    
    def test_feature_importance(self) -> bool:
        """피처 중요도 테스트"""
        print("\n5. 피처 중요도 테스트")
        print("-" * 30)
        
        try:
            response = self.session.get(f"{self.base_url}/api/feature-importance?top_n=10")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 피처 중요도 조회 완료")
                print(f"  전체 피처 수: {data['total_features']}")
                print(f"  상위 {data['top_n']}개 피처:")
                
                for item in data['feature_importance']:
                    print(f"    {item['rank']:2d}. {item['feature']:<25}: {item['importance']:.4f}")
                
                return True
            else:
                print(f"❌ 피처 중요도 조회 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 피처 중요도 조회 오류: {e}")
            return False
    
    def test_model_info(self) -> bool:
        """모델 정보 테스트"""
        print("\n6. 모델 정보 테스트")
        print("-" * 30)
        
        try:
            response = self.session.get(f"{self.base_url}/api/model/info")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 모델 정보 조회 완료")
                print(f"  모델 로딩 상태: {'✅' if data['model_loaded'] else '❌'}")
                print(f"  피처 수: {data['feature_count']}")
                print(f"  최적 임계값: {data['optimal_threshold']:.3f}")
                print(f"  클래스 가중치: {data['scale_pos_weight']:.3f}")
                print(f"  xAI 기능:")
                for xai_type, available in data['xai_capabilities'].items():
                    if xai_type != 'feature_importance':
                        print(f"    {xai_type}: {'✅' if available else '❌'}")
                
                return True
            else:
                print(f"❌ 모델 정보 조회 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 모델 정보 조회 오류: {e}")
            return False
    
    def test_batch_prediction(self) -> bool:
        """배치 예측 테스트"""
        print("\n7. 배치 예측 테스트")
        print("-" * 30)
        
        # 테스트용 배치 데이터
        batch_data = [
            {
                "EmployeeNumber": "BATCH_001",
                "Age": 25,
                "Department": "Research & Development",
                "JobSatisfaction": 1,
                "OverTime": "Yes",
                "MonthlyIncome": 3000,
                "WorkLifeBalance": 1,
                "StockOptionLevel": 0,
                "YearsAtCompany": 1
            },
            {
                "EmployeeNumber": "BATCH_002", 
                "Age": 45,
                "Department": "Sales",
                "JobSatisfaction": 4,
                "OverTime": "No",
                "MonthlyIncome": 8000,
                "WorkLifeBalance": 3,
                "StockOptionLevel": 2,
                "YearsAtCompany": 10
            }
        ]
        
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/api/predict/batch", json=batch_data)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 배치 예측 완료 (소요시간: {end_time - start_time:.3f}초)")
                print(f"  처리된 직원 수: {data['statistics']['total_employees']}")
                print(f"  성공한 예측: {data['statistics']['successful_predictions']}")
                print(f"  평균 이직 확률: {data['statistics']['average_probability']:.3f}")
                print(f"  고위험군: {data['statistics']['high_risk_count']}명")
                
                print(f"  개별 결과:")
                for pred in data['predictions']:
                    if 'error' not in pred:
                        print(f"    {pred['employee_number']}: {pred['attrition_probability']:.3f} ({pred['risk_category']})")
                    else:
                        print(f"    {pred['employee_number']}: 오류 - {pred['error']}")
                
                return True
            else:
                print(f"❌ 배치 예측 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 배치 예측 오류: {e}")
            return False
    
    def test_employee_analysis(self) -> bool:
        """개별 직원 심층 분석 테스트"""
        print("\n8. 개별 직원 심층 분석 테스트")
        print("-" * 40)
        
        # 테스트용 직원 데이터
        employee_data = {
            "Age": 28,
            "Department": "Research & Development",
            "JobSatisfaction": 2,
            "OverTime": "Yes",
            "MonthlyIncome": 4000,
            "WorkLifeBalance": 1,
            "StockOptionLevel": 0,
            "YearsAtCompany": 2,
            "EnvironmentSatisfaction": 2,
            "JobInvolvement": 2
        }
        
        employee_number = "ANALYSIS_001"
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/employee/analysis/{employee_number}", 
                json=employee_data
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 심층 분석 완료 (소요시간: {end_time - start_time:.3f}초)")
                print(f"  직원 번호: {data['employee_number']}")
                print(f"  이직 확률: {data['attrition_probability']:.3f}")
                print(f"  위험 범주: {data['risk_category']}")
                
                # 상세 분석 정보
                if 'detailed_analysis' in data:
                    detail = data['detailed_analysis']
                    print(f"  다음 위험도까지 거리: {detail.get('distance_to_next_level', 'N/A')}")
                
                # 권장사항
                if 'recommendations' in data:
                    print(f"  권장사항:")
                    for i, rec in enumerate(data['recommendations'][:3], 1):
                        print(f"    {i}. {rec}")
                
                return True
            else:
                print(f"❌ 심층 분석 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 심층 분석 오류: {e}")
            return False
    
    def test_react_integration_examples(self):
        """React 연동 예시 코드 출력"""
        print(f"\n7. React 연동 예시 코드")
        print("-" * 40)
        
        print("// React 컴포넌트에서 사용 예시")
        print()
        print("// 1. 이직 예측")
        print("const predictAttrition = async (employeeData) => {")
        print("  try {")
        print(f"    const response = await fetch('{self.base_url}/api/predict', {{")
        print("      method: 'POST',")
        print("      headers: { 'Content-Type': 'application/json' },")
        print("      body: JSON.stringify(employeeData)")
        print("    });")
        print("    const prediction = await response.json();")
        print("    console.log('이직 확률:', prediction.attrition_probability);")
        print("    return prediction;")
        print("  } catch (error) {")
        print("    console.error('예측 실패:', error);")
        print("  }")
        print("};")
        print()
        
        print("// 2. 예측 설명 (xAI)")
        print("const explainPrediction = async (employeeData) => {")
        print("  try {")
        print(f"    const response = await fetch('{self.base_url}/api/explain', {{")
        print("      method: 'POST',")
        print("      headers: { 'Content-Type': 'application/json' },")
        print("      body: JSON.stringify(employeeData)")
        print("    });")
        print("    const explanation = await response.json();")
        print("    console.log('SHAP 값:', explanation.shap_values);")
        print("    console.log('위험 요인:', explanation.top_risk_factors);")
        print("    return explanation;")
        print("  } catch (error) {")
        print("    console.error('설명 생성 실패:', error);")
        print("  }")
        print("};")
        print()
        
        print("// 3. 피처 중요도")
        print("const getFeatureImportance = async () => {")
        print("  try {")
        print(f"    const response = await fetch('{self.base_url}/api/feature-importance');")
        print("    const data = await response.json();")
        print("    return data.feature_importance;")
        print("  } catch (error) {")
        print("    console.error('피처 중요도 조회 실패:', error);")
        print("  }")
        print("};")
    
    def run_full_test(self):
        """전체 테스트 실행"""
        print("=" * 60)
        print("🧪 Structura HR 예측 API 전체 테스트 시작")
        print("=" * 60)
        
        # 테스트용 직원 데이터
        test_employee = {
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
            "OverTime": "No",
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 3,
            "StockOptionLevel": 1,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 2,
            "YearsAtCompany": 5,
            "YearsInCurrentRole": 3,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 2
        }
        
        # 1. 헬스체크
        if not self.test_health_check():
            print("\n❌ 헬스체크 실패. 서버가 실행 중인지 확인하세요.")
            return False
        
        # 2. 모델 훈련 (선택적)
        print("\n모델 훈련을 진행하시겠습니까? (y/n): ", end="")
        # 자동 테스트를 위해 기본값 설정
        train_model = True  # input().lower().startswith('y')
        
        if train_model:
            if not self.test_model_training():
                print("\n❌ 모델 훈련 실패")
                return False
        
        # 3. 모델 정보 확인
        if not self.test_model_info():
            print("\n❌ 모델 정보 조회 실패")
            return False
        
        # 4. 예측 테스트
        prediction_result = self.test_prediction()
        if not prediction_result:
            print("\n❌ 예측 테스트 실패")
            return False
        
        # 5. 예측 설명 테스트 (xAI)
        explanation_result = self.test_explanation(test_employee)
        if not explanation_result:
            print("\n⚠️ 예측 설명 실패 (xAI 라이브러리 확인 필요)")
        
        # 6. 피처 중요도 테스트
        if not self.test_feature_importance():
            print("\n❌ 피처 중요도 테스트 실패")
            return False
        
        # 7. 배치 예측 테스트
        if not self.test_batch_prediction():
            print("\n❌ 배치 예측 테스트 실패")
            return False
        
        # 8. 개별 직원 심층 분석 테스트
        if not self.test_employee_analysis():
            print("\n❌ 개별 직원 심층 분석 테스트 실패")
            return False
        
        # 9. React 연동 예시
        self.test_react_integration_examples()
        
        print("\n" + "=" * 60)
        print("✅ 모든 테스트 성공!")
        print("🤖 xAI 기능이 포함된 HR 예측 API 준비 완료!")
        print("=" * 60)
        return True

def main():
    """메인 함수"""
    
    # 서버 URL 설정
    base_url = "http://localhost:5003"
    
    print("Structura HR 예측 API 테스트를 시작합니다...")
    print(f"서버 URL: {base_url}")
    print("xAI 기능 테스트 포함 (SHAP, LIME, Feature Importance)")
    print()
    
    # 테스터 생성 및 실행
    tester = StructuraAPITester(base_url)
    
    try:
        success = tester.run_full_test()
        
        if success:
            print("\n🎉 모든 API 테스트가 성공적으로 완료되었습니다!")
            print("🚀 React 프론트엔드와 연동할 준비가 되었습니다!")
            print("🤖 xAI 기능으로 설명 가능한 AI 서비스 제공!")
        else:
            print("\n⚠️ 일부 테스트가 실패했습니다. 로그를 확인하세요.")
            
    except KeyboardInterrupt:
        print("\n테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
