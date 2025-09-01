"""
HR Attrition Predictor 사용 예시
"""

from app.hr_attrition_backend import HRAttritionPredictor
import pandas as pd
import numpy as np


def example_full_training():
    """전체 훈련 파이프라인 예시"""
    print("=== 전체 훈련 파이프라인 예시 ===")
    
    # 예측기 초기화
    predictor = HRAttritionPredictor(data_path="data/IBM_HR.csv")
    
    # 전체 파이프라인 실행 (하이퍼파라미터 최적화 포함)
    metrics = predictor.run_full_pipeline(
        optimize_hp=True,   # 하이퍼파라미터 최적화 사용
        n_trials=20         # 빠른 테스트를 위해 20회로 설정
    )
    
    # 모델 저장
    predictor.save_model("trained_model.pkl")
    
    print(f"훈련 완료! 최종 성능: ROC-AUC = {metrics['ROC_AUC']:.4f}")


def example_quick_training():
    """빠른 훈련 예시 (기본 하이퍼파라미터 사용)"""
    print("=== 빠른 훈련 예시 ===")
    
    predictor = HRAttritionPredictor(data_path="data/IBM_HR.csv")
    
    # 하이퍼파라미터 최적화 없이 빠른 훈련
    metrics = predictor.run_full_pipeline(
        optimize_hp=False   # 기본 하이퍼파라미터 사용
    )
    
    predictor.save_model("quick_model.pkl")
    print(f"빠른 훈련 완료! ROC-AUC = {metrics['ROC_AUC']:.4f}")


def example_load_and_predict():
    """저장된 모델 로딩 및 예측 예시"""
    print("=== 모델 로딩 및 예측 예시 ===")
    
    # 새로운 예측기 인스턴스 생성
    predictor = HRAttritionPredictor()
    
    # 저장된 모델 로딩
    try:
        predictor.load_model("trained_model.pkl")
        print("모델 로딩 성공!")
    except FileNotFoundError:
        print("저장된 모델이 없습니다. 먼저 훈련을 실행하세요.")
        return
    
    # 새로운 데이터로 예측 (예시로 원본 데이터 사용)
    df = predictor.load_data()
    X, y_true = predictor.preprocess_data(df)
    
    # 처음 10개 샘플로 예측
    sample_data = X.head(10)
    
    # 이진 예측 (0 또는 1)
    predictions = predictor.predict(sample_data)
    
    # 확률 예측
    probabilities = predictor.predict(sample_data, return_proba=True)
    
    print("\n예측 결과:")
    print("샘플 | 실제값 | 예측값 | 확률")
    print("-" * 35)
    for i in range(len(sample_data)):
        actual = y_true.iloc[i]
        pred = predictions[i]
        prob = probabilities[i]
        print(f"{i+1:4d} | {actual:6d} | {pred:6d} | {prob:.3f}")


def example_feature_importance():
    """피처 중요도 확인 예시"""
    print("=== 피처 중요도 확인 예시 ===")
    
    predictor = HRAttritionPredictor()
    
    try:
        predictor.load_model("trained_model.pkl")
    except FileNotFoundError:
        print("저장된 모델이 없습니다. 먼저 훈련을 실행하세요.")
        return
    
    # 상위 15개 피처 중요도
    importance_df = predictor.get_feature_importance(top_n=15)
    
    print("\n상위 15개 중요 피처:")
    print(importance_df.to_string(index=False))


def example_single_prediction():
    """단일 샘플 예측 예시"""
    print("=== 단일 샘플 예측 예시 ===")
    
    predictor = HRAttritionPredictor()
    
    try:
        predictor.load_model("trained_model.pkl")
    except FileNotFoundError:
        print("저장된 모델이 없습니다. 먼저 훈련을 실행하세요.")
        return
    
    # 예시 직원 데이터 생성 (실제 사용 시에는 실제 데이터 입력)
    sample_employee = pd.DataFrame({
        'Age': [35],
        'DistanceFromHome': [10],
        'MonthlyIncome': [5000],
        'TrainingTimesLastYear': [3],
        'YearsAtCompany': [5],
        'YearsInCurrentRole': [3],
        'YearsWithCurrManager': [2],
        'Education': [3],  # Bachelor
        'JobLevel': [2],   # Junior
        'JobInvolvement': [3],  # High
        'JobSatisfaction': [4],  # Very High
        'EnvironmentSatisfaction': [3],  # High
        'RelationshipSatisfaction': [4],  # Very High
        'WorkLifeBalance': [3],  # Better
        'PerformanceRating': [3],  # Excellent
        'StockOptionLevel': [1],  # Level 1
        'BusinessTravel': ['Travel_Rarely'],
        'Department': ['Research & Development'],
        'EducationField': ['Life Sciences'],
        'Gender': ['Male'],
        'JobRole': ['Research Scientist'],
        'MaritalStatus': ['Married'],
        'OverTime': ['No']
    })
    
    # 예측 실행
    prediction = predictor.predict(sample_employee)[0]
    probability = predictor.predict(sample_employee, return_proba=True)[0]
    
    print(f"\n직원 이탈 예측:")
    print(f"예측 결과: {'이탈 위험' if prediction == 1 else '이탈 낮음'}")
    print(f"이탈 확률: {probability:.1%}")
    
    if probability > 0.7:
        print("⚠️  높은 이탈 위험! 즉시 대응이 필요합니다.")
    elif probability > 0.4:
        print("⚡ 중간 이탈 위험. 모니터링이 필요합니다.")
    else:
        print("✅ 낮은 이탈 위험. 현재 상태 유지.")


def main():
    """메인 실행 함수"""
    print("HR Attrition Predictor 사용 예시\n")
    
    while True:
        print("\n다음 중 실행할 예시를 선택하세요:")
        print("1. 전체 훈련 파이프라인 (하이퍼파라미터 최적화 포함)")
        print("2. 빠른 훈련 (기본 파라미터)")
        print("3. 모델 로딩 및 예측")
        print("4. 피처 중요도 확인")
        print("5. 단일 샘플 예측")
        print("0. 종료")
        
        choice = input("\n선택 (0-5): ").strip()
        
        if choice == '1':
            example_full_training()
        elif choice == '2':
            example_quick_training()
        elif choice == '3':
            example_load_and_predict()
        elif choice == '4':
            example_feature_importance()
        elif choice == '5':
            example_single_prediction()
        elif choice == '0':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 다시 시도하세요.")


if __name__ == "__main__":
    main()
