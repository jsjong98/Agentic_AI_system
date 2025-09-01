# HR Attrition Prediction Backend

XGBoost 기반의 HR 직원 이탈 예측 자동화 백엔드 시스템입니다.

## 주요 기능

- **자동화된 데이터 전처리**: 결측값 처리, 범주형 변수 인코딩, 피처 엔지니어링
- **XGBoost 모델 훈련**: 클래스 불균형 처리 및 조기 종료 기능
- **하이퍼파라미터 최적화**: Optuna를 사용한 자동 최적화
- **최적 임계값 탐색**: F1 점수 기준 임계값 자동 조정
- **모델 저장/로딩**: 훈련된 모델의 영구 저장 및 재사용
- **예측 서비스**: 새로운 데이터에 대한 이탈 확률 예측

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 데이터 파일 준비:
   - `data/IBM_HR.csv` 파일이 있는지 확인

## 사용 방법

### 1. 기본 사용법

```python
from hr_attrition_backend import HRAttritionPredictor

# 예측기 초기화
predictor = HRAttritionPredictor(data_path="data/IBM_HR.csv")

# 전체 파이프라인 실행
metrics = predictor.run_full_pipeline(
    optimize_hp=True,  # 하이퍼파라미터 최적화 사용
    n_trials=30        # 최적화 시행 횟수
)

# 모델 저장
predictor.save_model("my_model.pkl")
```

### 2. 저장된 모델로 예측

```python
# 새로운 예측기 인스턴스
predictor = HRAttritionPredictor()

# 모델 로딩
predictor.load_model("my_model.pkl")

# 예측 실행
predictions = predictor.predict(new_data)
probabilities = predictor.predict(new_data, return_proba=True)
```

### 3. 예시 실행

```bash
python example_usage.py
```

## 파일 구조

```
├── hr_attrition_backend.py    # 메인 백엔드 클래스
├── example_usage.py           # 사용 예시
├── requirements.txt           # 필요 패키지 목록
├── README.md                  # 이 파일
└── data/
    └── IBM_HR.csv            # 훈련 데이터
```

## 모델 성능

- **ROC-AUC**: ~0.85
- **PR-AUC**: ~0.60
- **F1 Score**: ~0.45
- **정확도**: ~0.84

## 주요 피처

모델이 중요하게 고려하는 상위 피처들:

1. **MonthlyIncome** - 월급
2. **Age** - 나이
3. **YearsAtCompany** - 재직 기간
4. **DistanceFromHome** - 집과의 거리
5. **JobLevel** - 직급
6. **OverTime** - 초과근무 여부
7. **JobRole** - 직무
8. **WorkLifeBalance** - 워라밸

## API 참조

### HRAttritionPredictor 클래스

#### 주요 메서드

- `run_full_pipeline(optimize_hp=True, n_trials=50)`: 전체 파이프라인 실행
- `predict(X, return_proba=False)`: 예측 실행
- `save_model(filepath)`: 모델 저장
- `load_model(filepath)`: 모델 로딩
- `get_feature_importance(top_n=20)`: 피처 중요도 반환

#### 개별 단계 실행

- `load_data()`: 데이터 로딩
- `preprocess_data(df)`: 데이터 전처리
- `split_data(X, y)`: 훈련/테스트 분할
- `optimize_hyperparameters(X_train, y_train, n_trials)`: 하이퍼파라미터 최적화
- `train_model(X_train, y_train, hyperparams)`: 모델 훈련
- `optimize_threshold(X_train, y_train)`: 최적 임계값 탐색
- `evaluate_model(X_test, y_test)`: 모델 평가

## 설정 옵션

### 전처리 설정

- **제거 컬럼**: 불필요하거나 노이즈가 많은 피처들 자동 제거
- **순서형 변수**: 교육 수준, 직급 등을 수치형으로 자동 변환
- **명목형 변수**: 부서, 성별 등을 카테고리 타입으로 처리
- **결측값 처리**: 수치형은 중앙값, 범주형은 'UNK'로 대체

### 모델 설정

- **클래스 불균형**: `scale_pos_weight` 자동 계산
- **조기 종료**: 검증 성능 기준 자동 종료
- **교차 검증**: 5-fold Stratified CV 사용

## 문제 해결

### 일반적인 오류

1. **FileNotFoundError**: 데이터 파일 경로 확인
2. **ImportError**: `pip install -r requirements.txt` 실행
3. **메모리 부족**: `n_trials` 값을 줄여서 실행

### 성능 개선 팁

1. **하이퍼파라미터 최적화**: `n_trials`를 50-100으로 증가
2. **피처 엔지니어링**: 도메인 지식을 활용한 새로운 피처 추가
3. **앙상블**: 여러 모델의 결과를 결합

## 라이선스

MIT License

## 기여

버그 리포트나 기능 제안은 이슈로 등록해 주세요.