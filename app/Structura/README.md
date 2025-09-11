# Structura - HR 이직 예측 시스템 (XAI 포함)

XGBoost + SHAP 기반 설명 가능한 HR 이직 예측 Flask 백엔드 서비스입니다.  
**노트북 기반 최신 모델 적용** | **EmployeeNumber별 개별 XAI 설명** | **Probability 중심 예측**

## 🚀 주요 특징

### 🤖 머신러닝 기능
- **XGBoost 모델**: 고성능 그래디언트 부스팅 알고리즘
- **자동 하이퍼파라미터 최적화**: Optuna 기반 (선택적)
- **클래스 불균형 처리**: 자동 가중치 조정
- **교차 검증**: 5-fold Stratified Cross Validation

### 🔍 XAI (설명 가능한 AI) 기능 - 업그레이드됨
- **SHAP (SHapley Additive exPlanations)**: EmployeeNumber별 개별 예측 설명
- **변수 중요도 분석**: 각 직원별 위험/보호 요인 식별
- **Feature Importance**: 전역적 피처 중요도
- **Probability 중심**: 이직 확률만 반환하는 간소화된 예측
- **권장사항 생성**: XAI 분석 기반 개선 방안 제시

### 🌐 React 연동 최적화
- **Flask + CORS**: React 개발 서버와 완벽 호환
- **RESTful API**: 표준 REST 인터페이스
- **JSON 응답**: 한글 지원 및 구조화된 데이터
- **에러 처리**: React에서 쉽게 처리할 수 있는 표준화된 에러 응답

## 📋 시스템 요구사항

- Python 3.8+
- 최소 4GB RAM 권장
- IBM HR 데이터셋 (CSV 형식)

## 🛠️ 설치 및 설정

### 1. 의존성 설치

```bash
cd app/Structura
pip install -r requirements.txt
```

### 2. 데이터 준비

IBM HR 데이터셋을 `../data/IBM_HR.csv` 경로에 배치하거나, 코드에서 경로를 수정하세요.

### 3. 서버 실행

```bash
python run_structura_server.py
```

또는

```bash
python structura_flask_backend.py
```

## 📡 API 엔드포인트

### 기본 정보
- **서버 주소**: `http://localhost:5001`
- **Content-Type**: `application/json`
- **CORS**: React 개발 서버 지원

### 주요 엔드포인트 (업데이트됨)

#### 1. 헬스체크
```http
GET /api/health
```

**응답 예시:**
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "xai_status": {
    "shap_available": true,
    "lime_available": true
  },
  "dependencies": {
    "shap": true,
    "lime": true,
    "optuna": true
  }
}
```

#### 2. 모델 훈련
```http
POST /api/train
Content-Type: application/json

{
  "optimize_hyperparameters": false,
  "n_trials": 30
}
```

**응답 예시:**
```json
{
  "status": "success",
  "message": "모델 훈련 완료",
  "metrics": {
    "roc_auc": 0.8542,
    "f1_score": 0.7123,
    "precision": 0.6891,
    "recall": 0.7367,
    "accuracy": 0.8234
  },
  "xai_enabled": {
    "shap": true,
    "lime": true
  }
}
```

#### 3. 이직 예측 (XAI 포함)
```http
POST /api/predict
Content-Type: application/json

{
  "EmployeeNumber": "EMP_001",
  "Age": 35,
  "JobSatisfaction": 3,
  "WorkLifeBalance": 2,
  "OverTime": "Yes",
  "MonthlyIncome": 5000,
  // ... 기타 피처들
}
```

**응답 예시:**
```json
{
  "employee_number": "EMP_001",
  "attrition_probability": 0.742,
  "risk_category": "HIGH",
  "explanation": {
    "employee_number": "EMP_001",
    "global_feature_importance": {
      "StockOptionLevel": 0.100,
      "JobSatisfaction": 0.083
    },
    "individual_explanation": {
      "variable_importance": {
        "OverTime": 0.245,
        "JobSatisfaction": -0.123
      },
      "top_risk_factors": [
        {
          "feature": "OverTime",
          "impact": 0.245
        }
      ],
      "top_protective_factors": [
        {
          "feature": "JobSatisfaction", 
          "impact": 0.123
        }
      ]
    }
  }
}
```

#### 4. 예측 설명 (xAI)
```http
POST /api/explain
Content-Type: application/json

{
  "Age": 35,
  "JobSatisfaction": 3,
  // ... 직원 데이터
}
```

**응답 예시:**
```json
{
  "employee_id": null,
  "feature_importance": {
    "OverTime": 0.1234,
    "JobSatisfaction": 0.0987,
    "WorkLifeBalance": 0.0876
  },
  "shap_values": {
    "OverTime": 0.245,
    "JobSatisfaction": -0.123,
    "WorkLifeBalance": 0.089
  },
  "lime_explanation": {
    "features": ["OverTime", "JobSatisfaction", "Age"],
    "values": [0.245, -0.123, 0.067],
    "intercept": 0.156
  },
  "top_risk_factors": [
    {
      "feature": "OverTime",
      "impact": 0.245,
      "type": "risk"
    }
  ],
  "top_protective_factors": [
    {
      "feature": "JobSatisfaction",
      "impact": 0.123,
      "type": "protective"
    }
  ]
}
```

#### 5. 피처 중요도
```http
GET /api/feature-importance?top_n=10
```

**응답 예시:**
```json
{
  "feature_importance": [
    {
      "feature": "OverTime",
      "importance": 0.1234,
      "rank": 1
    },
    {
      "feature": "JobSatisfaction",
      "importance": 0.0987,
      "rank": 2
    }
  ],
  "total_features": 25,
  "top_n": 10
}
```

#### 6. 모델 정보
```http
GET /api/model/info
```

#### 7. 배치 예측 (신규)
```http
POST /api/predict/batch
Content-Type: application/json

[
  {
    "EmployeeNumber": "EMP_001",
    "Age": 25,
    "JobSatisfaction": 1,
    "OverTime": "Yes"
  },
  {
    "EmployeeNumber": "EMP_002", 
    "Age": 45,
    "JobSatisfaction": 4,
    "OverTime": "No"
  }
]
```

**응답 예시:**
```json
{
  "predictions": [
    {
      "employee_number": "EMP_001",
      "attrition_probability": 0.85,
      "risk_category": "HIGH"
    },
    {
      "employee_number": "EMP_002",
      "attrition_probability": 0.15,
      "risk_category": "LOW"
    }
  ],
  "statistics": {
    "total_employees": 2,
    "successful_predictions": 2,
    "average_probability": 0.50,
    "high_risk_count": 1
  }
}
```

#### 8. 개별 직원 심층 분석 (신규)
```http
POST /api/employee/analysis/{employee_number}
Content-Type: application/json

{
  "Age": 28,
  "JobSatisfaction": 2,
  "OverTime": "Yes"
}
```

**응답 예시:**
```json
{
  "employee_number": "EMP_001",
  "attrition_probability": 0.742,
  "risk_category": "HIGH",
  "detailed_analysis": {
    "probability_score": 0.742,
    "distance_to_next_level": null,
    "risk_thresholds": {
      "LOW": 0.4,
      "MEDIUM": 0.7,
      "HIGH": 1.0
    }
  },
  "recommendations": [
    "즉시 면담을 통한 이직 의도 파악 필요",
    "업무 환경 및 만족도 개선 방안 논의",
    "JobSatisfaction 개선을 위한 구체적 액션 플랜 수립"
  ]
}
```

## 🧪 테스트

### API 테스트 실행

```bash
python test_structura_api.py
```

테스트 스크립트는 다음을 검증합니다:
- 서버 헬스체크
- 모델 훈련 (선택적)
- 이직 예측
- 예측 설명 (xAI)
- 피처 중요도
- 모델 정보 조회

## 🌐 React 연동 가이드

### 기본 사용법

```javascript
// 1. 이직 예측
const predictAttrition = async (employeeData) => {
  try {
    const response = await fetch('http://localhost:5001/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(employeeData)
    });
    const prediction = await response.json();
    console.log('이직 확률:', prediction.attrition_probability);
    return prediction;
  } catch (error) {
    console.error('예측 실패:', error);
  }
};

// 2. 예측 설명 (xAI)
const explainPrediction = async (employeeData) => {
  try {
    const response = await fetch('http://localhost:5001/api/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(employeeData)
    });
    const explanation = await response.json();
    console.log('SHAP 값:', explanation.shap_values);
    console.log('위험 요인:', explanation.top_risk_factors);
    return explanation;
  } catch (error) {
    console.error('설명 생성 실패:', error);
  }
};

// 3. 피처 중요도
const getFeatureImportance = async () => {
  try {
    const response = await fetch('http://localhost:5001/api/feature-importance');
    const data = await response.json();
    return data.feature_importance;
  } catch (error) {
    console.error('피처 중요도 조회 실패:', error);
  }
};
```

### React Hook 예시

```javascript
import { useState, useEffect } from 'react';

const useStructuraAPI = () => {
  const [serverHealth, setServerHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const baseURL = 'http://localhost:5001/api';
  
  const checkHealth = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${baseURL}/health`);
      const data = await response.json();
      setServerHealth(data);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  const predictAttrition = async (employeeData) => {
    try {
      setLoading(true);
      const response = await fetch(`${baseURL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(employeeData)
      });
      const data = await response.json();
      setError(null);
      return data;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  };
  
  const explainPrediction = async (employeeData) => {
    try {
      setLoading(true);
      const response = await fetch(`${baseURL}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(employeeData)
      });
      const data = await response.json();
      setError(null);
      return data;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    checkHealth();
  }, []);
  
  return {
    serverHealth,
    loading,
    error,
    checkHealth,
    predictAttrition,
    explainPrediction
  };
};
```

## 📊 xAI 기능 상세

### SHAP (SHapley Additive exPlanations)
- **목적**: 개별 예측에 대한 각 피처의 기여도 정량화
- **해석**: 양수 값은 이직 위험 증가, 음수 값은 이직 위험 감소
- **활용**: 개별 직원의 이직 위험 요인 파악

### LIME (Local Interpretable Model-agnostic Explanations)
- **목적**: 복잡한 모델의 지역적 설명 제공
- **해석**: 특정 예측 주변에서의 모델 동작 설명
- **활용**: 모델의 의사결정 과정 이해

### Feature Importance
- **목적**: 전역적 피처 중요도 제공
- **해석**: 전체 데이터셋에서 각 피처의 평균적 중요도
- **활용**: 전반적인 이직 예측 요인 파악

## 🚨 문제 해결

### 일반적인 문제

1. **xAI 라이브러리 설치 오류**
   ```bash
   pip install shap lime
   # 또는 conda 사용
   conda install -c conda-forge shap lime
   ```

2. **메모리 부족 오류**
   ```
   SHAP 분석 시 메모리 사용량이 높을 수 있습니다.
   샘플 크기를 줄이거나 서버 메모리를 증설하세요.
   ```

3. **모델 훈련 실패**
   ```
   데이터 경로 확인: ../data/IBM_HR.csv
   데이터 형식 확인: CSV 파일, Attrition 컬럼 포함
   ```

### React 연동 문제

1. **CORS 오류**
   ```
   Flask-CORS가 설치되어 있는지 확인
   React 개발 서버 주소가 CORS 설정에 포함되어 있는지 확인
   ```

2. **API 응답 지연**
   ```
   xAI 분석은 시간이 걸릴 수 있습니다.
   React에서 로딩 상태를 적절히 처리하세요.
   ```

## 📈 성능 특성

- **예측 속도**: 평균 0.1초/명
- **xAI 분석 속도**: 평균 1-3초/명 (SHAP + LIME)
- **메모리 사용량**: 기본 500MB, xAI 분석 시 1-2GB
- **모델 정확도**: ROC-AUC 0.85+ (데이터에 따라 변동)

## 🔧 확장 가능성

### 추가 기능 구현 가능
- **배치 예측**: 여러 직원 동시 분석
- **실시간 모니터링**: 정기적 모델 재훈련
- **A/B 테스트**: 다양한 모델 비교
- **알림 시스템**: 고위험 직원 자동 알림
- **대시보드 연동**: 실시간 시각화

### xAI 확장
- **SHAP 시각화**: 워터폴 차트, 포스 플롯
- **LIME 이미지**: 텍스트 설명을 이미지로 변환
- **커스텀 설명**: 비즈니스 로직 기반 설명 추가

---

**버전**: 1.0.0 (Structura Edition)  
**xAI 지원**: SHAP, LIME, Feature Importance  
**React 연동**: 최적화 완료  
**최종 업데이트**: 2025년
