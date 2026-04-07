# Integration - 종합 분석 및 레포트 생성 시스템

Integration은 HR 데이터의 Attrition 예측을 위한 임계값 설정, 가중치 최적화, 그리고 LLM 기반 개인별 레포트 생성을 수행하는 Flask 기반 백엔드 시스템입니다.

## 📋 주요 기능

### 1. 임계값 설정 (Threshold Setting)
- 각 Score별 최적 임계값 계산
- F1-score 기반 성능 최적화
- 개별 Score의 예측 성능 분석
- 혼동 행렬 및 성능 지표 제공

### 2. 가중치 최적화 (Weight Optimization)
- 여러 최적화 알고리즘 지원:
  - Grid Search
  - Bayesian Optimization (scikit-optimize)
  - Scipy Optimization
- 가중치 합=1 제약조건 적용
- 최종 예측 모델 생성

### 3. 위험도 분류
- 3단계 위험도 구간 분류:
  - 안전군 (0.0 ~ 0.3)
  - 주의군 (0.3 ~ 0.7)
  - 고위험군 (0.7 ~ 1.0)

### 4. 개별 직원 예측
- 실시간 Attrition 위험도 예측
- 임계값 기반 및 가중치 기반 예측 제공
- 위험도 구간 분류

### 5. LLM 기반 레포트 생성 ✨ NEW
- GPT-5-nano 모델을 활용한 지능형 분석
- 에이전트별 점수를 종합한 심층 인사이트 제공
- 개인별 맞춤형 개선 방안 및 모니터링 포인트 생성
- JSON 및 텍스트 형태의 레포트 지원
- 일괄 레포트 생성 기능

## 🚀 설치 및 실행

### 1. 의존성 설치
```bash
cd app/Integration
pip install -r requirements.txt
```

### 2. 데이터 준비
다음 파일을 `data/` 폴더에 배치:
- `Total_score.csv` (필수): 각 Score와 attrition 정보가 포함된 데이터

### 3. OpenAI API 키 준비 (LLM 기능 사용 시)
GPT-5-nano 모델 기반 레포트 생성을 사용하려면 OpenAI API 키가 필요합니다.

#### 방법 1: .env 파일 사용 (권장)
```bash
# Integration 폴더에 .env 파일 생성
cd app/Integration
echo "OPENAI_API_KEY=sk-your-gpt5nano-api-key-here" > .env
```

#### 방법 2: API 엔드포인트 사용
- 서버 실행 후 `/set_api_key` 엔드포인트로 설정

**참고사항:**
- [OpenAI 플랫폼](https://platform.openai.com/)에서 GPT-5-nano 호환 API 키 발급
- Sentio/Agora와 동일한 `client.responses.create()` 방식 사용
- .env 파일이 있으면 자동으로 로드됩니다

### 4. 서버 실행
```bash
python run_integration_server.py
```

또는 직접 실행:
```bash
python integration_flask_backend.py
```

서버는 `http://localhost:5007`에서 실행됩니다.

## 📡 API 엔드포인트

### 기본 정보
- **Base URL**: `http://localhost:5007`
- **Content-Type**: `application/json`

### 엔드포인트 목록

#### 1. 서버 상태 확인
```http
GET /health
```

**응답 예시:**
```json
{
  "status": "healthy",
  "service": "Integration",
  "timestamp": "2024-01-01T12:00:00",
  "version": "1.0.0"
}
```

#### 2. 데이터 로드
```http
POST /load_data
```

**요청 본문:**
```json
{
  "file_path": "Total_score.csv"
}
```

**응답 예시:**
```json
{
  "success": true,
  "message": "데이터가 성공적으로 로드되었습니다.",
  "statistics": {
    "total_rows": 1470,
    "total_columns": 7,
    "score_columns": ["Structura_score", "Cognita_score", "Chronos_score", "Sentio_score", "Agora_score"],
    "attrition_distribution": {"No": 1233, "Yes": 237}
  }
}
```

#### 3. 임계값 계산
```http
POST /calculate_thresholds
```

**요청 본문:**
```json
{
  "score_columns": ["Structura_score", "Cognita_score", "Chronos_score", "Sentio_score", "Agora_score"]
}
```

**응답 예시:**
```json
{
  "success": true,
  "message": "임계값 계산이 완료되었습니다.",
  "results": {
    "summary": [
      {
        "Score": "Structura_score",
        "Optimal_Threshold": 0.899,
        "F1_Score": 0.8306,
        "Precision": 0.9227,
        "Recall": 0.7553,
        "Accuracy": 0.9503
      }
    ],
    "thresholds": {
      "Structura_score": 0.899,
      "Cognita_score": 0.4752,
      "Chronos_score": 0.0101,
      "Sentio_score": 0.4658,
      "Agora_score": 0.2458
    }
  }
}
```

#### 4. 가중치 최적화
```http
POST /optimize_weights
```

**요청 본문:**
```json
{
  "method": "bayesian",
  "n_calls": 100
}
```

**지원하는 방법:**
- `"grid"`: Grid Search (파라미터: `n_points_per_dim`)
- `"bayesian"`: Bayesian Optimization (파라미터: `n_calls`)
- `"scipy"`: Scipy Optimization

**응답 예시:**
```json
{
  "success": true,
  "message": "가중치 최적화가 완료되었습니다.",
  "results": {
    "method": "bayesian",
    "optimal_weights": {
      "Structura_score_prediction": 0.3216,
      "Cognita_score_prediction": 0.1000,
      "Chronos_score_prediction": 0.3690,
      "Sentio_score_prediction": 0.1000,
      "Agora_score_prediction": 0.1094
    },
    "optimal_threshold": 0.3131,
    "best_f1_score": 0.9299,
    "performance_metrics": {
      "f1_score": 0.9299,
      "precision": 0.9359,
      "recall": 0.9241,
      "accuracy": 0.9776,
      "auc": 0.9620
    },
    "risk_statistics": {
      "counts": {
        "안전군": 1102,
        "주의군": 264,
        "고위험군": 104
      },
      "attrition_rates": {
        "안전군": 0.015,
        "주의군": 0.443,
        "고위험군": 1.000
      }
    }
  }
}
```

#### 5. 개별 직원 예측
```http
POST /predict_employee
```

**요청 본문:**
```json
{
  "scores": {
    "Structura_score": 0.8,
    "Cognita_score": 0.6,
    "Chronos_score": 0.7,
    "Sentio_score": 0.4,
    "Agora_score": 0.3
  }
}
```

**응답 예시:**
```json
{
  "success": true,
  "employee_scores": {
    "Structura_score": 0.8,
    "Cognita_score": 0.6,
    "Chronos_score": 0.7,
    "Sentio_score": 0.4,
    "Agora_score": 0.3
  },
  "predictions": {
    "weighted_prediction": {
      "weighted_score": 0.631,
      "final_prediction": 1,
      "prediction_label": "위험",
      "risk_level": "주의군",
      "risk_numeric": 2,
      "threshold_used": 0.3131
    },
    "threshold_predictions": {
      "Structura_score_prediction": "안전",
      "Cognita_score_prediction": "위험",
      "Chronos_score_prediction": "위험",
      "Sentio_score_prediction": "안전",
      "Agora_score_prediction": "위험"
    }
  }
}
```

#### 6. 최적화 방법 비교
```http
POST /compare_methods
```

**요청 본문:**
```json
{
  "methods": ["grid", "bayesian", "scipy"]
}
```

#### 7. 현재 결과 조회
```http
GET /get_results
```

#### 8. 결과 내보내기
```http
POST /export_results
```

**요청 본문:**
```json
{
  "format": "csv",
  "include_data": true
}
```

## 🧪 테스트

### 전체 테스트 실행
```bash
python test_final_calc_api.py
```

### 개별 테스트 실행
```bash
# 서버 상태 확인
python test_final_calc_api.py health

# 데이터 로드 테스트
python test_final_calc_api.py load

# 임계값 계산 테스트
python test_final_calc_api.py threshold

# 가중치 최적화 테스트
python test_final_calc_api.py weight bayesian

# 개별 직원 예측 테스트
python test_final_calc_api.py predict
```

## 📁 파일 구조

```
Final_calc/
├── __init__.py                    # 모듈 초기화
├── threshold_calculator.py       # 임계값 계산 모듈
├── weight_optimizer.py          # 가중치 최적화 모듈
├── final_calc_flask_backend.py  # Flask 백엔드 서버
├── test_final_calc_api.py       # API 테스트 스크립트
├── run_final_calc_server.py     # 서버 실행 스크립트
├── requirements.txt             # 의존성 목록
├── README.md                    # 문서 (이 파일)
└── outputs/                     # 결과 파일 저장 폴더
    ├── threshold_summary_*.csv
    ├── final_weighted_predictions_*.csv
    ├── optimal_weights_*.csv
    └── risk_criteria_*.csv
```

## 🔧 주요 클래스 및 함수

### ThresholdCalculator
- `find_optimal_threshold()`: F1-score 기반 최적 임계값 계산
- `calculate_thresholds_for_scores()`: 여러 Score에 대한 임계값 계산
- `predict_attrition()`: 개별 직원 예측
- `apply_thresholds_to_data()`: 데이터에 임계값 적용

### WeightOptimizer
- `optimize_weights()`: 가중치 최적화 (여러 방법 지원)
- `grid_search_weights_normalized()`: Grid Search 최적화
- `bayesian_optimize_weights_normalized()`: Bayesian Optimization
- `scipy_optimize_weights_normalized()`: Scipy 최적화
- `apply_optimal_weights()`: 최적 가중치 적용
- `classify_risk_level()`: 위험도 구간 분류

## 📊 출력 파일

시스템은 다음과 같은 결과 파일들을 생성합니다:

1. **임계값 요약** (`threshold_summary_*.csv`)
   - 각 Score별 최적 임계값과 성능 지표

2. **예측 데이터** (`data_with_predictions_*.csv`)
   - 원본 데이터 + 각 Score별 예측 결과

3. **최종 가중 예측** (`final_weighted_predictions_*.csv`)
   - 가중치 적용된 최종 예측 및 위험도 분류

4. **최적 가중치 정보** (`optimal_weights_*.csv`)
   - 최적화된 가중치와 성능 정보

5. **위험도 기준** (`risk_criteria_*.csv`)
   - 위험도 구간 분류 기준

## ⚙️ 설정 옵션

### 환경 변수
- `DATA_DIR`: 데이터 파일 디렉토리 (기본값: `../../data`)
- `OUTPUT_DIR`: 출력 파일 디렉토리 (기본값: `./outputs`)

### 최적화 파라미터
- **Grid Search**: `n_points_per_dim` (기본값: 5)
- **Bayesian Optimization**: `n_calls` (기본값: 100)
- **위험도 구간**: 안전군(0-0.3), 주의군(0.3-0.7), 고위험군(0.7-1.0)

## 🚨 주의사항

1. **데이터 형식**: 입력 데이터는 다음 컬럼을 포함해야 합니다:
   - Score 컬럼들 (Structura_score, Cognita_score, 등)
   - `attrition` 컬럼 ('Yes'/'No' 값)

2. **메모리 사용량**: 큰 데이터셋의 경우 Grid Search의 `n_points_per_dim`을 줄이세요.

3. **실행 시간**: Bayesian Optimization은 시간이 오래 걸릴 수 있습니다. 빠른 테스트를 위해 `n_calls`를 줄이세요.

4. **의존성**: scikit-optimize가 설치되지 않은 경우 Bayesian Optimization을 사용할 수 없습니다.

## 🤝 기여 방법

1. 이슈 리포트: 버그나 개선사항을 GitHub Issues에 등록
2. 기능 요청: 새로운 최적화 방법이나 기능 제안
3. 코드 기여: Pull Request를 통한 코드 개선

## 🤖 LLM 기반 레포트 생성

### API 키 설정
```http
POST /set_api_key
```

**요청 본문:**
```json
{
  "api_key": "sk-your-gpt5nano-compatible-api-key-here"
}
```

### 개별 직원 레포트 생성
```http
POST /generate_report
```

**요청 본문:**
```json
{
  "employee_id": "EMP001",
  "agent_scores": {
    "agora_score": 0.75,
    "chronos_score": 0.45,
    "cognita_score": 0.82,
    "sentio_score": 0.65,
    "structura_score": 0.38
  },
  "format": "text",
  "save_file": true,
  "use_llm": true
}
```

**응답 예시:**
```json
{
  "success": true,
  "employee_id": "EMP001",
  "format": "text",
  "llm_used": true,
  "report": "=== 직원 퇴사 위험도 분석 레포트 ===\n...",
  "saved_files": {
    "text": "reports/report_EMP001_20240101_120000.txt",
    "json": "reports/report_EMP001_20240101_120000.json"
  }
}
```

### 일괄 레포트 생성
```http
POST /generate_batch_reports
```

**요청 본문:**
```json
{
  "employees": [
    {
      "employee_id": "EMP001",
      "agent_scores": {
        "agora_score": 0.75,
        "chronos_score": 0.45,
        "cognita_score": 0.82,
        "sentio_score": 0.65,
        "structura_score": 0.38
      }
    }
  ]
}
```

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 지원

문제가 발생하거나 질문이 있으시면 다음을 확인해주세요:

1. **로그 확인**: 서버 실행 시 출력되는 로그 메시지
2. **테스트 실행**: `python test_integration_api.py`로 전체 시스템 테스트
3. **데이터 확인**: 입력 데이터 형식과 필수 컬럼 존재 여부
4. **의존성 확인**: `pip install -r requirements.txt`로 모든 패키지 설치
5. **API 키 확인**: LLM 기능 사용 시 유효한 OpenAI API 키 설정

---

**Integration v1.0.0** - HR Attrition 예측을 위한 종합 분석 및 GPT-5-nano 기반 레포트 생성 시스템
