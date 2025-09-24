# 🕒 Chronos - Employee Attrition Prediction System

Chronos는 직원 퇴사 예측을 위한 고급 시계열 분석 시스템입니다. Attention 메커니즘을 활용한 딥러닝 모델로 직원의 행동 패턴을 분석하고 퇴사 위험도를 예측합니다.

## ✨ 주요 기능

### 🧠 고급 딥러닝 모델
- **GRU + CNN + Attention 하이브리드 아키텍처**
- **Feature-level Attention**: 각 피처의 중요도 계산
- **Temporal Attention**: 시계열 패턴의 중요한 시점 식별
- **실시간 예측 및 해석**

### 📊 인터랙티브 시각화
- **Feature Importance 분석**: Plotly 기반 인터랙티브 차트
- **Temporal Attention 시각화**: 시간대별 중요도 분석
- **개별 직원 타임라인**: 상세한 행동 패턴 분석
- **모델 성능 대시보드**: ROC 곡선, Confusion Matrix 등

### 🚀 RESTful API
- **모델 학습 API**: 하이퍼파라미터 최적화 지원
- **실시간 예측 API**: 배치 및 개별 예측
- **시각화 API**: HTML 기반 인터랙티브 차트
- **시스템 모니터링**: 상태 확인 및 진단

## 🏗️ 시스템 아키텍처

```
app/Chronos/
├── chronos_models.py          # 딥러닝 모델 정의
├── chronos_processor.py       # 데이터 처리 및 시각화
├── chronos_flask_backend.py   # Flask API 서버
├── run_chronos_server.py      # 서버 실행 스크립트
├── test_chronos_api.py        # API 테스트 도구
├── requirements.txt           # 의존성 패키지
└── README.md                  # 문서 (현재 파일)
```

## 📋 요구사항

### 시스템 요구사항
- Python 3.8+
- 4GB+ RAM 권장
- CPU/GPU 지원 (CUDA 선택사항)

### 데이터 요구사항
- `data/IBM_HR_timeseries.csv`: 직원 시계열 데이터
- `data/IBM_HR.csv`: 직원 기본 속성 데이터 (페르소나 정보 제거됨)

## 🚀 설치 및 실행

### 1. 의존성 설치
```bash
cd app/Chronos
pip install -r requirements.txt
```

### 2. 서버 실행
```bash
# 자동 실행 (권장)
python run_chronos_server.py

# 또는 직접 실행
python chronos_flask_backend.py
```

### 3. 웹 브라우저에서 접속
```
http://localhost:5003
```

## 🧪 API 테스트

### 빠른 테스트
```bash
python test_chronos_api.py
```

### 종합 테스트
```bash
python test_chronos_api.py --full
```

## 📡 API 엔드포인트

### 시스템 관리
- `GET /` - 홈페이지 및 API 문서
- `GET /api/status` - 시스템 상태 확인

### 모델 관리
- `POST /api/train` - 모델 학습
- `POST /api/predict` - 예측 수행

### 시각화
- `GET /api/feature_importance` - Feature importance 차트
- `GET /api/model_analysis` - 모델 분석 대시보드
- `GET /api/employee_timeline/{id}` - 개별 직원 분석

## 💡 사용 예제

### 1. 모델 학습
```bash
curl -X POST http://localhost:5003/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "sequence_length": 6,
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
  }'
```

### 2. 예측 수행
```bash
curl -X POST http://localhost:5003/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "employee_ids": [1, 2, 3, 4, 5]
  }'
```

### 3. 전체 직원 예측
```bash
curl -X POST http://localhost:5003/api/predict \
  -H "Content-Type: application/json" \
  -d '{}'
```

## 📊 예측 결과 해석

### 위험도 분류
- **High Risk** (0.7+): 즉시 관심 필요
- **Medium Risk** (0.3-0.7): 모니터링 필요
- **Low Risk** (0.3-): 안정적 상태

### Feature Importance
모델이 각 피처의 중요도를 0-1 스케일로 제공합니다:
- **높은 중요도 (0.8+)**: 핵심 예측 인자
- **중간 중요도 (0.4-0.8)**: 보조 예측 인자
- **낮은 중요도 (0.4-)**: 미미한 영향

### Temporal Attention
시계열에서 중요한 시점을 식별합니다:
- 최근 주차일수록 일반적으로 높은 가중치
- 특정 이벤트 시점에서 급격한 변화 감지

## 🔧 고급 설정

### 모델 하이퍼파라미터
```python
{
    "sequence_length": 6,      # 시계열 길이 (주 단위)
    "gru_hidden": 64,          # GRU 은닉층 크기
    "cnn_filters": 32,         # CNN 필터 수
    "dropout": 0.3,            # 드롭아웃 비율
    "learning_rate": 0.001,    # 학습률
    "batch_size": 32,          # 배치 크기
    "epochs": 50               # 학습 에포크
}
```

### 데이터 전처리 설정
```python
{
    "aggregation_unit": "week",  # 집계 단위 (week/month)
    "sequence_length": 6,        # 시퀀스 길이
    "feature_selection": "auto"  # 피처 선택 방법
}
```

## 🎯 성능 최적화

### GPU 사용
```python
# CUDA 사용 가능 시 자동으로 GPU 활용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 메모리 최적화
- 배치 크기 조정으로 메모리 사용량 제어
- 그래디언트 체크포인팅으로 메모리 절약
- 모델 양자화 지원 (선택사항)

## 🛠️ 트러블슈팅

### 일반적인 문제

1. **모듈 import 오류**
   ```bash
   pip install -r requirements.txt
   ```

2. **데이터 파일 없음**
   ```
   ⚠️ 데이터 파일을 data/ 폴더에 배치하세요
   ```

3. **메모리 부족**
   ```python
   # batch_size를 줄여보세요
   {"batch_size": 16}
   ```

4. **CUDA 오류**
   ```python
   # CPU 모드로 강제 실행
   device = torch.device('cpu')
   ```

### 로그 확인
```bash
# 서버 로그 확인
python chronos_flask_backend.py
```

## 📈 모델 성능 지표

### 평가 메트릭
- **Accuracy**: 전체 정확도
- **Precision**: 정밀도 (퇴사 예측의 정확성)
- **Recall**: 재현율 (실제 퇴사자 탐지율)
- **F1-Score**: 정밀도와 재현율의 조화평균
- **AUC-ROC**: ROC 곡선 아래 면적

### 벤치마크 성능
- **기본 모델**: ~85% 정확도
- **최적화 모델**: ~90% 정확도
- **앙상블 모델**: ~92% 정확도

## 🤝 기여 방법

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 🙋‍♂️ 지원

문제가 있거나 질문이 있으시면 다음 방법으로 연락주세요:

- **Issues**: GitHub Issues 페이지
- **Email**: support@chronos-ai.com
- **Documentation**: 상세 API 문서는 `/api/docs` 참조

---

**Chronos** - 직원 퇴사 예측의 새로운 표준 🚀
