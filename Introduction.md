# 🔬 Agentic AI System - 개발자 가이드 및 전체 구조 개괄

**에이전틱 AI 기반 HR 분석 플랫폼의 완전한 개발 가이드**

이 문서는 Agentic AI System의 전체 폴더 구조, 데이터 생성 과정, 분석 파이프라인, 그리고 각 파일의 역할을 상세히 설명합니다. 처음 접하는 개발자도 시스템의 전체적인 흐름을 이해하고 개발에 참여할 수 있도록 작성되었습니다.

---

## 📁 전체 프로젝트 구조

```
Agentic_AI_system/
├── 📊 Data generation/          # 데이터 생성 및 전처리
├── 📈 Data analysis/            # 분석 모델 개발 및 실험
├── 🗄️ data/                    # 원본 및 처리된 데이터
├── 🤖 app/                      # 프로덕션 서버 애플리케이션
├── 🌐 Dashboard/                # React 프론트엔드
├── 📋 Test data/                # 테스트용 데이터셋
├── 🔧 optimized_models/         # 최적화된 모델 파일
├── 📝 logs/                     # 서버 실행 로그
├── 📄 README/                   # 상세 문서
└── 🚀 start_all_services_simple.py  # 통합 서버 실행 스크립트
```

---

## 🔄 데이터 플로우 및 개발 워크플로우

### 1단계: 데이터 생성 (`Data generation/`)
```
원본 IBM_HR.csv → 데이터 생성 스크립트 → 에이전트별 특화 데이터
```

### 2단계: 분석 및 모델 개발 (`Data analysis/`)
```
특화 데이터 → Jupyter Notebook 실험 → 모델 개발 → 성능 평가
```

### 3단계: 프로덕션 배포 (`app/`)
```
검증된 모델 → Flask 서버 → REST API → React 대시보드
```

---

## 📊 Data generation/ - 데이터 생성 및 전처리

### 🎯 주요 목적
원본 IBM HR 데이터를 각 AI 에이전트가 분석할 수 있는 형태로 변환하고 확장

### 📁 파일 구조 및 역할

#### 🏢 Structura 관련 (정형 데이터)
- **`Structura_data_assignment.ipynb`**
  - **클래스**: 없음 (Jupyter Notebook)
  - **주요 함수**: 데이터 전처리 및 페르소나 할당
  - **역할**: IBM_HR.csv를 기반으로 직원별 페르소나 할당 및 라벨링
  - **출력**: `IBM_HR_personas_assigned.csv`

#### 🕸️ Cognita 관련 (관계형 데이터)
- **`Cognita_data_assignment.ipynb`**
  - **클래스**: 없음 (Jupyter Notebook)
  - **주요 함수**: 관계형 데이터 생성 및 Neo4j 스키마 설계
  - **역할**: 직원 간 협업 관계, 보고 관계, 프로젝트 참여 관계 생성
  - **출력**: `employee_relationships.json`, `employee_relationships.xml`

- **`Cognita_graph_development.ipynb`**
  - **클래스**: 없음 (Jupyter Notebook)
  - **주요 함수**: Neo4j 그래프 데이터베이스 구축
  - **역할**: 관계형 데이터를 Neo4j에 삽입하고 그래프 구조 최적화
  - **출력**: Neo4j 그래프 데이터베이스

#### ⏰ Chronos 관련 (시계열 데이터)
- **`Chronos_data_generation.py`**
  - **클래스**: 
    - `BusinessCalendar`: 영업일 및 공휴일 관리
    - `TimeSeriesGenerator`: 시계열 데이터 생성 엔진
    - `EmployeeTimeSeriesGenerator`: 직원별 시계열 패턴 생성
  - **주요 함수**:
    - `generate_business_calendar()`: 영업일 캘린더 생성
    - `generate_employee_timeseries()`: 개별 직원 시계열 생성
    - `apply_seasonal_patterns()`: 계절성 패턴 적용
  - **역할**: 1,470명 직원의 6주간 시계열 데이터 생성
  - **출력**: `IBM_HR_timeseries.csv`

- **`Chronos_data_assignment.ipynb`**
  - **클래스**: 없음 (Jupyter Notebook)
  - **주요 함수**: 시계열 데이터 검증 및 품질 확인
  - **역할**: 생성된 시계열 데이터의 통계적 검증
  - **출력**: 검증 리포트 및 시각화

- **`create_1470_chronos.py`** & **`generate_full_chronos_1470.py`**
  - **클래스**: `ChronosDataGenerator`
  - **주요 함수**: 
    - `generate_full_dataset()`: 전체 데이터셋 생성
    - `validate_data_quality()`: 데이터 품질 검증
  - **역할**: 1,470명 전체 직원의 완전한 시계열 데이터 생성
  - **출력**: `chronos_test_1470employees.csv`

#### 📝 Sentio 관련 (텍스트 데이터)
- **`Sentio_data_generation.py`**
  - **클래스**: 
    - `HRTextGenerator`: HR 텍스트 생성 엔진
  - **주요 함수**:
    - `generate_persona_texts()`: 페르소나별 텍스트 생성
    - `generate_text_for_employee()`: 개별 직원 텍스트 생성
    - `save_to_csv()`: CSV 형태로 저장
  - **역할**: 10가지 페르소나별 HR 텍스트 데이터 생성
  - **출력**: `IBM_HR_text.csv`, `IBM_HR_text.json`

- **`Sentio_data_generation_rev.py`**
  - **클래스**: `HRTextGenerator`
  - **주요 함수**: 
    - `generate_realistic_texts()`: 현실적인 텍스트 생성
    - `apply_linguistic_variations()`: 언어적 변형 적용
  - **역할**: 개선된 버전의 텍스트 데이터 생성
  - **출력**: `IBM_HR_text_sample.csv`

---

## 📈 Data analysis/ - 분석 모델 개발 및 실험

### 🎯 주요 목적
각 AI 에이전트의 머신러닝 모델 개발, 실험, 최적화 및 성능 평가

### 📁 파일 구조 및 역할

#### 🏢 Structura 분석 (정형 데이터 ML)
- **`Structura.ipynb`**
  - **클래스**: 없음 (Jupyter Notebook)
  - **주요 함수**: 
    - XGBoost 모델 훈련 및 최적화
    - SHAP, LIME 기반 xAI 분석
    - 하이퍼파라미터 튜닝 (Optuna)
  - **역할**: 이직 예측 모델 개발 및 설명 가능한 AI 구현
  - **출력**: 훈련된 XGBoost 모델, xAI 분석 결과

- **`Structura_rev.ipynb`**
  - **클래스**: 없음 (Jupyter Notebook)
  - **주요 함수**: 개선된 모델 아키텍처 및 피처 엔지니어링
  - **역할**: Structura 모델의 성능 개선 버전
  - **출력**: 최적화된 모델 및 성능 리포트

#### 🕸️ Cognita 분석 (관계형 데이터)
- **`Cognita.ipynb`**
  - **클래스**: 없음 (Jupyter Notebook)
  - **주요 함수**:
    - Neo4j 그래프 분석 알고리즘
    - 네트워크 중심성 계산
    - 사회적 고립 지수 계산
  - **역할**: 그래프 기반 관계형 위험도 분석 모델 개발
  - **출력**: 관계형 분석 알고리즘 및 위험도 지표

#### ⏰ Chronos 분석 (시계열 딥러닝)
- **`Chronos_analysis.py`**
  - **클래스**: 
    - `GRUModel`: GRU 기반 시계열 모델
    - `CNN1DModel`: 1D CNN 모델
    - `GRU_CNN_HybridModel`: GRU+CNN 하이브리드 모델
  - **주요 함수**:
    - `build_hybrid_model()`: 하이브리드 딥러닝 모델 구축
    - `train_with_attention()`: Attention 메커니즘 훈련
    - `evaluate_temporal_patterns()`: 시계열 패턴 평가
  - **역할**: 시계열 기반 이직 예측 딥러닝 모델 개발
  - **출력**: `seq_model_gru.pt`, `seq_scaler.joblib`

- **`Chronos_analysis_fixed.py`**
  - **클래스**: 수정된 모델 클래스들
  - **주요 함수**: 
    - `fix_model_architecture()`: 모델 아키텍처 수정
    - `optimize_hyperparameters()`: 하이퍼파라미터 최적화
  - **역할**: Chronos 모델의 버그 수정 및 성능 개선
  - **출력**: 수정된 모델 파일

#### 📝 Sentio 분석 (텍스트 NLP)
- **`Sentio.ipynb`**
  - **클래스**: 없음 (Jupyter Notebook)
  - **주요 함수**:
    - 텍스트 전처리 및 토큰화
    - 감정 분석 모델 훈련
    - 키워드 추출 알고리즘
  - **역할**: NLP 기반 텍스트 감정 분석 모델 개발
  - **출력**: 감정 분석 모델 및 키워드 사전

- **`Sentio_keywords.py`**
  - **클래스**: 없음 (함수 기반 스크립트)
  - **주요 함수**:
    - `extract_nouns()`: 명사 중심 키워드 추출
    - `analyze_sentiment()`: 감정 점수 계산
    - `detect_risk_signals()`: 퇴직 위험 신호 탐지
  - **역할**: 키워드 기반 위험 신호 탐지 시스템
  - **출력**: 키워드 분석 결과

- **`Sentio_keywords_improved.py`**
  - **클래스**: 없음 (함수 기반 스크립트)
  - **주요 함수**: 
    - `extract_contextual_keywords()`: 문맥 기반 키워드 추출
    - `filter_noise_words()`: 노이즈 단어 필터링
  - **역할**: 개선된 키워드 추출 및 분석 시스템
  - **출력**: 정제된 키워드 분석 결과

#### 🌍 Agora 분석 (시장 데이터)
- **`Agora.ipynb`**
  - **클래스**: 없음 (Jupyter Notebook)
  - **주요 함수**:
    - 시장 데이터 수집 및 분석
    - 급여 격차 분석
    - 시장 압력 지수 계산
  - **역할**: 외부 시장 분석 모델 개발
  - **출력**: 시장 분석 알고리즘 및 지표

#### 🎯 통합 분석 (Integration)
- **`Threshold_setting.ipynb`**
  - **클래스**: 없음 (Jupyter Notebook)
  - **주요 함수**:
    - `calculate_optimal_thresholds()`: 최적 임계값 계산
    - `optimize_f1_score()`: F1-score 최적화
  - **역할**: 각 에이전트별 최적 임계값 설정
  - **출력**: `Optimal_weights_info.csv`

- **`Threshold_setting_DT.ipynb`**
  - **클래스**: 없음 (Jupyter Notebook)
  - **주요 함수**: 의사결정 트리 기반 임계값 설정
  - **역할**: 결정 트리를 활용한 임계값 최적화
  - **출력**: 의사결정 트리 모델

- **`Weight_setting.ipynb`**
  - **클래스**: 없음 (Jupyter Notebook)
  - **주요 함수**:
    - `optimize_weights()`: 가중치 최적화 (Grid Search, Bayesian)
    - `evaluate_weighted_performance()`: 가중치 기반 성능 평가
  - **역할**: 다중 에이전트 결과의 최적 가중치 계산
  - **출력**: `Total_score_with_weighted_predictions.csv`

#### 🔧 유틸리티 및 지원 파일
- **`Data split.py`**
  - **클래스**: 없음
  - **주요 함수**: 
    - `split_data_with_label_ratio()`: 라벨 비율 유지 데이터 분할
  - **역할**: Train/Test 데이터 분할 (6:4 비율)
  - **출력**: `IBM_HR_personas_train.csv`, `IBM_HR_personas_test.csv`

- **`example_usage.py`**
  - **클래스**: 없음
  - **주요 함수**: API 사용 예제 및 테스트 코드
  - **역할**: 개발자를 위한 사용 예제 제공
  - **출력**: 예제 실행 결과

---

## 🗄️ data/ - 데이터 저장소

### 📁 파일 구조 및 설명

#### 원본 데이터
- **`IBM_HR.csv`**: 1,470명 직원의 기본 HR 데이터 (원본)
- **`IBM_HR_report.csv`**: 분석 리포트가 포함된 확장 데이터

#### 페르소나 할당 데이터
- **`IBM_HR_personas_assigned.csv`**: 10가지 페르소나가 할당된 전체 데이터
- **`IBM_HR_personas_train.csv`**: 훈련용 데이터 (60%)
- **`IBM_HR_personas_test.csv`**: 테스트용 데이터 (40%)

#### 시계열 데이터
- **`IBM_HR_timeseries.csv`**: 1,470명 × 6주간 시계열 데이터

#### 텍스트 데이터
- **`IBM_HR_text.csv`**: HR 텍스트 데이터 (CSV 형식)
- **`IBM_HR_text.json`**: HR 텍스트 데이터 (JSON 형식)
- **`IBM_HR_text_sample.csv`**: 샘플 텍스트 데이터

#### 관계형 데이터
- **`employee_relationships.json`**: 직원 간 관계 데이터 (JSON)
- **`employee_relationships.xml`**: 직원 간 관계 데이터 (XML)

#### 통합 분석 결과
- **`Total_score.csv`**: 모든 에이전트의 점수가 통합된 최종 데이터

---

## 🤖 app/ - 프로덕션 서버 애플리케이션

### 🎯 주요 목적
개발된 모델들을 실제 서비스로 배포하기 위한 Flask 기반 REST API 서버

### 📁 에이전트별 서버 구조

#### 🏢 Structura/ (정형 데이터 분석 서버)
- **`structura_flask_backend.py`**
  - **클래스**: 
    - `StructuraHRPredictor`: XGBoost 예측 엔진
    - `XAIExplainer`: SHAP/LIME 설명 생성기
  - **주요 함수**:
    - `predict_attrition()`: 이직 확률 예측
    - `explain_prediction()`: xAI 기반 예측 설명
    - `get_feature_importance()`: 피처 중요도 분석
  - **API 엔드포인트**: `/api/predict`, `/api/explain`, `/api/feature-importance`
  - **포트**: 5001

- **`run_structura_server.py`**: 서버 실행 스크립트
- **`test_structura_api.py`**: API 테스트 스크립트

#### 🕸️ Cognita/ (관계형 데이터 분석 서버)
- **`cognita_flask_backend.py`**
  - **클래스**: 
    - `Neo4jConnector`: Neo4j 데이터베이스 연결
    - `RelationshipAnalyzer`: 관계형 분석 엔진
  - **주요 함수**:
    - `analyze_employee_relationships()`: 직원 관계 분석
    - `calculate_social_isolation()`: 사회적 고립 지수 계산
    - `analyze_department_risk()`: 부서별 위험도 분석
  - **API 엔드포인트**: `/api/analyze/employee/{id}`, `/api/analyze/department`
  - **포트**: 5002

- **`run_cognita_server.py`**: 서버 실행 스크립트
- **`test_cognita_api.py`**: API 테스트 스크립트

#### ⏰ Chronos/ (시계열 분석 서버)
- **`chronos_flask_backend.py`**
  - **클래스**: 
    - `ChronosPredictor`: 시계열 예측 엔진
    - `AttentionVisualizer`: Attention 시각화
  - **주요 함수**:
    - `predict_timeseries()`: 시계열 기반 예측
    - `visualize_attention()`: Attention 가중치 시각화
    - `train_model()`: 모델 훈련
  - **API 엔드포인트**: `/api/predict`, `/api/train`, `/api/visualize/attention`
  - **포트**: 5003

- **`chronos_models.py`**: 딥러닝 모델 정의
- **`chronos_processor_fixed.py`**: 데이터 전처리 (수정 버전)
- **`run_chronos_server.py`**: 서버 실행 스크립트
- **`test_chronos_api.py`**: API 테스트 스크립트

#### 📝 Sentio/ (텍스트 분석 서버)
- **`sentio_flask_backend.py`**
  - **클래스**: 
    - `SentioAnalyzer`: 텍스트 분석 엔진
    - `TextGenerator`: GPT 기반 텍스트 생성
  - **주요 함수**:
    - `analyze_text()`: 텍스트 감정 분석
    - `extract_keywords()`: 키워드 추출
    - `generate_persona_text()`: 페르소나 기반 텍스트 생성
  - **API 엔드포인트**: `/analyze/text`, `/analyze/keywords`, `/generate/text`
  - **포트**: 5004

- **`sentio_processor.py`**: 텍스트 전처리
- **`sentio_analyzer.py`**: `SentioKeywordAnalyzer` 클래스 포함
- **`sentio_generator.py`**: 텍스트 생성 엔진
- **`run_sentio_server.py`**: 서버 실행 스크립트
- **`test_sentio_api.py`**: API 테스트 스크립트

#### 🌍 Agora/ (시장 분석 서버)
- **`agora_flask_backend.py`**
  - **클래스**: 
    - `MarketAnalyzer`: 시장 분석 엔진
    - `LLMInterpreter`: GPT 기반 해석 생성
  - **주요 함수**:
    - `analyze_market_pressure()`: 시장 압력 분석
    - `calculate_compensation_gap()`: 보상 격차 계산
    - `generate_market_report()`: 시장 분석 리포트 생성
  - **API 엔드포인트**: `/analyze/market`, `/analyze/batch`, `/market/trends`
  - **포트**: 5005

- **`agora_processor.py`**: 시장 데이터 처리
- **`agora_analyzer.py`**: 시장 분석 엔진
- **`agora_llm_generator.py`**: LLM 기반 해석 생성
- **`run_agora_server.py`**: 서버 실행 스크립트
- **`test_agora_api.py`**: API 테스트 스크립트

#### 🎯 Supervisor/ (워크플로우 관리 서버)
- **`supervisor_flask_backend.py`**
  - **클래스**: 
    - `SupervisorAgent`: 워크플로우 오케스트레이터
    - `WorkflowManager`: LangGraph 워크플로우 관리
  - **주요 함수**:
    - `analyze_employee()`: 통합 직원 분석
    - `batch_analyze()`: 배치 분석
    - `manage_workflow()`: 워크플로우 상태 관리
  - **API 엔드포인트**: `/analyze_employee`, `/batch_analyze`, `/worker_health_check`
  - **포트**: 5006

- **`supervisor_agent.py`**: 슈퍼바이저 에이전트 핵심 로직
- **`langgraph_workflow.py`**: LangGraph 워크플로우 구현
- **`worker_integrator.py`**: 워커 에이전트 통합
- **`synthesizer.py`**: 결과 합성 엔진
- **`agent_state.py`**: 에이전트 상태 관리
- **`run_supervisor_server.py`**: 서버 실행 스크립트
- **`test_supervisor_api.py`**: API 테스트 스크립트

#### 🎯 Integration/ (최종 레포트 생성 서버)
- **`integration_flask_backend.py`**
  - **클래스**: 
    - `ThresholdCalculator`: 임계값 계산기
    - `WeightOptimizer`: 가중치 최적화기
    - `ReportGenerator`: GPT 기반 레포트 생성
  - **주요 함수**:
    - `calculate_thresholds()`: 최적 임계값 계산
    - `optimize_weights()`: 가중치 최적화
    - `generate_report()`: 개별 직원 레포트 생성
  - **API 엔드포인트**: `/calculate_thresholds`, `/optimize_weights`, `/generate_report`
  - **포트**: 5007

- **`threshold_calculator.py`**: 임계값 계산 모듈
- **`weight_optimizer.py`**: 가중치 최적화 모듈
- **`report_generator.py`**: 레포트 생성 모듈
- **`run_integration_server.py`**: 서버 실행 스크립트
- **`test_integration_api.py`**: API 테스트 스크립트

### 🔧 공통 유틸리티
- **`result_manager.py`**: 결과 관리 시스템
- **`hierarchical_result_manager.py`**: 계층적 결과 관리
- **`run_agentic_system.py`**: 통합 시스템 실행
- **`run_fixed_agents.py`**: 수정된 에이전트 실행
- **`install_dependencies.py`**: 의존성 자동 설치
- **`create_simplified_structure.py`**: 간소화된 구조 생성

---

## 🌐 Dashboard/ - React 프론트엔드

### 🎯 주요 목적
사용자 친화적인 웹 인터페이스를 통한 AI 분석 결과 시각화

### 📁 주요 구조
- **`src/`**: React 소스 코드
- **`public/`**: 정적 파일
- **`package.json`**: 의존성 및 스크립트 정의

### 🔗 백엔드 연동
- **프록시 설정**: `http://localhost:5006` (Supervisor)
- **주요 기능**: 
  - 실시간 분석 결과 시각화
  - 인터랙티브 대시보드
  - 다중 에이전트 결과 통합 표시

---

## 🧪 Test data/ - 테스트 데이터셋

### 📁 파일 구조
- **`Structura test.csv`**: Structura 테스트 데이터
- **`Cognita test.csv`**: Cognita 테스트 데이터  
- **`Chronos test.csv`**: Chronos 테스트 데이터
- **`Sentio test.csv`**: Sentio 테스트 데이터
- **`Total score.csv`**: 통합 테스트 데이터

### 🎯 사용 목적
- 각 에이전트의 개별 성능 테스트
- 통합 시스템의 end-to-end 테스트
- 성능 벤치마킹 및 회귀 테스트

---

## 🔧 optimized_models/ - 최적화된 모델 저장소

### 📁 파일 구조
- **`best_hybrid_model.pth`**: 최적화된 Chronos 딥러닝 모델
- **`best_params.json`**: 최적 하이퍼파라미터
- **`optimization_study.pkl`**: Optuna 최적화 연구 결과
- **`processor_info.pkl`**: 전처리기 정보

### 🎯 사용 목적
- 프로덕션 환경에서 사용할 최적화된 모델 보관
- 모델 버전 관리 및 롤백 지원
- 성능 최적화 기록 보존

---

## 📝 logs/ - 서버 실행 로그

### 📁 파일 구조
- **`structura_server.log`**: Structura 서버 로그
- **`cognita_server.log`**: Cognita 서버 로그
- **`chronos_server.log`**: Chronos 서버 로그
- **`sentio_server.log`**: Sentio 서버 로그
- **`agora_server.log`**: Agora 서버 로그
- **`supervisor_server.log`**: Supervisor 서버 로그
- **`integration_server.log`**: Integration 서버 로그

### 🎯 사용 목적
- 서버 상태 모니터링
- 오류 디버깅 및 문제 해결
- 성능 분석 및 최적화

---

## 🚀 실행 스크립트

### **`start_all_services_simple.py`**
- **클래스**: 없음
- **주요 함수**:
  - `start_agent_background()`: 에이전트 백그라운드 실행
  - `check_service_health()`: 서비스 상태 확인
  - `monitor_services()`: 서비스 모니터링
  - `cleanup_processes()`: 프로세스 정리
- **역할**: 모든 백엔드 서버를 한 번에 실행하고 모니터링
- **실행 방법**: `C:/Users/OJH/anaconda3/envs/nlp/python.exe start_all_services_simple.py`

---

## 🔄 개발 워크플로우 가이드

### 1단계: 환경 설정
```bash
# Anaconda 환경 활성화
conda activate nlp

# 환경 변수 설정
export NEO4J_URI="bolt://YOUR_NEO4J_HOST:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="YOUR_NEO4J_PASSWORD"
export OPENAI_API_KEY="your-gpt5nano-api-key"
```

### 2단계: 데이터 생성 (선택적)
```bash
# 새로운 데이터 생성이 필요한 경우
cd "Data generation"
python Chronos_data_generation.py
python Sentio_data_generation.py
```

### 3단계: 모델 개발 및 실험
```bash
# Jupyter Notebook으로 모델 실험
cd "Data analysis"
jupyter notebook Structura.ipynb
jupyter notebook Chronos_analysis.py
```

### 4단계: 서버 실행
```bash
# 모든 서버 한 번에 실행
python start_all_services_simple.py

# 또는 개별 서버 실행
cd app/Structura && python run_structura_server.py
```

### 5단계: 프론트엔드 실행
```bash
cd Dashboard
npm install
npm start
```

### 6단계: 테스트 및 검증
```bash
# API 테스트
cd app/Structura && python test_structura_api.py
cd app/Supervisor && python test_supervisor_api.py
```

---

## 🔍 주요 클래스 및 함수 참조

### 데이터 생성 관련
- **`BusinessCalendar`** (Chronos_data_generation.py): 영업일 캘린더 관리
- **`HRTextGenerator`** (Sentio_data_generation.py): HR 텍스트 생성 엔진
- **`RealisticNoiseGenerator`** (Chronos_data_generation.py): 현실적인 노이즈 생성

### 분석 모델 관련
- **`GRU_CNN_HybridModel`** (Chronos_analysis.py): GRU+CNN 하이브리드 모델
- **`SentioKeywordAnalyzer`** (sentio_analyzer.py): 키워드 분석 엔진
- **`ThresholdCalculator`** (threshold_calculator.py): 임계값 계산기

### 서버 관련
- **`StructuraHRPredictor`** (structura_flask_backend.py): XGBoost 예측 엔진
- **`SupervisorAgent`** (supervisor_agent.py): 워크플로우 오케스트레이터
- **`Neo4jConnector`** (cognita_flask_backend.py): Neo4j 연결 관리
- **`ReportGenerator`** (report_generator.py): GPT 기반 레포트 생성

### 유틸리티 관련
- **`WeightOptimizer`** (weight_optimizer.py): 가중치 최적화
- **`ResultManager`** (result_manager.py): 결과 관리
- **`HierarchicalResultManager`** (hierarchical_result_manager.py): 계층적 결과 관리

---

## 🚨 주의사항 및 문제 해결

### 환경 설정 관련
1. **Python 환경**: 반드시 `conda activate nlp` 실행
2. **API 키**: OpenAI API 키가 올바르게 설정되었는지 확인
3. **Neo4j 연결**: Neo4j 데이터베이스 연결 정보 확인

### 파일 경로 관련
1. **상대 경로**: 모든 스크립트는 프로젝트 루트에서 실행
2. **데이터 파일**: `data/` 폴더에 필요한 데이터 파일 존재 확인
3. **모델 파일**: `optimized_models/` 폴더의 모델 파일 확인

### 서버 실행 관련
1. **포트 충돌**: 각 서버의 포트가 사용 중이지 않은지 확인
2. **의존성**: 각 에이전트 폴더의 `requirements.txt` 설치 확인
3. **로그 확인**: `logs/` 폴더의 서버 로그로 오류 진단

---

## 📚 추가 문서

- **`README.md`**: 전체 시스템 사용 가이드
- **`app/Launch.md`**: 빠른 실행 가이드
- **각 에이전트 폴더의 README.md**: 에이전트별 상세 가이드
- **`README/`**: 특정 컴포넌트별 상세 문서

---

**개발자 가이드 버전**: 1.0.0  
**최종 업데이트**: 2025년 9월  

이 문서는 Agentic AI System의 전체 구조를 이해하고 개발에 참여하기 위한 완전한 가이드입니다. 추가 질문이나 문제가 있으면 각 에이전트별 README 파일을 참조하거나 로그 파일을 확인하세요.
