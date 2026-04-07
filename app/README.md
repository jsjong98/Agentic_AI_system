# Agentic AI System - HR Analytics Platform

PwC RA팀의 **에이전틱 AI 기반 HR 분석 플랫폼**입니다. 

## 🤖 에이전틱 아키텍처

```
                    ┌─────────────────────────────────────┐
                    │      🎯 Supervisor Agent           │
                    │         (포트: 5006)              │
                    │   LangGraph 워크플로우 관리         │
                    └─────────────────────────────────────┘
                                    │
        ┌───────────┬───────────┬───────────┬───────────┬───────────┐
        │  워커1    │  워커2    │  워커3    │  워커4    │  워커5    │
        │  정형     │  관계형   │  시계열   │  자연어   │  외부     │
        │  데이터   │  데이터   │  데이터   │  데이터   │  시장     │
        │  분석     │  분석     │  분석     │  분석     │  분석     │
        │  ✅      │  ✅      │  ✅      │  ✅      │  ✅      │
        └───────────┴───────────┴───────────┴───────────┴───────────┘
                                    │
                    ┌─────────────────────────────────────┐
                    │       🎯 Integration 시스템         │
                    │         (포트: 5007)              │
                    │    GPT-5-nano 기반 최종 레포트     │
                    └─────────────────────────────────────┘
```

**현재 구현**: 5개 워커 에이전트 + Supervisor + Integration  
**완료 상태**: 모든 에이전트 구현 완료 (GPT-5-nano, LangGraph 적용)

---

## 🏗️ 프로젝트 구조

```
app/
├── Structura/          # 정형 데이터 분석 (XGBoost + xAI)
│   ├── structura_flask_backend.py
│   ├── hr_attrition_backend.py
│   ├── run_structura_server.py
│   ├── test_structura_api.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
│
├── Cognita/           # 관계형 데이터 분석 (Neo4j + Graph)
│   ├── cognita_flask_backend.py
│   ├── run_cognita_server.py
│   ├── test_cognita_api.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
│
├── Chronos/           # 시계열 데이터 분석 (GRU+CNN+Attention)
│   ├── chronos_flask_backend.py
│   ├── chronos_models.py
│   ├── chronos_processor.py
│   ├── run_chronos_server.py
│   ├── test_chronos_api.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
│
├── Sentio/            # 텍스트 감정 분석 (NLP + GPT-5-nano)
│   ├── sentio_flask_backend.py
│   ├── sentio_processor.py
│   ├── sentio_analyzer.py
│   ├── sentio_generator.py
│   ├── run_sentio_server.py
│   ├── test_sentio_api.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
│
├── Agora/             # 외부 시장 분석 (시장 분석 + GPT-5-nano)
│   ├── agora_flask_backend.py
│   ├── agora_analyzer.py
│   ├── agora_processor.py
│   ├── agora_llm_generator.py
│   ├── run_agora_server.py
│   ├── test_agora_api.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
│
├── Supervisor/        # 워커 통합 관리 (LangGraph + GPT-5-nano)
│   ├── supervisor_flask_backend.py
│   ├── supervisor_processor.py
│   ├── langgraph_workflow.py
│   ├── run_supervisor_server.py
│   ├── test_supervisor_api.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
│
├── Integration/       # 최종 종합 레포트 (GPT-5-nano)
│   ├── integration_flask_backend.py
│   ├── threshold_calculator.py
│   ├── weight_optimizer.py
│   ├── report_generator.py
│   ├── run_integration_server.py
│   ├── test_integration_api.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
│
├── Dockerfile                     # GPU 버전 통합 Docker 이미지
├── Dockerfile.cpu                 # CPU 버전 통합 Docker 이미지
├── requirements_agentic.txt       # 통합 의존성
├── requirements-docker.txt        # Docker용 의존성
└── README.md                      # 이 파일
```

---

## 🚀 빠른 시작

### 방법 1: Docker Compose 사용 (권장 ⭐)

```bash
# 프로젝트 루트에서 실행
cd ..  # Agentic_AI_system 루트로 이동

# Docker Compose로 모든 서비스 실행
docker-compose up -d

# 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs -f supervisor
```

**접속**:
- **Supervisor**: http://localhost:5006
- **Structura**: http://localhost:5001
- **Cognita**: http://localhost:5002
- **Chronos**: http://localhost:5003
- **Sentio**: http://localhost:5004
- **Agora**: http://localhost:5005
- **Integration**: http://localhost:5007

---

### 방법 2: Docker 개별 실행

각 에이전트를 개별 Docker 컨테이너로 실행:

```bash
# Structura
cd Structura
docker build -t structura:latest .
docker run -d -p 5001:5001 --name structura structura:latest

# Cognita
cd Cognita
docker build -t cognita:latest .
docker run -d -p 5002:5002 \
  -e NEO4J_URI="bolt://host.docker.internal:7687" \
  -e NEO4J_USERNAME="neo4j" \
  -e NEO4J_PASSWORD="your-password" \
  --name cognita cognita:latest

# Chronos
cd Chronos
docker build -t chronos:latest .
docker run -d -p 5003:5003 --name chronos chronos:latest

# Sentio
cd Sentio
docker build -t sentio:latest .
docker run -d -p 5004:5004 \
  -e OPENAI_API_KEY="your-api-key" \
  --name sentio sentio:latest

# Agora
cd Agora
docker build -t agora:latest .
docker run -d -p 5005:5005 \
  -e OPENAI_API_KEY="your-api-key" \
  --name agora agora:latest

# Supervisor
cd Supervisor
docker build -t supervisor:latest .
docker run -d -p 5006:5006 \
  -e OPENAI_API_KEY="your-api-key" \
  --name supervisor supervisor:latest

# Integration
cd Integration
docker build -t integration:latest .
docker run -d -p 5007:5007 \
  -e OPENAI_API_KEY="your-api-key" \
  --name integration integration:latest
```

---

### 방법 3: Python 직접 실행 (개발 환경)

#### 1. 통합 시스템 실행 (모든 에이전트 동시 실행)

```bash
# 1. 의존성 설치
cd app
pip install -r requirements_agentic.txt

# 2. 환경변수 설정
export NEO4J_URI="bolt://YOUR_NEO4J_HOST:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="YOUR_NEO4J_PASSWORD"
export OPENAI_API_KEY="your-gpt5nano-api-key"  # Sentio, Agora, Supervisor, Integration용

# 3. 통합 서버 실행 (모든 워커 에이전트 동시 실행)
cd ..
python start_all_services_simple.py
```

**접속**: 각 에이전트는 개별 포트에서 실행됩니다 (5001-5007)

---

#### 2. 개별 워커 실행 (개발/디버깅용)

각 워커를 개별적으로 실행할 수 있습니다:

##### Structura 워커 (정형 데이터 분석)
```bash
cd app/Structura
pip install -r requirements.txt
python run_structura_server.py  # 포트: 5001
```

##### Cognita 워커 (관계형 데이터 분석)
```bash
cd app/Cognita
pip install -r requirements.txt

# Neo4j 환경변수 설정 (필수)
export NEO4J_URI="bolt://YOUR_NEO4J_HOST:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="YOUR_NEO4J_PASSWORD"

python run_cognita_server.py    # 포트: 5002
```

##### Chronos 워커 (시계열 데이터 분석)
```bash
cd app/Chronos
pip install -r requirements.txt
python run_chronos_server.py    # 포트: 5003
```

##### Sentio 워커 (텍스트 감정 분석)
```bash
cd app/Sentio
pip install -r requirements.txt

# OpenAI API 키 설정 (필수)
export OPENAI_API_KEY="your-gpt5nano-api-key"

python run_sentio_server.py     # 포트: 5004
```

##### Agora 워커 (외부 시장 분석)
```bash
cd app/Agora
pip install -r requirements.txt

# OpenAI API 키 설정 (필수)
export OPENAI_API_KEY="your-gpt5nano-api-key"

python run_agora_server.py      # 포트: 5005
```

##### Supervisor (워커 통합 관리)
```bash
cd app/Supervisor
pip install -r requirements.txt

# OpenAI API 키 설정 (필수)
export OPENAI_API_KEY="your-gpt5nano-api-key"

python run_supervisor_server.py # 포트: 5006
```

##### Integration (최종 종합 레포트)
```bash
cd app/Integration
pip install -r requirements.txt

# OpenAI API 키 설정 (필수)
export OPENAI_API_KEY="your-gpt5nano-api-key"

python run_integration_server.py # 포트: 5007
```

---

## 🚀 시스템 개요

### 🤖 Structura - HR 이직 예측 시스템
**포트**: `5001` | **기술**: XGBoost + xAI (SHAP, LIME)

- **목적**: 개별 직원의 이직 가능성 예측 및 설명
- **특징**: 
  - 설명 가능한 AI (xAI) 기반 예측
  - SHAP, LIME을 통한 투명한 의사결정
  - 실시간 예측 및 위험 요인 분석
- **데이터**: IBM HR 데이터셋 (CSV)
- **주요 기능**:
  - 이직 확률 예측
  - 개별 직원 위험 요인 분석
  - 피처 중요도 분석
  - 모델 설명 및 해석

---

### 🕸️ Cognita - 관계형 위험도 분석 시스템
**포트**: `5002` | **기술**: Neo4j + Graph Analytics

- **목적**: 직원 간 관계 네트워크 기반 이직 위험도 분석
- **특징**:
  - 사회적 네트워크 분석
  - 관계형 데이터 기반 위험도 평가
  - 팀 역학 및 조직 구조 분석
- **데이터**: Neo4j 그래프 데이터베이스
- **주요 기능**:
  - 사회적 고립 지수 계산
  - 네트워크 중심성 분석
  - 관리자 안정성 평가
  - 부서별 위험도 분석

---

### ⏰ Chronos - 시계열 데이터 분석 시스템
**포트**: `5003` | **기술**: GRU+CNN+Attention 하이브리드 딥러닝

- **목적**: 직원 행동 패턴의 시간적 변화 추적 및 예측
- **특징**:
  - 시계열 패턴 분석
  - Attention 메커니즘으로 중요 시점 식별
  - 딥러닝 기반 고정밀 예측
- **데이터**: 시계열 CSV 데이터
- **주요 기능**:
  - 6주 단위 시계열 분석
  - Feature/Temporal Attention 시각화
  - 하이퍼파라미터 최적화
  - 인터랙티브 차트 제공

---

### 📝 Sentio - 텍스트 감정 분석 시스템
**포트**: `5004` | **기술**: NLP + 키워드 분석 + GPT-5-nano

- **목적**: HR 텍스트의 감정 분석 및 퇴직 위험 신호 탐지
- **특징**:
  - 명사 중심 키워드 추출
  - 5가지 퇴직 원인별 위험 신호 분석
  - GPT-5-nano 기반 텍스트 생성
  - .env 파일을 통한 API 키 관리
- **데이터**: HR 텍스트 데이터
- **주요 기능**:
  - 텍스트 감정 점수 계산
  - 페르소나 기반 텍스트 생성
  - JD-R 모델 기반 분석
  - client.responses.create() API 호출

---

### 🌍 Agora - 외부 시장 분석 시스템
**포트**: `5005` | **기술**: 시장 데이터 분석 + GPT-5-nano LLM 해석

- **목적**: 외부 시장 상황을 고려한 이직 위험도 분석
- **특징**:
  - 시장 압력 지수 계산
  - 보상 격차 분석
  - GPT-5-nano 기반 자연스러운 해석
  - .env 파일을 통한 API 키 관리
- **데이터**: 채용 공고 API 데이터
- **주요 기능**:
  - 직무별 시장 분석
  - 경쟁력 평가
  - 시장 트렌드 분석
  - LLM 기반 해석 제공

---

### 🎯 Supervisor - 워커 통합 관리 시스템
**포트**: `5006` | **기술**: LangGraph + GPT-5-nano 워크플로우

- **목적**: 5개 워커 에이전트의 결과를 종합 분석 및 관리
- **특징**:
  - LangGraph 기반 워크플로우 자동화
  - GPT-5-nano 기반 지능형 의사결정
  - 다중 에이전트 결과 합성
  - .env 파일을 통한 API 키 관리
- **데이터**: 워커 에이전트 결과 통합
- **주요 기능**:
  - 종합 분석 워크플로우
  - 결과 합성 및 우선순위 결정
  - 워크플로우 상태 관리
  - 품질 관리 및 일관성 보장

---

### 🎯 Integration - 최종 종합 레포트 시스템
**포트**: `5007` | **기술**: GPT-5-nano 기반 종합 분석 및 레포트 생성

- **목적**: 모든 에이전트 결과를 종합한 최종 퇴사 위험 레포트 생성
- **특징**:
  - 임계값 설정 및 F1-score 최적화
  - 가중치 최적화 (Grid Search, Bayesian)
  - GPT-5-nano 기반 맞춤형 레포트
  - .env 파일을 통한 API 키 관리
- **데이터**: 에이전트 점수 및 직원 데이터
- **주요 기능**:
  - 개별 직원 레포트 생성
  - 일괄 레포트 생성
  - 3단계 위험도 분류
  - Fallback 시스템 제공

---

## 📋 데이터 준비 가이드

### 🏢 Structura 데이터 준비
```bash
# IBM HR 데이터셋 배치
cp IBM_HR.csv ../data/IBM_HR.csv

# 필수 컬럼 확인
# Age, JobSatisfaction, OverTime, MonthlyIncome, WorkLifeBalance, Attrition 등
```

### 🕸️ Cognita 데이터베이스 설정
```bash
# Neo4j 환경변수 설정 (필수)
export NEO4J_URI="bolt://YOUR_NEO4J_HOST:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="YOUR_NEO4J_PASSWORD"

# 또는 .env 파일에 설정
echo "NEO4J_URI=bolt://YOUR_NEO4J_HOST:7687" >> .env
echo "NEO4J_USERNAME=neo4j" >> .env
echo "NEO4J_PASSWORD=YOUR_NEO4J_PASSWORD" >> .env
```

### ⏰ Chronos 시계열 데이터 준비
```bash
# 시계열 데이터 배치
cp IBM_HR_timeseries.csv ../data/IBM_HR_timeseries.csv

# 데이터 형식: employee_id, week, 시계열 피처들
# 6주 단위 시퀀스 데이터 필요
```

### 📝 Sentio 텍스트 데이터 준비
```bash
# HR 텍스트 데이터 배치
cp IBM_HR_text.csv ../data/IBM_HR_text.csv

# OpenAI API 키 설정 (GPT-5-nano용)
export OPENAI_API_KEY="your-gpt5nano-api-key"
# 또는 .env 파일에 설정
echo "OPENAI_API_KEY=your-gpt5nano-api-key" >> .env
```

### 🌍 Agora 시장 데이터 설정
```bash
# Structura의 IBM_HR.csv 데이터를 자동으로 활용
# 별도 데이터 준비 불필요 (Structura 데이터에서 직무, 급여 정보 추출)

# OpenAI API 키 설정 (GPT-5-nano용)
export OPENAI_API_KEY="your-gpt5nano-api-key"
```

### 🎯 Supervisor & Integration 설정
```bash
# OpenAI API 키 설정 (GPT-5-nano용)
export OPENAI_API_KEY="your-gpt5nano-api-key"

# 워커 에이전트들의 결과를 자동으로 수집
# 별도 데이터 준비 불필요
```

---

## 🌐 React 연동 가이드

### 통합 API 사용 예시

```javascript
// 개별 직원 통합 분석 (Supervisor → 5개 워커 → Integration)
const analyzeEmployeeIntegrated = async (employeeData, textData) => {
  const response = await fetch('http://localhost:5006/analyze_employee', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      employee_id: employeeData.EmployeeNumber,
      session_id: `session_${Date.now()}`,
      employee_data: employeeData,
      text_data: textData
    })
  });
  
  const result = await response.json();
  
  console.log('최종 레포트:', result.final_report);
  console.log('위험 점수:', result.final_report.risk_score);
  console.log('권장사항:', result.final_report.recommendations);
  
  return result;
};

// 배치 분석
const batchAnalyze = async (employeeIds) => {
  const response = await fetch('http://localhost:5006/batch_analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      employee_ids: employeeIds
    })
  });
  return response.json();
};
```

---

## 📊 시스템 비교

| 특징 | Structura | Cognita | Chronos | Sentio | Agora | Integration |
|------|-----------|---------|---------|--------|-------|-------------|
| **분석 방식** | 개별 직원 데이터 | 관계형 네트워크 | 시계열 패턴 | 텍스트 감정 | 외부 시장 | 최종 레포트 |
| **데이터 소스** | CSV (IBM HR) | Neo4j 그래프 DB | 시계열 CSV | HR 텍스트 | 채용 공고 API | 에이전트 점수 |
| **주요 기술** | XGBoost + xAI | Graph Analytics | GRU+CNN+Attention | NLP + GPT-5-nano | 시장 분석 + GPT-5-nano | GPT-5-nano 레포트 |
| **포트** | 5001 | 5002 | 5003 | 5004 | 5005 | 5007 |
| **예측 대상** | 이직 확률 | 관계형 위험도 | 시계열 예측 | 퇴직 위험 신호 | 시장 압력 지수 | 종합 위험도 분석 |

---

## 🔧 주요 API 엔드포인트

### 🎯 Supervisor Agent (포트 5006) - 최상위 관리자

| 메서드 | 엔드포인트 | 설명 |
|--------|------------|------|
| `GET` | `/health` | 시스템 전체 상태 확인 |
| `GET` | `/worker_health_check` | 워커 에이전트 상태 조회 |
| `POST` | `/analyze_employee` | **개별 직원 통합 분석** (5개 워커 + Integration) |
| `POST` | `/batch_analyze` | **배치 분석** (여러 직원 동시) |
| `GET` | `/get_workflow_status/{session_id}` | 워크플로우 상태 조회 |

### 개별 워커 API (개발/디버깅용)

#### Structura 워커 (포트 5001)
- `GET /api/health` - 서버 상태 확인
- `POST /api/train` - 모델 훈련
- `POST /api/predict` - 이직 예측
- `POST /api/explain` - 예측 설명 (xAI)
- `GET /api/feature-importance` - 피처 중요도

#### Cognita 워커 (포트 5002)
- `GET /api/health` - 서버 상태 확인
- `GET /api/employees` - 직원 목록
- `GET /api/departments` - 부서 목록
- `GET /api/analyze/employee/{id}` - 직원 분석
- `POST /api/analyze/department` - 부서 분석

#### Chronos 워커 (포트 5003)
- `GET /api/health` - 서버 상태 확인
- `POST /api/train` - 모델 학습
- `POST /api/predict` - 시계열 예측
- `POST /api/predict_batch` - 배치 예측
- `GET /api/visualize/attention` - Attention 시각화

#### Sentio 워커 (포트 5004)
- `GET /health` - 서버 상태 확인
- `POST /analyze/text` - 텍스트 분석
- `POST /analyze/keywords` - 키워드 분석
- `POST /analyze/risk` - 퇴직 위험 분석
- `POST /generate/text` - 텍스트 생성

#### Agora 워커 (포트 5005)
- `GET /health` - 서버 상태 확인
- `POST /analyze/market` - 개별 직원 시장 분석
- `POST /analyze/job_market` - 직무별 시장 분석
- `POST /analyze/batch` - 배치 시장 분석
- `GET /market/report/<job_role>` - 직무별 시장 보고서

#### Integration 시스템 (포트 5007)
- `GET /health` - 서버 상태 확인
- `POST /set_api_key` - OpenAI API 키 설정
- `POST /load_data` - 데이터 로드 및 전처리
- `POST /calculate_thresholds` - 임계값 계산
- `POST /optimize_weights` - 가중치 최적화
- `POST /generate_report` - 개별 직원 레포트 생성

---

## 🧪 테스트 실행

### 통합 시스템 테스트 (권장)

```bash
# 통합 에이전틱 시스템 테스트
cd app
python test_agentic_system.py
```

### 개별 워커 테스트 (개발/디버깅용)

```bash
# Structura 워커 테스트
cd app/Structura && python test_structura_api.py

# Cognita 워커 테스트
cd app/Cognita && python test_cognita_api.py

# Chronos 워커 테스트
cd app/Chronos && python test_chronos_api.py

# Sentio 워커 테스트
cd app/Sentio && python test_sentio_api.py

# Agora 워커 테스트
cd app/Agora && python test_agora_api.py

# Supervisor 워커 테스트
cd app/Supervisor && python test_supervisor_api.py

# Integration 시스템 테스트
cd app/Integration && python test_integration_api.py
```

### 개별 기능 테스트 (curl)

```bash
# 헬스체크
curl http://localhost:5001/api/health  # Structura
curl http://localhost:5002/api/health  # Cognita
curl http://localhost:5003/api/health  # Chronos
curl http://localhost:5004/health      # Sentio
curl http://localhost:5005/health      # Agora
curl http://localhost:5006/health      # Supervisor
curl http://localhost:5007/health      # Integration

# 간단한 예측 테스트 (Structura)
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 35, "JobSatisfaction": 3, "OverTime": "Yes"}'

# 직원 분석 테스트 (Cognita)
curl http://localhost:5002/api/analyze/employee/1

# 시계열 예측 테스트 (Chronos)
curl -X POST http://localhost:5003/api/predict \
  -H "Content-Type: application/json" \
  -d '{"employee_id": 1, "sequence_data": []}'

# 텍스트 분석 테스트 (Sentio)
curl -X POST http://localhost:5004/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "업무량이 너무 많아서 번아웃이 올 것 같습니다.", "employee_id": "test_001"}'
```

---

## 📈 성능 특성

### Structura
- **예측 속도**: ~0.1초/명
- **xAI 분석**: ~1-3초/명
- **메모리 사용**: 500MB-2GB
- **정확도**: ROC-AUC 0.85+

### Cognita
- **분석 속도**: ~0.82초/명 (v1.1.0 최적화)
- **부서 분석**: 3-15초 (샘플링 최적화)
- **처리량**: ~4,400명/시간
- **메모리 사용**: 300MB-1GB

### Chronos
- **모델 아키텍처**: GRU + CNN + Dual Attention
- **시퀀스 길이**: 6주 단위 시계열 분석
- **예측 속도**: 딥러닝 기반 실시간 예측
- **시각화**: Plotly 인터랙티브 차트

### Sentio
- **키워드 추출**: 명사 중심, 500+ 불용어 필터링
- **텍스트 생성**: GPT-5-nano 기반 고품질 생성
- **API 호출**: client.responses.create() 방식

### Agora
- **응답 시간**: < 500ms (개별 분석)
- **배치 처리**: 100명/분 (LLM 미사용)
- **캐시 적중률**: > 80% (1시간 TTL)
- **LLM 해석**: GPT-5-nano 기반

---

## 🚨 문제 해결

### 공통 문제

#### 1. 포트 충돌
```bash
# 포트 사용 확인
netstat -an | grep :5001
netstat -an | grep :5002

# 프로세스 종료
kill -9 $(lsof -ti:5001)
kill -9 $(lsof -ti:5002)
```

#### 2. 의존성 설치 오류
```bash
# 가상환경 사용 권장
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 3. CORS 오류 (React 연동)
```bash
# Flask-CORS 설치 확인
pip install flask-cors

# React 개발 서버 주소 확인 (localhost:3000)
```

### Docker 관련 문제

#### 1. Docker 이미지 빌드 실패
```bash
# 캐시 없이 다시 빌드
docker build --no-cache -t agent-name:latest .

# 빌드 로그 확인
docker build -t agent-name:latest . 2>&1 | tee build.log
```

#### 2. 컨테이너 실행 실패
```bash
# 컨테이너 로그 확인
docker logs agent-name

# 컨테이너 내부 진입
docker exec -it agent-name /bin/bash
```

#### 3. 환경 변수 전달 문제
```bash
# 환경 변수 확인
docker exec agent-name env | grep OPENAI_API_KEY

# Docker Compose에서 환경 변수 확인
docker-compose config
```

### 에이전트별 특정 문제

#### Structura
```bash
# xAI 라이브러리 오류
pip install shap lime

# 데이터 파일 확인
ls -la ../data/IBM_HR.csv
```

#### Cognita
```bash
# Neo4j 연결 실패
# 환경 변수 확인
echo $NEO4J_URI
echo $NEO4J_USERNAME
echo $NEO4J_PASSWORD

# Neo4j 서버 연결 테스트
curl -u neo4j:password http://neo4j-host:7474/db/data/
```

#### Sentio / Agora / Supervisor / Integration
```bash
# OpenAI API 키 확인
echo $OPENAI_API_KEY

# .env 파일 확인
cat .env

# API 키 테스트
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

---

## 🔮 향후 계획

### 단기 (1-2개월)
- [x] Docker 지원 - 완료 ✅
- [x] Docker Compose 지원 - 완료 ✅
- [ ] Kubernetes 배포 설정
- [ ] 모델 성능 모니터링
- [ ] 자동화된 테스트 스위트

### 중기 (3-6개월)
- [ ] 성능 최적화 (멀티프로세싱, 캐싱)
- [ ] 추가 xAI 기법 도입
- [ ] 클라우드 배포 (AWS/Azure)
- [ ] API 버전 관리

### 장기 (6개월+)
- [ ] 실시간 스트리밍 분석
- [ ] 다중 조직 지원
- [ ] AI 추천 시스템
- [ ] 자동화된 인사 정책 제안

---

**버전**: 2.0.0 (Docker Compose Support)  
**최종 업데이트**: 2025년 10월  
**기술 스택**: Python, Flask, XGBoost, Neo4j, PyTorch, GPT-5-nano, LangGraph, Docker  
**아키텍처**: Multi-Agent AI System (Supervisor + 5 Workers + Integration)
