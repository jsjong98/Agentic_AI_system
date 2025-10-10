# 🤖 Agentic AI System - HR Analytics Platform

**에이전틱 AI 기반 HR 분석 플랫폼**

차세대 인사 분석을 위한 다중 에이전트 AI 시스템으로, 정형 데이터, 관계형 데이터, 시계열 데이터, 텍스트 데이터를 동시에 분석하여 종합적인 인사 인사이트를 제공합니다.

## 🌟 주요 특징

- **🤖 에이전틱 아키텍처**: Supervisor가 5개 워커 에이전트를 관리하고 Integration이 최종 레포트 생성
- **🔍 다차원 분석**: 정형 + 관계형 + 시계열 + 텍스트 + 시장 데이터 통합 분석
- **⚡ 실시간 처리**: 개별 직원부터 전체 조직까지 즉시 분석
- **🔬 설명 가능한 AI**: SHAP, LIME, Attention 기반 투명한 의사결정
- **🕸️ 관계형 인사이트**: Neo4j 기반 조직 네트워크 분석
- **📝 텍스트 감정 분석**: NLP 기반 퇴직 위험 신호 탐지
- **📈 시계열 패턴 분석**: 딥러닝 기반 시간적 변화 추적
- **🌍 외부 시장 분석**: 시장 압력 지수 및 경쟁력 평가
- **🌐 React 대시보드**: 현대적 웹 인터페이스 완벽 지원
- **🐳 Docker 지원**: Docker Compose로 간편한 배포 및 관리

---

## 🏗️ 시스템 아키텍처

```
                    ┌─────────────────────────────────────┐
                    │      🎯 Supervisor Agent           │
                    │         (포트: 5006)              │
                    │   LangGraph 워크플로우 관리         │
                    └─────────────────────────────────────┘
                                    │
        ┌───────────┬───────────┬───────────┬───────────┬───────────┐
        │           │           │           │           │           │
┌───────▼───────┐ ┌─▼─────┐ ┌──▼──┐ ┌─────▼─────┐ ┌───▼───┐
│  🏢 Structura  │ │🕸️Cognita│ │⏰Chronos│ │📝 Sentio │ │🌍Agora│
│   워커 에이전트  │ │워커 에이전트│ │워커 에이전트│ │ 워커 에이전트│ │워커 에이전트│
│  (포트: 5001)  │ │(포트:5002)│ │(포트:5003)│ │(포트: 5004)│ │(포트:5005)│
│               │ │         │ │         │ │           │ │         │
│ 정형 데이터 분석 │ │관계형 데이터│ │시계열 데이터│ │텍스트 감정 │ │외부 시장 │
│ XGBoost + xAI │ │Neo4j+   │ │GRU+CNN+ │ │NLP+GPT-5│ │분석+GPT-5│
│ SHAP + LIME   │ │Graph    │ │Attention│ │키워드+생성│ │LLM 해석 │
└───────────────┘ └─────────┘ └─────────┘ └───────────┘ └─────────┘
        │               │           │           │           │
        ▼               ▼           ▼           ▼           ▼
┌───────────────┐ ┌─────────────┐ ┌─────────┐ ┌───────────┐ ┌─────────┐
│ 📊 IBM HR CSV │ │🗄️ Neo4j    │ │📈 시계열│ │📝 HR      │ │🌐 채용  │
│   데이터셋     │ │Graph DB    │ │CSV 데이터│ │텍스트     │ │공고 API │
└───────────────┘ └─────────────┘ └─────────┘ └───────────┘ └─────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │      🎯 Integration System         │
                    │         (포트: 5007)              │
                    │   GPT-5 기반 최종 종합 레포트       │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │      🌐 React Dashboard            │
                    │         (포트: 3000)              │
                    │   현대적 UI + 통합 분석 인터페이스   │
                    └─────────────────────────────────────┘
```

---

## 🚀 빠른 시작

### 방법 1: Docker Compose 사용 (가장 간편한 방법 🌟)

```bash
# 1. 저장소 클론
git clone <repository-url>
cd Agentic_AI_system

# 2. 환경 변수 설정
cp env.example .env
# .env 파일을 열어서 다음 값들을 설정하세요:
# - OPENAI_API_KEY: OpenAI API 키
# - NEO4J_PASSWORD: Neo4j 비밀번호

# 3. Docker Compose로 전체 시스템 실행
docker-compose up -d

# 4. 서비스 상태 확인
docker-compose ps

# 5. 로그 확인
docker-compose logs -f
```

**접속**:
- **React Dashboard**: http://localhost:3000
- **Supervisor API**: http://localhost:5006
- **Neo4j Browser**: http://localhost:7474

**중지 및 정리**:
```bash
# 서비스 중지
docker-compose down

# 볼륨까지 삭제 (데이터 초기화)
docker-compose down -v
```

> 💡 **자세한 Docker 가이드는 [DOCKER_SETUP.md](DOCKER_SETUP.md) 파일을 참조하세요.**

---

### 방법 2: Python 직접 실행 (개발 환경)

#### 1️⃣ 백엔드 서버 실행

```bash
# 1. 저장소 클론
git clone <repository-url>
cd Agentic_AI_system

# 2. Anaconda 환경 활성화 (필수)
conda activate nlp

# 3. 환경 변수 설정
export NEO4J_URI="bolt://54.162.43.24:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="resident-success-moss"
export OPENAI_API_KEY="your-gpt5nano-api-key"  # Sentio, Agora, Supervisor, Integration용

# 4. 모든 백엔드 서버 실행 🚀
python start_all_services_simple.py
```

**실행되는 서버들**:
- **Supervisor (포트 5006)**: LangGraph 워크플로우 관리 (최상위 관리자)
- **Structura (포트 5001)**: 정형 데이터 ML 분석
- **Cognita (포트 5002)**: 네트워크 관계 분석  
- **Chronos (포트 5003)**: 시계열 딥러닝 분석
- **Sentio (포트 5004)**: 텍스트 감정 분석
- **Agora (포트 5005)**: 시장 분석 + LLM
- **Integration (포트 5007)**: 최종 종합 레포트 생성

#### 2️⃣ 프론트엔드 실행

```bash
# 별도 터미널에서 실행
cd Dashboard
npm install
npm start
```

**접속**: http://localhost:3000 (React 대시보드)

#### 3️⃣ 상세 실행 가이드

전체 실행 가이드는 **[Launch.md](app/Launch.md)** 파일을 참조하세요.

---

## 🌐 React Dashboard 주요 기능

### 🎨 최신 UI/UX 개선사항

- **✅ 통일된 타이포그래피**: CSS 변수 기반 일관된 폰트 시스템
- **✅ 향상된 그룹 통계**: 부서/직무/직급별 Top 3 위험 요인 분석
- **✅ 실시간 분석**: 개별 직원부터 배치 분석까지 즉시 처리
- **✅ 인터랙티브 차트**: Recharts 기반 동적 시각화
- **✅ 반응형 디자인**: 모든 화면 크기 지원

### 📊 주요 페이지

1. **Home (홈)**: 시스템 소개 및 AI 챗봇
2. **Dashboard (대시보드)**: 전체 시스템 통계 및 현황
3. **FileUpload (파일 업로드)**: 데이터 업로드 및 검증
4. **BatchAnalysis (배치 분석)**: 대량 직원 분석 및 리포트
5. **EmployeePrediction (개별 예측)**: 단일 직원 상세 분석
6. **GroupStatistics (그룹 통계)**: 부서/직무/직급별 통계 분석
7. **RelationshipAnalysis (관계 분석)**: 네트워크 기반 조직 분석
8. **PostAnalysis (사후 분석)**: 예측 모델 평가 및 개선
9. **WeightOptimization (가중치 최적화)**: 에이전트 가중치 조정
10. **SupervisorWorkflow (워크플로우)**: Supervisor 작업 모니터링
11. **ReportGeneration (레포트 생성)**: 맞춤형 레포트 생성
12. **ExportResults (결과 내보내기)**: 분석 결과 다운로드

---

## 📋 데이터 요구사항

### 🏢 Structura - 정형 데이터 분석
**필수 데이터**: `data/IBM_HR.csv`
- **형식**: CSV 파일
- **필수 컬럼**: Age, JobSatisfaction, OverTime, MonthlyIncome, WorkLifeBalance 등
- **데이터 크기**: 1,470명 직원 데이터
- **자동 로드**: 서버 시작 시 자동으로 데이터 로드 및 모델 훈련

### 🕸️ Cognita - 관계형 데이터 분석
**필수 설정**: Neo4j 데이터베이스 연결
- **환경변수 필수**:
  ```bash
  NEO4J_URI=bolt://your-neo4j-host:7687
  NEO4J_USERNAME=neo4j
  NEO4J_PASSWORD=your-password
  ```
- **데이터 구조**: Employee, Department, Project 노드 및 COLLABORATES_WITH, REPORTS_TO 관계
- **자동 연결**: 환경변수 설정 시 자동으로 Neo4j 연결

### ⏰ Chronos - 시계열 데이터 분석
**필수 데이터**: `data/IBM_HR_timeseries.csv`
- **형식**: 시계열 CSV 파일
- **필수 컬럼**: employee_id, week, 각종 시계열 피처들
- **시퀀스 길이**: 6주 단위 시계열 데이터
- **자동 로드**: 서버 시작 시 자동으로 데이터 로드 및 전처리

### 📝 Sentio - 텍스트 감정 분석
**필수 데이터**: `data/IBM_HR_text.csv` 또는 `sample_hr_texts.csv`
- **형식**: CSV 파일 (텍스트 컬럼 포함)
- **필수 컬럼**: employee_id, text, text_type
- **API 키**: GPT-5-nano 사용을 위한 OpenAI API 키 (.env 파일 지원)
- **자동 로드**: 서버 시작 시 자동으로 텍스트 데이터 로드

### 🌍 Agora - 외부 시장 분석
**데이터 소스**: Structura의 IBM_HR.csv 데이터 활용
- **기본 데이터**: `data/IBM_HR.csv`에서 직무, 급여 정보 추출
- **외부 API**: 채용 공고 및 시장 데이터 (시뮬레이션)
- **API 키**: GPT-5-nano 사용을 위한 OpenAI API 키 (.env 파일 지원)
- **자동 연동**: Structura 데이터를 기반으로 시장 분석 수행

### 🎯 Supervisor - 워커 통합 관리
**데이터 소스**: 워커 에이전트들의 분석 결과
- **입력**: 5개 워커 에이전트의 API 응답 결과
- **API 키**: GPT-5-nano 워크플로우를 위한 OpenAI API 키 (.env 파일 지원)
- **자동 통합**: 워커 에이전트 결과를 자동으로 수집 및 합성

### 🎯 Integration - 최종 종합 레포트
**데이터 소스**: 모든 에이전트의 점수 및 직원 기본 정보
- **입력**: 각 에이전트별 점수 (structura_score, cognita_score 등)
- **기본 데이터**: 직원 기본 정보 (employee_id, 이름, 부서 등)
- **API 키**: GPT-5-nano 레포트 생성을 위한 OpenAI API 키 (.env 파일 지원)
- **자동 처리**: 점수 입력 시 자동으로 임계값 계산 및 레포트 생성

---

## 🤖 워커 에이전트 상세

### 🏢 Structura - 정형 데이터 분석 에이전트

**포트**: `5001` | **기술**: XGBoost + xAI (SHAP, LIME)

#### 🎯 주요 기능
- **이직 확률 예측**: 개별 직원의 이직 가능성 정량화
- **설명 가능한 AI**: SHAP, LIME을 통한 투명한 의사결정
- **위험 요인 분석**: 이직에 영향을 미치는 핵심 요소 식별
- **실시간 예측**: 즉시 분석 결과 제공

#### 📊 성능 지표
- **예측 속도**: ~0.1초/명
- **xAI 분석**: ~1-3초/명  
- **정확도**: ROC-AUC 0.85+
- **메모리 사용**: 500MB-2GB

#### 🔗 주요 API
- `POST /api/predict` - 이직 확률 예측
- `POST /api/explain` - 예측 설명 (SHAP/LIME)
- `GET /api/feature-importance` - 피처 중요도 분석

---

### 🕸️ Cognita - 관계형 데이터 분석 에이전트

**포트**: `5002` | **기술**: Neo4j + Graph Analytics

#### 🎯 주요 기능
- **사회적 네트워크 분석**: 직원 간 관계 패턴 분석
- **관계형 위험도 평가**: 네트워크 기반 이직 위험 예측
- **조직 구조 분석**: 팀 역학 및 관리 체계 평가
- **부서별 종합 분석**: 조직 단위 위험도 모니터링

#### 📈 성능 지표 (v1.1.0 최적화)
- **분석 속도**: 평균 0.82초/명 (30% 향상)
- **부서 분석**: 3-15초 (샘플링 최적화)
- **처리량**: ~4,400명/시간
- **메모리 사용**: 300MB-1GB (40% 감소)

#### 🔗 주요 API
- `GET /api/analyze/employee/{id}` - 개별 직원 관계형 분석
- `POST /api/analyze/department` - 부서별 위험도 분석
- `GET /api/employees` - 직원 목록 조회
- `GET /api/departments` - 부서 목록 조회

---

### ⏰ Chronos - 시계열 데이터 분석 에이전트

**포트**: `5003` | **기술**: GRU+CNN+Attention 하이브리드 딥러닝

#### 🎯 주요 기능
- **시계열 패턴 분석**: 직원 행동 패턴의 시간적 변화 추적
- **Attention 메커니즘**: 중요한 시점과 피처 자동 식별
- **딥러닝 예측**: GRU+CNN 하이브리드 모델로 고정밀 예측
- **인터랙티브 시각화**: Plotly 기반 동적 차트 제공

#### 📈 성능 지표
- **모델 아키텍처**: GRU + CNN + Feature/Temporal Attention
- **시퀀스 길이**: 6주 단위 시계열 분석
- **예측 정확도**: 딥러닝 기반 고정밀 예측
- **시각화**: 실시간 인터랙티브 차트

#### 🔗 주요 API
- `POST /api/train` - 모델 학습 및 하이퍼파라미터 최적화
- `POST /api/predict` - 시계열 기반 이직 예측
- `POST /api/predict_batch` - 배치 예측 처리
- `GET /api/visualize/attention` - Attention 가중치 시각화
- `GET /api/visualize/features` - 피처 중요도 시각화

---

### 📝 Sentio - 텍스트 감정 분석 에이전트

**포트**: `5004` | **기술**: NLP + 키워드 분석 + GPT-5-nano 텍스트 생성

#### 🎯 주요 기능
- **텍스트 감정 분석**: HR 텍스트의 감정 점수 및 퇴직 위험 탐지
- **키워드 분석**: 명사 중심 정확한 키워드 추출 (노이즈 제거)
- **퇴직 위험 신호 탐지**: 5가지 퇴직 원인별 위험 신호 분석
- **페르소나 기반 텍스트 생성**: 10가지 직원 유형별 맞춤 텍스트 생성
- **JD-R 모델 기반 분석**: Job Demands-Resources 모델 적용

#### 📊 성능 지표
- **키워드 추출**: 명사 중심, 500+ 불용어 필터링
- **감정 분석**: 0.0~1.0 점수 체계
- **퇴직 위험 분석**: 5가지 원인 그룹별 세부 분석
- **텍스트 생성**: GPT-5-nano 기반 고품질 생성 (.env 지원)
- **API 호출**: client.responses.create() 방식 사용

#### 🔗 주요 API
- `POST /analyze/text` - 텍스트 분석 (키워드 + 감정 + 위험도)
- `POST /analyze/keywords` - 퇴직자 vs 재직자 차별적 키워드 분석
- `POST /analyze/risk` - 배치 퇴직 위험 분석
- `POST /generate/text` - 페르소나 기반 텍스트 생성
- `GET /data/personas` - 페르소나 정보 조회

---

### 🌍 Agora - 외부 시장 분석 에이전트

**포트**: `5005` | **기술**: 시장 데이터 분석 + 경쟁력 평가 + GPT-5-nano LLM 해석

#### 🎯 주요 기능
- **시장 압력 지수 계산**: 외부 시장의 채용 수요 및 경쟁 상황 분석
- **보상 격차 분석**: 현재 급여와 시장 평균 급여 간의 격차 계산
- **이직 위험도 평가**: 시장 상황을 고려한 직원별 이직 위험도 산출
- **직무별 시장 분석**: 특정 직무의 채용 공고, 급여 수준, 트렌드 분석
- **경쟁력 평가**: 개별 직원의 시장 대비 경쟁력 종합 평가
- **GPT-5-nano 기반 해석**: 최신 LLM을 활용한 자연스러운 분석 결과 해석

#### 📊 성능 지표
- **응답 시간**: < 500ms (개별 분석)
- **배치 처리**: 100명/분 (LLM 미사용)
- **캐시 적중률**: > 80% (1시간 TTL)
- **API 안정성**: 99.9% 가용성

#### 🔗 주요 API
- `POST /analyze/market` - 개별 직원 시장 분석
- `POST /analyze/job_market` - 직무별 시장 분석
- `POST /analyze/batch` - 배치 시장 분석
- `GET /market/report/<job_role>` - 직무별 시장 보고서
- `GET /market/trends` - 전체 시장 트렌드
- `POST /market/competitive_analysis` - 경쟁력 분석

---

### 🎯 Integration - 최종 종합 레포트 생성 시스템

**포트**: `5007` | **기술**: GPT-5-nano 기반 종합 분석 및 레포트 생성

#### 🎯 주요 기능
- **임계값 설정**: 각 Score별 최적 임계값 계산 및 F1-score 최적화
- **가중치 최적화**: Grid Search, Bayesian Optimization, Scipy 최적화 지원
- **위험도 분류**: 3단계 위험도 구간 분류 (안전군/주의군/고위험군)
- **GPT-5-nano 레포트 생성**: 개별 직원별 맞춤형 퇴사 위험 분석 레포트
- **일괄 레포트 생성**: 여러 직원의 레포트 동시 생성

#### 📊 성능 지표
- **임계값 계산**: F1-score 기반 최적화
- **가중치 최적화**: 다양한 알고리즘 지원
- **레포트 생성**: GPT-5-nano 기반 고품질 분석
- **API 호출**: client.responses.create() 방식 사용
- **Fallback 시스템**: LLM 실패 시 규칙 기반 분석 제공

#### 🔗 주요 API
- `POST /set_api_key` - OpenAI API 키 설정
- `POST /load_data` - 데이터 로드 및 전처리
- `POST /calculate_thresholds` - 임계값 계산
- `POST /optimize_weights` - 가중치 최적화
- `POST /generate_report` - 개별 직원 레포트 생성 (LLM 지원)
- `POST /generate_batch_reports` - 일괄 레포트 생성
- `POST /load_employee_data` - 직원 기본 데이터 로드
- `GET /get_employee_list` - 직원 목록 조회

---

## 🌐 React 연동 가이드

### 통합 API 사용 예시

```javascript
// 🤖 개별 직원 통합 분석 (Supervisor → 5개 워커 에이전트 → Integration)
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
  
  // 🔍 Supervisor를 통한 통합 결과 활용
  console.log('최종 레포트:', result.final_report);
  console.log('위험 점수:', result.final_report.risk_score);
  console.log('위험 등급:', result.final_report.risk_grade);
  console.log('이탈 확률:', result.final_report.attrition_probability);
  console.log('권장사항:', result.final_report.recommendations);
  
  return result;
};

// 🎯 배치 분석
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

| 특징 | 🏢 Structura | 🕸️ Cognita | ⏰ Chronos | 📝 Sentio | 🌍 Agora | 🎯 Integration |
|------|-------------|------------|-----------|-----------|-----------|---------------|
| **분석 방식** | 개별 직원 데이터 | 관계형 네트워크 | 시계열 패턴 | 텍스트 감정 분석 | 외부 시장 분석 | 최종 종합 레포트 |
| **데이터 소스** | CSV (IBM HR) | Neo4j 그래프 DB | 시계열 CSV | HR 텍스트 데이터 | 채용 공고 API | 에이전트 점수 |
| **주요 기술** | XGBoost + xAI | Graph Analytics | GRU+CNN+Attention | NLP + GPT-5-nano | 시장 분석 + GPT-5-nano | GPT-5-nano 레포트 |
| **포트** | 5001 | 5002 | 5003 | 5004 | 5005 | 5007 |
| **예측 대상** | 이직 확률 | 관계형 위험도 | 시계열 기반 예측 | 퇴직 위험 신호 | 시장 압력 지수 | 종합 위험도 분석 |
| **설명 가능성** | SHAP, LIME | 네트워크 분석 | Attention 메커니즘 | 키워드 + LLM 분석 | 시장 + LLM 해석 | LLM 기반 레포트 |

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

### 🌐 React Dashboard (포트 3000)

| 페이지 | 경로 | 설명 |
|--------|------|------|
| Home | `/` | 시스템 소개 및 AI 챗봇 |
| Dashboard | `/dashboard` | 전체 시스템 통계 |
| 파일 업로드 | `/file-upload` | 데이터 업로드 및 검증 |
| 배치 분석 | `/batch-analysis` | 대량 직원 분석 |
| 그룹 통계 | `/group-statistics` | 부서/직무/직급별 통계 |
| 관계 분석 | `/relationship-analysis` | 네트워크 분석 |

---

## 📁 결과 저장 시스템

### 🗂️ 체계적 폴더 구조

```
results/
├── 📁 employees/                    # 개별 직원별 결과
│   └── 📁 employee_{ID}/
│       ├── 📄 employee_info.json    # 기본 정보
│       ├── 📄 analysis_summary.csv  # 통합 요약
│       ├── 📄 structura_result.json # Structura 상세 결과
│       ├── 📄 cognita_result.json   # Cognita 상세 결과
│       └── 📁 visualizations/       # 시각화 파일들
│
├── 📁 departments/                  # 부서별 집계 결과
│   └── 📁 {부서명}/
│       └── 📄 department_summary.csv
│
├── 📁 reports/                      # 종합 보고서
│   └── 📄 {부서명}_report.json
│
└── 📁 models/                       # 저장된 모델들
    ├── 📄 structura_model.pkl
    ├── 📄 chronos_model.pth
    └── 📄 chronos_scaler.joblib
```

---

## 🧪 테스트 실행

### 통합 시스템 테스트

```bash
# 🤖 통합 에이전틱 시스템 테스트
cd app
python test_agentic_system.py

# 📁 결과 저장 시스템 테스트
python test_result_system.py
```

### 개별 워커 테스트

```bash
# 각 에이전트별 테스트
cd app/Structura && python test_structura_api.py
cd app/Cognita && python test_cognita_api.py
cd app/Chronos && python test_chronos_api.py
cd app/Sentio && python test_sentio_api.py
cd app/Agora && python test_agora_api.py
cd app/Supervisor && python test_supervisor_api.py
cd app/Integration && python test_integration_api.py
```

---

## 📁 프로젝트 구조

```
Agentic_AI_system/
├── 📁 app/                          # 백엔드 애플리케이션
│   ├── 📁 Structura/                # 정형 데이터 분석 워커
│   ├── 📁 Cognita/                  # 관계형 데이터 분석 워커
│   ├── 📁 Chronos/                  # 시계열 데이터 분석 워커
│   ├── 📁 Sentio/                   # 텍스트 감정 분석 워커
│   ├── 📁 Agora/                    # 외부 시장 분석 워커
│   ├── 📁 Supervisor/               # 워커 통합 관리 에이전트
│   ├── 📁 Integration/              # 최종 종합 레포트 시스템
│   ├── 🐳 Dockerfile                # GPU 버전 Docker 이미지
│   ├── 🐳 Dockerfile.cpu            # CPU 버전 Docker 이미지
│   └── 📋 requirements_agentic.txt  # 통합 의존성
│
├── 📁 Dashboard/                    # React 프론트엔드
│   ├── 📁 src/
│   │   ├── 📁 components/           # React 컴포넌트
│   │   ├── 📁 styles/               # CSS 스타일
│   │   └── 📄 App.js                # 메인 앱
│   ├── 🐳 Dockerfile                # Dashboard Docker 이미지
│   └── 📦 package.json              # npm 의존성
│
├── 📁 data/                         # 데이터셋
│   ├── 📊 IBM_HR.csv                # IBM HR 데이터셋
│   ├── 📊 IBM_HR_timeseries.csv     # 시계열 데이터
│   ├── 📝 IBM_HR_text.csv           # HR 텍스트 데이터
│   └── 🕸️ employee_relationships.json
│
├── 📁 results/                      # 분석 결과
│   ├── 📁 employees/                # 개별 직원 결과
│   ├── 📁 departments/              # 부서별 결과
│   └── 📁 reports/                  # 종합 보고서
│
├── 🐳 docker-compose.yml            # Docker Compose 설정
├── 📄 env.example                   # 환경 변수 예시
├── 📖 DOCKER_SETUP.md               # Docker 설치 가이드
├── 🚀 start_all_services_simple.py  # 전체 서비스 실행 스크립트
└── 📖 README.md                     # 이 파일
```

---

## 📈 성능 특성

### 🏢 Structura 성능
- **예측 속도**: ~0.1초/명
- **xAI 분석**: ~1-3초/명
- **메모리 사용**: 500MB-2GB
- **정확도**: ROC-AUC 0.85+
- **처리량**: ~36,000명/시간

### 🕸️ Cognita 성능 (v1.1.0 최적화)
- **분석 속도**: 평균 0.82초/명 (30% 향상)
- **부서 분석**: 3-15초 (샘플링 최적화)
- **처리량**: ~4,400명/시간
- **메모리 사용**: 300MB-1GB (40% 감소)

### ⏰ Chronos 성능
- **모델 아키텍처**: GRU + CNN + Dual Attention
- **시퀀스 길이**: 6주 단위 시계열 분석
- **예측 속도**: 딥러닝 기반 실시간 예측
- **시각화**: Plotly 인터랙티브 차트

### 📝 Sentio 성능
- **키워드 추출**: 명사 중심, 500+ 불용어 필터링
- **텍스트 생성**: GPT-5-nano 기반 고품질 생성
- **API 호출**: client.responses.create() 방식

### 🌍 Agora 성능
- **응답 시간**: < 500ms (개별 분석)
- **배치 처리**: 100명/분 (LLM 미사용)
- **캐시 적중률**: > 80% (1시간 TTL)
- **LLM 해석**: GPT-5-nano 기반

### 확장성 평가
- **소규모 조직** (100명 미만): 실시간 분석 가능
- **중간 규모 조직** (1,000명 미만): 준실시간 분석 가능  
- **대규모 조직** (1,000명 이상): 배치 분석 권장

---

## 🚨 문제 해결

### Docker 관련 문제

#### 1. Docker 서비스 시작 실패
```bash
# Docker 데몬 상태 확인
docker info

# Docker Compose 로그 확인
docker-compose logs -f

# 특정 서비스 로그 확인
docker-compose logs -f supervisor
```

#### 2. 포트 충돌
```bash
# 사용 중인 포트 확인
netstat -an | grep :3000  # Dashboard
netstat -an | grep :5006  # Supervisor

# Docker 컨테이너 중지
docker-compose down
```

#### 3. 환경 변수 문제
```bash
# .env 파일 확인
cat .env

# 환경 변수가 컨테이너에 전달되었는지 확인
docker-compose exec supervisor env | grep OPENAI_API_KEY
```

### Python 실행 관련 문제

#### 1. 의존성 설치 오류
```bash
# 가상환경 사용 권장
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r app/requirements_agentic.txt
```

#### 2. Neo4j 연결 문제
```bash
# Neo4j 연결 확인
export NEO4J_URI="bolt://54.162.43.24:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="resident-success-moss"

# 연결 테스트
curl -u neo4j:resident-success-moss http://54.162.43.24:7474/db/data/
```

#### 3. OpenAI API 키 문제
```bash
# API 키 설정 확인
export OPENAI_API_KEY="your-gpt5nano-api-key"
echo $OPENAI_API_KEY

# .env 파일 확인
cat app/Sentio/.env
```

---

## 🔮 향후 계획

### 단기 (1-2주)
- [x] **Docker Compose 지원**: 간편한 배포 및 관리 - 완료 ✅
- [x] **React 프론트엔드 개선**: UI/UX 개선 및 폰트 통일 - 완료 ✅
- [x] **그룹 통계 기능 강화**: 부서별 위험 요인 분석 - 완료 ✅
- [ ] **실시간 모니터링**: 시스템 성능 및 분석 결과 추적
- [ ] **배치 처리 최적화**: 대용량 데이터 처리 개선

### 중기 (3-4주)
- [ ] **성능 최적화**: 워커 에이전트 간 통신 및 처리 속도 향상
- [ ] **고급 분석 기능**: 예측 모델 정확도 개선
- [ ] **클라우드 배포**: AWS/Azure 기반 확장

### 장기 (6개월+)
- [ ] **실시간 스트리밍**: 실시간 데이터 처리
- [ ] **AI 추천 시스템**: 자동화된 인사 정책 제안
- [ ] **다국어 지원**: 영어/한국어 UI 지원

---

## 🏆 주요 성과

### 기술적 성과
- ✅ **7개 에이전트 아키텍처** 구현 (5개 워커 + Supervisor + Integration)
- ✅ **설명 가능한 AI** (SHAP, LIME, Attention) 적용
- ✅ **관계형 네트워크 분석** 시스템 구축
- ✅ **텍스트 감정 분석** 시스템 구축 (Sentio - GPT-5-nano)
- ✅ **시계열 딥러닝 분석** 시스템 구축 (Chronos)
- ✅ **외부 시장 분석** 시스템 구축 (Agora - GPT-5-nano)
- ✅ **워커 통합 관리** 시스템 구축 (Supervisor - LangGraph)
- ✅ **최종 종합 레포트** 시스템 구축 (Integration - GPT-5-nano)
- ✅ **React 대시보드** 완성 및 UI/UX 개선
- ✅ **Docker Compose 지원** - 간편한 배포
- ✅ **환경변수 지원** (.env 파일 통합 관리)

### 성능 성과
- ✅ **Structura**: 0.1초/명 예측 속도 달성
- ✅ **Cognita**: 30% 성능 향상 (v1.1.0)
- ✅ **Sentio**: GPT-5-nano 기반 고품질 텍스트 분석
- ✅ **Chronos**: GRU+CNN+Attention 하이브리드 모델 구현
- ✅ **Agora**: GPT-5-nano 기반 시장 분석 및 해석
- ✅ **Supervisor**: LangGraph 기반 워크플로우 자동화
- ✅ **Integration**: GPT-5-nano 기반 맞춤형 레포트 생성
- ✅ **통합 시스템**: 7개 에이전트 동시 실행
- ✅ **확장성**: 대규모 조직 지원 가능

---

## 👥 기여자

**개발팀**
- 시스템 아키텍처 설계
- AI 모델 개발 및 최적화
- 웹 서비스 구현
- 성능 튜닝 및 최적화
- Docker 컨테이너화

---

## 📄 라이선스

이 프로젝트는 오픈소스 연구 프로젝트입니다.

---

## 📞 지원 및 문의

프로젝트 관련 문의사항이나 기술 지원이 필요한 경우:

1. **GitHub Issues**: 버그 리포트 및 기능 요청
2. **기술 문서**: 
   - [DOCKER_SETUP.md](DOCKER_SETUP.md) - Docker 설치 및 실행 가이드
   - [Launch.md](app/Launch.md) - 상세 실행 가이드
   - [README/](README/) 폴더 - 각종 기술 문서
3. **개별 워커 문서**: 각 워커 디렉토리의 README 참조

---

**버전**: 2.0.0 (Docker Compose & UI Enhancement)  
**최종 업데이트**: 2025년 10월  
**기술 스택**: Python, Flask, XGBoost, Neo4j, PyTorch, GPT-5-nano, LangGraph, React, Docker, Docker Compose  
**아키텍처**: Multi-Agent AI System (Supervisor + 5 Workers + Integration) + React Dashboard  
