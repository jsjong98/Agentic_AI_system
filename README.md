# 🤖 Agentic AI System - HR Analytics Platform

**에이전틱 AI 기반 HR 분석 플랫폼**

차세대 인사 분석을 위한 다중 에이전트 AI 시스템으로, 정형 데이터, 관계형 데이터, 시계열 데이터, 텍스트 데이터를 동시에 분석하여 종합적인 인사 인사이트를 제공합니다.

## 🌟 주요 특징

- **🤖 에이전틱 아키텍처**: 5개 AI 에이전트가 협력하여 분석
- **🔍 다차원 분석**: 정형 + 관계형 + 시계열 + 텍스트 + 시장 데이터 통합 분석
- **⚡ 실시간 처리**: 개별 직원부터 전체 조직까지 즉시 분석
- **🔬 설명 가능한 AI**: SHAP, LIME, Attention 기반 투명한 의사결정
- **🕸️ 관계형 인사이트**: Neo4j 기반 조직 네트워크 분석
- **📝 텍스트 감정 분석**: NLP 기반 퇴직 위험 신호 탐지
- **📈 시계열 패턴 분석**: 딥러닝 기반 시간적 변화 추적
- **🌍 외부 시장 분석**: 시장 압력 지수 및 경쟁력 평가
- **🌐 React 연동**: 현대적 웹 인터페이스 완벽 지원

---

## 🏗️ 시스템 아키텍처

```
                    ┌─────────────────────────────────────┐
                    │      🎯 Agentic Master Server      │
                    │         (포트: 8000)              │
                    │      통합 조정 및 결과 합성         │
                    └─────────────────────────────────────┘
                                    │
        ┌───────────┬───────────┬───────────┬───────────┬───────────┐
        │           │           │           │           │           │
┌───────▼───────┐ ┌─▼─────┐ ┌──▼──┐ ┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐
│  🏢 Structura  │ │🕸️Cognita│ │⏰Chronos│ │📝 Sentio │ │🌍Agora│ │  🔮 Future │
│   워커 에이전트  │ │워커 에이전트│ │워커 에이전트│ │ 워커 에이전트│ │워커 에이전트│ │   워커 에이전트 │
│  (포트: 5001)  │ │(포트:5000)│ │(포트:5002)│ │(포트: 5003)│ │(포트:5004)│ │  (포트: 50XX) │
│               │ │         │ │         │ │           │ │         │ │             │
│ 정형 데이터 분석 │ │관계형 데이터│ │시계열 데이터│ │텍스트 감정 │ │외부 시장 │ │ 향후 확장 예정 │
│ XGBoost + xAI │ │Neo4j+   │ │GRU+CNN+ │ │NLP+키워드│ │분석+경쟁력│ │             │
│ SHAP + LIME   │ │Graph    │ │Attention│ │분석+생성 │ │평가+LLM │ │             │
└───────────────┘ └─────────┘ └─────────┘ └───────────┘ └─────────┘ └─────────────┘
        │               │           │           │           │             │
        ▼               ▼           ▼           ▼           ▼             ▼
┌───────────────┐ ┌─────────────┐ ┌─────────┐ ┌───────────┐ ┌─────────┐ ┌─────────────┐
│ 📊 IBM HR CSV │ │🗄️ Neo4j    │ │📈 시계열│ │📝 HR      │ │🌐 채용  │ │ 🔮 미래 데이터│
│   데이터셋     │ │Graph DB    │ │CSV 데이터│ │텍스트     │ │공고 API │ │   소스       │
└───────────────┘ └─────────────┘ └─────────┘ └───────────┘ └─────────┘ └─────────────┘
```

---

## 🚀 빠른 시작

### 1️⃣ 통합 시스템 실행 (권장)

```bash
# 1. 저장소 클론
git clone <repository-url>
cd Agentic_AI_system

# 2. 통합 의존성 설치
cd app
pip install -r requirements_agentic.txt

# 3. 환경 변수 설정
export NEO4J_URI="bolt://54.162.43.24:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="resident-success-moss"
export OPENAI_API_KEY="your-openai-api-key"  # Sentio용

# 4. 통합 마스터 서버 실행 🚀
python run_agentic_system.py
```

**접속**: http://localhost:8000 (통합 마스터 서버)

### 2️⃣ 시스템 테스트

```bash
# 별도 터미널에서 실행
cd app
python test_agentic_system.py
```

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

**포트**: `5000` | **기술**: Neo4j + Graph Analytics

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

**포트**: `5002` | **기술**: GRU+CNN+Attention 하이브리드 딥러닝

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

**포트**: `5003` | **기술**: NLP + 키워드 분석 + GPT-4 텍스트 생성

#### 🎯 주요 기능
- **텍스트 감정 분석**: HR 텍스트의 감정 점수 및 퇴직 위험 탐지
- **키워드 분석**: 명사 중심 정확한 키워드 추출 (노이즈 제거)
- **퇴직 위험 신호 탐지**: 5가지 퇴직 원인별 위험 신호 분석
- **페르소나 기반 텍스트 생성**: 10가지 직원 유형별 맞춤 텍스트 생성

#### 📊 성능 지표
- **키워드 추출**: 명사 중심, 500+ 불용어 필터링
- **감정 분석**: 0.0~1.0 점수 체계
- **퇴직 위험 분석**: 5가지 원인 그룹별 세부 분석
- **텍스트 생성**: OpenAI GPT-4 기반 고품질 생성

#### 🔗 주요 API
- `POST /analyze/text` - 텍스트 분석 (키워드 + 감정 + 위험도)
- `POST /analyze/keywords` - 퇴직자 vs 재직자 차별적 키워드 분석
- `POST /analyze/risk` - 배치 퇴직 위험 분석
- `POST /generate/text` - 페르소나 기반 텍스트 생성
- `GET /data/personas` - 페르소나 정보 조회

---

### 🌍 Agora - 외부 시장 분석 에이전트

**포트**: `5004` | **기술**: 시장 데이터 분석 + 경쟁력 평가 + LLM 해석

#### 🎯 주요 기능
- **시장 압력 지수 계산**: 외부 시장의 채용 수요 및 경쟁 상황 분석
- **보상 격차 분석**: 현재 급여와 시장 평균 급여 간의 격차 계산
- **이직 위험도 평가**: 시장 상황을 고려한 직원별 이직 위험도 산출
- **직무별 시장 분석**: 특정 직무의 채용 공고, 급여 수준, 트렌드 분석
- **경쟁력 평가**: 개별 직원의 시장 대비 경쟁력 종합 평가
- **LLM 기반 해석**: OpenAI GPT를 활용한 자연스러운 분석 결과 해석

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

## 🌐 React 연동 가이드

### 통합 API 사용 예시

```javascript
// 🤖 개별 직원 통합 분석 (5개 워커 에이전트 동시 실행)
const analyzeEmployeeIntegrated = async (employeeData, textData) => {
  const response = await fetch('http://localhost:8000/api/analyze/individual', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...employeeData,
      text_data: textData,
      market_data: { use_llm: false },  // 시장 분석 옵션
      use_structura: true,  // 정형 데이터 분석
      use_cognita: true,    // 관계형 데이터 분석
      use_sentio: true,     // 텍스트 감정 분석
      use_chronos: true,    // 시계열 데이터 분석
      use_agora: true       // 외부 시장 분석
    })
  });
  
  const result = await response.json();
  
  // 🔍 통합 결과 활용
  console.log('Structura 결과:', result.structura_result);
  console.log('Cognita 결과:', result.cognita_result);
  console.log('Sentio 결과:', result.sentio_result);
  console.log('Chronos 결과:', result.chronos_result);
  console.log('Agora 결과:', result.agora_result);
  console.log('통합 분석:', result.combined_analysis);
  
  return result;
};

// 📝 텍스트 분석 전용
const analyzeText = async (text, employeeId) => {
  const response = await fetch('http://localhost:5003/analyze/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: text,
      employee_id: employeeId,
      text_type: "SELF_REVIEW"
    })
  });
  return response.json();
};

// 🎭 페르소나 기반 텍스트 생성
const generatePersonaText = async (employeeData, textType) => {
  const response = await fetch('http://localhost:5003/generate/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      employee_data: employeeData,
      text_type: textType
    })
  });
  return response.json();
};
```

### React Hook 예시

```javascript
import { useState, useEffect } from 'react';

const useAgenticAI = () => {
  const [systemHealth, setSystemHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const baseURL = 'http://localhost:8000/api';
  
  // 🤖 통합 직원 분석 (4개 워커 에이전트)
  const analyzeEmployee = async (employeeData, textData) => {
    try {
      setLoading(true);
      const response = await fetch(`${baseURL}/analyze/individual`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...employeeData,
          text_data: textData,
          use_structura: true,
          use_cognita: true,
          use_sentio: true,
          use_chronos: true
        })
      });
      const result = await response.json();
      setError(null);
      return result;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  };
  
  // 📝 텍스트 전용 분석
  const analyzeTextOnly = async (text, employeeId) => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5003/analyze/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          employee_id: employeeId
        })
      });
      const result = await response.json();
      setError(null);
      return result;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  };
  
  // 🔍 시스템 헬스체크
  const checkHealth = async () => {
    try {
      const response = await fetch(`${baseURL}/health`);
      const data = await response.json();
      setSystemHealth(data);
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  };
  
  useEffect(() => {
    checkHealth();
  }, []);
  
  return {
    systemHealth,
    loading,
    error,
    analyzeEmployee,
    analyzeTextOnly,
    checkHealth
  };
};
```

---

## 📊 시스템 비교

| 특징 | 🏢 Structura | 🕸️ Cognita | ⏰ Chronos | 📝 Sentio | 🌍 Agora |
|------|-------------|------------|-----------|-----------|-----------|
| **분석 방식** | 개별 직원 데이터 | 관계형 네트워크 | 시계열 패턴 | 텍스트 감정 분석 | 외부 시장 분석 |
| **데이터 소스** | CSV (IBM HR) | Neo4j 그래프 DB | 시계열 CSV | HR 텍스트 데이터 | 채용 공고 API |
| **주요 기술** | XGBoost + xAI | Graph Analytics | GRU+CNN+Attention | NLP + GPT-4 | 시장 분석 + LLM |
| **포트** | 5001 | 5000 | 5002 | 5003 | 5004 |
| **예측 대상** | 이직 확률 | 관계형 위험도 | 시계열 기반 예측 | 퇴직 위험 신호 | 시장 압력 지수 |
| **설명 가능성** | SHAP, LIME | 네트워크 분석 | Attention 메커니즘 | 키워드 분석 | 시장 해석 |
| **분석 범위** | 개별 중심 | 조직/팀 중심 | 시간적 패턴 중심 | 텍스트 중심 | 시장 중심 |
| **실시간성** | 즉시 예측 | 실시간 관계 분석 | 딥러닝 예측 | 실시간 텍스트 분석 | 실시간 시장 분석 |
| **성능** | 0.1초/명 | 0.82초/명 | 딥러닝 기반 | 키워드 기반 | 0.5초/명 |

---

## 🔧 주요 API 엔드포인트

### 🎯 통합 마스터 서버 (포트 8000) - 권장

| 메서드 | 엔드포인트 | 설명 |
|--------|------------|------|
| `GET` | `/api/health` | 시스템 전체 상태 확인 |
| `GET` | `/api/workers/status` | 워커 에이전트 상태 조회 |
| `POST` | `/api/analyze/individual` | **개별 직원 통합 분석** (5개 워커 동시) |
| `POST` | `/api/analyze/department` | **부서별 통합 분석** (5개 워커 동시) |
| `GET` | `/api/task/{task_id}/result` | 작업 결과 조회 |
| `GET` | `/api/results/employee/{id}` | **직원 결과 조회** (저장된 모든 결과) |
| `GET` | `/api/results/employee/{id}/visualizations` | **직원 시각화 목록** (PNG 파일들) |
| `GET` | `/api/results/department/{name}/report` | **부서별 종합 보고서** |

### 🏢 Structura 워커 (포트 5001)

| 메서드 | 엔드포인트 | 설명 |
|--------|------------|------|
| `GET` | `/api/health` | 서버 상태 확인 |
| `POST` | `/api/train` | 모델 훈련 |
| `POST` | `/api/predict` | 이직 예측 |
| `POST` | `/api/explain` | 예측 설명 (xAI) |
| `GET` | `/api/feature-importance` | 피처 중요도 |

### 🕸️ Cognita 워커 (포트 5000)

| 메서드 | 엔드포인트 | 설명 |
|--------|------------|------|
| `GET` | `/api/health` | 서버 상태 확인 |
| `GET` | `/api/employees` | 직원 목록 |
| `GET` | `/api/departments` | 부서 목록 |
| `GET` | `/api/analyze/employee/{id}` | 직원 분석 |
| `POST` | `/api/analyze/department` | 부서 분석 |

### ⏰ Chronos 워커 (포트 5002)

| 메서드 | 엔드포인트 | 설명 |
|--------|------------|------|
| `GET` | `/api/health` | 서버 상태 확인 |
| `POST` | `/api/train` | 모델 학습 |
| `POST` | `/api/predict` | 시계열 예측 |
| `POST` | `/api/predict_batch` | 배치 예측 |
| `GET` | `/api/visualize/attention` | Attention 시각화 |
| `GET` | `/api/visualize/features` | 피처 중요도 시각화 |

### 📝 Sentio 워커 (포트 5003)

| 메서드 | 엔드포인트 | 설명 |
|--------|------------|------|
| `GET` | `/health` | 서버 상태 확인 |
| `POST` | `/analyze/text` | 텍스트 분석 |
| `POST` | `/analyze/keywords` | 키워드 분석 |
| `POST` | `/analyze/risk` | 퇴직 위험 분석 |
| `POST` | `/generate/text` | 텍스트 생성 |
| `GET` | `/data/personas` | 페르소나 정보 |

### 🌍 Agora 워커 (포트 5004)

| 메서드 | 엔드포인트 | 설명 |
|--------|------------|------|
| `GET` | `/health` | 서버 상태 확인 |
| `POST` | `/analyze/market` | 개별 직원 시장 분석 |
| `POST` | `/analyze/job_market` | 직무별 시장 분석 |
| `POST` | `/analyze/batch` | 배치 시장 분석 |
| `GET` | `/market/report/<job_role>` | 직무별 시장 보고서 |
| `GET` | `/market/trends` | 전체 시장 트렌드 |
| `POST` | `/market/competitive_analysis` | 경쟁력 분석 |

---

## 📁 결과 저장 시스템

### 🗂️ 체계적 폴더 구조

모든 분석 결과는 다음과 같은 체계적인 폴더 구조로 저장됩니다:

```
results/
├── 📁 employees/                    # 개별 직원별 결과
│   └── 📁 employee_{ID}/
│       ├── 📄 employee_info.json    # 기본 정보
│       ├── 📄 analysis_summary.csv  # 통합 요약
│       ├── 📄 structura_result.json # Structura 상세 결과
│       ├── 📄 cognita_result.json   # Cognita 상세 결과
│       ├── 📄 sentio_result.json    # Sentio 상세 결과
│       ├── 📄 chronos_result.json   # Chronos 상세 결과
│       ├── 📄 agora_result.json     # Agora 상세 결과
│       └── 📁 visualizations/       # 시각화 파일들
│           ├── 🖼️ feature_importance.png  # 피처 중요도
│           ├── 🖼️ shap_analysis.png       # SHAP 분석
│           ├── 🖼️ temporal_attention.png  # 시간적 Attention
│           └── 🖼️ feature_attention.png   # 피처 Attention
│
├── 📁 departments/                  # 부서별 집계 결과
│   └── 📁 {부서명}/
│       └── 📄 department_summary.csv
│
├── 📁 positions/                    # 직급별 집계 결과
│   └── 📁 {직급명}/
│       └── 📄 position_summary.csv
│
├── 📁 reports/                      # 종합 보고서
│   └── 📄 {부서명}_report.json
│
└── 📁 models/                       # 저장된 모델들
    ├── 📄 structura_model.pkl
    ├── 📄 chronos_model.pth
    └── 📄 chronos_scaler.joblib
```

### 📊 저장되는 결과 유형

#### 🏢 Structura (정형 데이터 분석)
- **JSON 결과**: 이직 확률, 예측값, 피처 중요도, SHAP 값, LIME 설명
- **PNG 시각화**: 
  - `feature_importance.png` - 피처 중요도 차트
  - `shap_analysis.png` - SHAP 값 분석 차트

#### 🕸️ Cognita (관계형 데이터 분석)
- **JSON 결과**: 관계형 위험도, 네트워크 중심성, 사회적 고립도, 위험 요인
- **CSV 집계**: 부서별/직급별 관계형 위험도 통계

#### ⏰ Chronos (시계열 데이터 분석)
- **JSON 결과**: 시계열 예측, 확률, Attention 가중치
- **PNG 시각화**:
  - `temporal_attention.png` - 시간적 Attention 가중치
  - `feature_attention.png` - 피처별 Attention 가중치

#### 📝 Sentio (텍스트 감정 분석)
- **JSON 결과**: 감정 점수, 키워드 분석, 퇴직 위험 신호, 위험 수준
- **CSV 저장**: 키워드 분석 결과, 감정 점수 이력

### 🔍 결과 조회 방법

#### 1. API를 통한 조회
```bash
# 개별 직원 결과 조회
curl http://localhost:8000/api/results/employee/1001

# 직원 시각화 파일 목록
curl http://localhost:8000/api/results/employee/1001/visualizations

# 부서별 종합 보고서
curl "http://localhost:8000/api/results/department/Research%20%26%20Development/report"
```

#### 2. 파일 시스템을 통한 직접 접근
```bash
# 직원 1001의 모든 결과 확인
ls results/employees/employee_1001/

# 시각화 파일들 확인
ls results/employees/employee_1001/visualizations/

# 부서별 요약 확인
cat results/departments/Research\ \&\ Development/department_summary.csv
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
# 🏢 Structura 워커 테스트
cd app/Structura && python test_structura_api.py

# 🕸️ Cognita 워커 테스트  
cd app/Cognita && python test_cognita_api.py

# 📝 Sentio 워커 테스트
cd app/Sentio && python test_sentio_api.py

# ⏰ Chronos 워커 테스트
cd app/Chronos && python test_chronos_api.py

# 🌍 Agora 워커 테스트
cd app/Agora && python test_agora_api.py
```

### 간단한 API 테스트

```bash
# 헬스체크
curl http://localhost:8000/api/health    # 통합 시스템
curl http://localhost:5001/api/health    # Structura
curl http://localhost:5000/api/health    # Cognita
curl http://localhost:5003/health        # Sentio
curl http://localhost:5002/api/health    # Chronos

# 간단한 예측 테스트 (Structura)
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 35, "JobSatisfaction": 3, "OverTime": "Yes"}'

# 텍스트 분석 테스트 (Sentio)
curl -X POST http://localhost:5003/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "업무량이 너무 많아서 번아웃이 올 것 같습니다.", "employee_id": "test_001"}'

# 직원 분석 테스트 (Cognita)
curl http://localhost:5000/api/analyze/employee/1

# 시계열 예측 테스트 (Chronos)
curl -X POST http://localhost:5002/api/predict \
  -H "Content-Type: application/json" \
  -d '{"employee_id": 1, "sequence_data": []}'
```

---

## 📁 프로젝트 구조

```
Agentic_AI_system/
├── 📁 app/                          # 메인 애플리케이션
│   ├── 🤖 agentic_master_server.py   # 통합 마스터 서버
│   ├── 🚀 run_agentic_system.py      # 시스템 실행 스크립트
│   ├── 🧪 test_agentic_system.py     # 통합 테스트
│   ├── 📋 requirements_agentic.txt   # 통합 의존성
│   │
│   ├── 📁 Structura/                # 정형 데이터 분석 워커
│   │   ├── 🏢 structura_flask_backend.py
│   │   ├── 🔬 hr_attrition_backend.py
│   │   ├── 🚀 run_structura_server.py
│   │   ├── 🧪 test_structura_api.py
│   │   ├── 📋 requirements.txt
│   │   └── 📖 README.md
│   │
│   ├── 📁 Cognita/                  # 관계형 데이터 분석 워커
│   │   ├── 🕸️ cognita_flask_backend.py
│   │   ├── 🚀 run_cognita_server.py
│   │   ├── 🧪 test_cognita_api.py
│   │   ├── 📋 requirements.txt
│   │   └── 📖 README.md
│   │
│   ├── 📁 Sentio/                   # 텍스트 감정 분석 워커
│   │   ├── 📝 sentio_flask_backend.py
│   │   ├── 🧠 sentio_processor.py
│   │   ├── 📊 sentio_analyzer.py
│   │   ├── ✍️ sentio_generator.py
│   │   ├── 🚀 run_sentio_server.py
│   │   ├── 🧪 test_sentio_api.py
│   │   ├── 📋 requirements.txt
│   │   └── 📖 README.md
│   │
│   └── 📁 Chronos/                  # 시계열 데이터 분석 워커
│       ├── ⏰ chronos_flask_backend.py
│       ├── 🧠 chronos_models.py
│       ├── 📊 chronos_processor.py
│       ├── 🚀 run_chronos_server.py
│       ├── 🧪 test_chronos_api.py
│       ├── 📋 requirements.txt
│       └── 📖 README.md
│
├── 📁 data/                         # 데이터셋
│   ├── 📊 IBM_HR.csv                # IBM HR 데이터셋
│   ├── 📊 IBM_HR_timeseries.csv     # 시계열 데이터
│   ├── 📝 IBM_HR_text.csv           # HR 텍스트 데이터
│   ├── 🕸️ employee_relationships.json # 관계형 데이터
│   └── 🕸️ employee_relationships.xml
│
├── 📁 Data analysis/               # 분석 노트북
│   ├── 🏢 IBM_HR_XGB.ipynb         # Structura 분석
│   ├── 🕸️ Cognita.ipynb            # Cognita 분석
│   ├── 📝 Sentio.ipynb             # 텍스트 분석
│   ├── ⏰ Chronos.ipynb            # 시계열 분석
│   └── 📊 example_usage.py         # 사용 예시
│
├── 📁 Data generation/             # 데이터 생성
│   ├── 🏭 Chronos_data_generation.py
│   ├── 📝 Sentio_data_generation.py
│   ├── 🕸️ Cognita_graph_development.ipynb
│   └── 📊 Structura_data_assignment.ipynb
│
├── 📁 README/                      # 문서
│   ├── 🕸️ Cognita_Data_README.md
│   └── 🕸️ Cognita_graph_README.md
│
└── 📖 README.md                    # 이 파일
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
- **자동 인덱스**: 성능 최적화 자동 설정

### 📝 Sentio 성능
- **키워드 추출**: 명사 중심, 500+ 불용어 필터링
- **텍스트 분석**: 실시간 감정 및 위험도 분석
- **페르소나 분석**: 10가지 직원 유형별 특성 분석
- **텍스트 생성**: OpenAI GPT-4 기반 고품질 생성
- **퇴직 원인 분석**: 5가지 주요 원인 그룹별 세부 분석

### ⏰ Chronos 성능
- **모델 아키텍처**: GRU + CNN + Dual Attention
- **시퀀스 길이**: 6주 단위 시계열 분석
- **학습 시간**: 하이퍼파라미터 최적화 지원
- **예측 속도**: 딥러닝 기반 실시간 예측
- **시각화**: Plotly 인터랙티브 차트

### 확장성 평가
- **소규모 조직** (100명 미만): 실시간 분석 가능
- **중간 규모 조직** (1,000명 미만): 준실시간 분석 가능  
- **대규모 조직** (1,000명 이상): 배치 분석 권장

---

## 🚨 문제 해결

### 공통 문제

#### 1. 포트 충돌
```bash
# 포트 사용 확인
netstat -an | grep :8000  # 마스터 서버
netstat -an | grep :5000  # Cognita
netstat -an | grep :5001  # Structura
netstat -an | grep :5002  # Chronos
netstat -an | grep :5003  # Sentio

# 프로세스 종료
kill -9 $(lsof -ti:8000)
kill -9 $(lsof -ti:5000)
kill -9 $(lsof -ti:5001)
kill -9 $(lsof -ti:5002)
kill -9 $(lsof -ti:5003)
```

#### 2. 의존성 설치 오류
```bash
# 가상환경 사용 권장
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r app/requirements_agentic.txt
```

#### 3. CORS 오류 (React 연동)
```bash
# Flask-CORS 설치 확인
pip install flask-cors

# React 개발 서버 주소 확인 (localhost:3000)
```

### Sentio 특정 문제

#### 1. OpenAI API 오류
```bash
# API 키 설정 확인
export OPENAI_API_KEY="your-openai-api-key"
echo $OPENAI_API_KEY

# API 키 유효성 확인
```

#### 2. 텍스트 데이터 없음
```bash
# HR 텍스트 데이터를 data/ 폴더에 배치
# sample_hr_texts.csv 또는 IBM_HR_text.csv 확인
```

---

## 🔮 향후 계획

### 단기 (1-2주)
- [ ] **React 프론트엔드 개발**: 통합 대시보드 구현
- [ ] **실시간 모니터링**: 시스템 성능 및 분석 결과 추적
- [ ] **배치 처리 시스템**: 대용량 데이터 처리 최적화
- [x] **워커 에이전트 3**: 텍스트 감정 분석 (Sentio) - 완료
- [x] **워커 에이전트 4**: 시계열 데이터 분석 (Chronos) - 완료
- [x] **워커 에이전트 5**: 외부 시장 분석

### 중기 (3-4주)
- [ ] **Supervisor 에이전트**: 전체 워커 조정 및 관리
- [ ] **최종 종합 에이전트**: 모든 분석 결과 통합
- [ ] **클라우드 배포**: AWS/Azure 기반 확장

### 장기 (6개월+)
- [ ] **실시간 스트리밍**: 실시간 데이터 처리
- [ ] **AI 추천 시스템**: 자동화된 인사 정책 제안

---

## 🏆 주요 성과

### 기술적 성과
- ✅ **5개 워커 에이전트 아키텍처** 구현
- ✅ **설명 가능한 AI** (SHAP, LIME, Attention) 적용
- ✅ **관계형 네트워크 분석** 시스템 구축
- ✅ **텍스트 감정 분석** 시스템 구축 (Sentio)
- ✅ **시계열 딥러닝 분석** 시스템 구축 (Chronos)
- ✅ **외부 시장 분석** 시스템 구축 (Agora)
- ✅ **실시간 통합 분석** 플랫폼 완성
- ✅ **React 연동** 최적화

### 성능 성과
- ✅ **Structura**: 0.1초/명 예측 속도 달성
- ✅ **Cognita**: 30% 성능 향상 (v1.1.0)
- ✅ **Sentio**: 명사 중심 키워드 추출로 정확도 향상
- ✅ **Chronos**: GRU+CNN+Attention 하이브리드 모델 구현
- ✅ **Agora**: 0.5초/명 시장 분석 속도 달성
- ✅ **통합 시스템**: 5개 워커 에이전트 동시 실행
- ✅ **확장성**: 대규모 조직 지원 가능

---

## 👥 기여자

**개발팀**
- 시스템 아키텍처 설계
- AI 모델 개발 및 최적화
- 웹 서비스 구현
- 성능 튜닝 및 최적화

---

## 📄 라이선스

이 프로젝트는 오픈소스 연구 프로젝트입니다.

---

## 📞 지원 및 문의

프로젝트 관련 문의사항이나 기술 지원이 필요한 경우:

1. **GitHub Issues**: 버그 리포트 및 기능 요청
2. **기술 문서**: `README/` 폴더 참조
3. **개별 워커 문서**: 각 워커 디렉토리의 README 참조

---

**버전**: 1.4.0 (Agora Integration)  
**최종 업데이트**: 2025년  
**기술 스택**: Python, Flask, XGBoost, Neo4j, PyTorch, OpenAI GPT-4, 시장 분석 API, React  
**아키텍처**: Multi-Agent AI System (5 Workers)  