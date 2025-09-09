# 🤖 Agentic AI System - HR Analytics Platform

**PwC RA팀의 에이전틱 AI 기반 HR 분석 플랫폼**

차세대 인사 분석을 위한 다중 에이전트 AI 시스템으로, 정형 데이터와 관계형 데이터를 동시에 분석하여 종합적인 인사 인사이트를 제공합니다.

## 🌟 주요 특징

- **🤖 에이전틱 아키텍처**: 다중 AI 에이전트가 협력하여 분석
- **🔍 다차원 분석**: 정형 데이터 + 관계형 네트워크 동시 분석
- **⚡ 실시간 처리**: 개별 직원부터 전체 조직까지 즉시 분석
- **🔬 설명 가능한 AI**: SHAP, LIME 기반 투명한 의사결정
- **🕸️ 관계형 인사이트**: Neo4j 기반 조직 네트워크 분석
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
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼───────┐          ┌────────▼────────┐         ┌───────────────┐
│  🏢 Structura  │          │  🕸️ Cognita     │         │  ⏳ 향후 확장   │
│   워커 에이전트  │          │   워커 에이전트   │         │   워커 에이전트  │
│  (포트: 5001)  │          │  (포트: 5000)   │         │  (시계열/NLP)  │
│               │          │                │         │               │
│ 정형 데이터 분석 │          │ 관계형 데이터 분석│         │  추가 분석 모듈  │
│ XGBoost + xAI │          │ Neo4j + Graph  │         │               │
│ SHAP + LIME   │          │ Network Analytics│         │               │
└───────────────┘          └─────────────────┘         └───────────────┘
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐          ┌─────────────────┐         ┌───────────────┐
│ 📊 IBM HR CSV │          │ 🗄️ Neo4j Graph  │         │ 📈 시계열 DB   │
│   데이터셋     │          │   데이터베이스    │         │   (향후 구현)   │
└───────────────┘          └─────────────────┘         └───────────────┘
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

# 3. Neo4j 연결 설정
export NEO4J_URI="bolt://34.227.31.16:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="cover-site-establishment"

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

## 🌐 React 연동 가이드

### 통합 API 사용 예시

```javascript
// 🤖 개별 직원 통합 분석 (Structura + Cognita 동시 실행)
const analyzeEmployeeIntegrated = async (employeeData) => {
  const response = await fetch('http://localhost:8000/api/analyze/individual', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...employeeData,
      use_structura: true,  // 정형 데이터 분석 사용
      use_cognita: true     // 관계형 데이터 분석 사용
    })
  });
  
  const result = await response.json();
  
  // 🔍 통합 결과 활용
  console.log('Structura 결과:', result.structura_result);
  console.log('Cognita 결과:', result.cognita_result);
  console.log('통합 분석:', result.combined_analysis);
  
  return result;
};

// 🏢 부서별 통합 분석
const analyzeDepartmentIntegrated = async (departmentName) => {
  const response = await fetch('http://localhost:8000/api/analyze/department', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      department_name: departmentName,
      sample_size: 20,
      use_structura: true,
      use_cognita: true
    })
  });
  return response.json();
};

// 🔍 시스템 상태 확인
const checkSystemHealth = async () => {
  const response = await fetch('http://localhost:8000/api/health');
  const health = await response.json();
  
  console.log('시스템 상태:', health.status);
  console.log('워커 에이전트:', health.workers);
  
  return health;
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
  
  // 🤖 통합 직원 분석
  const analyzeEmployee = async (employeeData) => {
    try {
      setLoading(true);
      const response = await fetch(`${baseURL}/analyze/individual`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...employeeData,
          use_structura: true,
          use_cognita: true
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
    checkHealth
  };
};
```

---

## 📊 시스템 비교

| 특징 | 🏢 Structura | 🕸️ Cognita |
|------|-------------|------------|
| **분석 방식** | 개별 직원 데이터 | 관계형 네트워크 |
| **데이터 소스** | CSV (IBM HR) | Neo4j 그래프 DB |
| **주요 기술** | XGBoost + xAI | Graph Analytics |
| **포트** | 5001 | 5000 |
| **예측 대상** | 이직 확률 | 관계형 위험도 |
| **설명 가능성** | SHAP, LIME | 네트워크 분석 |
| **분석 범위** | 개별 중심 | 조직/팀 중심 |
| **실시간성** | 즉시 예측 | 실시간 관계 분석 |
| **성능** | 0.1초/명 | 0.82초/명 |

---

## 🔧 주요 API 엔드포인트

### 🎯 통합 마스터 서버 (포트 8000) - 권장

| 메서드 | 엔드포인트 | 설명 |
|--------|------------|------|
| `GET` | `/api/health` | 시스템 전체 상태 확인 |
| `GET` | `/api/workers/status` | 워커 에이전트 상태 조회 |
| `POST` | `/api/analyze/individual` | **개별 직원 통합 분석** (Structura + Cognita) |
| `POST` | `/api/analyze/department` | **부서별 통합 분석** (Structura + Cognita) |
| `GET` | `/api/task/{task_id}/result` | 작업 결과 조회 |

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

---

## 🧪 테스트 실행

### 통합 시스템 테스트

```bash
# 🤖 통합 에이전틱 시스템 테스트
cd app
python test_agentic_system.py
```

### 개별 워커 테스트

```bash
# 🏢 Structura 워커 테스트
cd app/Structura && python test_structura_api.py

# 🕸️ Cognita 워커 테스트  
cd app/Cognita && python test_cognita_api.py
```

### 간단한 API 테스트

```bash
# 헬스체크
curl http://localhost:8000/api/health    # 통합 시스템
curl http://localhost:5001/api/health    # Structura
curl http://localhost:5000/api/health    # Cognita

# 간단한 예측 테스트 (Structura)
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 35, "JobSatisfaction": 3, "OverTime": "Yes"}'

# 직원 분석 테스트 (Cognita)
curl http://localhost:5000/api/analyze/employee/1
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
│   └── 📁 Cognita/                  # 관계형 데이터 분석 워커
│       ├── 🕸️ cognita_flask_backend.py
│       ├── 🚀 run_cognita_server.py
│       ├── 🧪 test_cognita_api.py
│       ├── 📋 requirements.txt
│       └── 📖 README.md
│
├── 📁 data/                         # 데이터셋
│   ├── 📊 IBM_HR.csv                # IBM HR 데이터셋
│   ├── 📊 IBM_HR_timeseries.csv     # 시계열 데이터
│   ├── 🕸️ employee_relationships.json # 관계형 데이터
│   └── 🕸️ employee_relationships.xml
│
├── 📁 Data analysis/               # 분석 노트북
│   ├── 🏢 IBM_HR_XGB.ipynb         # Structura 분석
│   ├── 🕸️ Cognita.ipynb            # Cognita 분석
│   ├── ⏰ Chronos.ipynb            # 시계열 분석
│   └── 📊 example_usage.py         # 사용 예시
│
├── 📁 Data generation/             # 데이터 생성
│   ├── 🏭 Chronos_data_generation.py
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

# 프로세스 종료
kill -9 $(lsof -ti:8000)
kill -9 $(lsof -ti:5000)
kill -9 $(lsof -ti:5001)
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

### Structura 특정 문제

#### 1. xAI 라이브러리 오류
```bash
pip install shap lime
# 또는
conda install -c conda-forge shap lime
```

#### 2. 데이터 파일 없음
```bash
# IBM HR 데이터셋을 data/IBM_HR.csv에 배치
# 또는 코드에서 경로 수정
```

### Cognita 특정 문제

#### 1. Neo4j 연결 실패
```bash
# Neo4j 서버 실행 확인
# 연결 정보 확인 (URI, 사용자명, 비밀번호)
# 방화벽 설정 확인
```

#### 2. 그래프 데이터 없음
```bash
# Neo4j에 Employee, Department, Project 노드 확인
# COLLABORATES_WITH, REPORTS_TO 관계 확인
```

---

## 🔮 향후 계획

### 단기 (1-2주)
- [ ] **React 프론트엔드 개발**: 통합 대시보드 구현
- [ ] **실시간 모니터링**: 시스템 성능 및 분석 결과 추적
- [ ] **배치 처리 시스템**: 대용량 데이터 처리 최적화
- [ ] **모델 성능 개선**: 예측 정확도 향상
- [ ] **워커 에이전트 3**: 시계열 데이터 분석 (Chronos)
- [ ] **워커 에이전트 4**: 자연어 데이터 분석 (NLP)
- [ ] **워커 에이전트 5**: 외부 시장 분석

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
- ✅ **다중 에이전트 아키텍처** 구현
- ✅ **설명 가능한 AI** (SHAP, LIME) 적용
- ✅ **관계형 네트워크 분석** 시스템 구축
- ✅ **실시간 통합 분석** 플랫폼 완성
- ✅ **React 연동** 최적화

### 성능 성과
- ✅ **Structura**: 0.1초/명 예측 속도 달성
- ✅ **Cognita**: 30% 성능 향상 (v1.1.0)
- ✅ **통합 시스템**: 다중 워커 동시 실행
- ✅ **확장성**: 대규모 조직 지원 가능

---

## 👥 기여자

**오종환**
- 시스템 아키텍처 설계
- AI 모델 개발 및 최적화
- 웹 서비스 구현
- 성능 튜닝 및 최적화

---

## 📞 지원 및 문의

프로젝트 관련 문의사항이나 기술 지원이 필요한 경우:

1. **GitHub Issues**: 버그 리포트 및 기능 요청
2. **기술 문서**: `README/` 폴더 참조

---

**버전**: 1.1.0 (Performance Optimized)  
**최종 업데이트**: 2025년   
**기술 스택**: Python, Flask, XGBoost, Neo4j, React  
**아키텍처**: Multi-Agent AI System  
