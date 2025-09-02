# Agentic AI System - HR Analytics Platform

PwC RA팀의 **에이전틱 AI 기반 HR 분석 플랫폼**입니다. 

## 🤖 에이전틱 아키텍처

```
                    ┌─────────────────────────────────────┐
                    │         Supervisor 에이전트         │
                    │          (향후 구현)               │
                    └─────────────────────────────────────┘
                                    │
        ┌───────────┬───────────┬───────────┬───────────┬───────────┐
        │  워커1    │  워커2    │  워커3    │  워커4    │  워커5    │
        │  정형     │  관계형   │  시계열   │  자연어   │  외부     │
        │  데이터   │  데이터   │  데이터   │  데이터   │  시장     │
        │  분석     │  분석     │  분석     │  분석     │  분석     │
        │  ✅      │  ✅      │  ⏳      │  ⏳      │  ⏳      │
        └───────────┴───────────┴───────────┴───────────┴───────────┘
                                    │
                    ┌─────────────────────────────────────┐
                    │       최종 종합 에이전트            │
                    │          (향후 구현)               │
                    └─────────────────────────────────────┘
```

**현재 구현**: 워커 에이전트 1, 2가 **마스터 서버**에서 **동시 실행**  
**향후 확장**: Supervisor, 워커 3-5, 최종 종합 에이전트

## 🏗️ 프로젝트 구조

```
app/
├── Structura/          # 일반 데이터 분석 (XGBoost + xAI)
│   ├── structura_flask_backend.py
│   ├── hr_attrition_backend.py
│   ├── run_structura_server.py
│   ├── test_structura_api.py
│   ├── requirements.txt
│   └── README.md
│
├── Cognita/           # 관계형 데이터 분석 (Neo4j + Graph)
│   ├── cognita_flask_backend.py
│   ├── run_cognita_server.py
│   ├── test_cognita_api.py
│   ├── requirements.txt
│   └── README.md
│
└── README.md          # 이 파일
```

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

### 🕸️ Cognita - 관계형 위험도 분석 시스템
**포트**: `5000` | **기술**: Neo4j + Graph Analytics

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

## 🚀 빠른 시작 (통합 에이전틱 시스템)

### 1. 통합 시스템 실행 (권장)

```bash
# 1. 의존성 설치
cd app
pip install -r requirements_agentic.txt

# 2. Neo4j 연결 설정 (환경변수)
export NEO4J_URI="bolt://your-neo4j-host:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-password"

# 3. 통합 마스터 서버 실행 (모든 워커 에이전트 동시 실행)
python run_agentic_system.py

# 4. 통합 테스트 (별도 터미널)
python test_agentic_system.py
```

**접속**: http://localhost:8000 (통합 마스터 서버)

### 2. 개별 워커 실행 (개발/디버깅용)

#### Structura 워커 (정형 데이터 분석)
```bash
cd app/Structura
pip install -r requirements.txt
python run_structura_server.py  # 포트: 5001
```

#### Cognita 워커 (관계형 데이터 분석)
```bash
cd app/Cognita
pip install -r requirements.txt
python run_cognita_server.py    # 포트: 5000
```

## 🌐 React 연동 가이드 (통합 에이전틱 시스템)

**통합 마스터 서버**를 통해 모든 워커 에이전트에 접근할 수 있습니다.

### 통합 API 사용 예시

```javascript
// 개별 직원 통합 분석 (Structura + Cognita 동시 실행)
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
  
  // 통합 결과 활용
  console.log('Structura 결과:', result.structura_result);
  console.log('Cognita 결과:', result.cognita_result);
  console.log('통합 분석:', result.combined_analysis);
  
  return result;
};

// 부서별 통합 분석
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

// 시스템 상태 확인
const checkSystemHealth = async () => {
  const response = await fetch('http://localhost:8000/api/health');
  const health = await response.json();
  
  console.log('시스템 상태:', health.status);
  console.log('워커 에이전트:', health.workers);
  
  return health;
};
```

### React Hook 예시 (통합 시스템용)

```javascript
import { useState, useEffect } from 'react';

const useAgenticAI = () => {
  const [systemHealth, setSystemHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const baseURL = 'http://localhost:8000/api';
  
  // 통합 직원 분석
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
  
  return {
    systemHealth,
    loading,
    error,
    analyzeEmployee
  };
};
```

## 📊 시스템 비교

| 특징 | Structura | Cognita |
|------|-----------|---------|
| **분석 방식** | 개별 직원 데이터 | 관계형 네트워크 |
| **데이터 소스** | CSV (IBM HR) | Neo4j 그래프 DB |
| **주요 기술** | XGBoost + xAI | Graph Analytics |
| **포트** | 5001 | 5000 |
| **예측 대상** | 이직 확률 | 관계형 위험도 |
| **설명 가능성** | SHAP, LIME | 네트워크 분석 |
| **분석 범위** | 개별 중심 | 조직/팀 중심 |
| **실시간성** | 즉시 예측 | 실시간 관계 분석 |

## 🔧 주요 API 엔드포인트

### 통합 마스터 서버 (포트 8000) - 권장
- `GET /api/health` - 시스템 전체 상태 확인
- `GET /api/workers/status` - 워커 에이전트 상태 조회
- `POST /api/analyze/individual` - **개별 직원 통합 분석** (Structura + Cognita)
- `POST /api/analyze/department` - **부서별 통합 분석** (Structura + Cognita)
- `GET /api/task/{task_id}/result` - 작업 결과 조회

### 개별 워커 API (개발/디버깅용)

#### Structura 워커 (포트 5001)
- `GET /api/health` - 서버 상태 확인
- `POST /api/train` - 모델 훈련
- `POST /api/predict` - 이직 예측
- `POST /api/explain` - 예측 설명 (xAI)
- `GET /api/feature-importance` - 피처 중요도

#### Cognita 워커 (포트 5000)
- `GET /api/health` - 서버 상태 확인
- `GET /api/employees` - 직원 목록
- `GET /api/departments` - 부서 목록
- `GET /api/analyze/employee/{id}` - 직원 분석
- `POST /api/analyze/department` - 부서 분석

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
```

### 개별 기능 테스트

```bash
# 헬스체크
curl http://localhost:5001/api/health  # Structura
curl http://localhost:5000/api/health  # Cognita

# 간단한 예측 테스트 (Structura)
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 35, "JobSatisfaction": 3, "OverTime": "Yes"}'

# 직원 분석 테스트 (Cognita)
curl http://localhost:5000/api/analyze/employee/1
```

## 📈 성능 특성

### Structura
- **예측 속도**: ~0.1초/명
- **xAI 분석**: ~1-3초/명
- **메모리 사용**: 500MB-2GB
- **정확도**: ROC-AUC 0.85+

### Cognita
- **분석 속도**: ~1.06초/명
- **부서 분석**: 5-30초 (샘플 크기에 따라)
- **처리량**: ~3,400명/시간
- **메모리 사용**: 500MB-2GB

## 🚨 문제 해결

### 공통 문제

1. **포트 충돌**
   ```bash
   # 포트 사용 확인
   netstat -an | grep :5000
   netstat -an | grep :5001
   
   # 프로세스 종료
   kill -9 $(lsof -ti:5000)
   kill -9 $(lsof -ti:5001)
   ```

2. **의존성 설치 오류**
   ```bash
   # 가상환경 사용 권장
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **CORS 오류 (React 연동)**
   ```bash
   # Flask-CORS 설치 확인
   pip install flask-cors
   
   # React 개발 서버 주소 확인 (localhost:3000)
   ```

### Structura 특정 문제

1. **xAI 라이브러리 오류**
   ```bash
   pip install shap lime
   # 또는
   conda install -c conda-forge shap lime
   ```

2. **데이터 파일 없음**
   ```bash
   # IBM HR 데이터셋을 ../data/IBM_HR.csv에 배치
   # 또는 코드에서 경로 수정
   ```

### Cognita 특정 문제

1. **Neo4j 연결 실패**
   ```bash
   # Neo4j 서버 실행 확인
   # 연결 정보 확인 (URI, 사용자명, 비밀번호)
   # 방화벽 설정 확인
   ```

2. **그래프 데이터 없음**
   ```bash
   # Neo4j에 Employee, Department, Project 노드 확인
   # COLLABORATES_WITH, REPORTS_TO 관계 확인
   ```

## 🔮 향후 계획

### 단기 (1-2개월)
- [ ] React 프론트엔드 개발
- [ ] 실시간 대시보드 구현
- [ ] 배치 처리 시스템 추가
- [ ] 모델 성능 모니터링

### 중기 (3-6개월)
- [ ] 두 시스템 통합 분석
- [ ] 예측 정확도 개선
- [ ] 추가 xAI 기법 도입
- [ ] 클라우드 배포

### 장기 (6개월+)
- [ ] 실시간 스트리밍 분석
- [ ] 다중 조직 지원
- [ ] AI 추천 시스템
- [ ] 자동화된 인사 정책 제안

---

**버전**: 1.0.0  
**최종 업데이트**: 2025년  