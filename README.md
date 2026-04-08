# Agentic AI System — HR Analytics Platform

**에이전틱 AI 기반 HR 이직 예측 플랫폼**

다중 에이전트가 정형·관계형·시계열·텍스트·시장 데이터를 동시에 분석해 종합적인 HR 인사이트를 제공합니다.

**GitHub**: [github.com/jsjong98/Agentic_AI_system](https://github.com/jsjong98/Agentic_AI_system)

---

## 시스템 아키텍처

```
                    ┌──────────────────────┐
                    │   React Dashboard    │  :3000
                    │  (ReactFlow + Chat)  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Supervisor Agent   │  :5006
                    │  GPT-4o-mini / nano  │
                    └──┬───┬───┬───┬───┬──┘
                       │   │   │   │   │
              ┌────────┘   │   │   │   └────────┐
              │        ┌───┘   └───┐        │
        ┌─────▼──┐ ┌───▼──┐ ┌──▼───┐ ┌──▼──┐ ┌──▼──────┐
        │ Agora  │ │Chronos│ │Cognita│ │Sentio│ │Structura│
        │ :5005  │ │ :5003 │ │ :5002 │ │:5004 │ │  :5001  │
        │시장분석 │ │시계열 │ │관계형 │ │텍스트│ │  정형   │
        └────────┘ └───────┘ └───────┘ └──────┘ └─────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Integration System  │  :5007
                    │   최종 종합 레포트   │
                    └──────────────────────┘
```

---

## 빠른 시작

### Docker Compose (권장)

```bash
git clone https://github.com/jsjong98/Agentic_AI_system.git
cd Agentic_AI_system

cp env.example .env
# .env에 OPENAI_API_KEY, NEO4J_PASSWORD 설정

docker-compose up -d
```

접속: http://localhost:3000

### 직접 실행 (개발 환경)

```bash
# 백엔드 (conda 환경 필요)
conda activate nlp
export OPENAI_API_KEY="your-api-key"
python start_all_services_simple.py

# 프론트엔드 (별도 터미널)
cd Dashboard
npm install && npm start
```

전체 가이드: [app/Launch.md](app/Launch.md)

---

## React Dashboard 주요 기능

### 홈 화면 — AI 워크플로우 + 채팅

- **ReactFlow 에이전트 그래프**: Supervisor → 5개 워커 → Synthesize → Report 흐름 실시간 시각화
- **워크플로우 자동 트리거**: 분석 메시지 전송 시 에이전트 애니메이션 자동 시작 (단순 인사 제외)
- **직원 스캔 카운터**: 분석 중 `직원 #001~#1470 스캔 중...` 실시간 표시
- **대화 히스토리 메모리**: 최근 10개 메시지를 LLM에 전달, 문맥 유지 답변
- **Admin 권한 분리**: Admin 계정은 개별 직원 이름/위험도 등 전체 정보 조회 가능
- **라이트/다크 모드**: 시스템 테마에 따라 캔버스·노드 색상 자동 적용

### 분석 페이지

| 페이지 | 설명 |
|--------|------|
| Dashboard | 전체 이직 위험도 통계 및 현황 |
| BatchAnalysis | 전체 직원 대량 분석 및 고위험군 리포트 |
| EmployeePrediction | 단일 직원 상세 예측 (SHAP/LIME 설명) |
| GroupStatistics | 부서/직무/직급별 Top 3 위험 요인 |
| RelationshipAnalysis | Neo4j 기반 조직 네트워크 분석 |
| WeightOptimization | Bayesian/Grid Search 가중치 조정 |
| ReportGeneration | 맞춤형 레포트 생성 |
| ExportResults | 분석 결과 CSV/JSON 다운로드 |

---

## 워커 에이전트

### Structura (`:5001`) — 정형 데이터 분석
- XGBoost + SHAP/LIME 기반 이직 확률 예측
- ROC-AUC 0.85+, 예측 속도 ~0.1초/명

### Cognita (`:5002`) — 관계형 데이터 분석
- Neo4j 그래프 DB 기반 사회적 네트워크 분석
- 조직 내 관계 패턴 및 팀 역학 평가

### Chronos (`:5003`) — 시계열 분석
- GRU + CNN + Attention 하이브리드 딥러닝
- 6주 단위 직원 행동 패턴 시간적 변화 추적

### Sentio (`:5004`) — 텍스트 감정 분석
- NLP 키워드 분석 + GPT-5-nano 텍스트 생성
- JD-R 모델 기반 5가지 퇴직 원인 분석

### Agora (`:5005`) — 외부 시장 분석
- 시장 압력 지수·보상 격차·직무별 채용 트렌드 분석
- GPT-5-nano 기반 시장 경쟁력 해석

### Supervisor (`:5006`) — 워크플로우 관리
- GPT-4o-mini (fallback: GPT-5-nano) 기반 멀티에이전트 조율
- 채팅 API: 대화 히스토리 기반 문맥 유지 답변

### Integration (`:5007`) — 최종 레포트
- 임계값 계산, 가중치 최적화, GPT-5-nano 개별 레포트 생성

---

## 데이터 요구사항

| 에이전트 | 필요 데이터 |
|----------|------------|
| Structura | `data/IBM_HR.csv` (1,470명) |
| Chronos | `data/IBM_HR_timeseries.csv` |
| Sentio | `data/IBM_HR_text.csv` |
| Cognita | Neo4j DB 연결 (환경변수 필요) |
| Agora | IBM_HR.csv에서 직무/급여 정보 추출 |

---

## 환경 변수

```env
OPENAI_API_KEY=your-openai-api-key
NEO4J_URI=bolt://your-neo4j-host:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
```

---

## 포트 요약

| 서비스 | 포트 |
|--------|------|
| React Dashboard | 3000 |
| Structura | 5001 |
| Cognita | 5002 |
| Chronos | 5003 |
| Sentio | 5004 |
| Agora | 5005 |
| Supervisor | 5006 |
| Integration | 5007 |
| Neo4j Browser | 7474 |

---

**개발**: PwC RA Team | **버전**: 3.0.0 | **라이센스**: MIT
