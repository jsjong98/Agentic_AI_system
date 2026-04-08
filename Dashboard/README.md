# Agentic AI Dashboard

HR 이직 예측을 위한 다중 에이전트 시스템의 통합 대시보드입니다.

**GitHub**: [jsjong98/Agentic_AI_system](https://github.com/jsjong98/Agentic_AI_system)

---

## 주요 기능

### 홈 (AI 워크플로우 + 채팅)
- **ReactFlow 에이전트 워크플로우**: Supervisor → 5개 워커 에이전트 → Synthesize → Report 흐름을 실시간 시각화
- **워크플로우 자동 트리거**: 분석 관련 메시지 전송 시 자동으로 에이전트 워크플로우 애니메이션 시작
- **직원 스캔 카운터**: 에이전트 분석 중 직원 #001~#1470 실시간 스캔 표시
- **AI 채팅 (대화 기억)**: 현재 세션의 대화 히스토리를 LLM에 전달해 문맥을 유지하며 답변
- **Admin 권한**: Admin 계정은 개별 직원 상세 정보 포함 전체 데이터 조회 가능
- **라이트/다크 모드 대응**: 시스템 설정에 따라 캔버스 및 노드 색상 자동 조정

### 분석 대시보드
- **전체 통계**: 이직 위험도 분포, 부서별/직무별 위험 요인
- **배치 분석**: 전체 직원 대량 분석 및 고위험군 리포트
- **개별 예측**: 단일 직원 상세 분석 및 설명 가능한 AI (SHAP/LIME)
- **그룹 통계**: 부서/직급별 Top 3 위험 요인 비교
- **관계 분석**: Neo4j 기반 조직 네트워크 분석
- **가중치 최적화**: Bayesian/Grid Search 기반 에이전트 가중치 조정
- **레포트 생성**: 맞춤형 PDF/CSV 레포트 다운로드

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| UI Framework | React 18 |
| 워크플로우 시각화 | ReactFlow v11 |
| UI 컴포넌트 | Ant Design 5 |
| 차트 | Recharts |
| HTTP | Fetch API |
| 스타일 | CSS Variables (라이트/다크 모드) |

---

## 시작하기

```bash
cd Dashboard
npm install
npm start
```

개발 서버: http://localhost:3000

---

## 환경 변수

```env
REACT_APP_SUPERVISOR_URL=http://localhost:5006
```

---

## 페이지 구성

| 경로 | 페이지 | 설명 |
|------|--------|------|
| `/` | Home | AI 워크플로우 + 채팅 인터페이스 |
| `/dashboard` | Dashboard | 전체 통계 현황 |
| `/upload` | FileUpload | 데이터 업로드 |
| `/batch` | BatchAnalysis | 대량 직원 분석 |
| `/predict` | EmployeePrediction | 개별 직원 예측 |
| `/group` | GroupStatistics | 그룹별 통계 |
| `/relationship` | RelationshipAnalysis | 네트워크 분석 |
| `/weight` | WeightOptimization | 가중치 최적화 |
| `/report` | ReportGeneration | 레포트 생성 |
| `/export` | ExportResults | 결과 내보내기 |

---

## 에이전트 워크플로우 구조

```
         [Supervisor]
              │
    ┌────┬────┼────┬────┐
    │    │    │    │    │
[Agora][Chronos][Cognita][Sentio][Structura]
    │    │    │    │    │
    └────┴────┼────┴────┘
              │
        [Synthesize]
              │
          [Report]
```

- 채팅 메시지 전송 시 워크플로우 자동 시작 (단순 인사말 제외)
- 각 에이전트는 고유 색상으로 활성/비활성 상태 표시
- 10초 사이클 완료 후 자동으로 대기 상태로 복귀

---

**개발**: PwC RA Team | **버전**: 3.0.0
