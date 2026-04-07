# Supervisor Agent - 슈퍼바이저 에이전트

LangGraph 기반의 지능형 워크플로우 오케스트레이터로, 5개의 워커 에이전트(Structura, Cognita, Chronos, Sentio, Agora)를 조율하여 HR Attrition 예측을 수행합니다.

## 🏗️ 아키텍처 개요

### 슈퍼바이저-워커 아키텍처
- **슈퍼바이저**: 워크플로우 전체를 지휘하는 중추 신경계 역할
- **워커 에이전트들**: 각자의 전문 분야에서 분석 수행
- **LangGraph**: 복잡한 조건부 워크플로우 구현
- **AgentState**: 모든 노드에서 공유되는 글로벌 상태 객체

### 핵심 구성요소

#### 1. AgentState (글로벌 객체)
```python
class AgentState(BaseModel):
    employee_id: str                    # 분석 대상 직원 ID
    session_id: str                     # 세션 고유 식별자
    analysis_checklist: Dict[str, AnalysisStatus]  # 워커별 진행 상태
    worker_results: Dict[str, WorkerResult]        # 워커별 결과 저장
    error_log: List[ErrorLog]           # 오류 기록
    final_report: Optional[FinalReport] # 최종 종합 보고서
    # ... 기타 메타데이터
```

#### 2. 조건부 라우터 (SupervisorAgent)
- LLM 기반 지능적 라우팅 결정
- 워크플로우 상태 분석 및 다음 단계 결정
- 오류 분석 및 재시도 가능성 판단

#### 3. 워커 통합기 (WorkerIntegrator)
- 5개 워커 에이전트와의 HTTP 통신
- 병렬/순차 분석 실행
- 헬스체크 및 오류 처리

#### 4. LangGraph 워크플로우
- 노드: start → worker_dispatcher → retry_handler → synthesizer
- 조건부 엣지: 상태 기반 동적 라우팅
- 복원력: 오류 처리 및 재시도 메커니즘

## 📁 프로젝트 구조

```
app/Supervisor/
├── __init__.py                     # 패키지 초기화
├── agent_state.py                  # AgentState 글로벌 객체 정의
├── supervisor_agent.py             # 슈퍼바이저 핵심 로직
├── worker_integrator.py            # 워커 에이전트 통합 모듈
├── synthesizer.py                  # 최종 종합 분석 에이전트
├── langgraph_workflow.py           # LangGraph 워크플로우 구현
├── supervisor_flask_backend.py     # Flask REST API 서버
├── run_supervisor_server.py        # 서버 실행 스크립트
├── test_supervisor_api.py          # API 테스트 스크립트
├── mcp_server.py                   # 🆕 MCP 프로토콜 서버
├── mcp_tools.py                    # 🆕 MCP 도구 인터페이스
├── mcp_adapter.py                  # 🆕 MCP 어댑터 레이어
├── mcp_config.json                 # 🆕 MCP 서버 설정
├── test_mcp_integration.py         # 🆕 MCP 통합 테스트
├── MCP_INTEGRATION_GUIDE.md        # 🆕 MCP 통합 가이드
├── requirements.txt                # Python 의존성
└── README.md                       # 이 파일
```

## 🚀 설치 및 실행

### 1. 의존성 설치
```bash
cd app/Supervisor
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일 생성:
```env
# OpenAI API (필수)
OPENAI_API_KEY=your_openai_api_key_here

# 서버 설정
SUPERVISOR_PORT=5006
FLASK_DEBUG=False
MAX_RETRY_COUNT=3
TIMEOUT_MINUTES=30
MAX_BATCH_SIZE=2000

# 워커 서버 URL
STRUCTURA_URL=http://localhost:5001
COGNITA_URL=http://localhost:5002
CHRONOS_URL=http://localhost:5003
SENTIO_URL=http://localhost:5004
AGORA_URL=http://localhost:5005
```

### 3. 워커 서버 실행
슈퍼바이저를 실행하기 전에 모든 워커 서버가 실행되어야 합니다:
```bash
# 각 워커 디렉토리에서 실행
cd app/Structura && python run_structura_server.py  # 포트: 5001
cd app/Cognita && python run_cognita_server.py     # 포트: 5002
cd app/Chronos && python run_chronos_server.py     # 포트: 5003
cd app/Sentio && python run_sentio_server.py       # 포트: 5004
cd app/Agora && python run_agora_server.py         # 포트: 5005
```

### 4. 슈퍼바이저 서버 실행
```bash
cd app/Supervisor
python run_supervisor_server.py
```

서버가 성공적으로 시작되면:
```
🚀 Supervisor 에이전트 서버 시작
==================================================
🔍 워커 서버 상태 확인:
  ✅ Structura: http://localhost:5001 - 온라인
  ✅ Cognita: http://localhost:5002 - 온라인
  ✅ Chronos: http://localhost:5003 - 온라인
  ✅ Sentio: http://localhost:5004 - 온라인
  ✅ Agora: http://localhost:5005 - 온라인

📊 사용 가능한 워커: 5/5

🌐 서버 정보:
  📡 주소: http://localhost:5006
  🔧 디버그: False
  🤖 사용 가능한 워커: 5/5

🔥 서버 시작 중...
```

## 📡 API 엔드포인트

### 기본 엔드포인트

#### `GET /health`
서버 상태 확인
```json
{
  "status": "healthy",
  "service": "Supervisor",
  "workflow_initialized": true,
  "available_workers": ["structura", "cognita", "chronos", "sentio", "agora"]
}
```

#### `GET /worker_health_check`
워커 서버 상태 확인
```json
{
  "success": true,
  "worker_status": {
    "structura": {"healthy": true, "status": "online"},
    "cognita": {"healthy": true, "status": "online"}
  },
  "summary": {
    "total_workers": 5,
    "healthy_workers": 5,
    "health_rate": 1.0
  }
}
```

### 분석 엔드포인트

#### `POST /analyze_employee`
단일 직원 분석
```json
// 요청
{
  "employee_id": "EMP001",
  "session_id": "optional_session_id"
}

// 응답
{
  "success": true,
  "employee_id": "EMP001",
  "session_id": "uuid-session-id",
  "final_report": {
    "risk_score": 75.5,
    "risk_grade": "B",
    "attrition_probability": 0.755,
    "summary": "직원의 이탈 위험도가 높음 수준으로 평가됩니다...",
    "recommendations": ["정기 면담 실시", "업무 조정 검토"],
    "confidence_score": 0.85
  },
  "execution_summary": {
    "successful_workers": 5,
    "failed_workers": 0,
    "error_count": 0
  }
}
```

#### `POST /batch_analyze`
배치 분석 (여러 직원 동시)
```json
// 요청
{
  "employee_ids": ["EMP001", "EMP002", "EMP003"]
}

// 응답
{
  "success": true,
  "batch_results": [...],
  "summary": {
    "total_employees": 3,
    "successful_analyses": 3,
    "success_rate": 1.0
  }
}
```

### 상태 조회 엔드포인트

#### `GET /get_workflow_status/<session_id>`
워크플로우 상태 조회
```json
{
  "success": true,
  "session_id": "uuid-session-id",
  "status": {
    "current_step": "COMPLETED",
    "is_completed": true,
    "elapsed_time_minutes": 2.5,
    "completion_status": {
      "completion_rate": 1.0,
      "successful_workers": 5
    }
  }
}
```

#### `GET /list_active_sessions`
활성 세션 목록
```json
{
  "success": true,
  "active_sessions": [
    {
      "session_id": "uuid-1",
      "employee_id": "EMP001",
      "status": "completed",
      "created_at": "2024-01-15T10:30:00"
    }
  ],
  "total_sessions": 1
}
```

## 🔄 워크플로우 동작 원리

### 1. 워크플로우 시작
```
직원 ID 입력 → AgentState 초기화 → 워커 헬스체크
```

### 2. 병렬 분석 실행
```
Supervisor → WorkerDispatcher → [Structura, Cognita, Chronos, Sentio, Agora]
                                      ↓
                               각 워커가 병렬로 분석 수행
```

### 3. 조건부 라우팅
```python
# 슈퍼바이저의 라우팅 결정
if all_workers_completed:
    return "FINALIZE"
elif has_retryable_failures:
    return "RETRY_FAILED"
elif has_pending_workers:
    return "CONTINUE_ANALYSIS"
else:
    return "ERROR_CRITICAL"
```

### 4. 최종 종합 분석
```
성공한 워커 결과들 → Synthesizer → AHP 가중치 적용 → 최종 보고서
```

### 5. 오류 처리 및 복원력
- **재시도 메커니즘**: 최대 3회 재시도
- **부분 결과 처리**: 일부 워커 실패 시에도 가능한 결과로 분석
- **타임아웃 처리**: 30분 초과 시 부분 결과 반환
- **오류 분류**: 네트워크, 데이터, 시스템, 치명적 오류로 분류

## 🧪 테스트

### API 테스트 실행
```bash
cd app/Supervisor
python test_supervisor_api.py
```

테스트 결과:
```
🧪 Supervisor API 전체 테스트 시작
============================================================
🏥 Health Check 테스트...
  ✅ 서버 상태: healthy
  📊 워크플로우 초기화: True
  🤖 사용 가능한 워커: 5 (structura, cognita, chronos, sentio, agora)

👤 직원 분석 테스트 (ID: test_employee_001)...
  📤 분석 요청 전송...
  ✅ 분석 완료 (45.2초)
  📄 최종 보고서:
    위험 점수: 67.5/100
    위험 등급: B
    이탈 확률: 67.5%
    신뢰도: 85.0%

============================================================
📋 테스트 결과 요약:
  ✅ PASS health_check
  ✅ PASS system_info
  ✅ PASS worker_health_check
  ✅ PASS employee_analysis
  ✅ PASS workflow_status
  ✅ PASS batch_analysis

🎯 전체 결과: 6/6 테스트 통과 (100.0%)
🎉 모든 테스트가 성공했습니다!
```

## 🔧 설정 및 커스터마이징

### AHP 가중치 조정
`synthesizer.py`에서 워커별 가중치 조정:
```python
self.ahp_weights = {
    WorkerType.STRUCTURA: 0.35,  # 정형 데이터 - 가장 높은 가중치
    WorkerType.COGNITA: 0.20,    # 관계 분석
    WorkerType.CHRONOS: 0.25,    # 시계열 패턴
    WorkerType.SENTIO: 0.15,     # 감성 분석
    WorkerType.AGORA: 0.05       # 외부 시장 - 참고용
}
```

### 재시도 및 타임아웃 설정
환경변수로 조정:
```env
MAX_RETRY_COUNT=3        # 최대 재시도 횟수
TIMEOUT_MINUTES=30       # 전체 워크플로우 타임아웃
MAX_BATCH_SIZE=2000        # 배치 분석 최대 크기
```

### LLM 모델 변경
`supervisor_flask_backend.py`에서 모델 변경:
```python
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",  # 또는 "gpt-3.5-turbo"
    temperature=0.1
)
```

## 📊 모니터링 및 로깅

### 로그 레벨 설정
```python
# DEBUG 모드에서 상세 로그
FLASK_DEBUG=True

# 프로덕션에서는 INFO 레벨
FLASK_DEBUG=False
```

### 주요 로그 메시지
- `Starting workflow for employee {employee_id}`: 워크플로우 시작
- `Dispatching workers for employee {employee_id}`: 워커 분배
- `Routing decision: {decision} -> {next_node}`: 라우팅 결정
- `Synthesis completed for employee {employee_id}`: 종합 분석 완료

## 🚨 문제 해결

### 일반적인 문제

#### 1. 워커 서버 연결 실패
```
❌ Structura: http://localhost:5001 - 연결 실패
```
**해결방법**: 해당 워커 서버가 실행 중인지 확인

#### 2. OpenAI API 키 오류
```
⚠️ 경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.
```
**해결방법**: `.env` 파일에 유효한 OpenAI API 키 설정

#### 3. 분석 타임아웃
```
⏰ 분석 타임아웃 (30분 초과)
```
**해결방법**: `TIMEOUT_MINUTES` 환경변수 증가 또는 워커 성능 최적화

#### 4. 메모리 부족
```
Error in synthesis: Memory allocation failed
```
**해결방법**: 시스템 메모리 증가 또는 배치 크기 감소

### 디버깅 팁

1. **상세 로그 활성화**:
   ```env
   FLASK_DEBUG=True
   ```

2. **개별 워커 테스트**:
   ```bash
   curl http://localhost:5001/health  # Structura
   curl http://localhost:5002/health  # Cognita
   ```

3. **워크플로우 상태 추적**:
   ```bash
   curl http://localhost:5006/get_workflow_status/{session_id}
   ```

## 🔄 MCP (Model Context Protocol) 통합

### 개요
Supervisor Agent에 MCP 지원이 추가되어 Claude Desktop, IDE, 그리고 다른 MCP 클라이언트와 원활하게 통합할 수 있습니다.

### 주요 기능
- ✅ **8개의 MCP 도구**: 5개 워커 + 3개 관리 도구
- ✅ **표준화된 인터페이스**: MCP 프로토콜 완전 지원
- ✅ **하위 호환성**: 기존 REST API와 병행 사용 가능
- ✅ **세션 관리**: 멀티 세션 동시 처리 지원

### 사용 가능한 MCP 도구

1. **analyze_structura** - 정형 데이터 분석
2. **analyze_cognita** - 지식 그래프 분석
3. **analyze_chronos** - 시계열 분석
4. **analyze_sentio** - NLP 감성 분석
5. **analyze_agora** - 외부 시장 분석
6. **route_workflow** - 워크플로우 라우팅
7. **get_workflow_status** - 상태 조회
8. **create_error_report** - 오류 보고서 생성

### 빠른 시작

#### 1. Claude Desktop 연동
`claude_desktop_config.json`에 추가:
```json
{
  "mcpServers": {
    "supervisor-mcp-server": {
      "command": "python",
      "args": ["-m", "app.Supervisor.mcp_server"],
      "env": {
        "OPENAI_API_KEY": "your_key_here"
      }
    }
  }
}
```

#### 2. Python 프로그래밍 방식
```python
from app.Supervisor.mcp_adapter import SupervisorMCPAdapter

# 어댑터 초기화
adapter = SupervisorMCPAdapter(enable_mcp_server=False)

# MCP를 통한 워크플로우 실행
result = await adapter.execute_workflow(
    employee_id="EMP001",
    use_mcp=True
)
```

### 자세한 정보
MCP 통합에 대한 상세 가이드는 [MCP_INTEGRATION_GUIDE.md](./MCP_INTEGRATION_GUIDE.md)를 참조하세요.

## 🔮 향후 개선사항

### 성능 최적화
- [ ] 워커 결과 캐싱
- [ ] 비동기 배치 처리 개선
- [ ] 메모리 사용량 최적화

### 기능 확장
- [ ] 실시간 워크플로우 모니터링 대시보드
- [ ] 워커별 성능 메트릭 수집
- [ ] A/B 테스트를 위한 다중 AHP 가중치 지원
- [x] **MCP (Model Context Protocol) 통합** ✨ **NEW!**

### 운영 개선
- [ ] Kubernetes 배포 지원
- [ ] 프로메테우스 메트릭 내보내기
- [ ] 자동 스케일링 지원

---

**Supervisor Agent v1.1.0** - LangGraph 기반의 지능형 HR Attrition 예측 워크플로우 오케스트레이터 (MCP 지원 추가)
