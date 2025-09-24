# 🚀 Agentic AI System - 빠른 실행 가이드

이 가이드는 Agentic AI System의 모든 백엔드 서버와 프론트엔드를 빠르게 실행하는 방법을 설명합니다.

## 📋 시스템 구성

### 백엔드 서버 (7개)
- **Structura** (포트 5001): 정형 데이터 ML 분석
- **Cognita** (포트 5002): 네트워크 관계 분석  
- **Chronos** (포트 5003): 시계열 딥러닝 분석
- **Sentio** (포트 5004): 텍스트 감정 분석
- **Agora** (포트 5005): 시장 분석 + LLM
- **Supervisor** (포트 5006): LangGraph 워크플로우 관리
- **Integration** (포트 5007): 결과 통합 및 최적화

### 프론트엔드
- **Dashboard** (포트 3000): React 기반 웹 대시보드

## 🛠️ 사전 준비

### 1. Python 환경 설정
```bash
# Anaconda 환경 활성화 (필수)
conda activate nlp
```

### 2. 환경 변수 설정
```bash
# Neo4j 연결 설정 (Cognita용)
export NEO4J_URI="bolt://54.162.43.24:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="resident-success-moss"

# OpenAI API 키 설정 (Sentio, Agora, Supervisor, Integration용)
export OPENAI_API_KEY="your-gpt5nano-api-key"
```

또는 각 에이전트 폴더에 `.env` 파일 생성:
```env
# Cognita/.env
NEO4J_URI=bolt://54.162.43.24:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=resident-success-moss

# Sentio/.env, Agora/.env, Supervisor/.env, Integration/.env
OPENAI_API_KEY=your-gpt5nano-api-key
```

## 🚀 백엔드 서버 실행

### 방법 1: 통합 스크립트 사용 (권장)

프로젝트 루트 디렉토리에서:

```bash
# 모든 백엔드 서버를 한 번에 실행
C:/Users/OJH/anaconda3/envs/nlp/python.exe start_all_services_simple.py
```

**실행 결과:**
```
🤖 Agentic AI System - 통합 서비스 관리자 (현재 환경)
======================================================================

🐍 Python 실행 파일: C:/Users/OJH/anaconda3/envs/nlp/python.exe
🌍 Python 버전: 3.x.x
🔧 Conda 환경: nlp

🚀 Structura 서버 시작 중... (포트 5001)
✅ Structura 서버 시작됨 (PID: xxxx)

🚀 Cognita 서버 시작 중... (포트 5002)
✅ Cognita 서버 시작됨 (PID: xxxx)

🚀 Chronos 서버 시작 중... (포트 5003)
✅ Chronos 서버 시작됨 (PID: xxxx)

🚀 Sentio 서버 시작 중... (포트 5004)
✅ Sentio 서버 시작됨 (PID: xxxx)

🚀 Agora 서버 시작 중... (포트 5005)
✅ Agora 서버 시작됨 (PID: xxxx)

🚀 Supervisor 서버 시작 중... (포트 5006)
✅ Supervisor 서버 시작됨 (PID: xxxx)

🚀 Integration 서버 시작 중... (포트 5007)
✅ Integration 서버 시작됨 (PID: xxxx)

🎉 모든 에이전트 서버 시작 완료!
======================================================================

📡 실행 중인 서버들:
  • Structura    → http://localhost:5001
  • Cognita      → http://localhost:5002
  • Chronos      → http://localhost:5003
  • Sentio       → http://localhost:5004
  • Agora        → http://localhost:5005
  • Supervisor   → http://localhost:5006
  • Integration  → http://localhost:5007

🌐 React 대시보드:
  cd Dashboard && npm start

💡 메인 API 엔드포인트:
  Supervisor → http://localhost:5006 (모든 기능 통합)
  - 파일 업로드/관리
  - 모든 워커 에이전트 API 프록시
  - Integration 에이전트 API 프록시

⚠️  제어 명령:
  - Ctrl+C: 모든 서비스 종료
  - 서비스 상태는 자동으로 모니터링됩니다
```

### 방법 2: 개별 서버 실행 (개발/디버깅용)

각 서버를 개별적으로 실행하려면:

```bash
# 각 에이전트 디렉토리에서 실행
cd app/Structura && C:/Users/OJH/anaconda3/envs/nlp/python.exe run_structura_server.py
cd app/Cognita && C:/Users/OJH/anaconda3/envs/nlp/python.exe run_cognita_server.py
cd app/Chronos && C:/Users/OJH/anaconda3/envs/nlp/python.exe run_chronos_server.py
cd app/Sentio && C:/Users/OJH/anaconda3/envs/nlp/python.exe run_sentio_server.py
cd app/Agora && C:/Users/OJH/anaconda3/envs/nlp/python.exe run_agora_server.py
cd app/Supervisor && C:/Users/OJH/anaconda3/envs/nlp/python.exe run_supervisor_server.py
cd app/Integration && C:/Users/OJH/anaconda3/envs/nlp/python.exe run_integration_server.py
```

## 🌐 프론트엔드 실행

### 1. Dashboard 디렉토리로 이동
```bash
cd Dashboard
```

### 2. 의존성 설치 (최초 1회만)
```bash
npm install
```

### 3. 개발 서버 실행
```bash
npm start
```

**실행 결과:**
```
Compiled successfully!

You can now view agentic-ai-dashboard in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000

Note that the development build is not optimized.
To create a production build, use npm run build.

webpack compiled with warnings in 15834ms
```

### 4. 브라우저에서 접속
```
http://localhost:3000
```

## ✅ 서버 상태 확인

### 백엔드 서버 헬스체크
```bash
# 각 서버의 상태 확인
curl http://localhost:5001/api/health  # Structura
curl http://localhost:5002/api/health  # Cognita
curl http://localhost:5003/api/status  # Chronos
curl http://localhost:5004/health      # Sentio
curl http://localhost:5005/health      # Agora
curl http://localhost:5006/health      # Supervisor
curl http://localhost:5007/health      # Integration
```

### 통합 상태 확인
```bash
# Supervisor를 통한 전체 워커 상태 확인
curl http://localhost:5006/worker_health_check
```

## 🔧 문제 해결

### 1. 백엔드 서버 문제

#### 포트 충돌
```bash
# 포트 사용 확인
netstat -an | findstr :5001
netstat -an | findstr :5002
# ... 기타 포트들

# 프로세스 종료 (Windows)
taskkill /F /PID <PID번호>
```

#### Python 환경 문제
```bash
# 현재 conda 환경 확인
conda info --envs

# nlp 환경 활성화
conda activate nlp

# Python 경로 확인
which python  # Linux/Mac
where python  # Windows
```

#### 의존성 문제
```bash
# 각 에이전트 폴더에서 의존성 재설치
cd app/Structura && pip install -r requirements.txt
cd app/Cognita && pip install -r requirements.txt
# ... 기타 에이전트들
```

### 2. 프론트엔드 문제

#### Node.js 버전 확인
```bash
node --version  # v16+ 권장
npm --version   # v8+ 권장
```

#### 의존성 설치 문제
```bash
# 캐시 클리어 후 재설치
npm cache clean --force
rm -rf node_modules package-lock.json  # Linux/Mac
rmdir /s node_modules & del package-lock.json  # Windows
npm install
```

#### 프록시 설정 확인
`Dashboard/package.json`에서 프록시 설정 확인:
```json
{
  "proxy": "http://localhost:5006"
}
```

### 3. API 연결 문제

#### CORS 오류
- 백엔드 서버들이 모두 Flask-CORS로 설정되어 있는지 확인
- 브라우저 개발자 도구에서 네트워크 탭 확인

#### 환경 변수 누락
```bash
# 필수 환경 변수 확인
echo $NEO4J_URI
echo $OPENAI_API_KEY
```

## 📊 시스템 모니터링

### 로그 확인
통합 스크립트 사용 시 로그는 `logs/` 폴더에 저장됩니다:
```
logs/
├── structura_server.log
├── cognita_server.log
├── chronos_server.log
├── sentio_server.log
├── agora_server.log
├── supervisor_server.log
└── integration_server.log
```

### 실시간 모니터링
통합 스크립트는 15초마다 서비스 상태를 자동으로 체크합니다:
```
📊 서비스 상태 체크:
   ✅ Structura    → 정상 (포트 5001)
   ✅ Cognita      → 정상 (포트 5002)
   ✅ Chronos      → 정상 (포트 5003)
   ✅ Sentio       → 정상 (포트 5004)
   ✅ Agora        → 정상 (포트 5005)
   ✅ Supervisor   → 정상 (포트 5006)
   ✅ Integration  → 정상 (포트 5007)
🎉 모든 서비스가 정상 작동 중입니다!
```

## 🛑 서비스 종료

### 통합 스크립트 종료
통합 스크립트 실행 중인 터미널에서:
```bash
Ctrl+C
```

자동으로 모든 서버가 안전하게 종료됩니다:
```
🛑 모든 서비스를 종료합니다...
   ⏹️  Structura 서버 종료 중...
   ⏹️  Cognita 서버 종료 중...
   ⏹️  Chronos 서버 종료 중...
   ⏹️  Sentio 서버 종료 중...
   ⏹️  Agora 서버 종료 중...
   ⏹️  Supervisor 서버 종료 중...
   ⏹️  Integration 서버 종료 중...
✅ 모든 서비스가 종료되었습니다.
```

### 프론트엔드 종료
프론트엔드 실행 중인 터미널에서:
```bash
Ctrl+C
```

## 📚 추가 정보

### API 문서
- **Supervisor API**: http://localhost:5006 (통합 API)
- **각 에이전트 API**: 각 에이전트의 README.md 참조

### 데이터 준비
- **IBM HR 데이터**: `data/IBM_HR.csv`
- **시계열 데이터**: `data/IBM_HR_timeseries.csv`
- **텍스트 데이터**: `data/IBM_HR_text.csv`

### 성능 최적화
- **메모리**: 최소 8GB RAM 권장
- **CPU**: 멀티코어 프로세서 권장
- **GPU**: CUDA 지원 GPU (Chronos 딥러닝용, 선택사항)

---

**개발팀**: PwC RA Team  
**버전**: 1.0.0  
**최종 업데이트**: 2025년 1월

🎯 **빠른 시작**: `start_all_services_simple.py` → `cd Dashboard && npm start` → `http://localhost:3000`
