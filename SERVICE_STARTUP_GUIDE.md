# 🚀 Agentic AI System 서비스 시작 가이드

## 📋 개요

마스터 서버가 제거되고 **Supervisor**가 모든 기능을 통합 제공하는 새로운 아키텍처입니다.

```
🎯 통합된 Agentic AI System
├── Supervisor (통합 API 서버) - 포트 5006 ⭐
│   ├── LangGraph 워크플로우 관리
│   ├── 파일 업로드/관리 기능
│   ├── 모든 워커 에이전트 API 프록시
│   └── Integration 에이전트 API 프록시
├── Integration (최적화 엔진) - 포트 5007
│   ├── 임계치/가중치 최적화
│   ├── 배치 분석 처리
│   └── LLM 레포트 생성
└── 5개 워커 에이전트 (포트 5001-5005)
    ├── Structura (정형 데이터)
    ├── Cognita (관계 분석)
    ├── Chronos (시계열)
    ├── Sentio (감정 분석)
    └── Agora (시장 분석)
```

## 🐍 Conda 환경 설정

**중요**: 모든 에이전트는 `nlp` conda 환경에서 실행되어야 합니다.

### 1. conda 환경 생성 (최초 1회)
```bash
conda create -n nlp python=3.9
conda activate nlp
pip install -r requirements.txt  # 각 에이전트 디렉토리에서 실행
```

### 2. 환경 활성화 확인
```bash
conda activate nlp
python --version  # Python 3.9.x 확인
```

## 🚀 서비스 시작 방법

### 방법 1: 통합 서비스 관리자 (권장) 🌟

#### Windows:
```bash
# conda 환경 자동 처리
start_all_services_conda.bat

# 또는 수동으로
conda activate nlp
python start_all_services_conda.py
```

#### Linux/Mac:
```bash
conda activate nlp
python start_all_services_conda.py
```

**특징:**
- ✅ conda nlp 환경 자동 감지
- ✅ 모든 서비스 백그라운드 실행
- ✅ 실시간 상태 모니터링
- ✅ 로그 파일 자동 생성 (`logs/` 디렉토리)
- ✅ Ctrl+C로 모든 서비스 일괄 종료

### 방법 2: 개별 터미널 창 실행

#### Windows:
```bash
run_all_agents.bat  # conda activate nlp 자동 포함
```

#### Linux/Mac:
```bash
python run_all_agents.py
```

**특징:**
- ✅ 각 서비스가 독립적인 터미널 창에서 실행
- ✅ 개별 서비스 로그 확인 용이
- ⚠️ 각 터미널 창을 개별적으로 종료해야 함

### 방법 3: 기존 스크립트 (수정됨)

```bash
# conda 환경 활성화 후
conda activate nlp
python start_all_services.py
```

## 🔍 서비스 상태 확인

### 1. Health Check
```bash
curl http://localhost:5006/health  # Supervisor
curl http://localhost:5001/health  # Structura
curl http://localhost:5002/health  # Cognita
curl http://localhost:5003/health  # Chronos
curl http://localhost:5004/health  # Sentio
curl http://localhost:5005/health  # Agora
curl http://localhost:5007/health  # Integration
```

### 2. 통합 워커 상태 확인
```bash
curl http://localhost:5006/api/workers/health_check_all
```

### 3. 로그 파일 확인
```bash
# 로그 디렉토리
ls logs/
- structura_server.log
- cognita_server.log
- chronos_server.log
- sentio_server.log
- agora_server.log
- supervisor_server.log
- integration_server.log
```

## 💡 메인 API 엔드포인트

**Supervisor (포트 5006)** - 모든 기능 통합 제공:

### 📊 분석 기능
- `GET /health` - 서버 상태 확인
- `POST /analyze_employee` - 직원 분석
- `POST /batch_analyze` - 배치 분석
- `GET /worker_health_check` - 워커 상태 확인

### 📁 파일 관리
- `POST /upload_file` - 파일 업로드
- `GET /list_uploaded_files` - 업로드된 파일 목록
- `GET /download_file/<filename>` - 파일 다운로드
- `DELETE /delete_file/<filename>` - 파일 삭제

### 🔗 워커 에이전트 API 프록시
- `POST /api/workers/structura/predict` - Structura 예측
- `GET /api/workers/cognita/analyze/<employee_id>` - Cognita 분석
- `POST /api/workers/chronos/predict` - Chronos 예측
- `POST /api/workers/sentio/analyze_sentiment` - Sentio 감정 분석
- `POST /api/workers/agora/comprehensive_analysis` - Agora 시장 분석

### 🔧 Integration 에이전트 API 프록시
- `POST /api/workers/integration/load_data` - 데이터 로드
- `POST /api/workers/integration/calculate_thresholds` - 임계값 계산
- `POST /api/workers/integration/optimize_weights` - 가중치 최적화
- `POST /api/workers/integration/predict_employee` - 개별 직원 예측
- `POST /api/workers/integration/generate_report` - 레포트 생성

## 🌐 React Dashboard

```bash
cd Dashboard
npm install  # 최초 1회
npm start    # 포트 3000에서 실행
```

Dashboard는 Supervisor (포트 5006)와 연동됩니다.

## ⚠️ 문제 해결

### 1. conda 환경 문제
```bash
# 환경 재생성
conda remove -n nlp --all
conda create -n nlp python=3.9
conda activate nlp
```

### 2. 포트 충돌
```bash
# Windows에서 포트 사용 확인
netstat -ano | findstr :5006

# Linux/Mac에서 포트 사용 확인
lsof -i :5006
```

### 3. 서비스 시작 실패
- `logs/` 디렉토리의 로그 파일 확인
- conda nlp 환경이 활성화되어 있는지 확인
- 필요한 패키지가 설치되어 있는지 확인

### 4. 의존성 문제
```bash
conda activate nlp
cd app/Supervisor && pip install -r requirements.txt
cd ../Integration && pip install -r requirements.txt
cd ../Structura && pip install -r requirements.txt
# ... 각 에이전트 디렉토리에서 실행
```

## 📝 주요 변경사항

1. **마스터 서버 제거**: 더 이상 `agentic_master_server.py` 실행 불필요
2. **Supervisor 통합**: 모든 기능이 Supervisor (포트 5006)로 통합
3. **conda 환경 지원**: 모든 스크립트가 conda nlp 환경 사용
4. **로그 시스템**: 각 서비스의 로그가 `logs/` 디렉토리에 저장
5. **상태 모니터링**: 실시간 서비스 상태 확인 기능

## 🎯 권장 워크플로우

1. **conda nlp 환경 활성화**
2. **`start_all_services_conda.py` 실행** (권장)
3. **서비스 상태 확인** (`http://localhost:5006/health`)
4. **React Dashboard 시작** (`cd Dashboard && npm start`)
5. **분석 작업 수행**
6. **Ctrl+C로 모든 서비스 종료**
