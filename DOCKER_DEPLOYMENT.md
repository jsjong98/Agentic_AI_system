# 🐳 Agentic AI System - Docker 배포 가이드

## 📋 개요
Agentic AI System의 Docker 컨테이너 배포 가이드입니다. 7개의 AI 에이전트(Structura, Cognita, Chronos, Sentio, Agora, Supervisor, Integration)와 React Dashboard를 통합하여 제공하며, GPU 지원 환경과 CPU 전용 환경 모두를 지원합니다.

## 🤖 시스템 구성

### 백엔드 에이전트 (7개)
- **Structura** (포트 5001): 정형 데이터 ML 분석 (XGBoost, SHAP, LIME)
- **Cognita** (포트 5002): 네트워크 관계 분석 (Neo4j 그래프 DB)
- **Chronos** (포트 5003): 시계열 딥러닝 분석 (PyTorch, LSTM)
- **Sentio** (포트 5004): 텍스트 감정 분석 (Transformers, OpenAI)
- **Agora** (포트 5005): 시장 분석 + LLM (yfinance, OpenAI)
- **Supervisor** (포트 5006): LangGraph 워크플로우 관리 (메인 API)
- **Integration** (포트 5007): 결과 통합 및 최적화 (Optuna)

### 프론트엔드
- **Dashboard** (포트 80): React 기반 웹 대시보드 (Nginx)

## 🔧 환경 요구사항

### 최소 요구사항
- Docker 20.10 이상
- Docker Compose 2.0 이상
- 최소 16GB RAM (7개 에이전트 동시 실행)
- 최소 20GB 디스크 공간

### GPU 지원 (권장)
- NVIDIA GPU 드라이버 450.80.02 이상
- NVIDIA Container Toolkit
- CUDA 11.7 이상
- 최소 8GB GPU 메모리

### 패키지 의존성
- **딥러닝**: PyTorch 2.0.1, transformers, accelerate
- **머신러닝**: scikit-learn, xgboost, optuna, imbalanced-learn
- **시계열 분석**: statsmodels 0.14.0
- **그래프 DB**: Neo4j 5.13.0
- **LLM**: OpenAI, LangChain, LangGraph
- **웹 프레임워크**: Flask, React
- **시각화**: matplotlib, seaborn, plotly

## 🚀 빠른 시작

### 1. 환경 변수 설정 (필수)

시스템 실행 전에 필요한 API 키와 데이터베이스 연결 정보를 설정하세요:

```bash
# .env 파일 생성
cat > .env << EOF
# OpenAI API 설정 (Sentio, Agora, Supervisor, Integration용)
OPENAI_API_KEY=your-openai-api-key-here

# Neo4j 데이터베이스 설정 (Cognita용)
NEO4J_URI=bolt://54.162.43.24:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=resident-success-moss

# Flask 환경 설정
FLASK_ENV=production
PYTHONUNBUFFERED=1
EOF
```

### 2. GPU 지원 환경 (권장)

#### NVIDIA Container Toolkit 설치
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### 시스템 실행
```bash
# GPU 지원 docker-compose 사용
docker-compose up -d

# 로그 확인 (모든 에이전트)
docker-compose logs -f

# 특정 에이전트 로그 확인
docker-compose logs -f agentic-backend
```

### 3. CPU 전용 환경

```bash
# CPU 전용 docker-compose 사용
docker-compose -f docker-compose.cpu.yml up -d

# 로그 확인
docker-compose -f docker-compose.cpu.yml logs -f
```

### 4. 시스템 접속

시스템이 시작되면 다음 주소로 접속할 수 있습니다:

- **메인 대시보드**: http://localhost (React Dashboard)
- **Supervisor API**: http://localhost:5006 (메인 API 엔드포인트)
- **개별 에이전트 API**: 
  - Structura: http://localhost:5001
  - Cognita: http://localhost:5002
  - Chronos: http://localhost:5003
  - Sentio: http://localhost:5004
  - Agora: http://localhost:5005
  - Integration: http://localhost:5007

## 📊 서비스 구성

### 백엔드 (agentic-backend)
- **포트**: 5001-5007 (7개 에이전트)
- **이미지**: 커스텀 빌드 (PyTorch + Flask + 모든 AI 라이브러리)
- **에이전트별 기능**: 
  - **Structura**: ML 분석, 설명 가능한 AI (SHAP, LIME)
  - **Cognita**: 그래프 네트워크 분석 (Neo4j)
  - **Chronos**: 시계열 딥러닝 (PyTorch, LSTM)
  - **Sentio**: 자연어 감정 분석 (Transformers)
  - **Agora**: 시장 데이터 분석 + LLM
  - **Supervisor**: 워크플로우 관리 (LangGraph)
  - **Integration**: 결과 통합 및 최적화 (Optuna)
- **볼륨**: 에이전트별 모델, 캐시, 결과 파일 영구 저장
- **헬스체크**: `/health` 엔드포인트 (Supervisor 기준)

### 프론트엔드 (agentic-dashboard)
- **포트**: 80, 443
- **이미지**: Nginx + React 빌드
- **기능**:
  - 통합 AI 에이전트 대시보드
  - 실시간 분석 결과 시각화
  - 파일 업로드 및 관리
  - 워크플로우 모니터링
- **프록시**: Supervisor API (5006) 자동 라우팅
- **헬스체크**: HTTP 응답 확인

## 🔍 상태 확인

### 컨테이너 상태 확인
```bash
# 실행 중인 컨테이너 확인
docker-compose ps

# 헬스체크 상태 확인
docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
```

### API 엔드포인트 테스트
```bash
# 메인 시스템 헬스체크 (Supervisor)
curl http://localhost:5006/health

# 모든 워커 에이전트 상태 확인
curl http://localhost:5006/worker_health_check

# 개별 에이전트 헬스체크
curl http://localhost:5001/health  # Structura
curl http://localhost:5002/health  # Cognita
curl http://localhost:5003/api/status  # Chronos
curl http://localhost:5004/health  # Sentio
curl http://localhost:5005/health  # Agora
curl http://localhost:5007/health  # Integration

# 시스템 정보 확인
curl http://localhost:5006/system_info
```

### 로그 모니터링
```bash
# 전체 로그
docker-compose logs

# 특정 서비스 로그
docker-compose logs agentic-backend
docker-compose logs agentic-dashboard

# 실시간 로그 추적
docker-compose logs -f --tail=100

# 에이전트별 로그 필터링
docker-compose logs agentic-backend | grep -i "structura"
docker-compose logs agentic-backend | grep -i "cognita"
docker-compose logs agentic-backend | grep -i "chronos"
docker-compose logs agentic-backend | grep -i "sentio"
docker-compose logs agentic-backend | grep -i "agora"
docker-compose logs agentic-backend | grep -i "supervisor"
docker-compose logs agentic-backend | grep -i "integration"

# 오류 로그만 확인
docker-compose logs agentic-backend | grep -i "error\|exception\|failed"
```

### 리소스 사용량 확인
```bash
# 컨테이너 리소스 사용량
docker stats

# GPU 사용량 (GPU 환경)
nvidia-smi
```

## 🔧 환경 설정

### 환경 변수 설정
```bash
# .env 파일 생성
cat > .env << EOF
# OpenAI API 설정 (필수 - Sentio, Agora, Supervisor, Integration용)
OPENAI_API_KEY=your-openai-api-key-here

# Neo4j 데이터베이스 설정 (Cognita용)
NEO4J_URI=bolt://54.162.43.24:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=resident-success-moss

# Flask 환경 설정
FLASK_ENV=production
PYTHONUNBUFFERED=1

# 에이전트 포트 설정 (기본값)
STRUCTURA_PORT=5001
COGNITA_PORT=5002
CHRONOS_PORT=5003
SENTIO_PORT=5004
AGORA_PORT=5005
SUPERVISOR_PORT=5006
INTEGRATION_PORT=5007

# GPU 설정 (GPU 환경에서만)
NVIDIA_VISIBLE_DEVICES=all
CUDA_VISIBLE_DEVICES=0

# CPU 설정 (CPU 환경에서만)
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4

# 로그 레벨
LOG_LEVEL=INFO

# 캐시 설정
CACHE_SIZE=1000
PREDICTION_CACHE_TTL=3600
EOF
```

### 포트 변경
```yaml
# docker-compose.yml에서 포트 수정
services:
  agentic-dashboard:
    ports:
      - "8080:80"  # 포트 80 대신 8080 사용
      - "8443:443"  # 포트 443 대신 8443 사용
  agentic-backend:
    ports:
      - "6001:5001"  # Structura 포트 변경
      - "6002:5002"  # Cognita 포트 변경
      - "6003:5003"  # Chronos 포트 변경
      - "6004:5004"  # Sentio 포트 변경
      - "6005:5005"  # Agora 포트 변경
      - "6006:5006"  # Supervisor 포트 변경 (메인 API)
      - "6007:5007"  # Integration 포트 변경
```

## 💾 데이터 영속성

### 볼륨 관리
```bash
# 볼륨 목록 확인
docker volume ls

# 특정 볼륨 상세 정보 (프로젝트명에 따라 변경)
docker volume inspect agentic_ai_system_agentic_models
docker volume inspect agentic_ai_system_agentic_cache
docker volume inspect agentic_ai_system_structura_models
docker volume inspect agentic_ai_system_cognita_cache
docker volume inspect agentic_ai_system_integration_results

# 백업 생성 (주요 볼륨들)
docker run --rm -v agentic_ai_system_agentic_models:/data -v $(pwd):/backup alpine tar czf /backup/agentic-models-backup.tar.gz /data
docker run --rm -v agentic_ai_system_structura_models:/data -v $(pwd):/backup alpine tar czf /backup/structura-models-backup.tar.gz /data
docker run --rm -v agentic_ai_system_chronos_models:/data -v $(pwd):/backup alpine tar czf /backup/chronos-models-backup.tar.gz /data
docker run --rm -v agentic_ai_system_integration_results:/data -v $(pwd):/backup alpine tar czf /backup/integration-results-backup.tar.gz /data
```

### 데이터 마이그레이션
```bash
# 볼륨 데이터 복원
docker run --rm -v agentic_ai_system_agentic_models:/data -v $(pwd):/backup alpine tar xzf /backup/agentic-models-backup.tar.gz -C /
docker run --rm -v agentic_ai_system_structura_models:/data -v $(pwd):/backup alpine tar xzf /backup/structura-models-backup.tar.gz -C /
docker run --rm -v agentic_ai_system_chronos_models:/data -v $(pwd):/backup alpine tar xzf /backup/chronos-models-backup.tar.gz -C /
docker run --rm -v agentic_ai_system_integration_results:/data -v $(pwd):/backup alpine tar xzf /backup/integration-results-backup.tar.gz -C /
```

## 🔄 업데이트 및 재배포

### 1. 이미지 업데이트
```bash
# 이미지 다시 빌드
docker-compose build --no-cache

# 서비스 재시작
docker-compose up -d
```

### 2. 코드 변경 후 배포
```bash
# 개발 중인 변경사항 반영
docker-compose down
docker-compose build
docker-compose up -d
```

### 3. 롤링 업데이트
```bash
# 백엔드만 업데이트 (모든 에이전트)
docker-compose up -d --no-deps agentic-backend

# 프론트엔드만 업데이트
docker-compose up -d --no-deps agentic-dashboard

# 특정 서비스만 재시작 (컨테이너 내부에서 개별 에이전트 재시작은 불가)
docker-compose restart agentic-backend
docker-compose restart agentic-dashboard
```

## 🐛 문제 해결

### 1. 컨테이너 실행 오류
```bash
# 상세 로그 확인
docker-compose logs agentic-backend

# 컨테이너 내부 접속
docker exec -it agentic-backend bash

# 개별 에이전트 프로세스 확인 (컨테이너 내부에서)
docker exec -it agentic-backend ps aux | grep python

# 디스크 용량 확인
df -h
docker system df

# 컨테이너 리소스 사용량 확인
docker stats agentic-backend
```

### 2. GPU 인식 오류
```bash
# GPU 상태 확인
nvidia-smi

# NVIDIA Docker 런타임 확인
docker run --rm --gpus all nvidia/cuda:11.7-base-ubuntu20.04 nvidia-smi
```

### 3. 에이전트별 오류 진단
```bash
# 각 에이전트 패키지 확인
docker exec -it agentic-backend python -c "import torch; print(f'PyTorch: {torch.__version__}')"
docker exec -it agentic-backend python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
docker exec -it agentic-backend python -c "import neo4j; print(f'Neo4j: {neo4j.__version__}')"
docker exec -it agentic-backend python -c "import openai; print(f'OpenAI: {openai.__version__}')"
docker exec -it agentic-backend python -c "import langchain; print(f'LangChain: {langchain.__version__}')"

# 개별 에이전트 테스트
docker exec -it agentic-backend python -c "from Structura.structura_processor import StructuraProcessor; print('Structura OK')"
docker exec -it agentic-backend python -c "from Cognita.cognita_processor import CognitaProcessor; print('Cognita OK')"
docker exec -it agentic-backend python -c "from Chronos.chronos_processor_fixed import ChronosProcessor; print('Chronos OK')"
docker exec -it agentic-backend python -c "from Sentio.sentio_processor import SentioProcessor; print('Sentio OK')"

# 필요시 패키지 재설치
docker exec -it agentic-backend pip install --upgrade torch transformers langchain openai
```

### 4. 메모리 부족 오류
```bash
# 메모리 사용량 확인
docker stats --no-stream

# 컨테이너 메모리 제한 설정 (docker-compose.yml)
services:
  agentic-backend:
    mem_limit: 16g  # 7개 에이전트를 위한 충분한 메모리
    memswap_limit: 16g
    
# 개별 에이전트 메모리 사용량 확인 (컨테이너 내부)
docker exec -it agentic-backend top -p $(docker exec agentic-backend pgrep -f python)
```

## 📈 성능 최적화

### 1. 리소스 제한 설정
```yaml
# docker-compose.yml에 추가
services:
  agentic-backend:
    deploy:
      resources:
        limits:
          memory: 16G  # 7개 에이전트를 위한 충분한 메모리
          cpus: '8.0'  # 멀티 에이전트 처리를 위한 CPU
        reservations:
          memory: 8G
          cpus: '4.0'
  agentic-dashboard:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'
```

### 2. 이미지 크기 최적화
```bash
# 이미지 크기 확인
docker images | grep agentic

# 빌드 캐시 최적화
docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1

# 불필요한 이미지 정리
docker system prune -a
docker image prune -a
```

### 3. 볼륨 성능 향상
```bash
# SSD 사용 확인
lsblk -d -o name,rota

# 볼륨 마운트 옵션 최적화 (docker-compose.yml)
volumes:
  - agentic_models:/app/models:cached
  - structura_models:/app/Structura/models:cached
  - chronos_models:/app/Chronos/models:cached
  - integration_results:/app/Integration/results:cached
```

## 🔐 보안 설정

### 1. 네트워크 격리
```yaml
# docker-compose.yml에 커스텀 네트워크 추가 (이미 설정됨)
networks:
  agentic-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

services:
  agentic-backend:
    networks:
      - agentic-network
  agentic-dashboard:
    networks:
      - agentic-network
```

### 2. 환경 변수 보안
```bash
# 민감한 정보는 Docker secrets 사용
echo "your-openai-api-key" | docker secret create openai_api_key -
echo "your-neo4j-password" | docker secret create neo4j_password -

# .env 파일 권한 설정
chmod 600 .env

# 프로덕션에서는 환경 변수 파일 대신 secrets 사용 권장
```

## 📞 지원

### 로그 수집
```bash
# 전체 시스템 정보 수집
docker-compose logs > agentic-deployment-logs.txt
docker system info > system-info.txt
docker-compose ps --format json > container-status.json

# 에이전트별 로그 수집
docker-compose logs agentic-backend > backend-logs.txt
docker-compose logs agentic-dashboard > dashboard-logs.txt

# 시스템 리소스 정보 수집
docker stats --no-stream > resource-usage.txt
docker volume ls > volumes-list.txt
```

### 디버깅 모드
```bash
# 개발 모드로 실행 (환경 변수 설정)
FLASK_ENV=development docker-compose up

# 개별 에이전트 디버깅 (컨테이너 내부에서)
docker exec -it agentic-backend bash
cd /app/Structura && python run_structura_server.py  # 개별 실행 테스트
```

### 5. 주요 사용 사례

#### 직원 분석 워크플로우 테스트
```bash
# 1. 시스템 상태 확인
curl http://localhost:5006/health

# 2. 파일 업로드 (CSV 데이터)
curl -X POST -F "file=@employee_data.csv" http://localhost:5006/upload_file

# 3. 직원 분석 실행
curl -X POST -H "Content-Type: application/json" \
  -d '{"employee_id": "12345", "analysis_type": "comprehensive"}' \
  http://localhost:5006/analyze_employee

# 4. 결과 확인
curl http://localhost:5006/get_workflow_status/session_id
```

---

**© 2025 Agentic AI System. All rights reserved.**