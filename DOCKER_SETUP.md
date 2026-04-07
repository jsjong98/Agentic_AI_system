# 🐳 Docker Compose 실행 가이드

Agentic AI System을 Docker Compose로 한 번에 실행하는 방법입니다.

## 📋 사전 요구사항

- Docker Engine 20.10 이상
- Docker Compose V2 (2.0 이상)
- 최소 8GB RAM (권장: 16GB)
- 최소 20GB 디스크 공간
- (선택) NVIDIA GPU + nvidia-docker (GPU 가속 사용 시)

---

## 🚀 빠른 시작

### 1️⃣ 환경 변수 설정

```bash
# env.example을 .env로 복사
cp env.example .env

# .env 파일 편집 (필수!)
nano .env  # 또는 원하는 에디터 사용
```

**⚠️ 필수 설정**:
- `OPENAI_API_KEY`: OpenAI API 키 입력 (https://platform.openai.com/api-keys)

### 2️⃣ 전체 시스템 실행

```bash
# 모든 서비스 빌드 및 시작
docker-compose up --build

# 또는 백그라운드 실행
docker-compose up -d --build
```

### 3️⃣ 서비스 확인

실행 후 다음 URL에서 각 서비스 접근 가능:

| 서비스 | URL | 설명 |
|--------|-----|------|
| 🎨 **Dashboard** | http://localhost:3000 | 메인 웹 인터페이스 |
| 👑 **Supervisor** | http://localhost:5006 | 워크플로우 관리 API |
| 🤖 **Structura** | http://localhost:5001 | 정형 데이터 분석 |
| 🧠 **Cognita** | http://localhost:5002 | 관계망 분석 |
| ⏰ **Chronos** | http://localhost:5003 | 시계열 분석 |
| 💭 **Sentio** | http://localhost:5004 | 감정 분석 |
| 🏛️ **Agora** | http://localhost:5005 | 시장 분석 |
| 🔗 **Integration** | http://localhost:5007 | 결과 통합 |
| 🗄️ **Neo4j** | http://localhost:7474 | 그래프 DB 브라우저 |

---

## 🔧 주요 명령어

### 서비스 관리

```bash
# 전체 시스템 시작 (포그라운드)
docker-compose up

# 전체 시스템 시작 (백그라운드)
docker-compose up -d

# 전체 시스템 중지
docker-compose down

# 전체 시스템 중지 + 볼륨 삭제 (데이터 초기화)
docker-compose down -v

# 전체 시스템 재시작
docker-compose restart
```

### 개별 서비스 관리

```bash
# 특정 서비스만 시작
docker-compose up structura cognita

# 특정 서비스 재시작
docker-compose restart supervisor

# 특정 서비스 중지
docker-compose stop dashboard

# 특정 서비스 로그 확인
docker-compose logs -f supervisor
```

### 빌드 관리

```bash
# 전체 이미지 재빌드
docker-compose build

# 특정 서비스만 재빌드
docker-compose build supervisor

# 캐시 없이 빌드
docker-compose build --no-cache

# 빌드 후 즉시 실행
docker-compose up --build
```

### 로그 확인

```bash
# 전체 로그 실시간 확인
docker-compose logs -f

# 특정 서비스 로그만 확인
docker-compose logs -f supervisor integration

# 마지막 100줄만 확인
docker-compose logs --tail=100 supervisor
```

### 시스템 상태 확인

```bash
# 실행 중인 서비스 목록
docker-compose ps

# 서비스 상태 확인 (헬스체크)
docker-compose ps --format json | jq '.[].Health'

# 리소스 사용량 확인
docker stats
```

---

## 🔄 개발 워크플로우

### 코드 수정 후 반영

```bash
# 1. 특정 에이전트 코드 수정 후
docker-compose build supervisor
docker-compose restart supervisor

# 2. 프론트엔드 코드 수정 후
docker-compose build dashboard
docker-compose restart dashboard
```

### 데이터베이스 초기화

```bash
# Neo4j 데이터만 삭제
docker-compose down
docker volume rm agentic_ai_system_neo4j-data
docker-compose up -d neo4j

# 모든 데이터 삭제
docker-compose down -v
docker-compose up -d
```

---

## 🐛 트러블슈팅

### 포트 충돌 발생

```bash
# 사용 중인 포트 확인
netstat -ano | findstr :5006  # Windows
lsof -i :5006                 # Mac/Linux

# 포트 변경 (docker-compose.yml 수정)
ports:
  - "15006:5006"  # 호스트 포트를 15006으로 변경
```

### 서비스가 시작되지 않음

```bash
# 로그 확인
docker-compose logs supervisor

# 컨테이너 내부 접속
docker-compose exec supervisor /bin/bash

# 헬스체크 수동 확인
curl http://localhost:5006/health
```

### 메모리 부족

```bash
# Docker 메모리 제한 설정 (docker-compose.yml)
services:
  structura:
    deploy:
      resources:
        limits:
          memory: 2G
```

### 빌드 실패

```bash
# Docker 캐시 초기화
docker system prune -a

# 재빌드
docker-compose build --no-cache
```

---

## 📊 프로덕션 배포

### 최적화된 빌드

```bash
# 프로덕션 모드로 빌드
FLASK_DEBUG=false docker-compose up -d --build
```

### 리소스 제한 설정

`docker-compose.yml`에 추가:

```yaml
services:
  supervisor:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### GPU 사용 활성화

`docker-compose.yml`에서 Chronos 서비스의 주석 해제:

```yaml
chronos:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

---

## 🔐 보안 설정

### 환경 변수 보안

```bash
# .env 파일 권한 설정
chmod 600 .env

# Git에서 제외 (.gitignore에 추가)
echo ".env" >> .gitignore
```

### Neo4j 비밀번호 변경

`.env` 파일에서:
```bash
NEO4J_PASSWORD=your-strong-password-here
```

---

## 📈 모니터링

### 리소스 사용량 모니터링

```bash
# 실시간 통계
docker stats

# JSON 형식으로 출력
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

### 로그 수집

```bash
# 모든 로그를 파일로 저장
docker-compose logs > system-logs.txt

# 특정 기간 로그
docker-compose logs --since 2024-01-01 --until 2024-01-02
```

---

## 🧹 정리

### 시스템 완전 제거

```bash
# 컨테이너, 볼륨, 네트워크 모두 삭제
docker-compose down -v --remove-orphans

# 이미지도 삭제
docker-compose down --rmi all -v

# Docker 시스템 전체 정리
docker system prune -a --volumes
```

---

## 💡 팁

### 빠른 재시작

```bash
# 특정 서비스만 빠르게 재시작 (빌드 없이)
docker-compose restart supervisor
```

### 서비스 스케일링

```bash
# 동일 서비스 여러 개 실행 (로드 밸런싱)
docker-compose up -d --scale structura=3
```

### 로그 레벨 조정

`.env` 파일에 추가:
```bash
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

---

## 📞 문제 해결

문제가 지속되면 다음 정보와 함께 이슈 제기:

```bash
# 시스템 정보 수집
docker-compose version
docker version
docker-compose config  # 설정 검증
docker-compose logs > debug-logs.txt
```

---

**⚡ 빠른 시작 요약**:
```bash
cp env.example .env
nano .env  # OPENAI_API_KEY 설정
docker-compose up -d --build
```

**🌐 Dashboard 접속**: http://localhost:3000

