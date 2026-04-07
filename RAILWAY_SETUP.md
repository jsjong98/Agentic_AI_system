# Railway 배포 가이드 - Agentic AI System

## 아키텍처 개요

Railway에서 총 9개 서비스로 구성됩니다:

| 서비스 | Railway 서비스명 | Root Directory | 포트 |
|--------|-----------------|---------------|------|
| Structura | `structura` | `app/Structura` | 5001 |
| Cognita | `cognita` | `app/Cognita` | 5002 |
| Chronos | `chronos` | `app/Chronos` | 5003 |
| Sentio | `sentio` | `app/Sentio` | 5004 |
| Agora | `agora` | `app/Agora` | 5005 |
| Supervisor | `supervisor` | `app/Supervisor` | 5006 |
| Integration | `integration` | `app/Integration` | 5007 |
| Dashboard | `dashboard` | `Dashboard` | 80 |
| Neo4j | Railway Plugin | — | 7687 |

---

## Step 1: GitHub 정리 (처음 1회)

```bash
# 런타임 데이터 git 추적 제거 (12,000+ 파일 → 오래 걸림)
git rm -r --cached app/results/ app/uploads/ \
  app/Agora/data/market_cache.json data/market_cache.json \
  optimized_models/best_hybrid_model.pth

git add .gitignore
git commit -m "Remove runtime data from tracking, add Railway config"
git push origin main
```

---

## Step 2: Railway 프로젝트 생성

1. [railway.app](https://railway.app) 로그인 후 **New Project** 클릭
2. **Deploy from GitHub repo** 선택 → `jsjong98/Agentic_AI_system` 연결

---

## Step 3: Neo4j 서비스 추가

Railway 프로젝트에서:
1. **Add Service → Database → Neo4j** 클릭
2. 생성 후 `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` 변수 확인

또는 기존 Neo4j Aura 사용 시 해당 연결정보를 아래 Step 4에서 입력.

---

## Step 4: 백엔드 서비스 생성 (7개)

각 서비스를 아래 순서로 생성합니다.

### 공통 설정 방법
Railway 서비스 생성 → **Settings → Source → Root Directory** 설정 → **Dockerfile Path** = `Dockerfile.railway`

### Structura
- Root Directory: `app/Structura`
- Environment Variables:
  ```
  PORT=5001
  ```

### Cognita
- Root Directory: `app/Cognita`
- Environment Variables:
  ```
  PORT=5002
  NEO4J_URI=bolt://<neo4j-host>:7687
  NEO4J_USERNAME=neo4j
  NEO4J_PASSWORD=<your-neo4j-password>
  ```

### Chronos
- Root Directory: `app/Chronos`
- Environment Variables:
  ```
  PORT=5003
  ```

### Sentio
- Root Directory: `app/Sentio`
- Environment Variables:
  ```
  PORT=5004
  OPENAI_API_KEY=<your-key>
  ```

### Agora
- Root Directory: `app/Agora`
- Environment Variables:
  ```
  PORT=5005
  OPENAI_API_KEY=<your-key>
  ```

### Supervisor
- Root Directory: `app/Supervisor`
- Environment Variables:
  ```
  PORT=5006
  OPENAI_API_KEY=<your-key>
  STRUCTURA_URL=http://structura.railway.internal:5001
  COGNITA_URL=http://cognita.railway.internal:5002
  CHRONOS_URL=http://chronos.railway.internal:5003
  SENTIO_URL=http://sentio.railway.internal:5004
  AGORA_URL=http://agora.railway.internal:5005
  INTEGRATION_URL=http://integration.railway.internal:5007
  ```

### Integration
- Root Directory: `app/Integration`
- Environment Variables:
  ```
  PORT=5007
  OPENAI_API_KEY=<your-key>
  ```

---

## Step 5: Dashboard 서비스 생성

- Root Directory: `Dashboard`
- Environment Variables (Supervisor/Integration 배포 후 URL 확인하여 입력):
  ```
  REACT_APP_SUPERVISOR_URL=https://supervisor-production-e7c6.up.railway.app
  REACT_APP_STRUCTURA_URL=https://structura-production.up.railway.app
  REACT_APP_COGNITA_URL=https://cognita-production.up.railway.app
  REACT_APP_CHRONOS_URL=https://chronos-production-7ccc.up.railway.app
  REACT_APP_SENTIO_URL=https://sentio-production-937f.up.railway.app
  REACT_APP_AGORA_URL=https://agora-production-6df7.up.railway.app
  REACT_APP_INTEGRATION_URL=https://integration-production-1d5e.up.railway.app
  ```

> **참고**: Dashboard는 런타임 env 주입 방식을 사용하므로 변수 변경 시 재배포 없이 재시작만으로 반영됩니다.

---

## Railway 내부 네트워킹

백엔드 서비스 간 통신은 Railway private network를 사용합니다:
- 형식: `http://<서비스명>.railway.internal:<PORT>`
- Supervisor → 다른 에이전트 호출 시 이 URL 사용
- 외부에서는 공개 URL (`*.up.railway.app`) 사용

---

## Volume 설정 (선택)

Railway에서 영속성이 필요한 데이터의 경우:
- **Structura**: `/app/models` (훈련된 ML 모델)
- **Chronos**: `/app/models` (딥러닝 모델)
- **Neo4j**: 자동 관리됨

Railway 서비스 → **Settings → Volumes** 에서 마운트 경로 추가.

---

## 배포 후 확인

```bash
# 각 서비스 헬스체크
curl https://structura-production.up.railway.app/health
curl https://cognita-production.up.railway.app/health
curl https://chronos-production-7ccc.up.railway.app/api/status
curl https://sentio-production-937f.up.railway.app/health
curl https://agora-production-6df7.up.railway.app/health
curl https://supervisor-production-e7c6.up.railway.app/health
curl https://integration-production-1d5e.up.railway.app/health
curl https://retain-sentinel.up.railway.app/health
```

---

## 주의사항

- **Chronos**: Railway는 GPU 미지원 → CPU 전용 PyTorch 사용 (학습 속도 느림)
- **Agora**: Chromium 기반 web scraping은 Railway 메모리 제한에 주의
- **Neo4j**: Railway Neo4j 플러그인은 Community Edition (무료 플랜)
- **API 키**: `OPENAI_API_KEY`를 절대 코드에 하드코딩하지 말 것
