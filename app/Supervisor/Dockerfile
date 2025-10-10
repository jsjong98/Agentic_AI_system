# Supervisor Agent - LangGraph 워크플로우 관리 (메인 API)
FROM python:3.9-slim

# 비대화형 설치 + 타임존 설정
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

WORKDIR /app

# 시스템 의존성
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tzdata \
      build-essential \
      curl \
 && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && dpkg-reconfigure -f noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# requirements 파일 먼저 복사 (캐시 활용)
COPY Supervisor/requirements.txt* ./
COPY requirements-docker.txt ./

# Supervisor 전용 의존성 (LangGraph 포함)
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.1.1 \
    openai>=1.0.0 \
    langchain>=0.1.0 \
    langchain-openai>=0.0.5 \
    langchain-core>=0.1.0 \
    langgraph>=0.0.26 \
    flask>=2.3.0 \
    flask-cors>=4.0.0 \
    python-dotenv>=1.0.0 \
    requests>=2.31.0 \
    aiohttp>=3.8.0 \
    pydantic>=2.0.0 \
    matplotlib>=3.5.0

# Supervisor 코드 복사 (코드 변경 시에만 재빌드)
COPY Supervisor/ ./

# 필요한 디렉토리 생성
RUN mkdir -p logs cache temp uploads

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV SUPERVISOR_PORT=5006

# 워커 에이전트 URL 설정 (Docker 네트워크 내부 통신)
ENV STRUCTURA_URL=http://structura:5001
ENV COGNITA_URL=http://cognita:5002
ENV CHRONOS_URL=http://chronos:5003
ENV SENTIO_URL=http://sentio:5004
ENV AGORA_URL=http://agora:5005
ENV INTEGRATION_URL=http://integration:5007

# 포트 노출
EXPOSE 5006

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5006/health || exit 1

# 서버 실행
CMD ["python", "run_supervisor_server.py"]
