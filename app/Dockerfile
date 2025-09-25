# GPU 지원 PyTorch 베이스 이미지
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 비대화형 설치 + 타임존 설정
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

WORKDIR /app

# 시스템 의존성 (tzdata + distutils + dev 헤더)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tzdata \
      build-essential \
      libopenblas-dev \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev \
      python3-distutils \
      python3-dev \
      curl \
      wget \
      chromium-browser \
      chromium-chromedriver \
 && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && dpkg-reconfigure -f noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*

# Chrome/Chromium 환경 변수 설정 (Selenium용)
ENV CHROME_BIN=/usr/bin/chromium-browser
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# 1) pip/setuptools/wheel 최신화
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 2) 주요 과학 스택을 conda로 설치 (C-API 버전 충돌 방지)
RUN conda install -y \
      numpy=1.24.3 \
      pandas=2.1.1 \
      scikit-learn=1.3.0 \
      matplotlib=3.7.2 \
      seaborn=0.12.2 \
      python-dateutil=2.8.2 \
      pytz=2023.3 \
 && conda clean -afy

# 3) Docker 전용 requirements로 나머지 패키지 설치
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# 애플리케이션 코드 복사
COPY . .

# 로그·모델·업로드·결과 폴더 미리 생성 (모든 에이전트용)
RUN mkdir -p \
    logs \
    models \
    uploads \
    cache \
    results \
    temp \
    data \
    static/plots \
    static/reports \
    static/ma_plots \
    static/attention \
    predictions \
    holidays \
    Structura/models \
    Cognita/cache \
    Chronos/models \
    Sentio/cache \
    Agora/data \
    Supervisor/logs \
    Integration/results

# 모든 에이전트 포트 노출
EXPOSE 5001 5002 5003 5004 5005 5006 5007

# 통합 서비스 시작 스크립트 생성
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "🚀 Agentic AI System - Docker 컨테이너 시작"\n\
echo "==============================================="\n\
\n\
# 환경 변수 설정\n\
export PYTHONIOENCODING=utf-8\n\
export PYTHONUNBUFFERED=1\n\
\n\
# 각 에이전트를 백그라운드로 시작\n\
echo "🤖 Starting Structura Agent (Port 5001)..."\n\
cd /app/Structura && python run_structura_server.py &\n\
\n\
echo "🧠 Starting Cognita Agent (Port 5002)..."\n\
cd /app/Cognita && python run_cognita_server.py &\n\
\n\
echo "⏰ Starting Chronos Agent (Port 5003)..."\n\
cd /app/Chronos && python run_chronos_server.py &\n\
\n\
echo "💭 Starting Sentio Agent (Port 5004)..."\n\
cd /app/Sentio && python run_sentio_server.py &\n\
\n\
echo "🏛️ Starting Agora Agent (Port 5005)..."\n\
cd /app/Agora && python run_agora_server.py &\n\
\n\
echo "👑 Starting Supervisor Agent (Port 5006)..."\n\
cd /app/Supervisor && python run_supervisor_server.py &\n\
\n\
echo "🔗 Starting Integration Agent (Port 5007)..."\n\
cd /app/Integration && python run_integration_server.py &\n\
\n\
echo "✅ All agents started successfully!"\n\
echo "==============================================="\n\
echo "📡 Available Services:"\n\
echo "  • Structura    → http://localhost:5001"\n\
echo "  • Cognita      → http://localhost:5002"\n\
echo "  • Chronos      → http://localhost:5003"\n\
echo "  • Sentio       → http://localhost:5004"\n\
echo "  • Agora        → http://localhost:5005"\n\
echo "  • Supervisor   → http://localhost:5006"\n\
echo "  • Integration  → http://localhost:5007"\n\
echo "==============================================="\n\
\n\
# 모든 백그라운드 프로세스가 종료될 때까지 대기\n\
wait\n\
' > /app/start_all_agents.sh && chmod +x /app/start_all_agents.sh

CMD ["/app/start_all_agents.sh"]
