# Agora Agent - 시장 분석 + LLM
FROM python:3.9-slim

# 비대화형 설치 + 타임존 설정
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

WORKDIR /app

# 시스템 의존성 (웹 스크래핑용 추가)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tzdata \
      build-essential \
      curl \
      wget \
      chromium-browser \
      chromium-chromedriver \
 && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && dpkg-reconfigure -f noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*

# Chrome/Chromium 환경 변수 설정
ENV CHROME_BIN=/usr/bin/chromium-browser
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Agora 전용 의존성
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.1.1 \
    yfinance>=0.2.0 \
    beautifulsoup4>=4.12.0 \
    lxml>=4.9.0 \
    selenium>=4.15.0 \
    openai>=1.0.0 \
    langchain>=0.1.0 \
    langchain-openai>=0.0.5 \
    flask>=2.3.0 \
    flask-cors>=4.0.0 \
    python-dotenv>=1.0.0 \
    requests>=2.31.0 \
    matplotlib>=3.5.0 \
    plotly>=5.15.0

# Agora 코드 복사
COPY Agora/ ./
COPY requirements-docker.txt ./

# 필요한 디렉토리 생성
RUN mkdir -p data cache logs temp

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AGORA_PORT=5005

# OpenAI API 설정 (환경 변수로 주입)
# ENV OPENAI_API_KEY=your-api-key-here

# 포트 노출
EXPOSE 5005

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5005/health || exit 1

# 서버 실행
CMD ["python", "run_agora_server.py"]
