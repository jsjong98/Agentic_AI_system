# Structura Agent - 정형 데이터 ML 분석
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
      libopenblas-dev \
      curl \
 && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && dpkg-reconfigure -f noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Structura 전용 의존성
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.1.1 \
    scikit-learn==1.3.0 \
    xgboost>=1.6.0 \
    shap>=0.42.0 \
    lime>=0.2.0.1 \
    imbalanced-learn>=0.10.0 \
    flask>=2.3.0 \
    flask-cors>=4.0.0 \
    python-dotenv>=1.0.0 \
    requests>=2.31.0 \
    joblib>=1.3.0 \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    optuna>=3.0.0

# Structura 코드 복사
COPY Structura/ ./
COPY requirements-docker.txt ./

# 필요한 디렉토리 생성
RUN mkdir -p models cache logs temp

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV STRUCTURA_PORT=5001

# 포트 노출
EXPOSE 5001

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# 서버 실행
CMD ["python", "run_structura_server.py"]
