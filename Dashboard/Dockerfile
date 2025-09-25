# Multi-stage build for React Dashboard
FROM node:18-alpine AS build

WORKDIR /app

# 의존성 설치
COPY package*.json ./
RUN npm ci --only=production

# 소스 코드 복사 및 빌드
COPY . .

# 환경 변수 설정 (빌드 시점)
ENV REACT_APP_API_BASE_URL=http://localhost:5006
ENV REACT_APP_SUPERVISOR_URL=http://localhost:5006
ENV REACT_APP_STRUCTURA_URL=http://localhost:5001
ENV REACT_APP_COGNITA_URL=http://localhost:5002
ENV REACT_APP_CHRONOS_URL=http://localhost:5003
ENV REACT_APP_SENTIO_URL=http://localhost:5004
ENV REACT_APP_AGORA_URL=http://localhost:5005
ENV REACT_APP_INTEGRATION_URL=http://localhost:5007

RUN npm run build

# 프로덕션 환경을 위한 Nginx 설정
FROM nginx:alpine

# Nginx 설정 파일 생성
RUN echo 'server {\n\
    listen 80;\n\
    server_name localhost;\n\
    root /usr/share/nginx/html;\n\
    index index.html index.htm;\n\
\n\
    # React Router를 위한 설정 (SPA)\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
\n\
    # API 프록시 설정 (백엔드 서버로 전달)\n\
    location /api/ {\n\
        proxy_pass http://agentic-backend:5006;\n\
        proxy_http_version 1.1;\n\
        proxy_set_header Upgrade $http_upgrade;\n\
        proxy_set_header Connection "upgrade";\n\
        proxy_set_header Host $host;\n\
        proxy_set_header X-Real-IP $remote_addr;\n\
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n\
        proxy_set_header X-Forwarded-Proto $scheme;\n\
        proxy_cache_bypass $http_upgrade;\n\
        proxy_read_timeout 300s;\n\
        proxy_connect_timeout 75s;\n\
    }\n\
\n\
    # 정적 파일 캐싱\n\
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {\n\
        expires 1y;\n\
        add_header Cache-Control "public, immutable";\n\
    }\n\
\n\
    # 보안 헤더\n\
    add_header X-Frame-Options "SAMEORIGIN" always;\n\
    add_header X-XSS-Protection "1; mode=block" always;\n\
    add_header X-Content-Type-Options "nosniff" always;\n\
    add_header Referrer-Policy "no-referrer-when-downgrade" always;\n\
    add_header Content-Security-Policy "default-src '\''self'\'' http: https: data: blob: '\''unsafe-inline'\''" always;\n\
\n\
    # Gzip 압축\n\
    gzip on;\n\
    gzip_vary on;\n\
    gzip_min_length 1024;\n\
    gzip_proxied expired no-cache no-store private must-revalidate auth;\n\
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss application/javascript;\n\
}' > /etc/nginx/conf.d/default.conf

# 빌드된 React 앱 복사
COPY --from=build /app/build /usr/share/nginx/html

# 포트 노출
EXPOSE 80

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
