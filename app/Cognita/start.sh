#!/bin/sh
set -e

echo "=== Cognita startup ==="

# 1. Neo4j 초기화 (데이터가 없을 때만 XML 로드)
python /app/init_neo4j.py

# 2. Flask 서버 시작
exec python /app/run_cognita_server.py
