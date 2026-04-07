#!/bin/sh

echo "=== Cognita startup ==="

# 1. Neo4j 초기화 (데이터가 없을 때만 XML 로드; 연결 실패 시 경고 후 계속)
python /app/init_neo4j.py || echo "Neo4j init skipped, starting Flask anyway..."

# 2. Flask 서버 시작
exec python /app/run_cognita_server.py
