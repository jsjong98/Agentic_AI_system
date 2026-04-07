#!/bin/sh
set -e

# /app/results에 department_index.json이 없으면 seed 실행
if [ ! -f /app/results/Sales/department_index.json ]; then
    echo "📦 결과 데이터가 없습니다. seed_results_generator 실행..."
    python /app/seed_results_generator.py
else
    echo "✅ 결과 데이터가 이미 존재합니다. seed 건너뜀."
fi

exec python run_integration_server.py
