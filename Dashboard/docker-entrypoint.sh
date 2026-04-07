#!/bin/sh
# Runtime environment variable injection for React SPA
# Replaces placeholder strings baked into the JS bundle at runtime

set -e

echo "Injecting runtime environment variables..."

# Replace placeholder values in built JS files with actual env var values
find /usr/share/nginx/html -name "*.js" | while read file; do
  sed -i \
    -e "s|PLACEHOLDER_SUPERVISOR_URL|${REACT_APP_SUPERVISOR_URL:-http://localhost:5006}|g" \
    -e "s|PLACEHOLDER_STRUCTURA_URL|${REACT_APP_STRUCTURA_URL:-http://localhost:5001}|g" \
    -e "s|PLACEHOLDER_COGNITA_URL|${REACT_APP_COGNITA_URL:-http://localhost:5002}|g" \
    -e "s|PLACEHOLDER_CHRONOS_URL|${REACT_APP_CHRONOS_URL:-http://localhost:5003}|g" \
    -e "s|PLACEHOLDER_SENTIO_URL|${REACT_APP_SENTIO_URL:-http://localhost:5004}|g" \
    -e "s|PLACEHOLDER_AGORA_URL|${REACT_APP_AGORA_URL:-http://localhost:5005}|g" \
    -e "s|PLACEHOLDER_INTEGRATION_URL|${REACT_APP_INTEGRATION_URL:-http://localhost:5007}|g" \
    "$file"
done

echo "Environment injection complete."
echo "  SUPERVISOR_URL  = ${REACT_APP_SUPERVISOR_URL}"
echo "  STRUCTURA_URL   = ${REACT_APP_STRUCTURA_URL}"
echo "  INTEGRATION_URL = ${REACT_APP_INTEGRATION_URL}"

exec "$@"
