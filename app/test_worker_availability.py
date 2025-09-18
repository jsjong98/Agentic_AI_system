#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
워커 에이전트 가용성 테스트
"""

import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

print('=== 워커 에이전트 가용성 테스트 ===')

# Structura 테스트
try:
    from Structura.structura_flask_backend import StructuraHRPredictor
    print('✅ Structura: 사용 가능')
    STRUCTURA_AVAILABLE = True
except Exception as e:
    print(f'❌ Structura: {e}')
    STRUCTURA_AVAILABLE = False

# Cognita 테스트
try:
    from Cognita.cognita_flask_backend import CognitaRiskAnalyzer
    print('✅ Cognita: 사용 가능')
    COGNITA_AVAILABLE = True
except Exception as e:
    print(f'❌ Cognita: {e}')
    COGNITA_AVAILABLE = False

# Chronos 테스트
try:
    from Chronos.chronos_processor_fixed import ChronosDataProcessor
    print('✅ Chronos: 사용 가능')
    CHRONOS_AVAILABLE = True
except Exception as e:
    print(f'❌ Chronos: {e}')
    CHRONOS_AVAILABLE = False

# Sentio 테스트
try:
    from Sentio.sentio_processor import SentioTextProcessor
    print('✅ Sentio: 사용 가능')
    SENTIO_AVAILABLE = True
except Exception as e:
    print(f'❌ Sentio: {e}')
    SENTIO_AVAILABLE = False

# Agora 테스트
try:
    from Agora.agora_processor import AgoraMarketProcessor
    print('✅ Agora: 사용 가능')
    AGORA_AVAILABLE = True
except Exception as e:
    print(f'❌ Agora: {e}')
    AGORA_AVAILABLE = False

print('\n=== 요약 ===')
available_count = sum([STRUCTURA_AVAILABLE, COGNITA_AVAILABLE, CHRONOS_AVAILABLE, SENTIO_AVAILABLE, AGORA_AVAILABLE])
print(f'사용 가능한 워커: {available_count}/5개')

if available_count == 0:
    print('⚠️ 모든 워커가 사용 불가능합니다. 의존성 문제가 있을 수 있습니다.')
elif available_count < 5:
    print('⚠️ 일부 워커가 사용 불가능합니다.')
else:
    print('✅ 모든 워커가 정상적으로 사용 가능합니다.')
