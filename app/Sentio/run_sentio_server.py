#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentio HR Text Analysis 서버 실행 스크립트
"""

import os
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 환경변수 설정
os.environ.setdefault('FLASK_APP', 'sentio_flask_backend.py')
os.environ.setdefault('FLASK_ENV', 'development')

def main():
    """Sentio 서버 실행"""
    print("Sentio HR Text Analysis 서버 시작...")
    print("=" * 60)
    
    try:
        # Flask 앱 import 및 실행
        from sentio_flask_backend import app, initialize_system
        
        print("✅ Sentio 모듈 로드 완료")
        
        # 시스템 초기화 (중요!)
        print("🔧 Sentio 시스템 초기화 중...")
        if initialize_system():
            print("✅ 시스템 초기화 완료")
        else:
            print("❌ 시스템 초기화 실패")
            print("⚠️ 일부 기능이 제한될 수 있습니다.")
        
        print("🌐 서버 주소: http://localhost:5004")
        print("📚 API 문서: http://localhost:5004/")
        print("🔍 헬스체크: http://localhost:5004/health")
        print("=" * 60)
        
        # 서버 실행
        port = int(os.environ.get('PORT', 5004))
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            threaded=True
        )
        
    except ImportError as e:
        print(f"❌ 모듈 import 오류: {e}")
        print("💡 다음을 확인해주세요:")
        print("   1. requirements.txt 패키지 설치: pip install -r requirements.txt")
        print("   2. OpenAI API 키 환경변수 설정: export OPENAI_API_KEY=your_key")
        print("   3. 데이터 파일 경로 확인")
        sys.exit(1)
        
    except Exception as e:
        print(f"서버 시작 오류: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
