# -*- coding: utf-8 -*-
"""
업데이트된 수정 사항 테스트 스크립트
- Chronos 300MB 파일 업로드 제한 테스트
- Agora와 Sentio OpenAI API 통일 테스트
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent))

def test_chronos_file_size_limit():
    """Chronos 파일 크기 제한 테스트"""
    print("=" * 60)
    print("🧪 Chronos 파일 크기 제한 테스트 (300MB)")
    print("=" * 60)
    
    try:
        from Chronos.chronos_flask_backend import app
        
        # Flask 앱 설정 확인
        print("1. Flask 앱 설정 확인...")
        max_content_length = app.config.get('MAX_CONTENT_LENGTH')
        
        if max_content_length:
            max_mb = max_content_length / (1024 * 1024)
            print(f"  ✅ 최대 파일 크기: {max_mb:.0f}MB")
            
            if max_mb == 300:
                print("  ✅ 300MB 제한 설정 확인")
                return True
            else:
                print(f"  ❌ 예상과 다른 크기 제한: {max_mb}MB (예상: 300MB)")
                return False
        else:
            print("  ❌ MAX_CONTENT_LENGTH 설정이 없습니다")
            return False
            
    except ImportError as e:
        print(f"❌ Chronos import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ Chronos 테스트 실패: {e}")
        return False

def test_agora_openai_api():
    """Agora OpenAI API 수정 사항 테스트"""
    print("\n" + "=" * 60)
    print("🧪 Agora OpenAI API 수정 사항 테스트")
    print("=" * 60)
    
    try:
        from Agora.agora_llm_generator import AgoraLLMGenerator
        
        # 1. 모델명 확인
        print("1. OpenAI 모델명 확인...")
        llm_generator = AgoraLLMGenerator("test_key")
        
        if llm_generator.model == "gpt-4o-mini":
            print(f"  ✅ 모델명 수정 확인: {llm_generator.model}")
        else:
            print(f"  ❌ 잘못된 모델명: {llm_generator.model} (예상: gpt-4o-mini)")
            return False
        
        # 2. API 사용법 확인 (코드 검사)
        print("2. API 사용법 확인...")
        
        # generate_market_interpretation 메서드의 소스 코드 확인
        import inspect
        source = inspect.getsource(llm_generator.generate_market_interpretation)
        
        if "chat.completions.create" in source:
            print("  ✅ chat.completions.create API 사용 확인")
        else:
            print("  ❌ 잘못된 API 사용법")
            return False
        
        if "max_tokens" in source and "temperature" in source:
            print("  ✅ 표준 파라미터 사용 확인")
        else:
            print("  ❌ 표준 파라미터 누락")
            return False
        
        # 3. 규칙 기반 해석 테스트
        print("3. 규칙 기반 해석 테스트...")
        sample_analysis = {
            'employee_id': 'TEST001',
            'job_role': 'Sales Executive',
            'market_pressure_index': 0.5,
            'compensation_gap': 0.2,
            'job_postings_count': 10,
            'risk_level': 'MEDIUM',
            'market_data': {'market_trend': 'STABLE'}
        }
        
        interpretation = llm_generator._generate_rule_based_interpretation(sample_analysis)
        if interpretation and len(interpretation) > 100:
            print(f"  ✅ 규칙 기반 해석 생성 성공 ({len(interpretation)} 문자)")
        else:
            print("  ❌ 규칙 기반 해석 생성 실패")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Agora import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ Agora 테스트 실패: {e}")
        return False

def test_sentio_openai_api():
    """Sentio OpenAI API 수정 사항 테스트"""
    print("\n" + "=" * 60)
    print("🧪 Sentio OpenAI API 수정 사항 테스트")
    print("=" * 60)
    
    try:
        from Sentio.sentio_generator import SentioTextGenerator
        
        # 1. 초기화 테스트
        print("1. Sentio 텍스트 생성기 초기화...")
        generator = SentioTextGenerator("test_key")
        print("  ✅ 초기화 성공")
        
        # 2. API 사용법 확인 (코드 검사)
        print("2. API 사용법 확인...")
        
        import inspect
        source = inspect.getsource(generator.generate_text_with_keywords)
        
        if "chat.completions.create" in source:
            print("  ✅ chat.completions.create API 사용 확인")
        else:
            print("  ❌ 잘못된 API 사용법")
            return False
        
        if "gpt-4o-mini" in source:
            print("  ✅ 모델명 수정 확인: gpt-4o-mini")
        else:
            print("  ❌ 모델명 수정 실패")
            return False
        
        if "max_tokens" in source and "temperature" in source:
            print("  ✅ 표준 파라미터 사용 확인")
        else:
            print("  ❌ 표준 파라미터 누락")
            return False
        
        # 3. 키워드 초기화 테스트
        print("3. 퇴직 키워드 초기화 확인...")
        if generator.attrition_keywords and len(generator.attrition_keywords) > 0:
            keyword_categories = list(generator.attrition_keywords.keys())
            print(f"  ✅ 키워드 카테고리 로드: {len(keyword_categories)}개")
            print(f"    - {', '.join(keyword_categories[:3])}...")
        else:
            print("  ❌ 키워드 초기화 실패")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Sentio import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ Sentio 테스트 실패: {e}")
        return False

def test_master_server_file_limit():
    """마스터 서버 파일 크기 제한 테스트"""
    print("\n" + "=" * 60)
    print("🧪 마스터 서버 파일 크기 제한 테스트")
    print("=" * 60)
    
    try:
        # create_app 함수를 직접 호출하여 앱 생성
        from agentic_master_server import create_app
        
        print("1. 마스터 서버 앱 생성...")
        app = create_app()
        
        # Flask 앱 설정 확인
        print("2. 파일 크기 제한 설정 확인...")
        max_content_length = app.config.get('MAX_CONTENT_LENGTH')
        
        if max_content_length:
            max_mb = max_content_length / (1024 * 1024)
            print(f"  ✅ 최대 파일 크기: {max_mb:.0f}MB")
            
            if max_mb == 300:
                print("  ✅ 300MB 제한 설정 확인")
                return True
            else:
                print(f"  ❌ 예상과 다른 크기 제한: {max_mb}MB (예상: 300MB)")
                return False
        else:
            print("  ❌ MAX_CONTENT_LENGTH 설정이 없습니다")
            return False
            
    except ImportError as e:
        print(f"❌ 마스터 서버 import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 마스터 서버 테스트 실패: {e}")
        return False

def test_api_consistency():
    """Agora와 Sentio API 일관성 테스트"""
    print("\n" + "=" * 60)
    print("🧪 Agora와 Sentio API 일관성 테스트")
    print("=" * 60)
    
    try:
        from Agora.agora_llm_generator import AgoraLLMGenerator
        from Sentio.sentio_generator import SentioTextGenerator
        
        # 1. 모델명 일관성 확인
        print("1. 모델명 일관성 확인...")
        agora_generator = AgoraLLMGenerator("test_key")
        sentio_generator = SentioTextGenerator("test_key")
        
        agora_model = agora_generator.model
        
        # Sentio의 모델명은 코드에서 확인
        import inspect
        sentio_source = inspect.getsource(sentio_generator.generate_text_with_keywords)
        
        if agora_model == "gpt-4o-mini" and "gpt-4o-mini" in sentio_source:
            print(f"  ✅ 모델명 일관성 확인: {agora_model}")
        else:
            print(f"  ❌ 모델명 불일치")
            print(f"    Agora: {agora_model}")
            print(f"    Sentio: {'gpt-4o-mini' if 'gpt-4o-mini' in sentio_source else '확인 불가'}")
            return False
        
        # 2. API 사용법 일관성 확인
        print("2. API 사용법 일관성 확인...")
        agora_source = inspect.getsource(agora_generator.generate_market_interpretation)
        
        agora_uses_chat = "chat.completions.create" in agora_source
        sentio_uses_chat = "chat.completions.create" in sentio_source
        
        if agora_uses_chat and sentio_uses_chat:
            print("  ✅ 두 에이전트 모두 chat.completions.create API 사용")
        else:
            print("  ❌ API 사용법 불일치")
            return False
        
        # 3. 파라미터 일관성 확인
        print("3. 파라미터 일관성 확인...")
        agora_has_params = "max_tokens" in agora_source and "temperature" in agora_source
        sentio_has_params = "max_tokens" in sentio_source and "temperature" in sentio_source
        
        if agora_has_params and sentio_has_params:
            print("  ✅ 두 에이전트 모두 표준 파라미터 사용")
        else:
            print("  ❌ 파라미터 사용 불일치")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ API 일관성 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 업데이트된 수정 사항 테스트 시작")
    print(f"📅 테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    
    # 개별 테스트 실행
    test_results['chronos_file_limit'] = test_chronos_file_size_limit()
    test_results['agora_openai'] = test_agora_openai_api()
    test_results['sentio_openai'] = test_sentio_openai_api()
    test_results['master_file_limit'] = test_master_server_file_limit()
    test_results['api_consistency'] = test_api_consistency()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
    
    print(f"\n📈 전체 결과: {passed}/{total} 테스트 통과 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 모든 업데이트 테스트 통과!")
        print("\n📋 적용된 수정 사항:")
        print("  • Chronos: 파일 업로드 크기 제한 300MB로 증가")
        print("  • Agora: OpenAI 모델명을 gpt-4o-mini로 변경")
        print("  • Agora: chat.completions.create API 사용으로 변경")
        print("  • Sentio: OpenAI 모델명을 gpt-4o-mini로 변경")
        print("  • Sentio: chat.completions.create API 사용으로 변경")
        print("  • Master: 파일 업로드 크기 제한 300MB로 설정")
        print("  • API 일관성: Agora와 Sentio 통일")
    else:
        print("⚠️ 일부 테스트 실패. 추가 확인이 필요합니다.")
    
    # 결과를 JSON 파일로 저장
    result_file = f"updated_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    test_summary = {
        'timestamp': datetime.now().isoformat(),
        'results': test_results,
        'summary': {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': passed / total * 100
        },
        'modifications': {
            'chronos_file_limit': '300MB',
            'agora_model': 'gpt-4o-mini',
            'sentio_model': 'gpt-4o-mini',
            'api_method': 'chat.completions.create',
            'master_file_limit': '300MB'
        }
    }
    
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(test_summary, f, indent=2, ensure_ascii=False)
        print(f"📄 테스트 결과가 {result_file}에 저장되었습니다.")
    except Exception as e:
        print(f"⚠️ 결과 파일 저장 실패: {e}")

if __name__ == "__main__":
    main()
