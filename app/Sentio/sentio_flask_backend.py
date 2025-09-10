# -*- coding: utf-8 -*-
"""
Sentio HR 텍스트 감정 분석 Flask 백엔드 서비스
키워드 분석 + 퇴직 위험 신호 탐지 + 텍스트 생성 시스템
React 연동에 최적화된 REST API 서버
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import logging
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
import re
import traceback

# 로컬 모듈 import
from sentio_processor import SentioTextProcessor
from sentio_analyzer import SentioKeywordAnalyzer
from sentio_generator import SentioTextGenerator

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 전역 변수
text_processor = None
keyword_analyzer = None
text_generator = None

# 데이터 경로 설정
DATA_PATH = {
    'hr_data': '../../data/IBM_HR_personas_assigned.csv',
    'text_data': '../../data/IBM_HR_text.csv',
    'sample_texts': '../../sample_hr_texts.csv'
}

MODEL_PATH = 'app/Sentio/models'
os.makedirs(MODEL_PATH, exist_ok=True)

@dataclass
class SentioAnalysisResult:
    """Sentio 분석 결과 데이터 클래스"""
    employee_id: str
    text_type: str
    original_text: str
    keywords: List[str]
    sentiment_score: float
    attrition_risk_score: float
    risk_factors: List[str]
    analysis_timestamp: str

@dataclass
class SentioGenerationResult:
    """Sentio 텍스트 생성 결과 데이터 클래스"""
    employee_id: str
    persona_code: str
    persona_name: str
    text_type: str
    generated_text: str
    keywords_used: List[str]
    generation_timestamp: str

def initialize_system():
    """
    Sentio 시스템 초기화
    """
    global text_processor, keyword_analyzer, text_generator
    
    try:
        logger.info("Sentio 시스템 초기화 시작...")
        
        # 텍스트 프로세서 초기화
        text_processor = SentioTextProcessor()
        logger.info("✅ 텍스트 프로세서 초기화 완료")
        
        # 키워드 분석기 초기화
        if os.path.exists(DATA_PATH['sample_texts']):
            keyword_analyzer = SentioKeywordAnalyzer(DATA_PATH['sample_texts'])
            keyword_analyzer.load_data()
            logger.info("✅ 키워드 분석기 초기화 완료")
        else:
            logger.warning("⚠️ 텍스트 데이터 파일을 찾을 수 없습니다.")
        
        # 텍스트 생성기 초기화 (API 키는 환경변수에서 가져오기)
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key and os.path.exists(DATA_PATH['hr_data']):
            text_generator = SentioTextGenerator(api_key, DATA_PATH['hr_data'])
            logger.info("✅ 텍스트 생성기 초기화 완료")
        else:
            logger.warning("⚠️ OpenAI API 키 또는 HR 데이터 파일을 찾을 수 없습니다.")
        
        logger.info("🎉 Sentio 시스템 초기화 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 시스템 초기화 실패: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# ============================================================================
# API 엔드포인트
# ============================================================================

@app.route('/')
def home():
    """홈페이지"""
    return jsonify({
        "service": "Sentio HR Text Analysis API",
        "version": "1.0.0",
        "status": "running",
        "description": "HR 텍스트 감정 분석 및 퇴직 위험 신호 탐지 서비스",
        "endpoints": {
            "/analyze/text": "텍스트 분석 (키워드 추출 + 감정 분석)",
            "/analyze/keywords": "키워드 분석 (퇴직자 vs 재직자)",
            "/analyze/risk": "퇴직 위험 신호 분석",
            "/generate/text": "페르소나 기반 텍스트 생성",
            "/health": "서비스 상태 확인"
        }
    })

@app.route('/health')
def health_check():
    """서비스 상태 확인"""
    status = {
        "service": "Sentio",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "text_processor": text_processor is not None,
            "keyword_analyzer": keyword_analyzer is not None,
            "text_generator": text_generator is not None
        }
    }
    
    # 전체 상태 확인
    all_healthy = all(status["components"].values())
    status["status"] = "healthy" if all_healthy else "degraded"
    
    return jsonify(status)

@app.route('/analyze/text', methods=['POST'])
def analyze_text():
    """
    텍스트 분석 API
    입력: 텍스트, 직원ID (선택)
    출력: 키워드, 감정점수, 퇴직위험점수
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "텍스트가 필요합니다."}), 400
        
        text = data['text']
        employee_id = data.get('employee_id', 'unknown')
        text_type = data.get('text_type', 'general')
        
        if not text_processor:
            return jsonify({"error": "텍스트 프로세서가 초기화되지 않았습니다."}), 500
        
        # 텍스트 분석 수행
        analysis_result = text_processor.analyze_text(
            text=text,
            employee_id=employee_id,
            text_type=text_type
        )
        
        # 결과 반환
        result = SentioAnalysisResult(
            employee_id=employee_id,
            text_type=text_type,
            original_text=text,
            keywords=analysis_result['keywords'],
            sentiment_score=analysis_result['sentiment_score'],
            attrition_risk_score=analysis_result['attrition_risk_score'],
            risk_factors=analysis_result['risk_factors'],
            analysis_timestamp=datetime.now().isoformat()
        )
        
        return jsonify(asdict(result))
        
    except Exception as e:
        logger.error(f"텍스트 분석 오류: {str(e)}")
        return jsonify({"error": f"분석 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/analyze/keywords', methods=['POST'])
def analyze_keywords():
    """
    키워드 분석 API
    입력: 분석 옵션
    출력: 퇴직자 vs 재직자 차별적 키워드
    """
    try:
        data = request.get_json() or {}
        min_frequency = data.get('min_frequency', 5)
        text_columns = data.get('text_columns', None)
        
        if not keyword_analyzer:
            return jsonify({"error": "키워드 분석기가 초기화되지 않았습니다."}), 500
        
        # 키워드 분석 수행
        results = keyword_analyzer.analyze_text_columns(text_columns)
        if not results:
            return jsonify({"error": "분석할 텍스트 데이터가 없습니다."}), 400
        
        # 차별적 키워드 찾기
        distinctive_keywords = keyword_analyzer.find_distinctive_keywords(
            results, min_frequency=min_frequency
        )
        
        # 결과 정리
        analysis_summary = {
            "analysis_timestamp": datetime.now().isoformat(),
            "min_frequency": min_frequency,
            "columns_analyzed": list(results.keys()),
            "distinctive_keywords": distinctive_keywords,
            "summary": {
                col: {
                    "resigned_total_keywords": data['resigned_total'],
                    "stayed_total_keywords": data['stayed_total'],
                    "resigned_unique_count": len(distinctive_keywords[col]['resigned_unique']),
                    "stayed_unique_count": len(distinctive_keywords[col]['stayed_unique'])
                }
                for col, data in results.items()
            }
        }
        
        return jsonify(analysis_summary)
        
    except Exception as e:
        logger.error(f"키워드 분석 오류: {str(e)}")
        return jsonify({"error": f"분석 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/analyze/risk', methods=['POST'])
def analyze_attrition_risk():
    """
    퇴직 위험 신호 분석 API
    입력: 직원 텍스트 데이터
    출력: 위험 점수 및 주요 신호
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({"error": "분석할 텍스트 목록이 필요합니다."}), 400
        
        texts = data['texts']  # List of {employee_id, text, text_type}
        
        if not text_processor:
            return jsonify({"error": "텍스트 프로세서가 초기화되지 않았습니다."}), 500
        
        risk_analysis_results = []
        
        for text_data in texts:
            employee_id = text_data.get('employee_id', 'unknown')
            text = text_data.get('text', '')
            text_type = text_data.get('text_type', 'general')
            
            if not text:
                continue
            
            # 위험 신호 분석
            risk_result = text_processor.analyze_attrition_risk(
                text=text,
                employee_id=employee_id
            )
            
            risk_analysis_results.append({
                "employee_id": employee_id,
                "text_type": text_type,
                "risk_score": risk_result['risk_score'],
                "risk_level": risk_result['risk_level'],
                "risk_factors": risk_result['risk_factors'],
                "keywords_detected": risk_result['keywords_detected']
            })
        
        # 전체 요약 통계
        risk_scores = [r['risk_score'] for r in risk_analysis_results]
        summary_stats = {
            "total_analyzed": len(risk_analysis_results),
            "average_risk_score": np.mean(risk_scores) if risk_scores else 0,
            "high_risk_count": len([r for r in risk_analysis_results if r['risk_level'] == 'high']),
            "medium_risk_count": len([r for r in risk_analysis_results if r['risk_level'] == 'medium']),
            "low_risk_count": len([r for r in risk_analysis_results if r['risk_level'] == 'low'])
        }
        
        return jsonify({
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": summary_stats,
            "individual_results": risk_analysis_results
        })
        
    except Exception as e:
        logger.error(f"위험 분석 오류: {str(e)}")
        return jsonify({"error": f"분석 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/generate/text', methods=['POST'])
def generate_text():
    """
    페르소나 기반 텍스트 생성 API
    입력: 직원 정보, 텍스트 타입
    출력: 생성된 텍스트
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "요청 데이터가 필요합니다."}), 400
        
        if not text_generator:
            return jsonify({"error": "텍스트 생성기가 초기화되지 않았습니다. OpenAI API 키를 확인해주세요."}), 500
        
        # 단일 직원 텍스트 생성
        if 'employee_data' in data:
            employee_data = data['employee_data']
            text_type = data.get('text_type', 'SELF_REVIEW')
            
            generated_text = text_generator.generate_text_for_employee(
                employee_data=employee_data,
                text_type=text_type
            )
            
            result = SentioGenerationResult(
                employee_id=employee_data.get('EmployeeNumber', 'unknown'),
                persona_code=employee_data.get('softmax_Persona_Code', 'unknown'),
                persona_name=employee_data.get('softmax_Persona', 'unknown'),
                text_type=text_type,
                generated_text=generated_text,
                keywords_used=text_generator.get_attrition_keywords_for_persona(
                    employee_data.get('softmax_Persona_Code', 'N01')
                )[:5],  # 상위 5개 키워드만
                generation_timestamp=datetime.now().isoformat()
            )
            
            return jsonify(asdict(result))
        
        # 배치 텍스트 생성
        elif 'batch_size' in data:
            batch_size = data['batch_size']
            text_types = data.get('text_types', ['SELF_REVIEW'])
            
            text_generator.generate_all_texts(
                text_types=text_types,
                sample_size=batch_size
            )
            
            return jsonify({
                "message": f"{batch_size}명에 대한 텍스트 생성이 완료되었습니다.",
                "generated_count": len(text_generator.generated_texts),
                "text_types": text_types,
                "generation_timestamp": datetime.now().isoformat()
            })
        
        else:
            return jsonify({"error": "employee_data 또는 batch_size가 필요합니다."}), 400
        
    except Exception as e:
        logger.error(f"텍스트 생성 오류: {str(e)}")
        return jsonify({"error": f"생성 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/data/personas', methods=['GET'])
def get_personas():
    """페르소나 정보 조회 API"""
    try:
        if not text_generator:
            return jsonify({"error": "텍스트 생성기가 초기화되지 않았습니다."}), 500
        
        # 페르소나별 통계 정보
        df = text_generator.df
        persona_stats = df.groupby(['softmax_Persona_Code', 'softmax_Persona']).agg({
            'EmployeeNumber': 'count',
            'Attrition': lambda x: (x == 'Yes').sum()
        }).reset_index()
        
        persona_stats.columns = ['persona_code', 'persona_name', 'total_count', 'attrition_count']
        persona_stats['attrition_rate'] = persona_stats['attrition_count'] / persona_stats['total_count']
        
        return jsonify({
            "personas": persona_stats.to_dict('records'),
            "total_employees": len(df),
            "total_attrition": (df['Attrition'] == 'Yes').sum()
        })
        
    except Exception as e:
        logger.error(f"페르소나 조회 오류: {str(e)}")
        return jsonify({"error": f"조회 중 오류가 발생했습니다: {str(e)}"}), 500

# ============================================================================
# 오류 처리
# ============================================================================

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """HTTP 예외 처리"""
    return jsonify({
        "error": e.description,
        "status_code": e.code
    }), e.code

@app.errorhandler(Exception)
def handle_general_exception(e):
    """일반 예외 처리"""
    logger.error(f"예상치 못한 오류: {str(e)}")
    logger.error(traceback.format_exc())
    
    return jsonify({
        "error": "서버 내부 오류가 발생했습니다.",
        "message": str(e)
    }), 500

# ============================================================================
# 앱 실행
# ============================================================================

if __name__ == '__main__':
    print("🚀 Sentio HR Text Analysis API 서버 시작...")
    print("=" * 60)
    
    # 시스템 초기화
    if initialize_system():
        print("✅ 시스템 초기화 완료")
        print("🌐 서버 주소: http://localhost:5003")
        print("📚 API 문서: http://localhost:5003/")
        print("=" * 60)
        
        app.run(
            host='0.0.0.0',
            port=5003,
            debug=True,
            threaded=True
        )
    else:
        print("❌ 시스템 초기화 실패")
        print("서버를 시작할 수 없습니다.")
