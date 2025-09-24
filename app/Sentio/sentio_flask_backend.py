# -*- coding: utf-8 -*-
"""
Sentio HR 텍스트 감정 분석 Flask 백엔드 서비스
키워드 분석 + 퇴직 위험 신호 탐지 + 텍스트 생성 시스템
React 연동에 최적화된 REST API 서버
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
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

# 데이터 경로 설정 - uploads 디렉토리에서 찾기
def get_sentio_data_paths(analysis_type='batch'):
    """uploads 디렉토리에서 Sentio 데이터 파일 찾기"""
    uploads_dir = f"../uploads/sentio/{analysis_type}"
    data_paths = {
        'hr_data': None,
        'text_data': None,
        'sample_texts': None
    }
    
    if os.path.exists(uploads_dir):
        files = [f for f in os.listdir(uploads_dir) if f.endswith('.csv')]
        if files:
            # 가장 최근 파일 사용 (타임스탬프 기준)
            files.sort(reverse=True)
            
            # 파일들을 분류
            for file in files:
                file_path = os.path.join(uploads_dir, file)
                if 'text' in file.lower():
                    if data_paths['text_data'] is None:
                        data_paths['text_data'] = file_path
                    if data_paths['sample_texts'] is None:
                        data_paths['sample_texts'] = file_path
                elif data_paths['hr_data'] is None:
                    data_paths['hr_data'] = file_path
    
    # batch에 파일이 없으면 post 디렉토리 확인
    if analysis_type == 'batch' and not any(data_paths.values()):
        post_paths = get_sentio_data_paths('post')
        for key, value in post_paths.items():
            if data_paths[key] is None:
                data_paths[key] = value
    
    # 기본값으로 fallback
    if data_paths['hr_data'] is None:
        data_paths['hr_data'] = '../../data/IBM_HR.csv'
    if data_paths['text_data'] is None:
        data_paths['text_data'] = '../../data/IBM_HR_text.csv'
    if data_paths['sample_texts'] is None:
        data_paths['sample_texts'] = '../../sample_hr_texts.csv'
        
    return data_paths

DATA_PATH = get_sentio_data_paths()

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
        
        # 키워드 분석기 초기화 (선택적)
        try:
            if os.path.exists(DATA_PATH['sample_texts']):
                keyword_analyzer = SentioKeywordAnalyzer(DATA_PATH['sample_texts'])
                keyword_analyzer.load_data()
                logger.info("✅ 키워드 분석기 초기화 완료")
            else:
                logger.info("⚠️ 텍스트 데이터 파일이 없습니다. 파일 업로드 후 분석 가능합니다.")
                keyword_analyzer = None
        except Exception as e:
            logger.warning(f"⚠️ 키워드 분석기 초기화 실패: {e}")
            keyword_analyzer = None
        
        # 텍스트 프로세서 초기화 (analyzer 연결)
        text_processor = SentioTextProcessor(analyzer=keyword_analyzer)
        logger.info("✅ 텍스트 프로세서 초기화 완료 (JD-R 모델 연결)")
        
        # 텍스트 생성기 초기화 (선택적)
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            try:
                # 페르소나 파일 없이 키워드 기반 텍스트 생성기 초기화
                text_generator = SentioTextGenerator(api_key, None)
                logger.info("✅ 텍스트 생성기 초기화 완료 (키워드 기반)")
            except Exception as e:
                logger.warning(f"⚠️ 텍스트 생성기 초기화 실패: {e}")
                text_generator = None
        else:
            logger.info("⚠️ OpenAI API 키가 없습니다. 텍스트 생성 기능은 비활성화됩니다.")
        
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
            "/generate/text": "키워드 기반 텍스트 생성",
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

@app.route('/upload/text_data', methods=['POST'])
def upload_text_data():
    """HR 텍스트 데이터 CSV 파일 업로드"""
    try:
        # 파일 확인
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "파일이 업로드되지 않았습니다."
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "파일이 선택되지 않았습니다."
            }), 400
        
        # 파일 확장자 확인
        if not file.filename.lower().endswith('.csv'):
            return jsonify({
                "success": False,
                "error": "CSV 파일만 업로드 가능합니다."
            }), 400
        
        # 파일 저장
        filename = secure_filename(file.filename)
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'Sentio')
        os.makedirs(upload_dir, exist_ok=True)
        
        # 타임스탬프를 포함한 파일명으로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(filename)[0]
        new_filename = f"{base_name}_{timestamp}.csv"
        file_path = os.path.join(upload_dir, new_filename)
        
        # 최신 파일 링크도 생성
        latest_link = os.path.join(upload_dir, 'latest_text_data.csv')
        file.save(file_path)
        
        # 최신 파일 링크 생성
        try:
            if os.path.exists(latest_link):
                os.remove(latest_link)
            import shutil
            shutil.copy2(file_path, latest_link)
        except Exception as e:
            logger.warning(f"최신 파일 링크 생성 실패: {e}")
        
        # 데이터 검증
        try:
            df = pd.read_csv(file_path)
            
            # 필수 컬럼 확인
            required_columns = ['employee_id', 'text']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return jsonify({
                    "success": False,
                    "error": f"필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}",
                    "required_columns": required_columns,
                    "found_columns": list(df.columns)
                }), 400
            
            # 텍스트 데이터 기본 통계
            text_stats = {
                "total_texts": len(df),
                "unique_employees": df['employee_id'].nunique(),
                "avg_text_length": df['text'].str.len().mean(),
                "text_types": df['text_type'].value_counts().to_dict() if 'text_type' in df.columns else {}
            }
            
            # 기존 시스템 초기화 (새 데이터로 재처리 필요)
            global text_processor, keyword_analyzer
            text_processor = None
            keyword_analyzer = None
            
            return jsonify({
                "success": True,
                "message": "텍스트 데이터가 성공적으로 업로드되었습니다.",
                "file_info": {
                    "original_filename": filename,
                    "saved_filename": new_filename,
                    "upload_path": upload_dir,
                    "file_path": file_path,
                    "latest_link": latest_link,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns)
                },
                "text_stats": text_stats,
                "note": "새로운 데이터로 시스템을 재초기화해주세요."
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"데이터 파일 읽기 오류: {str(e)}"
            }), 400
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"파일 업로드 오류: {str(e)}"
        }), 500

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
    키워드 기반 텍스트 생성 API
    입력: 텍스트 타입, 키워드 목록
    출력: 생성된 텍스트
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "요청 데이터가 필요합니다."}), 400
        
        if not text_generator:
            return jsonify({"error": "텍스트 생성기가 초기화되지 않았습니다. OpenAI API 키를 확인해주세요."}), 500
        
        # 키워드 기반 텍스트 생성
        if 'keywords' in data:
            keywords = data['keywords']
            text_type = data.get('text_type', 'SELF_REVIEW')
            employee_id = data.get('employee_id', 'unknown')
            analysis_type = data.get('analysis_type', 'batch')
            
            # 배치/사후 분석에서는 LLM 사용 안함 (API 비용 절약)
            use_llm = analysis_type not in ['batch', 'post']
            
            if use_llm:
                generated_text = text_generator.generate_text_from_keywords(
                    keywords=keywords,
                    text_type=text_type
                )
            else:
                # 배치/사후 분석에서는 간단한 키워드 기반 텍스트 반환
                generated_text = f"키워드 기반 분석 완료 (분석 타입: {analysis_type}, 키워드: {', '.join(keywords[:5])})"
            
            result = SentioGenerationResult(
                employee_id=employee_id,
                text_type=text_type,
                generated_text=generated_text,
                keywords_used=keywords[:5],  # 상위 5개 키워드만
                generation_timestamp=datetime.now().isoformat()
            )
            
            return jsonify(asdict(result))
        
        else:
            return jsonify({"error": "keywords가 필요합니다."}), 400
        
    except Exception as e:
        logger.error(f"텍스트 생성 오류: {str(e)}")
        return jsonify({"error": f"생성 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/data/keywords', methods=['GET'])
def get_available_keywords():
    """사용 가능한 키워드 목록 조회 API"""
    try:
        if not text_generator:
            return jsonify({"error": "텍스트 생성기가 초기화되지 않았습니다."}), 500
        
        # 사용 가능한 키워드 카테고리 반환
        available_keywords = text_generator.get_all_available_keywords()
        
        return jsonify({
            "keyword_categories": available_keywords,
            "total_categories": len(available_keywords)
        })
        
    except Exception as e:
        logger.error(f"키워드 조회 오류: {str(e)}")
        return jsonify({"error": f"조회 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/analyze/comprehensive_report', methods=['POST'])
def generate_comprehensive_report():
    """
    개별 직원의 모든 워커 에이전트 결과를 종합한 최종 레포트 생성 API (LLM 선택적 사용)
    입력: 한 직원의 모든 워커 분석 결과
    출력: 개별 직원 종합 분석 레포트
    """
    try:
        data = request.get_json()
        
        if not data or 'employee_id' not in data or 'worker_results' not in data:
            return jsonify({"error": "직원 ID와 모든 워커 분석 결과가 필요합니다."}), 400
        
        employee_id = data['employee_id']
        worker_results = data['worker_results']  # {structura: {...}, cognita: {...}, chronos: {...}, sentio: {...}}
        analysis_type = data.get('analysis_type', 'batch')
        
        # 배치/사후 분석에서는 LLM 사용 안함 (API 비용 절약)
        use_llm = data.get('use_llm', analysis_type not in ['batch', 'post'])
        
        if not keyword_analyzer:
            return jsonify({"error": "키워드 분석기가 초기화되지 않았습니다."}), 500
        
        # 종합 레포트 생성
        comprehensive_report = keyword_analyzer.generate_individual_comprehensive_report(
            employee_id=employee_id,
            all_worker_results=worker_results,
            use_llm=use_llm
        )
        
        # LLM 해석 추가 (선택적)
        if use_llm:
            llm_interpretation = keyword_analyzer.generate_comprehensive_llm_interpretation(
                comprehensive_report=comprehensive_report,
                use_llm=True
            )
            comprehensive_report['llm_interpretation'] = llm_interpretation
        else:
            # 규칙 기반 해석
            rule_based_interpretation = keyword_analyzer.generate_comprehensive_llm_interpretation(
                comprehensive_report=comprehensive_report,
                use_llm=False
            )
            comprehensive_report['rule_based_interpretation'] = rule_based_interpretation
        
        return jsonify(comprehensive_report)
        
    except Exception as e:
        logger.error(f"종합 레포트 생성 오류: {str(e)}")
        return jsonify({"error": f"레포트 생성 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    """
    Supervisor에서 호출하는 감정 분석 API
    입력: employee_id, 추가 데이터
    출력: 감정 분석 결과
    """
    global DATA_PATH, keyword_analyzer, text_processor
    
    try:
        data = request.get_json()
        
        # 단일 직원 분석과 배치 분석 모두 지원
        if 'employee_id' in data:
            # 단일 직원 분석
            employee_id = data['employee_id']
            analysis_type = data.get('analysis_type', 'batch')
            employees_data = [{'employee_id': employee_id, 'text_data': data.get('text_data', {})}]
        elif 'employees' in data:
            # 배치 분석
            employees_data = data['employees']
            analysis_type = data.get('analysis_type', 'batch')
            if not employees_data:
                return jsonify({"error": "employees 배열이 비어있습니다."}), 400
        else:
            return jsonify({"error": "employee_id 또는 employees 배열이 필요합니다."}), 400
        
        # 배치/사후 분석에서는 LLM 사용 안함 (API 비용 절약)
        use_llm = analysis_type not in ['batch', 'post']
        
        # 분석 타입에 따른 데이터 경로 확인 및 재로드
        new_data_paths = get_sentio_data_paths(analysis_type)
        current_data_paths = get_sentio_data_paths()
        
        if new_data_paths != current_data_paths:
            print(f"🔄 Sentio: {analysis_type} 분석을 위한 데이터 재로드")
            DATA_PATH = new_data_paths
            
            # 키워드 분석기 재초기화
            try:
                if new_data_paths['sample_texts'] and os.path.exists(new_data_paths['sample_texts']):
                    keyword_analyzer = SentioKeywordAnalyzer(new_data_paths['sample_texts'])
                    keyword_analyzer.load_data()
                    text_processor = SentioTextProcessor(analyzer=keyword_analyzer)
                    print(f"✅ Sentio {analysis_type} 데이터 재로드 완료")
            except Exception as e:
                print(f"⚠️ Sentio 데이터 재로드 실패: {e}")
        
        print(f"📊 Sentio {analysis_type} 분석 시작 - {len(employees_data)}명")
        
        if not text_processor:
            return jsonify({"error": "텍스트 프로세서가 초기화되지 않았습니다."}), 500
        
        # 배치 분석 결과 저장
        analysis_results = []
        
        for emp_data in employees_data:
            employee_id = emp_data.get('employee_id')
            text_data = emp_data.get('text_data', {})
            
            # 텍스트 데이터 추출
            if isinstance(text_data, str):
                # 단순 문자열인 경우
                combined_text = text_data
            else:
                # 딕셔너리인 경우
                self_review = text_data.get('self_review', '')
                peer_feedback = text_data.get('peer_feedback', '')
                weekly_survey = text_data.get('weekly_survey', '')
                combined_text = ' '.join([str(text) for text in [self_review, peer_feedback, weekly_survey] if text])
            
            # 텍스트가 없으면 기본값 사용
            if not combined_text or combined_text.strip() == '':
                combined_text = f"직원 {employee_id}의 기본 텍스트 데이터"
            
            try:
                # 실제 텍스트 분석 수행
                analysis_result = text_processor.analyze_text(
                    text=combined_text,
                    employee_id=employee_id,
                    text_type="comprehensive"
                )
                
                # 개별 결과 생성
                individual_result = {
                    "employee_id": employee_id,
                    "sentiment_score": analysis_result.get('sentiment_score', 0.5),
                    "risk_keywords": analysis_result.get('risk_factors', [])[:10],
                    "emotional_state": determine_emotional_state(analysis_result.get('sentiment_score', 0.5)),
                    "confidence_score": min(0.9, max(0.1, len(analysis_result.get('keywords', [])) / 20)),
                    "text_analysis_summary": f"JD-R 모델 기반 분석 - 위험도: {analysis_result.get('risk_level', 'MEDIUM')}, 키워드: {len(analysis_result.get('keywords', []))}개{' (분석 타입: ' + analysis_type + ')' if not use_llm else ''}",
                    "analysis_timestamp": datetime.now().isoformat(),
                    "detailed_analysis": {
                        "attrition_risk_score": analysis_result.get('attrition_risk_score', 0.5),
                        "risk_level": analysis_result.get('risk_level', 'MEDIUM'),
                        "keywords_count": len(analysis_result.get('keywords', [])),
                        "jd_r_indicators": analysis_result.get('jd_r_indicators', {})
                    },
                    # PostAnalysis.js에서 기대하는 필드 추가
                    "psychological_risk_score": analysis_result.get('attrition_risk_score', 0.5)
                }
                
                analysis_results.append(individual_result)
                
            except Exception as e:
                logger.warning(f"직원 {employee_id} 분석 실패: {str(e)}")
                # 실패한 경우 기본값 반환
                analysis_results.append({
                    "employee_id": employee_id,
                    "sentiment_score": 0.5,
                    "risk_keywords": ["analysis_error"],
                    "emotional_state": "neutral",
                    "confidence_score": 0.1,
                    "text_analysis_summary": f"분석 중 오류 발생: {str(e)}",
                    "analysis_timestamp": datetime.now().isoformat(),
                    "psychological_risk_score": 0.5
                })
        
        # 단일 직원인 경우 기존 형식으로 반환
        if len(employees_data) == 1:
            return jsonify(analysis_results[0])
        
        # 배치 분석인 경우 배열 형식으로 반환
        return jsonify({
            "success": True,
            "analysis_results": analysis_results,
            "total_analyzed": len(analysis_results),
            "analysis_type": analysis_type
        })
        
    except Exception as e:
        logger.error(f"감정 분석 오류 (직원 {employee_id}): {str(e)}")
        return jsonify({
            "sentiment_score": 0.5,
            "risk_keywords": ["analysis_error"],
            "emotional_state": "neutral",
            "confidence_score": 0.1,
            "text_analysis_summary": f"분석 중 오류 발생: {str(e)}",
            "analysis_timestamp": datetime.now().isoformat()
        }), 200  # 오류가 있어도 200으로 반환하여 워크플로우 중단 방지

def determine_emotional_state(sentiment_score):
    """감정 점수를 기반으로 감정 상태 결정"""
    if sentiment_score >= 0.7:
        return "positive"
    elif sentiment_score >= 0.4:
        return "neutral_positive"
    elif sentiment_score >= 0.3:
        return "neutral"
    else:
        return "negative"

@app.route('/analyze/batch_csv', methods=['POST'])
def generate_batch_csv():
    """
    대량 텍스트 데이터를 CSV로 빠르게 분석 (LLM 없이)
    입력: 텍스트 데이터 목록
    출력: CSV 파일 경로 및 분석 통계
    """
    try:
        data = request.get_json()
        
        if not data or 'text_data_list' not in data:
            return jsonify({"error": "텍스트 데이터 목록이 필요합니다."}), 400
        
        text_data_list = data['text_data_list']
        output_filename = data.get('output_filename', 'sentio_batch_analysis.csv')
        
        if not keyword_analyzer:
            return jsonify({"error": "키워드 분석기가 초기화되지 않았습니다."}), 500
        
        logger.info(f"배치 CSV 분석 시작: {len(text_data_list)}개 데이터")
        
        # 대량 분석 수행 (LLM 없이)
        start_time = datetime.now()
        df = keyword_analyzer.generate_csv_batch_analysis(text_data_list)
        
        # CSV 저장
        output_path = keyword_analyzer.save_analysis_to_csv(df, output_filename)
        end_time = datetime.now()
        
        # 통계 계산
        processing_time = (end_time - start_time).total_seconds()
        
        # 위험도별 분포
        risk_distribution = df['risk_level'].value_counts().to_dict()
        
        # 평균 점수
        avg_scores = {
            'psychological_risk_score': float(df['psychological_risk_score'].mean()),
            'job_demands_score': float(df['job_demands_score'].mean()),
            'job_resources_deficiency_score': float(df['job_resources_deficiency_score'].mean()),
            'sentiment_score': float(df['sentiment_score'].mean())
        }
        
        # 예측 분포
        prediction_distribution = df['attrition_prediction'].value_counts().to_dict()
        
        result = {
            "status": "success",
            "output_file": output_path,
            "processing_stats": {
                "total_processed": len(df),
                "processing_time_seconds": round(processing_time, 2),
                "records_per_second": round(len(df) / processing_time, 2),
                "analysis_timestamp": end_time.isoformat()
            },
            "analysis_summary": {
                "risk_distribution": risk_distribution,
                "average_scores": avg_scores,
                "prediction_distribution": {
                    "will_leave": prediction_distribution.get(1, 0),
                    "will_stay": prediction_distribution.get(0, 0)
                }
            },
            "message": f"✅ {len(df)}명의 텍스트 분석이 완료되었습니다. (LLM 미사용으로 빠른 처리)"
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"배치 CSV 분석 오류: {str(e)}")
        return jsonify({"error": f"분석 중 오류가 발생했습니다: {str(e)}"}), 500

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
        print("🌐 서버 주소: http://localhost:5004")
        print("📚 API 문서: http://localhost:5004/")
        print("=" * 60)
        
        app.run(
            host='0.0.0.0',
            port=5004,
            debug=True,
            threaded=True
        )
    else:
        print("❌ 시스템 초기화 실패")
        print("서버를 시작할 수 없습니다.")
