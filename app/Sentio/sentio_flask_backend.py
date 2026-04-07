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
from dotenv import load_dotenv

# .env 파일 로드 (OpenAI API 키 등)
load_dotenv()

# 로컬 모듈 import
from sentio_processor import SentioTextProcessor
from sentio_analyzer import SentioKeywordAnalyzer
from sentio_generator import SentioTextGenerator

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)

# 로깅 설정 - 파일과 콘솔 모두 출력
log_dir = "../../logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "sentio_server.log")

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 파일 핸들러
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)

# 콘솔 핸들러  
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 포맷터
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 핸들러 추가
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 전역 변수
text_processor = None
keyword_analyzer = None
text_generator = None

# 데이터 경로 설정 - uploads 디렉토리에서 찾기
def get_sentio_data_paths(analysis_type='batch'):
    """uploads 디렉토리에서 Sentio 데이터 파일 찾기"""
    _base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads', 'Sentio'))
    uploads_dir = os.path.join(_base, analysis_type)
    data_paths = {
        'hr_data': None,
        'text_data': None,
        'sample_texts': None
    }

    # Check subdirectory first, then root upload dir
    search_dirs = [uploads_dir, _base]
    print(f"Sentio 데이터 경로 확인: {search_dirs}")

    found_dir = None
    for d in search_dirs:
        if os.path.exists(d):
            files = [f for f in os.listdir(d) if f.endswith('.csv')]
            if files:
                found_dir = d
                break

    uploads_dir = found_dir if found_dir else uploads_dir

    if found_dir:
        files = [f for f in os.listdir(uploads_dir) if f.endswith('.csv')]
        if files:
            # 가장 최근 파일 사용 (수정 시간 기준)
            files_with_time = [(f, os.path.getmtime(os.path.join(uploads_dir, f))) for f in files]
            files_with_time.sort(key=lambda x: x[1], reverse=True)  # 수정 시간 기준 내림차순
            print(f"📁 발견된 파일들: {[f[0] for f in files_with_time]}")
            
            # 가장 최신 파일을 모든 용도로 사용 (Sentio 데이터는 통합 파일)
            latest_file = files_with_time[0][0]
            latest_file_path = os.path.join(uploads_dir, latest_file)
            
            data_paths['hr_data'] = latest_file_path
            data_paths['text_data'] = latest_file_path  
            data_paths['sample_texts'] = latest_file_path
            
            print(f"✅ 최신 파일 사용: {latest_file}")
    
    # Sentio는 배치 분석에서 post 데이터를 참고하지 않음 (키워드 분석은 독립적)
    # batch에 파일이 없으면 기본값 사용
    if analysis_type == 'batch' and not any(data_paths.values()):
        print("⚠️ batch 디렉토리에 Sentio 데이터가 없습니다. 기본값을 사용합니다.")
    
    # 기본값으로 fallback (파일이 없는 경우에만)
    if not any(data_paths.values()):
        print("⚠️ uploads 디렉토리에 파일이 없어서 기본 경로 사용")
        if data_paths['hr_data'] is None:
            data_paths['hr_data'] = '../../data/IBM_HR.csv'
        if data_paths['text_data'] is None:
            data_paths['text_data'] = '../../data/IBM_HR_text.csv'
        if data_paths['sample_texts'] is None:
            data_paths['sample_texts'] = '../../data/IBM_HR_text.csv'
    
    print(f"📊 최종 데이터 경로: {data_paths}")
    return data_paths

# 초기화 시 데이터가 있는 디렉토리를 찾아서 사용
def find_available_sentio_data():
    """사용 가능한 Sentio 데이터 경로 찾기"""
    # post 디렉토리 먼저 확인 (사후 분석 데이터가 더 최신)
    for analysis_type in ['post', 'batch']:
        paths = get_sentio_data_paths(analysis_type)
        if any(paths.values()) and any(os.path.exists(path) for path in paths.values() if path):
            print(f"✅ {analysis_type} 디렉토리에서 데이터 발견")
            return paths
    
    # 둘 다 없으면 기본값 반환
    print("⚠️ uploads 디렉토리에 데이터가 없어서 기본 경로 사용")
    return get_sentio_data_paths('batch')

DATA_PATH = find_available_sentio_data()

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
        
        # 키워드 분석기 초기화 (필수 - 점수 계산을 위해 반드시 필요)
        global keyword_analyzer
        keyword_analyzer = None
        try:
            sample_texts_path = DATA_PATH['sample_texts']
            print(f"키워드 분석기 초기화 시도: {sample_texts_path}")
            print(f"현재 작업 디렉토리: {os.getcwd()}")
            print(f"절대 경로: {os.path.abspath(sample_texts_path) if sample_texts_path else 'None'}")
            
            if sample_texts_path and os.path.exists(sample_texts_path):
                print(f"파일 존재 확인됨: {sample_texts_path}")
                keyword_analyzer = SentioKeywordAnalyzer(sample_texts_path)
                
                # 데이터 로드 시도
                load_success = keyword_analyzer.load_data()
                print(f"데이터 로드 결과: {load_success}")
                
                if load_success:
                    logger.info("키워드 분석기 초기화 완료")
                    print(f"키워드 분석기 데이터 로드 성공: {sample_texts_path}")
                    print(f"퇴직자: {len(keyword_analyzer.resigned_data)}명, 재직자: {len(keyword_analyzer.stayed_data)}명")
                else:
                    logger.error("키워드 분석기 데이터 로드 실패 - 점수 계산 불가")
                    keyword_analyzer = None
            else:
                logger.error(f"텍스트 데이터 파일이 없습니다: {sample_texts_path}")
                keyword_analyzer = None
                
        except Exception as e:
            logger.error(f"키워드 분석기 초기화 실패: {e}")
            print(f"키워드 분석기 오류: {str(e)}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
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
            
            # DATA_PATH 갱신 후 시스템 재초기화
            global DATA_PATH, text_processor, keyword_analyzer
            DATA_PATH = {
                'hr_data': file_path,
                'text_data': file_path,
                'sample_texts': file_path
            }
            text_processor = None
            keyword_analyzer = None
            try:
                initialize_system()
            except Exception as init_err:
                logger.warning(f"자동 초기화 실패 (무시): {init_err}")

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
            return jsonify({"error": "키워드 분석기가 초기화되지 않았습니다. JD-R 모델 기반 점수 계산이 불가능합니다. 서버를 재시작하거나 데이터 파일을 확인해주세요."}), 500
        
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
            # 배치 분석 - 프론트엔드에서 직접 텍스트 데이터를 보내는 경우
            employees_list = data['employees']
            analysis_type = data.get('analysis_type', 'batch')
            if not employees_list:
                return jsonify({"error": "employees 배열이 비어있습니다."}), 400
            
            # 프론트엔드에서 보낸 데이터가 이미 텍스트 데이터를 포함하고 있는지 확인
            if isinstance(employees_list[0], dict) and 'text_data' in employees_list[0]:
                # 이미 텍스트 데이터가 포함된 경우 (BatchAnalysis에서 보낸 경우)
                employees_data = employees_list
                logger.info(f"📝 프론트엔드에서 텍스트 데이터 직접 수신: {len(employees_data)}명")
            else:
                # ID만 있는 경우 기존 로직 사용 (CSV에서 읽어오기)
                employee_ids = employees_list
                
                # CSV 파일에서 해당 직원들의 데이터 읽어오기
                try:
                    csv_path = get_sentio_data_paths(analysis_type)['hr_data']
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        employees_data = []
                        for emp_id in employee_ids:
                            # EmployeeNumber 컬럼으로 검색 (CSV 파일의 실제 컬럼명)
                            emp_row = df[df['EmployeeNumber'] == int(emp_id)] if 'EmployeeNumber' in df.columns else df[df['employee_id'] == int(emp_id)] if df['employee_id'].dtype != 'object' else df[df['employee_id'] == emp_id]
                            if not emp_row.empty:
                                row_data = emp_row.iloc[0].to_dict()
                                # 여러 텍스트 컬럼을 합쳐서 사용 (persona 컬럼 및 텍스트 제외)
                                text_parts = []
                                # persona 관련 컬럼은 제외하고 텍스트 컬럼만 사용
                                text_columns_to_use = ['SELF_REVIEW_text', 'PEER_FEEDBACK_text', 'WEEKLY_SURVEY_text', 'text']
                                for col in text_columns_to_use:
                                    if col in row_data and pd.notna(row_data[col]):
                                        # persona 관련 텍스트 제외
                                        text_content = str(row_data[col])
                                        if not _is_persona_text(text_content):
                                            text_parts.append(text_content)
                                combined_text = ' '.join(text_parts) if text_parts else f"직원 {emp_id}의 기본 텍스트 데이터"
                                
                                employees_data.append({
                                    'employee_id': emp_id,
                                    'text_data': combined_text
                                })
                            else:
                                employees_data.append({
                                    'employee_id': emp_id,
                                    'text_data': f"직원 {emp_id}의 기본 텍스트 데이터"
                                })
                    else:
                        # CSV 파일이 없으면 기본 데이터로 구성
                        employees_data = [{'employee_id': emp_id, 'text_data': f"직원 {emp_id}의 기본 텍스트 데이터"} for emp_id in employee_ids]
                except Exception as e:
                    logger.warning(f"CSV 데이터 읽기 실패: {e}, 기본 데이터 사용")
                    employees_data = [{'employee_id': emp_id, 'text_data': f"직원 {emp_id}의 기본 텍스트 데이터"} for emp_id in employee_ids]
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
            
            # 키워드 분석기 재초기화 (필수 - 점수 계산을 위해 반드시 필요)
            global keyword_analyzer, text_processor
            try:
                sample_texts_path = new_data_paths['sample_texts']
                logger.info(f"🔍 {analysis_type} 분석용 키워드 분석기 재초기화: {sample_texts_path}")
                
                if sample_texts_path and os.path.exists(sample_texts_path):
                    logger.info(f"📁 {analysis_type} 파일 존재 확인됨: {sample_texts_path}")
                    keyword_analyzer = SentioKeywordAnalyzer(sample_texts_path)
                    
                    load_success = keyword_analyzer.load_data()
                    logger.info(f"📊 {analysis_type} 데이터 로드 결과: {load_success}")
                    
                    if load_success:
                        text_processor = SentioTextProcessor(analyzer=keyword_analyzer)
                        logger.info(f"✅ Sentio {analysis_type} 키워드 분석기 재로드 완료")
                        logger.info(f"📈 {analysis_type} 퇴직자: {len(keyword_analyzer.resigned_data)}명, 재직자: {len(keyword_analyzer.stayed_data)}명")
                    else:
                        logger.error(f"❌ Sentio {analysis_type} 키워드 분석기 데이터 로드 실패 - 점수 계산 불가")
                        keyword_analyzer = None
                        text_processor = SentioTextProcessor(analyzer=None)
                else:
                    logger.warning(f"❌ {analysis_type} 분석용 텍스트 데이터 파일 없음: {sample_texts_path}")
                    keyword_analyzer = None
                    text_processor = SentioTextProcessor(analyzer=None)
                    
            except Exception as e:
                logger.error(f"❌ Sentio {analysis_type} 데이터 재로드 실패: {e}")
                import traceback
                logger.error(f"상세 오류: {traceback.format_exc()}")
                keyword_analyzer = None
                text_processor = SentioTextProcessor(analyzer=None)
        
        logger.info(f"📊 Sentio {analysis_type} 분석 시작 - {len(employees_data)}명")
        
        if not text_processor:
            return jsonify({"error": "텍스트 프로세서가 초기화되지 않았습니다."}), 500
        
        # 배치 분석 결과 저장
        analysis_results = []
        
        for emp_data in employees_data:
            employee_id = emp_data.get('employee_id')
            text_data = emp_data.get('text_data', {})
            
            # 텍스트 데이터 추출 (persona 텍스트 제외)
            if isinstance(text_data, str):
                # 단순 문자열인 경우 - persona 텍스트 확인
                combined_text = text_data if not _is_persona_text(text_data) else ""
            else:
                # 딕셔너리인 경우 - 각 텍스트에서 persona 제외
                text_parts = []
                for text_key in ['self_review', 'peer_feedback', 'weekly_survey']:
                    text_content = text_data.get(text_key, '')
                    if text_content and not _is_persona_text(str(text_content)):
                        text_parts.append(str(text_content))
                combined_text = ' '.join(text_parts)
            
            # 텍스트가 없으면 기본값 사용
            if not combined_text or combined_text.strip() == '':
                combined_text = f"직원 {employee_id}의 기본 텍스트 데이터"
            
            try:
                logger.info(f"🔍 직원 {employee_id} 텍스트 분석 시작 (길이: {len(combined_text)}자)")
                # 실제 텍스트 분석 수행
                analysis_result = text_processor.analyze_text(
                    text=combined_text,
                    employee_id=employee_id,
                    text_type="comprehensive"
                )
                logger.info(f"🔍 분석 결과 타입: {type(analysis_result)}")
                logger.info(f"🔍 분석 결과 키들: {list(analysis_result.keys()) if isinstance(analysis_result, dict) else 'N/A'}")
                
                # analysis_result가 딕셔너리인지 확인
                if not isinstance(analysis_result, dict):
                    logger.error(f"❌ 분석 결과가 딕셔너리가 아닙니다: {type(analysis_result)}")
                    raise ValueError(f"분석 결과 타입 오류: {type(analysis_result)}")
                
                # 안전한 값 추출
                sentiment_score = analysis_result.get('sentiment_score', 0.5)
                risk_factors = analysis_result.get('risk_factors', [])
                keywords = analysis_result.get('keywords', [])
                risk_level = analysis_result.get('risk_level', 'MEDIUM')
                
                # 위험 점수 추출 - 여러 가능한 필드명 시도
                attrition_risk_score = (
                    analysis_result.get('psychological_risk_score') or
                    analysis_result.get('attrition_risk_score') or
                    analysis_result.get('risk_score') or
                    0.5
                )
                
                jd_r_indicators = analysis_result.get('jd_r_indicators', {})
                
                logger.info(f"✅ 직원 {employee_id} 분석 완료:")
                logger.info(f"  - 키워드: {len(keywords)}개")
                logger.info(f"  - 위험도: {risk_level}")
                logger.info(f"  - 위험 점수: {attrition_risk_score}")
                logger.info(f"  - 감정 점수: {sentiment_score}")
                
                # 개별 결과 생성
                individual_result = {
                    "employee_id": employee_id,
                    "sentiment_score": sentiment_score,
                    "risk_keywords": risk_factors[:10],
                    "emotional_state": determine_emotional_state(sentiment_score),
                    "confidence_score": min(0.9, max(0.1, len(keywords) / 20)),
                    "text_analysis_summary": f"JD-R 모델 기반 분석 - 위험도: {risk_level}, 키워드: {len(keywords)}개{' (분석 타입: ' + analysis_type + ')' if not use_llm else ''}",
                    "analysis_timestamp": datetime.now().isoformat(),
                    "detailed_analysis": {
                        "attrition_risk_score": attrition_risk_score,
                        "risk_level": risk_level,
                        "keywords_count": len(keywords),
                        "jd_r_indicators": jd_r_indicators
                    },
                    # PostAnalysis.js에서 기대하는 필드 추가
                    "psychological_risk_score": attrition_risk_score
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
        
        logger.info(f"🎉 Sentio {analysis_type} 분석 완료 - 총 {len(analysis_results)}명 처리")
        
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
        # employee_id가 정의되지 않은 경우를 대비
        emp_id = locals().get('employee_id', 'unknown')
        logger.error(f"감정 분석 오류 (직원 {emp_id}): {str(e)}")
        return jsonify({
            "sentiment_score": 0.5,
            "risk_keywords": ["analysis_error"],
            "emotional_state": "neutral",
            "confidence_score": 0.1,
            "text_analysis_summary": f"분석 중 오류 발생: {str(e)}",
            "analysis_timestamp": datetime.now().isoformat()
        }), 200  # 오류가 있어도 200으로 반환하여 워크플로우 중단 방지

def _is_persona_text(text_content):
    """persona 관련 텍스트인지 판별 (더 엄격한 기준)"""
    if not text_content or pd.isna(text_content):
        return False
    
    text_lower = str(text_content).lower()
    
    # 명확한 persona 관련 키워드만 체크 (더 엄격한 기준)
    strict_persona_indicators = [
        'persona', '페르소나', 'softmax_persona', 'persona_code', 'persona_type', 'persona_name',
        'argmax_persona', 'prob_p01', 'prob_p02', 'prob_p03', 'prob_p04', 'prob_s01', 'prob_s02', 'prob_n01',
        # 정확한 페르소나 코드만
        'p01', 'p02', 'p03', 'p04', 'p05', 'n01', 'n02', 'n03', 's01', 's02', 's03'
    ]
    
    # 엄격한 persona 텍스트 패턴 확인
    for indicator in strict_persona_indicators:
        if indicator in text_lower:
            return True
    
    # 페르소나 분류 결과가 포함된 텍스트인지 확인 (더 정확한 패턴)
    import re
    persona_result_patterns = [
        r'argmax_persona.*?:', r'softmax_persona.*?:', r'persona_code.*?:',
        r'prob_[pns]\d+', r'score_[pns]\d+', r'risktier.*?:'
    ]
    
    for pattern in persona_result_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

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
            return jsonify({"error": "키워드 분석기가 초기화되지 않았습니다. CSV 배치 분석 기능을 사용할 수 없습니다."}), 500
        
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

@app.route('/analyze/batch', methods=['POST'])
def batch_analyze():
    """
    배치 감정 분석 - 프론트엔드 배치 분석용
    입력: 직원 텍스트 데이터 목록
    출력: 각 직원별 감정 분석 결과
    """
    try:
        data = request.get_json()
        
        if not data or 'employees' not in data:
            return jsonify({"error": "직원 데이터 목록이 필요합니다."}), 400
        
        employees = data['employees']
        
        if not keyword_analyzer:
            return jsonify({"error": "키워드 분석기가 초기화되지 않았습니다."}), 500
        
        logger.info(f"배치 감정 분석 시작: {len(employees)}명")
        
        # 텍스트 데이터 추출 및 분석
        results = []
        for i, emp in enumerate(employees):
            try:
                emp_id = emp.get('employee_id')
                logger.debug(f"직원 {emp_id} 감정 분석 시작 ({i+1}/{len(employees)})")
                text_data = emp.get('text_data', {})
                
                # 텍스트 결합 (상세 디버깅 추가)
                self_review = text_data.get('self_review', '')
                peer_feedback = text_data.get('peer_feedback', '')
                weekly_survey = text_data.get('weekly_survey', '')
                
                logger.info(f"🔍 직원 {emp_id} 텍스트 데이터 상세:")
                logger.info(f"  - self_review 길이: {len(str(self_review))}자")
                logger.info(f"  - peer_feedback 길이: {len(str(peer_feedback))}자") 
                logger.info(f"  - weekly_survey 길이: {len(str(weekly_survey))}자")
                logger.info(f"  - self_review 내용 (처음 100자): '{str(self_review)[:100]}...'")
                logger.info(f"  - peer_feedback 내용 (처음 100자): '{str(peer_feedback)[:100]}...'")
                logger.info(f"  - weekly_survey 내용 (처음 100자): '{str(weekly_survey)[:100]}...'")
                
                combined_text = ' '.join([
                    str(self_review),
                    str(peer_feedback), 
                    str(weekly_survey)
                ]).strip()
                
                logger.info(f"📝 직원 {emp_id} 결합된 텍스트 길이: {len(combined_text)}자")
                logger.info(f"📝 결합된 텍스트 내용 (처음 200자): '{combined_text[:200]}...'")
                
                if not combined_text or combined_text == 'nannannan':
                    logger.warning(f"⚠️ 직원 {emp_id}의 텍스트가 비어있거나 모두 NaN입니다!")
                
                if not combined_text:
                    # 텍스트가 없는 경우 기본값
                    results.append({
                        "employee_id": str(emp_id),
                        "sentio_score": 0.5,
                        "risk_level": "MEDIUM",
                        "psychological_risk_score": 0.5,
                        "job_demands_score": 0.5,
                        "job_resources_deficiency_score": 0.5,
                        "sentiment_score": 0.0,
                        "attrition_prediction": 0
                    })
                    continue
                
                # 키워드 분석 수행 - 올바른 메서드 사용
                analysis_result = keyword_analyzer.analyze_employee_text(emp_id, 
                    text_data.get('self_review', ''),
                    text_data.get('peer_feedback', ''), 
                    text_data.get('weekly_survey', ''))
                
                # 강화된 디버깅: 분석 결과 상세 확인
                logger.info(f"🔍 직원 {emp_id} 분석 결과:")
                logger.info(f"  - 결과 타입: {type(analysis_result)}")
                logger.info(f"  - 텍스트 길이: {len(combined_text)}자")
                
                if isinstance(analysis_result, dict):
                    logger.info(f"  - psychological_risk_score: {analysis_result.get('psychological_risk_score', 'Missing')}")
                    logger.info(f"  - risk_level: {analysis_result.get('risk_level', 'Missing')}")
                    logger.info(f"  - detected_keywords: {analysis_result.get('detected_keywords', [])}")
                    logger.info(f"  - job_demands_matches: {analysis_result.get('job_demands_matches', [])}")
                    logger.info(f"  - job_resources_deficiency_matches: {analysis_result.get('job_resources_deficiency_matches', [])}")
                    
                    # JD-R 지표 상세 확인
                    jdr_indicators = analysis_result.get('jd_r_indicators', {})
                    if isinstance(jdr_indicators, dict):
                        logger.info(f"  - job_demands_score: {jdr_indicators.get('job_demands_score', 'Missing')}")
                        logger.info(f"  - job_resources_deficiency_score: {jdr_indicators.get('job_resources_deficiency_score', 'Missing')}")
                else:
                    logger.error(f"❌ 직원 {emp_id} 분석 결과가 딕셔너리가 아닙니다!")
                
                # JD-R 지표에서 점수 추출 (중첩된 구조 처리)
                jdr_indicators = analysis_result.get('jd_r_indicators', {})
                job_demands_score = jdr_indicators.get('job_demands_score', 0.5) if isinstance(jdr_indicators, dict) else 0.5
                job_resources_deficiency_score = jdr_indicators.get('job_resources_deficiency_score', 0.5) if isinstance(jdr_indicators, dict) else 0.5
                
                # 프론트엔드가 기대하는 형식으로 변환
                result = {
                    "employee_id": str(emp_id),
                    "sentio_score": round(analysis_result.get('psychological_risk_score', 0.5), 3),
                    "risk_level": analysis_result.get('risk_level', 'MEDIUM').upper(),
                    "psychological_risk_score": round(analysis_result.get('psychological_risk_score', 0.5), 3),
                    "job_demands_score": round(job_demands_score, 3),
                    "job_resources_deficiency_score": round(job_resources_deficiency_score, 3),
                    "sentiment_score": round(analysis_result.get('sentiment_score', 0.0), 3),
                    "attrition_prediction": analysis_result.get('attrition_prediction', 0)
                }
                results.append(result)
                logger.debug(f"직원 {emp_id} 분석 완료: sentio_score={result['sentio_score']}")
                
            except Exception as e:
                logger.warning(f"직원 {emp.get('employee_id')} 분석 실패: {str(e)}")
                # 실패 시 기본값 반환
                results.append({
                    "employee_id": str(emp.get('employee_id')),
                    "sentio_score": 0.5,
                    "risk_level": "MEDIUM",
                    "psychological_risk_score": 0.5,
                    "job_demands_score": 0.5,
                    "job_resources_deficiency_score": 0.5,
                    "sentiment_score": 0.0,
                    "attrition_prediction": 0
                })
        
        logger.info(f"배치 감정 분석 완료: {len(results)}명")
        
        return jsonify({
            "status": "success",
            "total_analyzed": len(results),
            "analysis_results": results,
            "analysis_timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"배치 감정 분석 오류: {str(e)}")
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
