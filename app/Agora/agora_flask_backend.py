# -*- coding: utf-8 -*-
"""
Agora HR Market Analysis API
외부 노동 시장 분석을 위한 Flask 백엔드 서비스
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

# 프로젝트 루트 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# Agora 모듈 import
try:
    from agora_processor import AgoraMarketProcessor
    from agora_analyzer import AgoraMarketAnalyzer
    from agora_llm_generator import AgoraLLMGenerator
    AGORA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Agora 모듈 import 실패: {e}")
    AGORA_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agora_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)

# 전역 변수
market_processor = None
market_analyzer = None
llm_generator = None

# 데이터 경로 설정
DATA_PATH = {
    'hr_data': 'data/IBM_HR.csv',
    'market_cache': 'data/market_cache.json'
}

def get_structura_data_path():
    """Structura에서 업로드된 데이터 경로 확인"""
    # app/uploads/Structura 경로 우선 확인
    upload_path = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'Structura', 'latest_hr_data.csv')
    if os.path.exists(upload_path):
        return upload_path
    
    # 기존 data 폴더 경로 확인
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'IBM_HR.csv')
    if os.path.exists(data_path):
        return data_path
    
    # 기본 경로 반환
    return DATA_PATH['hr_data']

@dataclass
class AgoraAnalysisResult:
    """Agora 분석 결과 데이터 클래스"""
    employee_id: str
    job_role: str
    department: str
    market_pressure_index: float
    compensation_gap: float
    job_postings_count: int
    market_competitiveness: str
    agora_score: float  # 0~1 범위의 종합 위험도 점수
    risk_level: str
    llm_interpretation: Optional[str]
    analysis_timestamp: str

@dataclass
class AgoraMarketReport:
    """Agora 시장 분석 보고서 데이터 클래스"""
    job_role: str
    total_postings: int
    average_salary: float
    salary_range: Dict[str, float]
    market_trend: str
    competition_level: str
    key_skills: List[str]
    report_timestamp: str

# ============================================================================
# 시스템 초기화
# ============================================================================

def initialize_system():
    """
    Agora 시스템 초기화
    """
    global market_processor, market_analyzer, llm_generator
    
    try:
        logger.info("Agora 시스템 초기화 시작...")
        
        # 시장 데이터 프로세서 초기화
        market_processor = AgoraMarketProcessor()
        logger.info("✅ 시장 데이터 프로세서 초기화 완료")
        
        # 시장 분석기 초기화 (Structura 업로드 데이터 우선 사용)
        hr_data_path = get_structura_data_path()
        if os.path.exists(hr_data_path):
            market_analyzer = AgoraMarketAnalyzer(hr_data_path)
            logger.info(f"✅ 시장 분석기 초기화 완료 (데이터: {hr_data_path})")
        else:
            logger.warning("⚠️ HR 데이터 파일을 찾을 수 없습니다.")
        
        # LLM 생성기 초기화 (API 키는 환경변수에서 가져오기)
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            llm_generator = AgoraLLMGenerator(api_key)
            logger.info("✅ LLM 생성기 초기화 완료")
        else:
            logger.warning("⚠️ OpenAI API 키가 없어 LLM 생성기를 초기화하지 않습니다.")
        
        logger.info("🎉 Agora 시스템 초기화 완료!")
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
    """API 홈페이지"""
    return jsonify({
        "service": "Agora HR Market Analysis API",
        "version": "1.0.0",
        "description": "외부 노동 시장 분석 및 채용 경쟁력 평가 서비스",
        "endpoints": {
            "health": "GET /health - 시스템 상태 확인",
            "analyze_market": "POST /analyze/market - 개별 직원 시장 분석",
            "analyze_job_market": "POST /analyze/job_market - 직무별 시장 분석",
            "batch_analysis": "POST /analyze/batch - 배치 시장 분석",
            "market_report": "GET /market/report/<job_role> - 직무별 시장 보고서",
            "market_trends": "GET /market/trends - 전체 시장 트렌드"
        },
        "status": "running"
    })

@app.route('/health')
def health_check():
    """시스템 헬스체크"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "market_processor": "available" if market_processor else "unavailable",
            "market_analyzer": "available" if market_analyzer else "unavailable", 
            "llm_generator": "available" if llm_generator else "unavailable"
        },
        "agora_available": AGORA_AVAILABLE
    }
    
    return jsonify(status)

@app.route('/refresh/data', methods=['POST'])
def refresh_structura_data():
    """Structura에서 업로드된 새로운 데이터로 시스템 새로고침"""
    try:
        global market_analyzer
        
        # 새로운 데이터 경로 확인
        hr_data_path = get_structura_data_path()
        
        if not os.path.exists(hr_data_path):
            return jsonify({
                "success": False,
                "error": "Structura 데이터 파일을 찾을 수 없습니다.",
                "searched_path": hr_data_path
            }), 404
        
        # 시장 분석기 재초기화
        try:
            market_analyzer = AgoraMarketAnalyzer(hr_data_path)
            
            # 데이터 통계 확인
            import pandas as pd
            df = pd.read_csv(hr_data_path)
            
            data_stats = {
                "total_employees": len(df),
                "unique_job_roles": df['JobRole'].nunique() if 'JobRole' in df.columns else 0,
                "unique_departments": df['Department'].nunique() if 'Department' in df.columns else 0,
                "salary_range": {
                    "min": df['MonthlyIncome'].min() if 'MonthlyIncome' in df.columns else 0,
                    "max": df['MonthlyIncome'].max() if 'MonthlyIncome' in df.columns else 0,
                    "avg": df['MonthlyIncome'].mean() if 'MonthlyIncome' in df.columns else 0
                } if 'MonthlyIncome' in df.columns else {}
            }
            
            return jsonify({
                "success": True,
                "message": "Structura 데이터로 시스템이 성공적으로 새로고침되었습니다.",
                "data_path": hr_data_path,
                "data_stats": data_stats,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"시장 분석기 재초기화 실패: {str(e)}"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"데이터 새로고침 오류: {str(e)}"
        }), 500

@app.route('/analyze/market', methods=['POST'])
def analyze_individual_market():
    """
    개별 직원 시장 분석 API
    입력: 직원 정보 (JobRole, Department, MonthlyIncome 등)
    출력: 시장 압력 지수, 보상 격차, 위험도 등
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "요청 데이터가 필요합니다."}), 400
        
        employee_id = data.get('EmployeeNumber', 'unknown')
        job_role = data.get('JobRole')
        department = data.get('Department')
        monthly_income = data.get('MonthlyIncome', 0)
        
        if not job_role:
            return jsonify({"error": "JobRole이 필요합니다."}), 400
        
        if not market_analyzer:
            return jsonify({"error": "시장 분석기가 초기화되지 않았습니다."}), 500
        
        # 시장 분석 수행
        analysis_result = market_analyzer.analyze_employee_market(
            employee_data=data,
            include_llm=data.get('use_llm', False)
        )
        
        # 결과 구성
        result = AgoraAnalysisResult(
            employee_id=str(employee_id),
            job_role=job_role,
            department=department or 'Unknown',
            market_pressure_index=analysis_result.get('market_pressure_index', 0.0),
            compensation_gap=analysis_result.get('compensation_gap', 0.0),
            job_postings_count=analysis_result.get('job_postings_count', 0),
            market_competitiveness=analysis_result.get('market_competitiveness', 'MEDIUM'),
            agora_score=analysis_result.get('agora_score', 0.0),  # 0~1 범위 점수
            risk_level=analysis_result.get('risk_level', 'MEDIUM'),
            llm_interpretation=analysis_result.get('llm_interpretation'),
            analysis_timestamp=datetime.now().isoformat()
        )
        
        return jsonify(asdict(result))
        
    except Exception as e:
        logger.error(f"개별 시장 분석 오류: {str(e)}")
        return jsonify({"error": f"분석 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/analyze/job_market', methods=['POST'])
def analyze_job_market():
    """
    직무별 시장 분석 API
    입력: 직무명, 지역, 경력 수준
    출력: 채용 공고 수, 평균 급여, 시장 트렌드 등
    """
    try:
        data = request.get_json()
        
        if not data or 'job_role' not in data:
            return jsonify({"error": "job_role이 필요합니다."}), 400
        
        job_role = data['job_role']
        location = data.get('location', '서울')
        experience_level = data.get('experience_level', 'mid')
        
        if not market_processor:
            return jsonify({"error": "시장 데이터 프로세서가 초기화되지 않았습니다."}), 500
        
        # 직무별 시장 분석
        market_data = market_processor.analyze_job_market(
            job_role=job_role,
            location=location,
            experience_level=experience_level
        )
        
        # 결과 구성
        result = AgoraMarketReport(
            job_role=job_role,
            total_postings=market_data.get('total_postings', 0),
            average_salary=market_data.get('average_salary', 0.0),
            salary_range=market_data.get('salary_range', {}),
            market_trend=market_data.get('market_trend', 'STABLE'),
            competition_level=market_data.get('competition_level', 'MEDIUM'),
            key_skills=market_data.get('key_skills', []),
            report_timestamp=datetime.now().isoformat()
        )
        
        return jsonify(asdict(result))
        
    except Exception as e:
        logger.error(f"직무별 시장 분석 오류: {str(e)}")
        return jsonify({"error": f"분석 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/analyze/batch', methods=['POST'])
def batch_market_analysis():
    """
    배치 시장 분석 API
    입력: 직원 데이터 목록
    출력: 각 직원별 시장 분석 결과
    """
    try:
        data = request.get_json()
        
        if not data or 'employees' not in data:
            return jsonify({"error": "직원 데이터 목록이 필요합니다."}), 400
        
        employees = data['employees']
        use_llm = data.get('use_llm', False)
        
        if not market_analyzer:
            return jsonify({"error": "시장 분석기가 초기화되지 않았습니다."}), 500
        
        logger.info(f"배치 시장 분석 시작: {len(employees)}명")
        
        # 배치 분석 수행
        results = market_analyzer.batch_analyze_market(
            employees_data=employees,
            include_llm=use_llm
        )
        
        # 통계 계산
        total_analyzed = len(results)
        high_risk_count = sum(1 for r in results if r.get('risk_level') == 'HIGH')
        avg_market_pressure = sum(r.get('market_pressure_index', 0) for r in results) / total_analyzed if total_analyzed > 0 else 0
        
        response = {
            "status": "success",
            "total_analyzed": total_analyzed,
            "high_risk_employees": high_risk_count,
            "average_market_pressure": round(avg_market_pressure, 3),
            "results": results,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"배치 시장 분석 오류: {str(e)}")
        return jsonify({"error": f"분석 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/market/report/<job_role>')
def get_market_report(job_role):
    """
    직무별 시장 보고서 조회 API
    """
    try:
        if not market_processor:
            return jsonify({"error": "시장 데이터 프로세서가 초기화되지 않았습니다."}), 500
        
        # 시장 보고서 생성
        report = market_processor.generate_market_report(job_role)
        
        return jsonify(report)
        
    except Exception as e:
        logger.error(f"시장 보고서 조회 오류: {str(e)}")
        return jsonify({"error": f"보고서 생성 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/market/trends')
def get_market_trends():
    """
    전체 시장 트렌드 조회 API
    """
    try:
        if not market_analyzer:
            return jsonify({"error": "시장 분석기가 초기화되지 않았습니다."}), 500
        
        # 시장 트렌드 분석
        trends = market_analyzer.analyze_market_trends()
        
        return jsonify(trends)
        
    except Exception as e:
        logger.error(f"시장 트렌드 조회 오류: {str(e)}")
        return jsonify({"error": f"트렌드 분석 중 오류가 발생했습니다: {str(e)}"}), 500

@app.route('/market/competitive_analysis', methods=['POST'])
def competitive_analysis():
    """
    경쟁력 분석 API
    입력: 직원 정보 및 비교 대상
    출력: 시장 대비 경쟁력 분석
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "요청 데이터가 필요합니다."}), 400
        
        if not market_analyzer:
            return jsonify({"error": "시장 분석기가 초기화되지 않았습니다."}), 500
        
        # 경쟁력 분석 수행
        analysis = market_analyzer.analyze_competitiveness(data)
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"경쟁력 분석 오류: {str(e)}")
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
    print("🚀 Agora HR Market Analysis API 서버 시작...")
    print("=" * 60)
    
    # 시스템 초기화
    if initialize_system():
        print("✅ 시스템 초기화 완료")
        print("🌐 서버 주소: http://localhost:5005")
        print("📚 API 문서: http://localhost:5005/")
        print("=" * 60)
        
        app.run(
            host='0.0.0.0',
            port=5005,
            debug=True,
            threaded=True
        )
    else:
        print("❌ 시스템 초기화 실패")
        print("서버를 시작할 수 없습니다.")
