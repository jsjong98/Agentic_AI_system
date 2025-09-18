#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 테스트용 Master Server
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_app():
    """테스트용 Flask 앱 생성"""
    app = Flask(__name__)
    CORS(app)
    
    # 설정
    app.config['JSON_AS_ASCII'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    
    @app.route('/api/health')
    def health():
        """헬스체크"""
        return jsonify({
            "status": "healthy",
            "service": "Test Master Server",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route('/api/analyze/batch', methods=['POST'])
    def analyze_batch():
        """배치 분석 테스트"""
        try:
            data = request.get_json()
            if not data or not data.get('employees'):
                return jsonify({"error": "직원 데이터가 필요합니다"}), 400
            
            employees = data['employees']
            
            # 간단한 테스트 결과 생성
            batch_results = []
            for idx, employee in enumerate(employees):
                batch_results.append({
                    'employee_number': employee.get('EmployeeNumber', str(idx)),
                    'analysis_result': {
                        'structura_result': {'attrition_probability': 0.25},
                        'cognita_result': {'overall_risk_score': 0.53},
                        'chronos_result': {'trend_score': 0.65},
                        'sentio_result': {'sentiment_score': -0.2},
                        'agora_result': {'market_pressure_index': 0.796},
                        'combined_analysis': {'overall_risk_level': 'MEDIUM'}
                    },
                    'progress': ((idx + 1) / len(employees)) * 100
                })
            
            return jsonify({
                'batch_id': f"batch_{int(time.time() * 1000)}",
                'total_employees': len(employees),
                'completed_employees': len(batch_results),
                'results': batch_results,
                'summary': {
                    'high_risk_count': 0,
                    'medium_risk_count': len(batch_results),
                    'low_risk_count': 0
                }
            })
            
        except Exception as e:
            logger.error(f"배치 분석 실패: {str(e)}")
            return jsonify({"error": f"배치 분석 실패: {str(e)}"}), 500
    
    return app

def run_test_server(host='0.0.0.0', port=8000):
    """테스트 서버 실행"""
    app = create_test_app()
    
    print("=" * 50)
    print("🧪 테스트 Master Server 시작")
    print("=" * 50)
    print(f"📡 서버 주소: http://{host}:{port}")
    print(f"🔗 헬스체크: http://{host}:{port}/api/health")
    print(f"🔗 배치 분석: http://{host}:{port}/api/analyze/batch")
    print()
    
    try:
        app.run(host=host, port=port, debug=True)
    except Exception as e:
        logger.error(f"서버 실행 실패: {e}")
        raise

if __name__ == '__main__':
    run_test_server()
