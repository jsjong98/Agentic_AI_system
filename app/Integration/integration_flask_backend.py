"""
Integration Flask 백엔드
임계값 설정, 가중치 최적화 및 LLM 기반 레포트 생성 API 서버
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import traceback
from typing import Dict, List, Any
from dotenv import load_dotenv

from threshold_calculator import ThresholdCalculator, load_and_process_data
from weight_optimizer import WeightOptimizer
from report_generator import ReportGenerator

# 환경변수 로드 (Sentio/Agora와 동일)
load_dotenv()

app = Flask(__name__)
CORS(app)

# 전역 변수
threshold_calculator = ThresholdCalculator()
weight_optimizer = WeightOptimizer()
report_generator = ReportGenerator()  # 환경변수에서 자동으로 API 키 로드 시도
current_data = None
current_results = {}

# 데이터 디렉토리 설정
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        'status': 'healthy',
        'service': 'Integration',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'llm_enabled': report_generator.client is not None
    })


@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    """OpenAI API 키 설정"""
    try:
        data = request.get_json()
        api_key = data.get('api_key')
        
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'API 키가 필요합니다.'
            }), 400
        
        # 새로운 ReportGenerator 인스턴스 생성
        global report_generator
        report_generator = ReportGenerator(api_key=api_key)
        
        return jsonify({
            'success': True,
            'message': 'API 키가 성공적으로 설정되었습니다.',
            'llm_enabled': report_generator.client is not None
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'API 키 설정 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/load_data', methods=['POST'])
def load_data():
    """데이터 로드"""
    global current_data
    
    try:
        data = request.get_json()
        file_path = data.get('file_path', 'Total_score.csv')
        
        # 절대 경로 생성
        if not os.path.isabs(file_path):
            file_path = os.path.join(DATA_DIR, file_path)
        
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': f'파일을 찾을 수 없습니다: {file_path}'
            }), 404
        
        # 데이터 로드
        current_data, score_columns = load_and_process_data(file_path)
        
        # 기본 통계 계산
        stats = {
            'total_rows': len(current_data),
            'total_columns': len(current_data.columns),
            'score_columns': score_columns,
            'attrition_distribution': current_data['attrition'].value_counts().to_dict() if 'attrition' in current_data.columns else {},
            'missing_values': current_data.isnull().sum().to_dict()
        }
        
        return jsonify({
            'success': True,
            'message': '데이터가 성공적으로 로드되었습니다.',
            'file_path': file_path,
            'statistics': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'데이터 로드 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/calculate_thresholds', methods=['POST'])
def calculate_thresholds():
    """임계값 계산"""
    global current_data, current_results
    
    try:
        if current_data is None:
            return jsonify({
                'success': False,
                'error': '데이터가 로드되지 않았습니다. /load_data를 먼저 호출하세요.'
            }), 400
        
        data = request.get_json()
        score_columns = data.get('score_columns', None)
        
        # Score 컬럼 자동 감지
        if score_columns is None:
            score_columns = [col for col in current_data.columns if col.endswith('_score')]
        
        if not score_columns:
            return jsonify({
                'success': False,
                'error': 'Score 컬럼을 찾을 수 없습니다.'
            }), 400
        
        # 임계값 계산
        results = threshold_calculator.calculate_thresholds_for_scores(current_data, score_columns)
        
        # 요약 테이블 생성
        summary_df = threshold_calculator.get_summary_table()
        thresholds_dict = threshold_calculator.get_thresholds_dict()
        
        # 예측 컬럼 추가
        data_with_predictions = threshold_calculator.apply_thresholds_to_data(current_data, score_columns)
        
        # 결과 저장
        current_results['threshold_results'] = results
        current_results['threshold_summary'] = summary_df.to_dict('records')
        current_results['thresholds_dict'] = thresholds_dict
        current_results['data_with_predictions'] = data_with_predictions
        
        # 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 요약 결과 저장
        summary_file = os.path.join(OUTPUT_DIR, f'threshold_summary_{timestamp}.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        
        # 예측 데이터 저장
        predictions_file = os.path.join(OUTPUT_DIR, f'data_with_predictions_{timestamp}.csv')
        data_with_predictions.to_csv(predictions_file, index=False, encoding='utf-8')
        
        return jsonify({
            'success': True,
            'message': '임계값 계산이 완료되었습니다.',
            'results': {
                'summary': current_results['threshold_summary'],
                'thresholds': thresholds_dict,
                'best_score': summary_df.loc[summary_df['F1_Score'].idxmax()].to_dict(),
                'total_predictions': len(data_with_predictions)
            },
            'files': {
                'summary': summary_file,
                'predictions': predictions_file
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'임계값 계산 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/optimize_weights', methods=['POST'])
def optimize_weights():
    """가중치 최적화"""
    global current_data, current_results
    
    try:
        if current_data is None:
            return jsonify({
                'success': False,
                'error': '데이터가 로드되지 않았습니다. /load_data를 먼저 호출하세요.'
            }), 400
        
        # 예측 데이터가 있는지 확인
        data_to_use = current_results.get('data_with_predictions', current_data)
        prediction_cols = [col for col in data_to_use.columns if col.endswith('_prediction')]
        
        if not prediction_cols:
            return jsonify({
                'success': False,
                'error': '예측 컬럼이 없습니다. /calculate_thresholds를 먼저 호출하세요.'
            }), 400
        
        data = request.get_json()
        method = data.get('method', 'bayesian')  # 'grid', 'bayesian', 'scipy'
        
        # 방법별 파라미터
        method_params = {}
        if method == 'grid':
            method_params['n_points_per_dim'] = data.get('n_points_per_dim', 5)
        elif method == 'bayesian':
            method_params['n_calls'] = data.get('n_calls', 100)
        
        # 가중치 최적화 실행
        optimization_results = weight_optimizer.optimize_weights(
            data_to_use, method=method, **method_params
        )
        
        if not optimization_results.get('best_weights'):
            return jsonify({
                'success': False,
                'error': '가중치 최적화에 실패했습니다.',
                'details': optimization_results.get('error', 'Unknown error')
            }), 500
        
        # 최적 가중치 적용
        data_with_weighted = weight_optimizer.apply_optimal_weights(data_to_use)
        data_with_risk = weight_optimizer.classify_risk_level(data_with_weighted)
        
        # 성능 요약
        performance_summary = weight_optimizer.get_performance_summary(data_with_risk)
        
        # 결과 저장
        current_results['weight_optimization'] = optimization_results
        current_results['final_data'] = data_with_risk
        current_results['performance_summary'] = performance_summary
        
        # 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 최종 데이터 저장
        final_data_file = os.path.join(OUTPUT_DIR, f'final_weighted_predictions_{timestamp}.csv')
        data_with_risk.to_csv(final_data_file, index=False, encoding='utf-8')
        
        # 가중치 정보 저장
        weights_info = pd.DataFrame([
            {'Variable': col, 'Weight': weight} 
            for col, weight in optimization_results['best_weights'].items()
        ])
        weights_info['Method'] = method
        weights_info['F1_Score'] = optimization_results['best_f1']
        weights_info['Threshold'] = optimization_results['best_threshold']
        
        weights_file = os.path.join(OUTPUT_DIR, f'optimal_weights_{timestamp}.csv')
        weights_info.to_csv(weights_file, index=False, encoding='utf-8')
        
        # 위험도 기준 정보 저장
        risk_criteria = pd.DataFrame([
            {'Risk_Level': '안전군', 'Score_Range': '0.0 ~ 0.3', 'Numeric_Code': 1},
            {'Risk_Level': '주의군', 'Score_Range': '0.3 ~ 0.7', 'Numeric_Code': 2},
            {'Risk_Level': '고위험군', 'Score_Range': '0.7 ~ 1.0', 'Numeric_Code': 3}
        ])
        
        risk_criteria_file = os.path.join(OUTPUT_DIR, f'risk_criteria_{timestamp}.csv')
        risk_criteria.to_csv(risk_criteria_file, index=False, encoding='utf-8')
        
        return jsonify({
            'success': True,
            'message': '가중치 최적화가 완료되었습니다.',
            'results': {
                'method': method,
                'optimal_weights': optimization_results['best_weights'],
                'optimal_threshold': optimization_results['best_threshold'],
                'best_f1_score': optimization_results['best_f1'],
                'performance_metrics': performance_summary['performance_metrics'],
                'risk_statistics': performance_summary['risk_statistics'],
                'total_records': len(data_with_risk)
            },
            'files': {
                'final_data': final_data_file,
                'weights_info': weights_file,
                'risk_criteria': risk_criteria_file
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'가중치 최적화 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/predict_employee', methods=['POST'])
def predict_employee():
    """개별 직원 예측"""
    try:
        data = request.get_json()
        employee_scores = data.get('scores', {})
        
        if not employee_scores:
            return jsonify({
                'success': False,
                'error': '직원 점수 데이터가 필요합니다.'
            }), 400
        
        results = {}
        
        # 임계값 기반 예측
        if threshold_calculator.optimal_thresholds:
            threshold_predictions = threshold_calculator.predict_attrition(employee_scores)
            results['threshold_predictions'] = threshold_predictions
        
        # 가중치 기반 예측
        if weight_optimizer.optimal_weights:
            # 예측 컬럼 생성 (임계값 기반)
            prediction_data = {}
            for score_name, score_value in employee_scores.items():
                if score_name in threshold_calculator.optimal_thresholds:
                    threshold = threshold_calculator.optimal_thresholds[score_name]
                    prediction_col = f"{score_name}_prediction"
                    prediction_data[prediction_col] = 1 if score_value >= threshold else 0
            
            if prediction_data:
                # 가중 점수 계산
                weighted_score = sum(
                    prediction_data[col] * weight 
                    for col, weight in weight_optimizer.optimal_weights.items()
                    if col in prediction_data
                )
                
                # 최종 예측
                final_prediction = 1 if weighted_score >= weight_optimizer.optimal_threshold else 0
                
                # 위험도 분류
                if weighted_score < 0.3:
                    risk_level = '안전군'
                    risk_numeric = 1
                elif weighted_score < 0.7:
                    risk_level = '주의군'
                    risk_numeric = 2
                else:
                    risk_level = '고위험군'
                    risk_numeric = 3
                
                results['weighted_prediction'] = {
                    'weighted_score': weighted_score,
                    'final_prediction': final_prediction,
                    'prediction_label': '위험' if final_prediction == 1 else '안전',
                    'risk_level': risk_level,
                    'risk_numeric': risk_numeric,
                    'threshold_used': weight_optimizer.optimal_threshold
                }
        
        if not results:
            return jsonify({
                'success': False,
                'error': '모델이 학습되지 않았습니다. 먼저 임계값 계산 및 가중치 최적화를 수행하세요.'
            }), 400
        
        return jsonify({
            'success': True,
            'employee_scores': employee_scores,
            'predictions': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'예측 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/get_results', methods=['GET'])
def get_results():
    """현재 결과 조회"""
    try:
        if not current_results:
            return jsonify({
                'success': False,
                'error': '계산된 결과가 없습니다.'
            }), 404
        
        # 민감한 데이터 제외하고 요약 정보만 반환
        summary = {
            'has_threshold_results': 'threshold_results' in current_results,
            'has_weight_optimization': 'weight_optimization' in current_results,
            'has_final_data': 'final_data' in current_results
        }
        
        if 'threshold_summary' in current_results:
            summary['threshold_summary'] = current_results['threshold_summary']
            summary['thresholds_dict'] = current_results['thresholds_dict']
        
        if 'performance_summary' in current_results:
            summary['performance_summary'] = current_results['performance_summary']
        
        return jsonify({
            'success': True,
            'results': summary
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'결과 조회 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/compare_methods', methods=['POST'])
def compare_methods():
    """여러 최적화 방법 비교"""
    global current_data, current_results
    
    try:
        if current_data is None:
            return jsonify({
                'success': False,
                'error': '데이터가 로드되지 않았습니다.'
            }), 400
        
        data_to_use = current_results.get('data_with_predictions', current_data)
        prediction_cols = [col for col in data_to_use.columns if col.endswith('_prediction')]
        
        if not prediction_cols:
            return jsonify({
                'success': False,
                'error': '예측 컬럼이 없습니다. /calculate_thresholds를 먼저 호출하세요.'
            }), 400
        
        data = request.get_json()
        methods = data.get('methods', ['grid', 'scipy'])  # bayesian은 시간이 오래 걸릴 수 있음
        
        comparison_results = []
        
        for method in methods:
            try:
                print(f"=== {method} 방법 테스트 중 ===")
                
                # 새로운 optimizer 인스턴스 생성
                temp_optimizer = WeightOptimizer()
                
                # 방법별 파라미터 설정
                method_params = {}
                if method == 'grid':
                    method_params['n_points_per_dim'] = 3  # 빠른 테스트를 위해 줄임
                elif method == 'bayesian':
                    method_params['n_calls'] = 50  # 빠른 테스트를 위해 줄임
                
                # 최적화 실행
                result = temp_optimizer.optimize_weights(
                    data_to_use, method=method, **method_params
                )
                
                if result.get('best_weights'):
                    # 성능 평가
                    data_temp = temp_optimizer.apply_optimal_weights(data_to_use)
                    performance = temp_optimizer.get_performance_summary(data_temp)
                    
                    comparison_results.append({
                        'method': method,
                        'best_weights': result['best_weights'],
                        'best_f1_score': result['best_f1'],
                        'best_threshold': result['best_threshold'],
                        'performance_metrics': performance['performance_metrics'],
                        'success': True
                    })
                else:
                    comparison_results.append({
                        'method': method,
                        'success': False,
                        'error': result.get('error', 'Unknown error')
                    })
                    
            except Exception as method_error:
                comparison_results.append({
                    'method': method,
                    'success': False,
                    'error': str(method_error)
                })
        
        # 최고 성능 방법 찾기
        successful_results = [r for r in comparison_results if r.get('success')]
        best_method = None
        
        if successful_results:
            best_method = max(successful_results, key=lambda x: x['best_f1_score'])
        
        return jsonify({
            'success': True,
            'message': '방법 비교가 완료되었습니다.',
            'comparison_results': comparison_results,
            'best_method': best_method,
            'total_methods_tested': len(methods),
            'successful_methods': len(successful_results)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'방법 비교 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/export_results', methods=['POST'])
def export_results():
    """결과 내보내기"""
    try:
        if not current_results:
            return jsonify({
                'success': False,
                'error': '내보낼 결과가 없습니다.'
            }), 404
        
        data = request.get_json()
        export_format = data.get('format', 'csv')  # 'csv', 'json'
        include_data = data.get('include_data', True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exported_files = []
        
        if export_format == 'csv':
            # 임계값 요약
            if 'threshold_summary' in current_results:
                threshold_df = pd.DataFrame(current_results['threshold_summary'])
                threshold_file = os.path.join(OUTPUT_DIR, f'threshold_export_{timestamp}.csv')
                threshold_df.to_csv(threshold_file, index=False, encoding='utf-8')
                exported_files.append(threshold_file)
            
            # 최종 데이터
            if include_data and 'final_data' in current_results:
                final_data_file = os.path.join(OUTPUT_DIR, f'final_data_export_{timestamp}.csv')
                current_results['final_data'].to_csv(final_data_file, index=False, encoding='utf-8')
                exported_files.append(final_data_file)
        
        elif export_format == 'json':
            # JSON 형태로 결과 저장
            export_data = {
                'timestamp': timestamp,
                'threshold_results': current_results.get('threshold_summary', []),
                'thresholds_dict': current_results.get('thresholds_dict', {}),
                'performance_summary': current_results.get('performance_summary', {})
            }
            
            if include_data and 'final_data' in current_results:
                export_data['final_data'] = current_results['final_data'].to_dict('records')
            
            json_file = os.path.join(OUTPUT_DIR, f'results_export_{timestamp}.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            exported_files.append(json_file)
        
        return jsonify({
            'success': True,
            'message': '결과가 성공적으로 내보내졌습니다.',
            'exported_files': exported_files,
            'format': export_format,
            'timestamp': timestamp
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'결과 내보내기 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/generate_report', methods=['POST'])
def generate_report():
    """개별 직원 레포트 생성"""
    try:
        data = request.get_json()
        employee_id = data.get('employee_id')
        agent_scores = data.get('agent_scores', {})
        format_type = data.get('format', 'json')  # 'json', 'text', 'both'
        save_file = data.get('save_file', False)
        use_llm = data.get('use_llm', True)  # LLM 사용 여부
        
        if not employee_id:
            return jsonify({
                'success': False,
                'error': '직원 ID가 필요합니다.'
            }), 400
        
        if not agent_scores:
            return jsonify({
                'success': False,
                'error': '에이전트 점수 데이터가 필요합니다.'
            }), 400
        
        # 점수 설정
        report_generator.set_agent_scores(employee_id, agent_scores)
        
        # 레포트 생성
        if format_type == 'text':
            report_content = report_generator.generate_text_report(employee_id, use_llm)
            result = {
                'success': True,
                'employee_id': employee_id,
                'format': 'text',
                'report': report_content,
                'llm_used': use_llm and report_generator.client is not None
            }
        else:
            report_content = report_generator.generate_employee_report(employee_id, use_llm)
            result = {
                'success': True,
                'employee_id': employee_id,
                'format': 'json',
                'report': report_content,
                'llm_used': use_llm and report_generator.client is not None
            }
        
        # 파일 저장 옵션
        if save_file:
            saved_files = report_generator.save_report(
                employee_id, 
                os.path.join(OUTPUT_DIR, 'reports'), 
                format_type
            )
            
            if 'error' in saved_files:
                result['file_save_error'] = saved_files['error']
            else:
                result['saved_files'] = saved_files
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'레포트 생성 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/generate_batch_reports', methods=['POST'])
def generate_batch_reports():
    """여러 직원의 레포트 일괄 생성"""
    try:
        data = request.get_json()
        employees_data = data.get('employees', [])  # [{'employee_id': 'EMP001', 'agent_scores': {...}}, ...]
        
        if not employees_data:
            return jsonify({
                'success': False,
                'error': '직원 데이터가 필요합니다.'
            }), 400
        
        # 각 직원의 점수 설정
        employee_ids = []
        for emp_data in employees_data:
            employee_id = emp_data.get('employee_id')
            agent_scores = emp_data.get('agent_scores', {})
            
            if employee_id and agent_scores:
                report_generator.set_agent_scores(employee_id, agent_scores)
                employee_ids.append(employee_id)
        
        if not employee_ids:
            return jsonify({
                'success': False,
                'error': '유효한 직원 데이터가 없습니다.'
            }), 400
        
        # 일괄 레포트 생성
        batch_results = report_generator.generate_batch_reports(
            employee_ids, 
            os.path.join(OUTPUT_DIR, 'reports')
        )
        
        return jsonify({
            'success': True,
            'message': '일괄 레포트 생성이 완료되었습니다.',
            'results': batch_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'일괄 레포트 생성 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/load_employee_data', methods=['POST'])
def load_employee_data():
    """직원 기본 데이터 로드 (레포트 생성용)"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({
                'success': False,
                'error': '파일 경로가 필요합니다.'
            }), 400
        
        # 절대 경로 생성
        if not os.path.isabs(file_path):
            file_path = os.path.join(DATA_DIR, file_path)
        
        # 데이터 로드
        success = report_generator.load_employee_data(file_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': '직원 데이터가 성공적으로 로드되었습니다.',
                'file_path': file_path,
                'total_employees': len(report_generator.employee_data) if report_generator.employee_data is not None else 0
            })
        else:
            return jsonify({
                'success': False,
                'error': '직원 데이터 로드에 실패했습니다.'
            }), 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'직원 데이터 로드 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/upload/employee_data', methods=['POST'])
def upload_employee_data():
    """직원 데이터 CSV 파일 업로드"""
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
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'Integration')
        os.makedirs(upload_dir, exist_ok=True)
        
        # 타임스탬프를 포함한 파일명으로 저장 (기존 파일 보존)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(filename)[0]
        new_filename = f"{base_name}_{timestamp}.csv"
        file_path = os.path.join(upload_dir, new_filename)
        
        # 최신 파일 링크도 생성
        latest_link = os.path.join(upload_dir, 'latest_employee_data.csv')
        file.save(file_path)
        
        # 최신 파일 링크 생성
        try:
            if os.path.exists(latest_link):
                os.remove(latest_link)
            import shutil
            shutil.copy2(file_path, latest_link)
        except Exception as e:
            print(f"최신 파일 링크 생성 실패: {e}")
        
        # 데이터 검증 및 로드
        try:
            df = pd.read_csv(file_path)
            
            # 필수 컬럼 확인
            required_columns = ['employee_id']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return jsonify({
                    "success": False,
                    "error": f"필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}",
                    "required_columns": required_columns,
                    "found_columns": list(df.columns)
                }), 400
            
            # ReportGenerator에 데이터 로드
            success = report_generator.load_employee_data(file_path)
            
            if success:
                # 데이터 통계
                employee_stats = {
                    "total_employees": len(df),
                    "unique_employees": df['employee_id'].nunique(),
                    "columns": len(df.columns),
                    "column_names": list(df.columns)
                }
                
                # 부서별 통계 (부서 컬럼이 있는 경우)
                if 'Department' in df.columns:
                    employee_stats["departments"] = df['Department'].value_counts().to_dict()
                
                return jsonify({
                    "success": True,
                    "message": "직원 데이터가 성공적으로 업로드되고 로드되었습니다.",
                    "file_info": {
                        "original_filename": filename,
                        "saved_filename": new_filename,
                        "upload_path": upload_dir,
                        "file_path": file_path,
                        "latest_link": latest_link
                    },
                    "employee_stats": employee_stats,
                    "note": "이제 에이전트 점수를 설정하고 레포트를 생성할 수 있습니다."
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "데이터 업로드는 성공했지만 시스템 로드에 실패했습니다."
                }), 500
                
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"데이터 파일 처리 오류: {str(e)}"
            }), 400
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"파일 업로드 오류: {str(e)}"
        }), 500


@app.route('/get_employee_list', methods=['GET'])
def get_employee_list():
    """로드된 직원 목록 조회"""
    try:
        if report_generator.employee_data is None:
            return jsonify({
                'success': False,
                'error': '직원 데이터가 로드되지 않았습니다.'
            }), 400
        
        # 직원 목록 추출
        if 'employee_id' in report_generator.employee_data.columns:
            employee_list = report_generator.employee_data['employee_id'].tolist()
        else:
            # employee_id 컬럼이 없으면 인덱스 사용
            employee_list = report_generator.employee_data.index.tolist()
        
        return jsonify({
            'success': True,
            'total_employees': len(employee_list),
            'employee_ids': employee_list[:100],  # 처음 100명만 반환
            'has_more': len(employee_list) > 100
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'직원 목록 조회 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    print("🚀 Integration Flask 서버 시작")
    print(f"📁 데이터 디렉토리: {DATA_DIR}")
    print(f"📁 출력 디렉토리: {OUTPUT_DIR}")
    print("🌐 서버 주소: http://localhost:5007")
    
    # API 키 상태 확인 (Sentio/Agora와 동일)
    if report_generator.client:
        print("✅ OpenAI API 연결 성공 - LLM 기반 분석 가능")
    else:
        print("⚠️  OpenAI API 키 없음 - 기본 분석 모드로 동작")
        print("   💡 .env 파일에 OPENAI_API_KEY를 설정하거나 /set_api_key 엔드포인트를 사용하세요")
    print("\n사용 가능한 엔드포인트:")
    print("  GET  /health - 서버 상태 확인")
    print("  POST /set_api_key - OpenAI API 키 설정")
    print("  POST /load_data - 데이터 로드")
    print("  POST /calculate_thresholds - 임계값 계산")
    print("  POST /optimize_weights - 가중치 최적화")
    print("  POST /predict_employee - 개별 직원 예측")
    print("  GET  /get_results - 현재 결과 조회")
    print("  POST /compare_methods - 최적화 방법 비교")
    print("  POST /export_results - 결과 내보내기")
    print("  POST /load_employee_data - 직원 기본 데이터 로드")
    print("  GET  /get_employee_list - 직원 목록 조회")
    print("  POST /generate_report - 개별 직원 레포트 생성 (LLM 지원)")
    print("  POST /generate_batch_reports - 일괄 레포트 생성")
    
    app.run(host='0.0.0.0', port=5007, debug=True)
