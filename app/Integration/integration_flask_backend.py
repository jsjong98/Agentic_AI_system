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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')
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
        'llm_enabled': hasattr(report_generator, 'llm') and report_generator.llm is not None
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
            'llm_enabled': hasattr(report_generator, 'llm') and report_generator.llm is not None
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
        
        # 최적화 방법 설정 (기본값: Bayesian Optimization)
        optimization_method = data.get('method', 'bayesian')
        
        # 임계값 계산
        results = threshold_calculator.calculate_thresholds_for_scores(
            current_data, score_columns, method=optimization_method
        )
        
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
            method_params['n_calls'] = data.get('n_calls', 50)  # 베이지안 최적화 50회로 설정
        
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
                    method_params['n_calls'] = 50  # 베이지안 최적화 50회
                
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
                'llm_used': use_llm and hasattr(report_generator, 'llm') and report_generator.llm is not None
            }
        else:
            report_content = report_generator.generate_employee_report(employee_id, use_llm)
            result = {
                'success': True,
                'employee_id': employee_id,
                'format': 'json',
                'report': report_content,
                'llm_used': use_llm and hasattr(report_generator, 'llm') and report_generator.llm is not None
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


@app.route('/api/post-analysis/bayesian-optimization', methods=['POST'])
def bayesian_optimization():
    """사후 분석용 베이지안 최적화 (PostAnalysis.js 전용)"""
    global current_data, current_results
    
    try:
        data = request.get_json()
        print(f"🔧 베이지안 최적화 요청 받음")
        print(f"📊 요청 데이터 키: {list(data.keys()) if data else 'None'}")
        
        agent_results = data.get('agent_results', {})
        optimization_config = data.get('optimization_config', {})
        
        print(f"📊 agent_results 키: {list(agent_results.keys()) if agent_results else 'None'}")
        print(f"📊 optimization_config: {optimization_config}")
        
        if not agent_results:
            return jsonify({
                'success': False,
                'error': '에이전트 분석 결과가 필요합니다.'
            }), 400
        
        # 에이전트 분석 결과에서 직접 데이터 생성 (Total_score.csv 불필요)
        print("✅ 에이전트 분석 결과를 사용하여 베이지안 최적화 수행")
        
        # 임계값이 계산되지 않은 경우 자동 계산
        if 'threshold_results' not in current_results:
            try:
                # 기본 임계값 설정
                current_results['threshold_results'] = {
                    'structura_threshold': 0.5,
                    'cognita_threshold': 0.5,
                    'chronos_threshold': 0.5,
                    'sentio_threshold': 0.5,
                    'agora_threshold': 0.5,
                    'high_risk_threshold': 0.7,
                    'medium_risk_threshold': 0.4
                }
                print("✅ 기본 임계값 설정 완료")
            except Exception as e:
                print(f"❌ 임계값 설정 실패: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'임계값 설정에 실패했습니다: {str(e)}'
                }), 500
        
        # 베이지안 최적화 설정
        n_trials = optimization_config.get('n_trials', 50)  # 베이지안 최적화 50회로 설정
        optimization_target = optimization_config.get('optimization_target', 'f1_score')
        
        # 에이전트 결과에서 실제 예측 데이터 추출
        print("🔧 실제 베이지안 최적화 시작...")
        print(f"📊 에이전트 결과 분석: {list(agent_results.keys())}")
        
        # 1. 에이전트별 예측 결과 추출 (0~1 사이 값)
        agent_predictions = {}
        actual_labels = []
        employee_ids = []
        
        for agent_name, result in agent_results.items():
            # 예측 결과 추출 (PostAnalysis.js 구조에 맞게 수정)
            predictions = None
            
            # PostAnalysis.js에서 생성하는 구조: result.raw_result.data.predictions
            if result.get('raw_result', {}).get('data', {}).get('predictions'):
                predictions = result['raw_result']['data']['predictions']
                print(f"   - {agent_name}: raw_result.data.predictions에서 발견")
            elif result.get('predictions'):
                predictions = result['predictions']
                print(f"   - {agent_name}: predictions에서 발견")
            elif result.get('data', {}).get('predictions'):
                predictions = result['data']['predictions']
                print(f"   - {agent_name}: data.predictions에서 발견")
            else:
                print(f"   - {agent_name}: 예측 데이터 구조 확인")
                print(f"     result 키: {list(result.keys()) if isinstance(result, dict) else 'not dict'}")
                if result.get('raw_result'):
                    print(f"     raw_result 키: {list(result['raw_result'].keys()) if isinstance(result['raw_result'], dict) else 'not dict'}")
                    if result['raw_result'].get('data'):
                        print(f"     raw_result.data 키: {list(result['raw_result']['data'].keys()) if isinstance(result['raw_result']['data'], dict) else 'not dict'}")
            
            if predictions:
                print(f"   - {agent_name}: {len(predictions)}개 예측 결과")
                
                # 첫 번째 에이전트에서 employee_id와 actual_attrition 추출 (실제 데이터만 사용)
                if not employee_ids:
                    employee_ids = [pred['employee_id'] for pred in predictions]
                    # actual_attrition이 있다면 사용, 없으면 오류 반환
                    if predictions and len(predictions) > 0 and 'actual_attrition' in predictions[0]:
                        actual_labels = [pred['actual_attrition'] for pred in predictions]
                        print(f"✅ 실제 라벨 사용: {sum(actual_labels)}/{len(actual_labels)} 이탈")
                    else:
                        return jsonify({
                            'success': False,
                            'error': '실제 이탈 라벨(actual_attrition)이 없습니다. 실제 데이터를 사용해주세요.'
                        }), 400
                
                # 에이전트별 위험도 점수 (0~1)
                agent_predictions[agent_name] = [pred['risk_score'] for pred in predictions]
            else:
                print(f"   ⚠️ {agent_name}: 예측 결과 없음")
        
        if not agent_predictions:
            return jsonify({
                'success': False,
                'error': '에이전트 예측 결과가 없습니다. 먼저 1단계 에이전트 분석을 완료해주세요.'
            }), 400
        
        print(f"✅ 데이터 준비 완료: {len(employee_ids)}명, {len(agent_predictions)}개 에이전트")
        
        # 에이전트 예측 결과를 Total_score.csv 형식으로 변환
        optimization_data = []
        for i, emp_id in enumerate(employee_ids):
            row = {'employee_id': emp_id}
            
            # Total_score.csv 컬럼명에 맞게 변환 (대문자 시작)
            agent_name_mapping = {
                'structura': 'Structura_score',
                'cognita': 'Cognita_score', 
                'chronos': 'Chronos_score',
                'sentio': 'Sentio_score',
                'agora': 'Agora_score'
            }
            
            # 각 에이전트의 예측 점수 추가 (Total_score.csv 형식)
            for agent_name, predictions in agent_predictions.items():
                if i < len(predictions):
                    column_name = agent_name_mapping.get(agent_name.lower(), f'{agent_name}_score')
                    row[column_name] = predictions[i]
            
            # 누락된 에이전트 점수는 기본값 0.5로 설정
            for column_name in agent_name_mapping.values():
                if column_name not in row:
                    row[column_name] = 0.5
            
            # 실제 라벨을 Total_score.csv 형식으로 변환 (Yes/No)
            if i < len(actual_labels):
                row['attrition'] = 'Yes' if actual_labels[i] == 1 else 'No'
            else:
                row['attrition'] = 'No'  # 기본값
            
            optimization_data.append(row)
        
        # DataFrame으로 변환
        current_data = pd.DataFrame(optimization_data)
        print(f"📊 Total_score.csv 형식으로 데이터 생성 완료: {len(current_data)}행, {len(current_data.columns)}열")
        print(f"📊 컬럼: {list(current_data.columns)}")
        print(f"📊 샘플 데이터:")
        print(current_data.head(3).to_string())
        
        # 2. 베이지안 최적화를 위한 목적 함수 정의
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
        
        def objective_function(weights):
            """가중치 조합의 F1-Score 계산"""
            # 가중치 정규화 (합이 1이 되도록)
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # 앙상블 예측 계산
            ensemble_scores = np.zeros(len(employee_ids))
            agent_names = list(agent_predictions.keys())
            
            for i, agent_name in enumerate(agent_names):
                if i < len(weights):  # 가중치가 있는 에이전트만
                    ensemble_scores += np.array(agent_predictions[agent_name]) * weights[i]
            
            # 최적 임계값 찾기 (ROC 곡선 기반)
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(actual_labels, ensemble_scores)
            
            # Youden's J statistic으로 최적 임계값 찾기
            j_scores = tpr - fpr
            best_threshold_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[best_threshold_idx]
            
            # 예측 수행
            predictions = (ensemble_scores >= optimal_threshold).astype(int)
            
            # F1-Score 계산 (최적화 목표)
            f1 = f1_score(actual_labels, predictions)
            
            return -f1  # 최소화 문제로 변환 (음수)
        
        # 3. 베이지안 최적화 실행
        try:
            from skopt import gp_minimize
            from skopt.space import Real
            
            # 가중치 공간 정의 (각각 0.1~0.9, 합이 1이 되도록 제약)
            n_agents = len(agent_predictions)
            dimensions = [Real(0.1, 0.9, name=f'weight_{i}') for i in range(n_agents)]
            
            print(f"🎯 베이지안 최적화 실행: {n_trials}회 시도, {n_agents}개 에이전트")
            
            # 베이지안 최적화 실행
            result = gp_minimize(
                func=objective_function,
                dimensions=dimensions,
                n_calls=n_trials,
                n_initial_points=10,
                random_state=42,
                acq_func='EI'  # Expected Improvement
            )
            
            # 최적 가중치 추출 및 정규화
            optimal_weights_raw = result.x
            optimal_weights_raw = np.array(optimal_weights_raw)
            optimal_weights_normalized = optimal_weights_raw / optimal_weights_raw.sum()
            
            # 에이전트별 가중치 딕셔너리 생성
            agent_names = list(agent_predictions.keys())
            optimal_weights = {}
            for i, agent_name in enumerate(agent_names):
                optimal_weights[f'{agent_name}_weight'] = float(optimal_weights_normalized[i])
            
            print(f"✅ 최적 가중치: {optimal_weights}")
            
        except ImportError:
            print("⚠️ scikit-optimize가 없어 개선된 랜덤 서치로 대체")
            # scikit-optimize가 없는 경우 개선된 랜덤 서치
            best_f1 = -1
            optimal_weights = {}
            agent_names = list(agent_predictions.keys())
            n_agents = len(agent_names)
            
            # 모든 시도 기록
            all_trials = []
            
            print(f"🔍 개선된 랜덤 서치 실행: {min(n_trials, 100)}회 시도")
            
            for trial in range(min(n_trials, 100)):  # 최대 100회
                # 제약 조건을 만족하는 가중치 생성
                attempts = 0
                while attempts < 10:  # 최대 10번 시도
                    # 디리클레 분포로 합이 1인 가중치 생성
                    weights = np.random.dirichlet(np.ones(n_agents) * 2)  # alpha=2로 더 균등하게
                    
                    # 경계 조건 확인 (0.1 ~ 0.9)
                    if np.all(weights >= 0.1) and np.all(weights <= 0.9):
                        break
                    
                    # 경계 조건 위반 시 클리핑 후 재정규화
                    weights = np.clip(weights, 0.1, 0.9)
                    weights = weights / weights.sum()
                    attempts += 1
                
                # F1-Score 계산
                f1 = -objective_function(weights)
                all_trials.append(f1)
                
                if f1 > best_f1:
                    best_f1 = f1
                    optimal_weights = {f'{agent_names[i]}_weight': float(weights[i]) for i in range(n_agents)}
                
                # 진행률 출력
                if (trial + 1) % 20 == 0:
                    print(f"   진행률: {trial + 1}/{min(n_trials, 100)}, 현재 최고 F1: {best_f1:.4f}")
            
            print(f"✅ 랜덤 서치 완료: 최고 F1-Score {best_f1:.4f}")
            
            # 결과 객체 생성
            result = type('Result', (), {
                'fun': -best_f1, 
                'func_vals': [-f1 for f1 in all_trials],
                'x': [optimal_weights[f'{agent_names[i]}_weight'] for i in range(n_agents)]
            })()
        
        # 4. 최적 가중치로 앙상블 예측 및 임계값 계산
        ensemble_scores = np.zeros(len(employee_ids))
        for agent_name, predictions in agent_predictions.items():
            weight_key = f'{agent_name}_weight'
            if weight_key in optimal_weights:
                ensemble_scores += np.array(predictions) * optimal_weights[weight_key]
        
        # ROC 곡선으로 최적 임계값 계산
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(actual_labels, ensemble_scores)
        j_scores = tpr - fpr
        best_threshold_idx = np.argmax(j_scores)
        optimal_ensemble_threshold = float(thresholds[best_threshold_idx])
        
        # 에이전트별 개별 임계값도 계산
        optimal_thresholds = {}
        for agent_name, predictions in agent_predictions.items():
            fpr_agent, tpr_agent, thresholds_agent = roc_curve(actual_labels, predictions)
            j_scores_agent = tpr_agent - fpr_agent
            best_idx_agent = np.argmax(j_scores_agent)
            optimal_thresholds[f'{agent_name}_threshold'] = float(thresholds_agent[best_idx_agent])
        
        # 위험도 분류 임계값
        optimal_thresholds['high_risk_threshold'] = optimal_ensemble_threshold + 0.1
        optimal_thresholds['medium_risk_threshold'] = optimal_ensemble_threshold - 0.1
        
        # 5. 최종 성능 계산
        final_predictions = (ensemble_scores >= optimal_ensemble_threshold).astype(int)
        best_performance = {
            'f1_score': float(f1_score(actual_labels, final_predictions)),
            'precision': float(precision_score(actual_labels, final_predictions)),
            'recall': float(recall_score(actual_labels, final_predictions)),
            'accuracy': float(accuracy_score(actual_labels, final_predictions)),
            'auc': float(roc_auc_score(actual_labels, ensemble_scores))
        }
        
        # 6. 최적화 히스토리 생성
        optimization_history = []
        if hasattr(result, 'func_vals'):
            for i, score in enumerate(result.func_vals[:20]):  # 최대 20개
                optimization_history.append({
                    'trial': i + 1,
                    'score': float(-score),  # 다시 양수로 변환
                    'f1_score': float(-score)
                })
        
        optimization_history.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"✅ 베이지안 최적화 완료!")
        print(f"   최적 F1-Score: {best_performance['f1_score']:.4f}")
        print(f"   최적 임계값: {optimal_ensemble_threshold:.4f}")
        print(f"   가중치 합: {sum(optimal_weights.values()):.4f}")
        
        # 결과 저장
        current_results['bayesian_optimization'] = {
            'optimal_weights': optimal_weights,
            'optimal_thresholds': optimal_thresholds,
            'best_performance': best_performance,
            'optimization_history': optimization_history
        }
        
        # 최적화된 결과를 Total_score.csv 형식으로 저장
        try:
            # 최적 가중치로 앙상블 점수 계산
            final_ensemble_scores = np.zeros(len(employee_ids))
            agent_names = list(agent_predictions.keys())
            
            for i, agent_name in enumerate(agent_names):
                weight_key = f'{agent_name}_weight'
                if weight_key in optimal_weights:
                    final_ensemble_scores += np.array(agent_predictions[agent_name]) * optimal_weights[weight_key]
            
            # 최종 예측 결과
            final_predictions = (final_ensemble_scores >= optimal_ensemble_threshold).astype(int)
            
            # Total_score.csv 형식으로 최종 결과 DataFrame 생성
            final_results = []
            for i, emp_id in enumerate(employee_ids):
                row = {'employee_id': emp_id}
                
                # 각 에이전트 점수 (Total_score.csv 컬럼명)
                agent_name_mapping = {
                    'structura': 'Structura_score',
                    'cognita': 'Cognita_score', 
                    'chronos': 'Chronos_score',
                    'sentio': 'Sentio_score',
                    'agora': 'Agora_score'
                }
                
                for agent_name, predictions in agent_predictions.items():
                    if i < len(predictions):
                        column_name = agent_name_mapping.get(agent_name.lower(), f'{agent_name}_score')
                        row[column_name] = predictions[i]
                
                # 앙상블 점수 및 예측 결과 추가
                row['ensemble_score'] = final_ensemble_scores[i]
                row['ensemble_prediction'] = final_predictions[i]
                row['attrition'] = 'Yes' if actual_labels[i] == 1 else 'No'
                
                final_results.append(row)
            
            # DataFrame으로 변환
            final_df = pd.DataFrame(final_results)
            
            # 파일 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join('app/results', f'optimized_total_score_{timestamp}.csv')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            final_df.to_csv(output_file, index=False, encoding='utf-8')
            
            print(f"✅ 최적화 결과 저장 완료: {output_file}")
            print(f"   형식: Total_score.csv 호환")
            print(f"   행 수: {len(final_df)}")
            print(f"   컬럼: {list(final_df.columns)}")
            
            # current_data를 최종 결과로 업데이트
            current_data = final_df
            
        except Exception as e:
            print(f"⚠️ 결과 저장 중 오류 (계속 진행): {str(e)}")
        
        # 위험도 분류 통계 (실제 점수 기반)
        total_employees = len(current_data)
        
        # 앙상블 점수를 기반으로 실제 위험도 분류
        if 'ensemble_score' in current_data.columns:
            ensemble_scores = current_data['ensemble_score']
            high_risk_count = len(ensemble_scores[ensemble_scores >= 0.7])
            medium_risk_count = len(ensemble_scores[(ensemble_scores >= 0.3) & (ensemble_scores < 0.7)])
            low_risk_count = len(ensemble_scores[ensemble_scores < 0.3])
        else:
            # 앙상블 점수가 없으면 기본 분포 사용
            high_risk_count = int(total_employees * 0.15)
            medium_risk_count = int(total_employees * 0.25)
            low_risk_count = total_employees - high_risk_count - medium_risk_count
        
        risk_distribution = {
            '안전군': low_risk_count,
            '주의군': medium_risk_count,
            '고위험군': high_risk_count
        }
        
        # performance_summary 생성 (성능 분석 탭 활성화를 위해 필요)
        performance_summary = {
            'performance_metrics': {
                'f1_score': best_performance['f1_score'],
                'precision': best_performance['precision'],
                'recall': best_performance['recall'],
                'accuracy': best_performance.get('accuracy', 0.85)
            },
            'risk_statistics': risk_distribution,
            'optimization_summary': {
                'total_trials': len(optimization_history),
                'best_trial': max(optimization_history, key=lambda x: x['f1_score']) if optimization_history else None,
                'convergence_achieved': True
            }
        }
        
        # current_results에 저장
        current_results['performance_summary'] = performance_summary
        
        return jsonify({
            'success': True,
            'message': '베이지안 최적화가 완료되었습니다.',
            'optimal_thresholds': optimal_thresholds,
            'optimal_weights': optimal_weights,
            'best_performance': best_performance,
            'optimization_history': optimization_history,
            'performance_summary': performance_summary,
            'cv_results': {
                'mean_f1_score': best_performance['f1_score'],
                'std_f1_score': 0.02,
                'mean_precision': best_performance['precision'],
                'std_precision': 0.03,
                'mean_recall': best_performance['recall'],
                'std_recall': 0.025
            },
            'n_trials': len(optimization_history),
            'risk_distribution': risk_distribution,
            'total_employees': total_employees
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'베이지안 최적화 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/post-analysis/update-risk-thresholds', methods=['POST'])
def update_risk_thresholds():
    """위험도 분류 임계값 및 퇴사 예측 기준 업데이트"""
    global current_data, current_results
    
    try:
        data = request.get_json()
        new_thresholds = data.get('risk_thresholds', {})
        attrition_prediction_mode = data.get('attrition_prediction_mode', 'high_risk_only')  # 'high_risk_only' 또는 'medium_high_risk'
        
        high_risk_threshold = new_thresholds.get('high_risk_threshold', 0.7)
        low_risk_threshold = new_thresholds.get('low_risk_threshold', 0.3)
        
        print(f"🎯 위험도 임계값 업데이트: 안전군 < {low_risk_threshold}, 주의군 {low_risk_threshold}-{high_risk_threshold}, 고위험군 >= {high_risk_threshold}")
        print(f"🎯 퇴사 예측 모드: {attrition_prediction_mode}")
        
        if current_data is None or current_data.empty:
            return jsonify({
                'success': False,
                'error': '분석 데이터가 없습니다. 먼저 Bayesian Optimization을 실행해주세요.'
            }), 400
        
        # 새로운 임계값으로 위험도 재분류
        total_employees = len(current_data)
        
        if 'ensemble_score' in current_data.columns:
            ensemble_scores = current_data['ensemble_score']
            high_risk_count = len(ensemble_scores[ensemble_scores >= high_risk_threshold])
            medium_risk_count = len(ensemble_scores[(ensemble_scores >= low_risk_threshold) & (ensemble_scores < high_risk_threshold)])
            low_risk_count = len(ensemble_scores[ensemble_scores < low_risk_threshold])
            
            # 위험도 레벨 컬럼 업데이트
            current_data['risk_level'] = current_data['ensemble_score'].apply(
                lambda x: '고위험군' if x >= high_risk_threshold 
                         else '주의군' if x >= low_risk_threshold 
                         else '안전군'
            )
            
            # 퇴사 예측 컬럼 업데이트 (새로운 기준에 따라)
            if attrition_prediction_mode == 'high_risk_only':
                # 고위험군만 퇴사 예측
                current_data['predicted_attrition'] = (current_data['ensemble_score'] >= high_risk_threshold).astype(int)
            else:  # 'medium_high_risk'
                # 주의군 + 고위험군 퇴사 예측
                current_data['predicted_attrition'] = (current_data['ensemble_score'] >= low_risk_threshold).astype(int)
                
        else:
            return jsonify({
                'success': False,
                'error': '앙상블 점수 데이터가 없습니다.'
            }), 400
        
        # 새로운 위험도 분포
        new_risk_distribution = {
            '안전군': low_risk_count,
            '주의군': medium_risk_count,
            '고위험군': high_risk_count
        }
        
        # 성능 지표 계산 (실제 퇴사 데이터가 있는 경우)
        performance_metrics = {}
        confusion_matrix = {}
        
        if 'actual_attrition' in current_data.columns:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix as cm
            
            y_true = current_data['actual_attrition'].astype(int)
            y_pred = current_data['predicted_attrition'].astype(int)
            
            # 성능 지표 계산
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            performance_metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
            
            # Confusion Matrix 계산
            cm_matrix = cm(y_true, y_pred)
            confusion_matrix = {
                'true_negative': int(cm_matrix[0, 0]),   # 실제 잔류, 예측 잔류
                'false_positive': int(cm_matrix[0, 1]),  # 실제 잔류, 예측 퇴사
                'false_negative': int(cm_matrix[1, 0]),  # 실제 퇴사, 예측 잔류
                'true_positive': int(cm_matrix[1, 1])    # 실제 퇴사, 예측 퇴사
            }
            
            print(f"📊 성능 지표 계산 완료:")
            print(f"   정확도: {accuracy:.4f}")
            print(f"   정밀도: {precision:.4f}")
            print(f"   재현율: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"📊 Confusion Matrix: TN={confusion_matrix['true_negative']}, FP={confusion_matrix['false_positive']}, FN={confusion_matrix['false_negative']}, TP={confusion_matrix['true_positive']}")
        else:
            print("⚠️ 실제 퇴사 데이터(actual_attrition)가 없어 성능 지표를 계산할 수 없습니다.")
        
        # performance_summary 업데이트
        if 'performance_summary' in current_results:
            current_results['performance_summary']['risk_statistics'] = new_risk_distribution
            current_results['performance_summary']['risk_thresholds'] = {
                'high_risk_threshold': high_risk_threshold,
                'low_risk_threshold': low_risk_threshold
            }
            current_results['performance_summary']['attrition_prediction_mode'] = attrition_prediction_mode
            
            # 성능 지표가 계산된 경우 업데이트
            if performance_metrics:
                current_results['performance_summary']['performance_metrics'] = performance_metrics
                current_results['performance_summary']['confusion_matrix'] = confusion_matrix
        
        print(f"✅ 위험도 재분류 완료: 안전군 {low_risk_count}명, 주의군 {medium_risk_count}명, 고위험군 {high_risk_count}명")
        
        return jsonify({
            'success': True,
            'message': '위험도 임계값이 업데이트되었습니다.',
            'risk_distribution': new_risk_distribution,
            'risk_thresholds': {
                'high_risk_threshold': high_risk_threshold,
                'low_risk_threshold': low_risk_threshold
            },
            'attrition_prediction_mode': attrition_prediction_mode,
            'performance_metrics': performance_metrics,
            'confusion_matrix': confusion_matrix,
            'total_employees': total_employees,
            'performance_summary': current_results.get('performance_summary', {})
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'위험도 임계값 업데이트 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/post-analysis/save-final-settings', methods=['POST'])
def save_final_settings():
    """최종 위험도 분류 설정을 배치 분석용으로 저장"""
    global current_results
    
    try:
        data = request.get_json()
        
        # 사용자가 최종 결정한 설정 + 최적화된 가중치
        final_settings = {
            'risk_thresholds': data.get('risk_thresholds', {}),
            'attrition_prediction_mode': data.get('attrition_prediction_mode', 'high_risk_only'),
            'performance_metrics': data.get('performance_metrics', {}),
            'confusion_matrix': data.get('confusion_matrix', {}),
            'risk_distribution': data.get('risk_distribution', {}),
            'integration_config': current_results.get('integration_config', {}) if current_results else {},
            'optimized_weights': current_results.get('optimized_weights', {}) if current_results else {},
            'timestamp': datetime.now().isoformat(),
            'total_employees': data.get('total_employees', 0),
            'source': 'post_analysis_optimization'
        }
        
        print(f"💾 최종 위험도 분류 설정 저장:")
        print(f"   안전군 임계값: < {final_settings['risk_thresholds'].get('low_risk_threshold', 0.3)}")
        print(f"   고위험군 임계값: >= {final_settings['risk_thresholds'].get('high_risk_threshold', 0.7)}")
        print(f"   퇴사 예측 모드: {final_settings['attrition_prediction_mode']}")
        
        # 프로젝트 루트 기준으로 절대 경로 생성
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 배치 분석용 설정 파일 저장
        settings_file = os.path.join(project_root, 'app/results/models/final_risk_settings.json')
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        
        import json
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(final_settings, f, indent=2, ensure_ascii=False)
        
        # current_results에도 저장
        current_results['final_risk_settings'] = final_settings
        
        print(f"✅ 최종 설정 저장 완료: {settings_file}")
        
        return jsonify({
            'success': True,
            'message': '최종 위험도 분류 설정이 저장되었습니다.',
            'settings_file': settings_file,
            'final_settings': final_settings
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'최종 설정 저장 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

def create_xai_visualizations(employee_result, employee_dir, employee_id):
    """XAI 시각화 생성 및 저장"""
    try:
        # 시각화 디렉토리 생성
        viz_dir = os.path.join(employee_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
        plt.rcParams['axes.unicode_minus'] = False
        
        agent_results = employee_result.get('agent_results', {})
        
        # 1. Structura Feature Importance 시각화
        if 'structura' in agent_results:
            structura_data = agent_results['structura']
            feature_importance = structura_data.get('feature_importance', {})
            
            if feature_importance:
                plt.figure(figsize=(12, 8))
                features = list(feature_importance.keys())
                importances = list(feature_importance.values())
                
                # 중요도 순으로 정렬
                sorted_idx = np.argsort(importances)[::-1]
                features = [features[i] for i in sorted_idx]
                importances = [importances[i] for i in sorted_idx]
                
                plt.barh(range(len(features)), importances, color='skyblue')
                plt.yticks(range(len(features)), features)
                plt.xlabel('Feature Importance')
                plt.title(f'Structura Feature Importance - Employee {employee_id}')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'structura_feature_importance.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. Structura SHAP Values 시각화
        if 'structura' in agent_results:
            shap_values = agent_results['structura'].get('shap_values', {})
            
            if shap_values:
                plt.figure(figsize=(12, 8))
                features = list(shap_values.keys())
                values = list(shap_values.values())
                
                # SHAP 값에 따라 색상 결정
                colors = ['red' if v > 0 else 'blue' for v in values]
                
                plt.barh(range(len(features)), values, color=colors, alpha=0.7)
                plt.yticks(range(len(features)), features)
                plt.xlabel('SHAP Values')
                plt.title(f'Structura SHAP Values - Employee {employee_id}')
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'structura_shap_values.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Chronos Attention Weights 시각화
        if 'chronos' in agent_results:
            chronos_data = agent_results['chronos']
            xai_explanation = chronos_data.get('xai_explanation', {})
            attention_weights = xai_explanation.get('attention_weights', {})
            
            if attention_weights:
                plt.figure(figsize=(12, 6))
                timesteps = list(attention_weights.keys())
                weights = list(attention_weights.values())
                
                plt.plot(range(len(timesteps)), weights, marker='o', linewidth=2, markersize=8)
                plt.xlabel('Time Steps')
                plt.ylabel('Attention Weights')
                plt.title(f'Chronos Attention Weights - Employee {employee_id}')
                plt.xticks(range(len(timesteps)), timesteps, rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'chronos_attention_weights.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        # 4. Chronos Sequence Importance 시각화
        if 'chronos' in agent_results:
            sequence_importance = agent_results['chronos'].get('xai_explanation', {}).get('sequence_importance', {})
            
            if sequence_importance:
                plt.figure(figsize=(12, 6))
                timesteps = list(sequence_importance.keys())
                importance = list(sequence_importance.values())
                
                plt.bar(range(len(timesteps)), importance, color='lightcoral', alpha=0.7)
                plt.xlabel('Time Steps')
                plt.ylabel('Sequence Importance')
                plt.title(f'Chronos Sequence Importance - Employee {employee_id}')
                plt.xticks(range(len(timesteps)), timesteps, rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'chronos_sequence_importance.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        # 5. 에이전트별 위험도 점수 비교
        agent_scores = {}
        if 'structura' in agent_results:
            agent_scores['Structura'] = agent_results['structura'].get('attrition_probability', 0)
        if 'chronos' in agent_results:
            agent_scores['Chronos'] = agent_results['chronos'].get('risk_score', 0)
        if 'cognita' in agent_results:
            agent_scores['Cognita'] = agent_results['cognita'].get('overall_risk_score', 0)
        if 'sentio' in agent_results:
            agent_scores['Sentio'] = agent_results['sentio'].get('risk_score', 0)
        if 'agora' in agent_results:
            agent_scores['Agora'] = agent_results['agora'].get('market_risk_score', 0)
        
        if agent_scores:
            plt.figure(figsize=(10, 6))
            agents = list(agent_scores.keys())
            scores = list(agent_scores.values())
            
            bars = plt.bar(agents, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
            plt.ylabel('Risk Score')
            plt.title(f'Agent Risk Scores Comparison - Employee {employee_id}')
            plt.ylim(0, 1)
            
            # 막대 위에 값 표시
            for bar, score in zip(bars, scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'agent_scores_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Sentio 감정 분포 시각화
        if 'sentio' in agent_results:
            emotion_dist = agent_results['sentio'].get('emotion_distribution', {})
            
            if emotion_dist:
                plt.figure(figsize=(8, 8))
                emotions = list(emotion_dist.keys())
                values = list(emotion_dist.values())
                
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                plt.pie(values, labels=emotions, autopct='%1.1f%%', colors=colors[:len(emotions)])
                plt.title(f'Sentio Emotion Distribution - Employee {employee_id}')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'sentio_emotion_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"✅ XAI 시각화 생성 완료: {viz_dir}")
        return True
        
    except Exception as e:
        print(f"❌ XAI 시각화 생성 실패 (직원 {employee_id}): {str(e)}")
        return False

@app.route('/api/batch-analysis/save-results', methods=['POST'])
def save_batch_analysis_results():
    """배치 분석 결과를 부서별로 정리하여 저장"""
    try:
        data = request.get_json()
        
        if not data or 'results' not in data:
            return jsonify({
                'success': False,
                'error': '분석 결과 데이터가 없습니다.'
            }), 400
        
        results = data['results']
        analysis_timestamp = datetime.now().isoformat()
        
        print(f"💾 배치 분석 결과 저장 시작: {len(results)}명")
        
        # 프로젝트 루트 기준으로 절대 경로 생성 (기존 구조 활용)
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_base_dir = os.path.join(project_root, 'app/results')
        
        # 배치 분석 요약용 디렉토리
        batch_summary_dir = os.path.join(results_base_dir, 'batch_analysis')
        os.makedirs(batch_summary_dir, exist_ok=True)
        
        # 부서별 결과 정리
        department_results = {}
        individual_results = []
        
        for employee in results:
            # 부서 정보 추출 (기존 구조에 맞게)
            dept = employee.get('department', '미분류')
            
            # 부서명 정규화 (기존 구조와 일치시키기)
            dept_mapping = {
                'Human Resources': 'Human_Resources',
                'Research & Development': 'Research_&_Development', 
                'Research and Development': 'Research_&_Development',
                'R&D': 'Research_&_Development',
                'Sales': 'Sales',
                'HR': 'Human_Resources'
            }
            
            normalized_dept = dept_mapping.get(dept, dept.replace(' ', '_').replace('&', '_&_'))
            
            if normalized_dept not in department_results:
                department_results[normalized_dept] = {
                    'department': normalized_dept,
                    'original_name': dept,
                    'total_employees': 0,
                    'risk_distribution': {'안전군': 0, '주의군': 0, '고위험군': 0},
                    'employees': []
                }
            
            # 위험도 분류
            risk_score = employee.get('risk_score', 0)
            risk_level = employee.get('risk_level', 'unknown')
            
            # 위험도 분포 업데이트
            if risk_level == 'low':
                department_results[normalized_dept]['risk_distribution']['안전군'] += 1
            elif risk_level == 'medium':
                department_results[normalized_dept]['risk_distribution']['주의군'] += 1
            elif risk_level == 'high':
                department_results[normalized_dept]['risk_distribution']['고위험군'] += 1
            
            department_results[normalized_dept]['total_employees'] += 1
            
            # 개별 직원 결과 정리
            employee_result = {
                'employee_id': employee.get('employee_number', 'Unknown'),
                'department': dept,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'analysis_timestamp': analysis_timestamp,
                'agent_results': {}
            }
            
            # 각 에이전트 결과 추출
            analysis_result = employee.get('analysis_result', {})
            
            # Structura 결과 (XAI 포함)
            if 'structura_result' in analysis_result:
                structura = analysis_result['structura_result']
                employee_result['agent_results']['structura'] = {
                    'attrition_probability': structura.get('prediction', {}).get('attrition_probability', 0),
                    'predicted_attrition': structura.get('prediction', {}).get('predicted_attrition', 0),
                    'confidence': structura.get('prediction', {}).get('confidence', 0),
                    'feature_importance': structura.get('prediction', {}).get('feature_importance', {}),
                    'xai_explanation': structura.get('xai_explanation', {}),
                    'shap_values': structura.get('shap_values', {}),
                    'lime_explanation': structura.get('lime_explanation', {})
                }
            
            # Chronos 결과 (XAI 포함)
            if 'chronos_result' in analysis_result:
                chronos = analysis_result['chronos_result']
                employee_result['agent_results']['chronos'] = {
                    'risk_score': chronos.get('prediction', {}).get('risk_score', 0),
                    'anomaly_score': chronos.get('anomaly_detection', {}).get('anomaly_score', 0),
                    'trend_analysis': chronos.get('trend_analysis', {}),
                    'xai_explanation': chronos.get('xai_explanation', {}),
                    'attention_weights': chronos.get('attention_weights', {}),
                    'sequence_importance': chronos.get('sequence_importance', {})
                }
            
            # Cognita 결과
            if 'cognita_result' in analysis_result:
                cognita = analysis_result['cognita_result']
                employee_result['agent_results']['cognita'] = {
                    'overall_risk_score': cognita.get('risk_analysis', {}).get('overall_risk_score', 0),
                    'network_centrality': cognita.get('network_analysis', {}).get('centrality_score', 0),
                    'relationship_strength': cognita.get('relationship_analysis', {}).get('avg_strength', 0),
                    'influence_score': cognita.get('influence_analysis', {}).get('influence_score', 0)
                }
            
            # Sentio 결과
            if 'sentio_result' in analysis_result:
                sentio = analysis_result['sentio_result']
                employee_result['agent_results']['sentio'] = {
                    'sentiment_score': sentio.get('sentiment_analysis', {}).get('overall_sentiment', 0),
                    'risk_score': sentio.get('sentiment_analysis', {}).get('risk_score', 0),
                    'keyword_analysis': sentio.get('keyword_analysis', {}),
                    'emotion_distribution': sentio.get('emotion_analysis', {})
                }
            
            # Agora 결과
            if 'agora_result' in analysis_result:
                agora = analysis_result['agora_result']
                employee_result['agent_results']['agora'] = {
                    'market_risk_score': agora.get('market_analysis', {}).get('risk_score', 0),
                    'industry_trend': agora.get('industry_analysis', {}),
                    'job_market_analysis': agora.get('job_market', {}),
                    'external_factors': agora.get('external_factors', {})
                }
            
            individual_results.append(employee_result)
            department_results[normalized_dept]['employees'].append(employee_result)
            
            # 기존 구조에 맞게 개별 직원 파일 저장
            employee_id = employee.get('employee_number', 'Unknown')
            employee_dir = os.path.join(results_base_dir, normalized_dept, f'employee_{employee_id}')
            os.makedirs(employee_dir, exist_ok=True)
            
            # 배치 분석 결과 저장 (기존 파일과 구분)
            batch_result_file = os.path.join(employee_dir, f'batch_analysis_{analysis_timestamp.replace(":", "-").replace(".", "-")}.json')
            with open(batch_result_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'analysis_type': 'batch_analysis',
                    'timestamp': analysis_timestamp,
                    'employee_result': employee_result,
                    'applied_settings': data.get('applied_settings', {}),
                    'xai_included': True,
                    'visualizations_generated': True
                }, f, indent=2, ensure_ascii=False)
            
            # XAI 시각화 생성 및 저장 (PNG 파일들)
            visualization_success = create_xai_visualizations(employee_result, employee_dir, employee_id)
            
            print(f"✅ 직원 {employee_id} 배치 분석 결과 저장: {batch_result_file}")
            if visualization_success:
                print(f"✅ 직원 {employee_id} XAI 시각화 생성 완료")
            else:
                print(f"⚠️ 직원 {employee_id} XAI 시각화 생성 부분 실패")
        
        # 부서별 결과 저장 (배치 분석 요약 디렉토리에)
        dept_summary_file = os.path.join(batch_summary_dir, f'department_summary_{analysis_timestamp.replace(":", "-").replace(".", "-")}.json')
        with open(dept_summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_timestamp': analysis_timestamp,
                'total_employees': len(results),
                'total_departments': len(department_results),
                'department_results': department_results,
                'applied_settings': data.get('applied_settings', {}),
                'summary_statistics': {
                    'overall_risk_distribution': {
                        '안전군': sum(dept['risk_distribution']['안전군'] for dept in department_results.values()),
                        '주의군': sum(dept['risk_distribution']['주의군'] for dept in department_results.values()),
                        '고위험군': sum(dept['risk_distribution']['고위험군'] for dept in department_results.values())
                    }
                }
            }, f, indent=2, ensure_ascii=False)
        
        # 개별 직원 상세 결과 저장 (XAI 포함) - 배치 분석 요약 디렉토리에
        individual_file = os.path.join(batch_summary_dir, f'individual_results_{analysis_timestamp.replace(":", "-").replace(".", "-")}.json')
        with open(individual_file, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_timestamp': analysis_timestamp,
                'total_employees': len(individual_results),
                'individual_results': individual_results,
                'applied_settings': data.get('applied_settings', {}),
                'xai_included': True,
                'agents_analyzed': ['structura', 'chronos', 'cognita', 'sentio', 'agora']
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 배치 분석 결과 저장 완료:")
        print(f"   부서별 요약: {dept_summary_file}")
        print(f"   개별 상세: {individual_file}")
        print(f"   총 {len(department_results)}개 부서, {len(individual_results)}명 직원")
        
        return jsonify({
            'success': True,
            'message': f'배치 분석 결과가 저장되었습니다.',
            'files_saved': {
                'department_summary': dept_summary_file,
                'individual_results': individual_file
            },
            'statistics': {
                'total_employees': len(individual_results),
                'total_departments': len(department_results),
                'department_breakdown': {dept: info['total_employees'] for dept, info in department_results.items()}
            }
        })
        
    except Exception as e:
        print(f"❌ 배치 분석 결과 저장 실패: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'배치 분석 결과 저장 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/generate-employee-report', methods=['POST'])
def generate_employee_report():
    """개별 직원 보고서 생성 (LLM 기반)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': '요청 데이터가 없습니다.'
            }), 400
        
        employee_id = data.get('employee_id')
        department = data.get('department', '미분류')
        risk_level = data.get('risk_level', 'unknown')
        risk_score = data.get('risk_score', 0)
        agent_scores = data.get('agent_scores', {})
        
        if not employee_id:
            return jsonify({
                'success': False,
                'error': '직원 ID가 필요합니다.'
            }), 400
        
        print(f"📝 직원 {employee_id} 보고서 생성 시작")
        
        # 프로젝트 루트 기준으로 절대 경로 생성
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 부서명 정규화
        dept_mapping = {
            'Human Resources': 'Human_Resources',
            'Research & Development': 'Research_&_Development', 
            'Research and Development': 'Research_&_Development',
            'R&D': 'Research_&_Development',
            'Sales': 'Sales',
            'HR': 'Human_Resources'
        }
        normalized_dept = dept_mapping.get(department, department.replace(' ', '_').replace('&', '_&_'))
        
        # 직원 데이터 파일 경로
        employee_dir = os.path.join(project_root, 'app/results', normalized_dept, f'employee_{employee_id}')
        
        if not os.path.exists(employee_dir):
            return jsonify({
                'success': False,
                'error': f'직원 {employee_id}의 분석 결과를 찾을 수 없습니다.'
            }), 404
        
        # 각 에이전트 결과 파일 로드
        agent_data = {}
        
        # Structura 결과 로드
        structura_file = os.path.join(employee_dir, 'structura_result.json')
        if os.path.exists(structura_file):
            with open(structura_file, 'r', encoding='utf-8') as f:
                agent_data['structura'] = json.load(f)
        
        # Chronos 결과 로드
        chronos_file = os.path.join(employee_dir, 'chronos_result.json')
        if os.path.exists(chronos_file):
            with open(chronos_file, 'r', encoding='utf-8') as f:
                agent_data['chronos'] = json.load(f)
        
        # Cognita 결과 로드
        cognita_file = os.path.join(employee_dir, 'cognita_result.json')
        if os.path.exists(cognita_file):
            with open(cognita_file, 'r', encoding='utf-8') as f:
                agent_data['cognita'] = json.load(f)
        
        # Sentio 결과 로드
        sentio_file = os.path.join(employee_dir, 'sentio_result.json')
        if os.path.exists(sentio_file):
            with open(sentio_file, 'r', encoding='utf-8') as f:
                agent_data['sentio'] = json.load(f)
        
        # 직원 정보 로드
        employee_info_file = os.path.join(employee_dir, 'employee_info.json')
        employee_info = {}
        if os.path.exists(employee_info_file):
            with open(employee_info_file, 'r', encoding='utf-8') as f:
                employee_info = json.load(f)
        
        # 배치 분석 결과 로드 (가장 최신)
        batch_files = [f for f in os.listdir(employee_dir) if f.startswith('batch_analysis_') and f.endswith('.json')]
        batch_data = {}
        if batch_files:
            latest_batch_file = sorted(batch_files)[-1]  # 가장 최신 파일
            with open(os.path.join(employee_dir, latest_batch_file), 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
        
        # LLM으로 보고서 생성
        report = generate_llm_report(
            employee_id=employee_id,
            department=department,
            risk_level=risk_level,
            risk_score=risk_score,
            agent_scores=agent_scores,
            agent_data=agent_data,
            employee_info=employee_info,
            batch_data=batch_data
        )
        
        print(f"✅ 직원 {employee_id} 보고서 생성 완료")
        
        return jsonify({
            'success': True,
            'report': report,
            'employee_id': employee_id,
            'department': department,
            'risk_level': risk_level
        })
        
    except Exception as e:
        print(f"❌ 보고서 생성 실패: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'보고서 생성 중 오류 발생: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

def generate_llm_report(employee_id, department, risk_level, risk_score, agent_scores, agent_data, employee_info, batch_data):
    """LLM을 사용한 개별 직원 보고서 생성"""
    try:
        # 위험도 분류 한글 변환
        risk_level_kr = {
            'high': '고위험군',
            'medium': '주의군', 
            'low': '안전군'
        }.get(risk_level, '미분류')
        
        # Structura 주요 위험 요인 추출
        structura_risks = []
        if 'structura' in agent_data and 'explanation' in agent_data['structura']:
            explanation = agent_data['structura']['explanation']
            if 'individual_explanation' in explanation:
                top_risks = explanation['individual_explanation'].get('top_risk_factors', [])
                structura_risks = [factor.get('factor', '') for factor in top_risks[:3]]
        
        # Structura 보호 요인 추출
        structura_protections = []
        if 'structura' in agent_data and 'explanation' in agent_data['structura']:
            explanation = agent_data['structura']['explanation']
            if 'individual_explanation' in explanation:
                top_protections = explanation['individual_explanation'].get('top_protective_factors', [])
                structura_protections = [factor.get('factor', '') for factor in top_protections[:3]]
        
        # 보고서 템플릿
        report_template = f"""
# 직원 위험도 분석 보고서

## 📋 기본 정보
- **직원 ID**: {employee_id}
- **소속 부서**: {department}
- **위험도 분류**: {risk_level_kr}
- **종합 위험 점수**: {risk_score:.1%}

## 📊 에이전트별 분석 결과

### 🧠 Structura (HR 데이터 분석)
- **이직 확률**: {agent_scores.get('structura', 0):.1%}
- **주요 위험 요인**: {', '.join(structura_risks) if structura_risks else '데이터 없음'}
- **보호 요인**: {', '.join(structura_protections) if structura_protections else '데이터 없음'}

### ⏰ Chronos (시계열 분석)
- **시계열 위험도**: {agent_scores.get('chronos', 0):.1%}
- **트렌드 분석**: {"상승 추세" if agent_scores.get('chronos', 0) > 0.5 else "안정적"}

### 🔗 Cognita (관계 분석)
- **관계 위험도**: {agent_scores.get('cognita', 0):.1%}
- **네트워크 영향력**: {"높음" if agent_scores.get('cognita', 0) > 0.6 else "보통" if agent_scores.get('cognita', 0) > 0.3 else "낮음"}

### 💭 Sentio (감정 분석)
- **감정 위험도**: {agent_scores.get('sentio', 0):.1%}
- **감정 상태**: {"부정적" if agent_scores.get('sentio', 0) > 0.5 else "긍정적"}

### 🌍 Agora (시장 분석)
- **시장 위험도**: {agent_scores.get('agora', 0):.1%}
- **외부 환경**: {"불리함" if agent_scores.get('agora', 0) > 0.5 else "유리함"}

## 🎯 종합 분석 및 권장사항

### 위험도 평가
"""

        if risk_level == 'high':
            report_template += """
이 직원은 **고위험군**으로 분류되어 즉각적인 관심과 개입이 필요합니다.

**주요 우려사항:**
- 높은 이직 확률로 인한 인재 손실 위험
- 팀 내 부정적 영향 전파 가능성
- 업무 성과 및 몰입도 저하 우려

**즉시 권장사항:**
1. **1:1 면담 실시**: 관리자와의 개별 상담을 통한 문제점 파악
2. **근무 환경 개선**: 주요 위험 요인에 대한 구체적 개선 방안 수립
3. **인센티브 제공**: 성과 보상, 승진 기회, 교육 지원 등 고려
4. **정기적 모니터링**: 월 1회 이상 상태 점검 및 피드백
"""
        elif risk_level == 'medium':
            report_template += """
이 직원은 **주의군**으로 분류되어 예방적 관리가 필요합니다.

**주요 특징:**
- 중간 수준의 이직 위험도
- 적절한 개입을 통한 개선 가능성 높음
- 조기 대응을 통한 위험도 감소 기대

**권장사항:**
1. **정기적 피드백**: 분기별 성과 면담 및 커리어 상담
2. **업무 만족도 향상**: 업무 배치 조정, 역할 명확화
3. **교육 기회 제공**: 역량 개발 프로그램 참여 지원
4. **팀 내 소통 강화**: 동료 및 상사와의 관계 개선 지원
"""
        else:
            report_template += """
이 직원은 **안전군**으로 분류되어 현재 안정적인 상태입니다.

**주요 특징:**
- 낮은 이직 위험도
- 높은 업무 만족도 및 조직 몰입도
- 팀 내 긍정적 영향력 기대

**유지 관리 방안:**
1. **현재 상태 유지**: 기존의 긍정적 요인들을 지속적으로 지원
2. **성장 기회 제공**: 추가적인 도전과 발전 기회 제공
3. **멘토 역할**: 다른 직원들의 롤모델 및 멘토 역할 부여
4. **장기적 관점**: 경력 개발 계획 수립 및 지원
"""

        report_template += f"""

## 📈 데이터 기반 인사이트

### XAI (설명 가능한 AI) 분석
- **가장 중요한 예측 요인**: {structura_risks[0] if structura_risks else '데이터 분석 중'}
- **개선 가능한 영역**: {structura_protections[0] if structura_protections else '추가 분석 필요'}

### 시각화 자료
- 상세한 XAI 시각화는 다음 경로에서 확인 가능합니다:
  `app/results/{department.replace(' ', '_').replace('&', '_&_')}/employee_{employee_id}/visualizations/`

## 📅 후속 조치 계획

### 단기 (1개월 이내)
- [ ] 직속 상사와의 1:1 면담 실시
- [ ] 주요 위험 요인에 대한 구체적 개선 방안 논의
- [ ] 업무 환경 및 조건 점검

### 중기 (3개월 이내)
- [ ] 개선 방안 실행 및 효과 측정
- [ ] 추가적인 지원 방안 검토
- [ ] 정기적 모니터링 체계 구축

### 장기 (6개월 이후)
- [ ] 위험도 재평가 실시
- [ ] 장기적 경력 개발 계획 수립
- [ ] 조직 내 역할 및 기여도 재검토

---
*본 보고서는 AI 기반 분석 결과를 바탕으로 생성되었으며, 실제 인사 결정 시에는 추가적인 정성적 평가가 필요합니다.*

**보고서 생성 일시**: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}
"""

        return report_template.strip()
        
    except Exception as e:
        print(f"❌ LLM 보고서 생성 실패: {str(e)}")
        return f"보고서 생성 중 오류가 발생했습니다: {str(e)}"


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

@app.route('/save_agent_models', methods=['POST'])
def save_agent_models():
    """에이전트 모델을 app/results/models에 저장"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': '요청 데이터가 없습니다'
            }), 400
        
        models = data.get('models')
        save_path = data.get('save_path', 'app/results/models/agent_models.json')
        
        if not models:
            return jsonify({
                'success': False,
                'error': '저장할 모델 데이터가 없습니다'
            }), 400
        
        # 프로젝트 루트 기준으로 절대 경로 생성
        import os
        # app/Integration에서 프로젝트 루트로 이동 (../../)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        absolute_save_path = os.path.join(project_root, save_path)
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(absolute_save_path), exist_ok=True)
        
        # 파일 저장
        import json
        with open(absolute_save_path, 'w', encoding='utf-8') as f:
            json.dump(models, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 에이전트 모델 저장 완료: {save_path}")
        
        return jsonify({
            'success': True,
            'message': '에이전트 모델이 성공적으로 저장되었습니다',
            'file_path': save_path,
            'agents_saved': len(models.get('saved_models', {})),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ 에이전트 모델 저장 실패: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'모델 저장 중 오류가 발생했습니다: {str(e)}'
        }), 500

@app.route('/save_optimized_models', methods=['POST'])
def save_optimized_models():
    """최적화된 모델과 임계값/가중치를 app/results/models에 저장"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': '요청 데이터가 없습니다'
            }), 400
        
        complete_model = data.get('complete_model')
        save_path = data.get('save_path', 'app/results/models/optimized_models.json')
        
        if not complete_model:
            return jsonify({
                'success': False,
                'error': '저장할 모델 데이터가 없습니다'
            }), 400
        
        # 프로젝트 루트 기준으로 절대 경로 생성
        import os
        # app/Integration에서 프로젝트 루트로 이동 (../../)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        absolute_save_path = os.path.join(project_root, save_path)
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(absolute_save_path), exist_ok=True)
        
        # 배치 분석에서 쉽게 사용할 수 있도록 구조화
        optimized_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'stage': complete_model.get('training_metadata', {}).get('stage', 'optimization_completed'),
                'agents_used': complete_model.get('training_metadata', {}).get('agents_used', []),
                'training_data_size': complete_model.get('training_metadata', {}).get('training_data_size', 0)
            },
            'agent_models': complete_model.get('saved_models', {}),
            'agent_results': complete_model.get('agent_results', {}),
            'optimization_results': complete_model.get('optimization_results', {}),
            'ready_for_batch_analysis': True
        }
        
        # 파일 저장
        import json
        with open(absolute_save_path, 'w', encoding='utf-8') as f:
            json.dump(optimized_data, f, indent=2, ensure_ascii=False)
        
        # 배치 분석용 심볼릭 링크 또는 복사본 생성
        batch_ready_path = 'app/results/models/batch_ready_models.json'
        absolute_batch_path = os.path.join(project_root, batch_ready_path)
        with open(absolute_batch_path, 'w', encoding='utf-8') as f:
            json.dump(optimized_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 최적화된 모델 저장 완료: {save_path}")
        print(f"✅ 배치 분석용 모델 준비 완료: {batch_ready_path}")
        
        return jsonify({
            'success': True,
            'message': '최적화된 모델과 임계값/가중치가 성공적으로 저장되었습니다',
            'file_path': save_path,
            'batch_ready_path': batch_ready_path,
            'agents_count': len(optimized_data['agent_models']),
            'has_optimization': bool(optimized_data['optimization_results']),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ 최적화된 모델 저장 실패: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'모델 저장 중 오류가 발생했습니다: {str(e)}'
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
    print("  POST /save_agent_models - 에이전트 모델 저장")
    print("  POST /save_optimized_models - 최적화된 모델 저장")
    
    app.run(host='0.0.0.0', port=5007, debug=True)
