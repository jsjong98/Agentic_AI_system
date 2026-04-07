# ============================================================================
# Chronos Flask 백엔드 API
# ============================================================================

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import traceback
from typing import Dict, List, Any
import warnings
import logging
warnings.filterwarnings('ignore')

def safe_json_serialize(obj):
    """
    NaN, Infinity 값을 안전하게 처리하는 JSON 직렬화 함수
    """
    if isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, (np.ndarray,)):
        return safe_json_serialize(obj.tolist())
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    else:
        return obj

# 하이퍼파라미터 최적화를 위한 Optuna import
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    print("✅ Optuna 사용 가능 - 베이지안 최적화 활성화")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna 미설치 - 고정 하이퍼파라미터 사용")

# 로컬 모듈 import
from chronos_models import GRU_CNN_HybridModel, ChronosModelTrainer, create_hybrid_model, create_attention_model
from chronos_processor_fixed import ProperTimeSeriesProcessor, ChronosVisualizer, employee_based_train_test_split

def time_series_cross_validation(X, y, employee_ids, n_splits=3):
    """
    시계열 교차 검증 - 시간 순서를 고려한 직원별 분할
    각 fold에서 이전 시점의 직원들로 학습하고 이후 시점의 직원들로 검증
    """
    unique_employees = np.unique(employee_ids)
    n_employees = len(unique_employees)
    
    # 직원을 시간 순서대로 정렬 (가정: employee_id가 시간 순서와 연관)
    # 실제로는 각 직원의 첫 시점을 기준으로 정렬해야 함
    sorted_employees = np.sort(unique_employees)
    
    fold_size = n_employees // n_splits
    cv_results = []
    
    for fold in range(n_splits):
        # 시간 기반 분할: 이전 시점 직원들로 학습, 이후 시점 직원들로 검증
        if fold == n_splits - 1:  # 마지막 fold
            train_employees = sorted_employees[:fold * fold_size]
            val_employees = sorted_employees[fold * fold_size:]
        else:
            train_employees = sorted_employees[:fold * fold_size + fold_size]
            val_employees = sorted_employees[fold * fold_size + fold_size:(fold + 1) * fold_size + fold_size]
        
        if len(train_employees) == 0 or len(val_employees) == 0:
            continue
            
        # 마스크 생성
        train_mask = np.isin(employee_ids, train_employees)
        val_mask = np.isin(employee_ids, val_employees)
        
        cv_results.append({
            'train_indices': np.where(train_mask)[0],
            'val_indices': np.where(val_mask)[0],
            'train_employees': train_employees,
            'val_employees': val_employees
        })
    
    return cv_results

app = Flask(__name__)
CORS(app)

# 모델 캐시 (최적화된 모델 재사용)
model_cache = {
    'trained_model': None,
    'trained_processor': None,
    'model_metadata': None,
    'training_timestamp': None,
    'data_hash': None  # 데이터 변경 감지용
}

# 파일 업로드 크기 제한을 300MB로 설정
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300MB

# 로깅 설정 - 파일과 콘솔 모두 출력
log_dir = "../../logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "chronos_server.log")

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
processor = None
model = None
visualizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모듈 import 시 자동 초기화 (함수 정의 후에 실행하기 위해 나중에 호출)

# 데이터 경로 설정 - uploads 디렉토리에서 찾기
def get_chronos_data_paths(analysis_type='batch'):
    """uploads 디렉토리에서 Chronos 데이터 파일 찾기"""
    uploads_dir = "app/uploads/Chronos"
    post_dir = "app/uploads/Chronos/post"
    batch_dir = "app/uploads/Chronos/batch"
    
    data_paths = {
        'timeseries': None,
        'personas': None
    }
    
    logger.info(f"🔍 Chronos 데이터 경로 탐색 중... (분석 타입: {analysis_type})")
    
    # 분석 타입에 따른 디렉토리 우선순위 설정
    if analysis_type == 'post':
        search_dirs = [post_dir, uploads_dir]
    elif analysis_type == 'batch':
        search_dirs = [batch_dir, post_dir, uploads_dir]
    else:
        search_dirs = [uploads_dir, post_dir, batch_dir]
    
    # 각 디렉토리에서 파일 찾기
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            files = [f for f in os.listdir(search_dir) if f.endswith('.csv')]
            if files:
                # 가장 최근 파일 사용 (수정 시간 기준)
                files_with_time = [(f, os.path.getmtime(os.path.join(search_dir, f))) for f in files]
                files_with_time.sort(key=lambda x: x[1], reverse=True)  # 수정 시간 기준 내림차순
                latest_file = os.path.join(search_dir, files_with_time[0][0])
                data_paths['timeseries'] = latest_file
                logger.info(f"📁 {search_dir}에서 최신 파일 발견: {files_with_time[0][0]}")
                break
    
    # latest 파일 확인
    if data_paths['timeseries'] is None and os.path.exists(uploads_dir):
        latest_timeseries = os.path.join(uploads_dir, 'latest_timeseries.csv')
        if os.path.exists(latest_timeseries):
            data_paths['timeseries'] = latest_timeseries
            print(f"📁 latest 파일 사용: {latest_timeseries}")
    
    # 기본값으로 fallback
    if data_paths['timeseries'] is None:
        data_paths['timeseries'] = 'data/IBM_HR_timeseries.csv'
        print(f"⚠️ 업로드된 파일 없음, 기본 경로 사용: {data_paths['timeseries']}")
    
    if data_paths['personas'] is None:
        data_paths['personas'] = 'data/IBM_HR_personas_assigned.csv'
        
    return data_paths

DATA_PATH = get_chronos_data_paths('post')

MODEL_PATH = 'app/Chronos/models'
os.makedirs(MODEL_PATH, exist_ok=True)

def load_model():
    """
    저장된 모델 로드
    """
    global model
    
    try:
        model_file = os.path.join(MODEL_PATH, 'chronos_attention_model.pth')
        scaler_file = os.path.join(MODEL_PATH, 'chronos_scaler.joblib')
        
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            # 모델 로드
            checkpoint = torch.load(model_file, map_location=device)
            model = create_hybrid_model(
                input_size=checkpoint['input_size'],
                gru_hidden=checkpoint.get('gru_hidden', 32),
                cnn_filters=checkpoint.get('cnn_filters', 16),
                kernel_sizes=checkpoint.get('kernel_sizes', [2, 3]),
                dropout=checkpoint.get('dropout', 0.2)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            # 스케일러 로드
            if processor:
                processor.scaler = joblib.load(scaler_file)
                processor.feature_columns = checkpoint.get('feature_columns', [])
            
            print("✅ 모델 및 스케일러 로드 완료")
            return True
        else:
            print("⚠️ 저장된 모델 파일을 찾을 수 없습니다.")
            return False
            
    except Exception as e:
        print(f"❌ 모델 로드 실패: {str(e)}")
        return False

def initialize_system():
    """
    시스템 초기화
    """
    global processor, model, visualizer, DATA_PATH
    
    try:
        logger.info("🚀 Chronos 시스템 초기화 중...")
        
        # 데이터 경로 업데이트 (업로드된 파일 확인 - post 우선)
        DATA_PATH = get_chronos_data_paths('post')
        logger.info(f"📁 데이터 경로: {DATA_PATH}")
        
        # 개선된 프로세서 및 시각화 도구 초기화
        processor = ProperTimeSeriesProcessor(sequence_length=50, aggregation_unit='week')
        visualizer = ChronosVisualizer()
        
        # 데이터 로드 및 전처리
        if os.path.exists(DATA_PATH['timeseries']):
            # personas 파일이 있으면 사용, 없으면 None
            personas_path = DATA_PATH['personas'] if DATA_PATH['personas'] and os.path.exists(DATA_PATH['personas']) else None
            processor.load_data(DATA_PATH['timeseries'], personas_path)
            processor.detect_columns()
            processor.preprocess_data()
            processor.identify_features()
            
            # processed_data 속성을 생성하기 위해 시퀀스 생성 호출
            try:
                processor.create_proper_sequences()
                logger.info("✅ 데이터 로드, 전처리 및 시퀀스 생성 완료")
            except Exception as e:
                logger.warning(f"⚠️ 시퀀스 생성 실패: {e}")
                # processed_data 속성을 수동으로 설정
                if hasattr(processor, 'ts_data') and processor.ts_data is not None:
                    processor.processed_data = processor.ts_data
                logger.info("✅ 데이터 로드 및 전처리 완료 (시퀀스 생성 제외)")
        else:
            logger.warning("⚠️ 시계열 데이터 파일을 찾을 수 없습니다. 학습된 모델만 사용 가능합니다.")
        
        # 기존 모델 로드 시도
        model_file = os.path.join(MODEL_PATH, 'chronos_attention_model.pth')
        if os.path.exists(model_file):
            load_model()
            print("✅ 기존 모델 로드 완료")
        else:
            print("ℹ️ 기존 모델이 없습니다. 새로운 모델을 학습해주세요.")
            
        logger.info("🎉 Chronos 시스템 초기화 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 시스템 초기화 실패: {str(e)}")
        traceback.print_exc()
        return False

# 모듈 import 시 자동 초기화 실행
def _auto_initialize():
    """모듈이 import될 때 자동으로 시스템 초기화"""
    global processor, model, visualizer
    if processor is None:
        try:
            print("🔄 Chronos 모듈 import 시 자동 초기화 중...")
            initialize_system()
        except Exception as e:
            print(f"⚠️ 자동 초기화 실패: {e}")

# 자동 초기화 실행
_auto_initialize()


def save_model(model, processor, additional_info=None):
    """
    모델 저장
    """
    try:
        model_file = os.path.join(MODEL_PATH, 'chronos_attention_model.pth')
        scaler_file = os.path.join(MODEL_PATH, 'chronos_scaler.joblib')
        
        # 모델 저장
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'input_size': model.input_size,
            'gru_hidden': model.gru_hidden,
            'cnn_filters': model.cnn_filters,
            'feature_columns': processor.feature_columns,
            'sequence_length': processor.sequence_length,
            'aggregation_unit': processor.aggregation_unit,
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_info:
            checkpoint.update(additional_info)
            
        torch.save(checkpoint, model_file)
        
        # 스케일러 저장
        joblib.dump(processor.scaler, scaler_file)
        
        print("✅ 모델 및 스케일러 저장 완료")
        return True
        
    except Exception as e:
        print(f"❌ 모델 저장 실패: {str(e)}")
        return False

@app.route('/')
def home():
    """
    홈페이지
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chronos - Employee Attrition Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .api-section { margin: 20px 0; padding: 20px; background: #ecf0f1; border-radius: 5px; }
            .endpoint { margin: 10px 0; padding: 10px; background: white; border-left: 4px solid #3498db; }
            .method { font-weight: bold; color: #e74c3c; }
            .status { padding: 10px; margin: 20px 0; border-radius: 5px; }
            .status.ready { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.warning { background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🕒 Chronos - Employee Attrition Prediction System</h1>
            
            <div class="status {{ status_class }}">
                <strong>시스템 상태:</strong> {{ status_message }}
            </div>
            
            <div class="api-section">
                <h2>📋 API 엔드포인트</h2>
                
                <div class="endpoint">
                    <span class="method">GET</span> <strong>/api/status</strong>
                    <p>시스템 상태 확인</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/api/train</strong>
                    <p>모델 학습 (JSON: {"sequence_length": 6, "epochs": 50})</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <strong>/api/predict</strong>
                    <p>예측 수행 (JSON: {"employee_ids": [1, 2, 3]})</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <strong>/api/feature_importance</strong>
                    <p>전체 평균 Feature importance 시각화</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <strong>/api/employee_feature_importance</strong>
                    <p>직원별 Attention & Gradient Feature Importance 비교</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <strong>/api/employee_attention_heatmap</strong>
                    <p>직원별 Attention Weights 히트맵</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <strong>/api/feature_importance_distribution</strong>
                    <p>Feature Importance 분포 분석</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <strong>/api/model_analysis</strong>
                    <p>모델 분석 대시보드</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <strong>/api/employee_timeline/{employee_id}</strong>
                    <p>개별 직원 시계열 분석</p>
                </div>
            </div>
            
            <div class="api-section">
                <h2>🚀 사용 방법</h2>
                <ol>
                    <li><strong>모델 학습:</strong> POST /api/train으로 모델을 먼저 학습시키세요</li>
                    <li><strong>예측 수행:</strong> POST /api/predict로 직원들의 퇴사 확률을 예측하세요</li>
                    <li><strong>전체 분석:</strong> GET /api/feature_importance로 전체 평균 중요 피처들을 확인하세요</li>
                    <li><strong>직원별 분석:</strong> GET /api/employee_feature_importance로 각 직원별 Attention & Gradient 중요도를 비교하세요</li>
                    <li><strong>시간별 패턴:</strong> GET /api/employee_attention_heatmap으로 직원별 시간대별 집중도를 확인하세요</li>
                    <li><strong>분포 분석:</strong> GET /api/feature_importance_distribution으로 피처 중요도 분포를 분석하세요</li>
                    <li><strong>개별 분석:</strong> GET /api/employee_timeline/{id}로 개별 직원을 상세 분석하세요</li>
                </ol>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 시스템 상태 확인
    if model is not None and processor is not None:
        status_class = "ready"
        status_message = "✅ 시스템이 준비되었습니다. 모든 기능을 사용할 수 있습니다."
    elif processor is not None:
        status_class = "warning"
        status_message = "⚠️ 데이터는 로드되었지만 모델이 학습되지 않았습니다. 먼저 모델을 학습시켜주세요."
    else:
        status_class = "warning"
        status_message = "⚠️ 시스템 초기화가 필요합니다. 데이터 파일을 확인해주세요."
    
    return render_template_string(html_template, 
                                status_class=status_class, 
                                status_message=status_message)

@app.route('/api/status')
def get_status():
    """
    시스템 상태 확인
    """
    try:
        status = {
            'system_initialized': processor is not None,
            'model_loaded': model is not None,
            'data_available': processor is not None and processor.ts_data is not None,
            'device': str(device),
            'timestamp': datetime.now().isoformat(),
            'data_paths': DATA_PATH,
            'timeseries_exists': os.path.exists(DATA_PATH['timeseries']) if DATA_PATH['timeseries'] else False,
            'personas_exists': os.path.exists(DATA_PATH['personas']) if DATA_PATH['personas'] else False
        }
        
        if processor:
            status.update({
                'sequence_length': processor.sequence_length,
                'aggregation_unit': processor.aggregation_unit,
                'feature_count': len(processor.feature_columns),
                'data_shape': processor.processed_data.shape if processor.processed_data is not None else None
            })
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """
    헬스체크 엔드포인트 (표준 health check)
    """
    try:
        # 기본 상태 정보
        health_status = {
            'status': 'healthy',
            'service': 'chronos',
            'timestamp': datetime.now().isoformat(),
            'system_initialized': processor is not None,
            'model_loaded': model is not None,
            'data_available': processor is not None and processor.ts_data is not None,
            'device': str(device)
        }
        
        # 모델 상태
        if model is not None:
            health_status['model_status'] = 'loaded'
            health_status['model_type'] = 'GRU_CNN_Hybrid'
        else:
            health_status['model_status'] = 'not_loaded'
        
        # 데이터 상태
        if processor is not None:
            health_status['data_status'] = {
                'processor_ready': True,
                'sequence_length': processor.sequence_length,
                'feature_count': len(processor.feature_columns) if processor.feature_columns else 0,
                'aggregation_unit': processor.aggregation_unit
            }
        else:
            health_status['data_status'] = {
                'processor_ready': False
            }
        
        # 의존성 상태
        health_status['dependencies'] = {
            'torch': True,  # 이미 import 되어 있음
            'optuna': OPTUNA_AVAILABLE,
            'sklearn': True  # chronos_processor_fixed에서 사용
        }
        
        return jsonify(health_status)
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'chronos',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/upload/timeseries', methods=['POST'])
def upload_timeseries_data():
    """시계열 데이터 CSV 파일 업로드"""
    try:
        # 파일 확인
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "시계열 파일이 업로드되지 않았습니다."
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "시계열 파일이 선택되지 않았습니다."
            }), 400
            
        # personas 파일 확인 (선택사항)
        personas_file = None
        if 'personas_file' in request.files:
            personas_file = request.files['personas_file']
            if personas_file.filename == '':
                personas_file = None
        
        # 파일 확장자 확인
        if not file.filename.lower().endswith('.csv'):
            return jsonify({
                "success": False,
                "error": "CSV 파일만 업로드 가능합니다."
            }), 400
        
        # 파일 저장
        filename = secure_filename(file.filename)
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'Chronos')
        os.makedirs(upload_dir, exist_ok=True)
        
        # 타임스탬프를 포함한 파일명으로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(filename)[0]
        new_filename = f"{base_name}_{timestamp}.csv"
        file_path = os.path.join(upload_dir, new_filename)
        
        # 최신 파일 링크도 생성
        latest_link = os.path.join(upload_dir, 'latest_timeseries.csv')
        file.save(file_path)
        
        # 최신 파일 링크 생성
        try:
            if os.path.exists(latest_link):
                os.remove(latest_link)
            import shutil
            shutil.copy2(file_path, latest_link)
        except Exception as e:
            print(f"최신 파일 링크 생성 실패: {e}")
        
        # 파일 크기 확인 (300MB 제한)
        file.seek(0, 2)  # 파일 끝으로 이동
        file_size = file.tell()
        file.seek(0)  # 파일 시작으로 되돌리기
        
        max_size = 300 * 1024 * 1024  # 300MB
        if file_size > max_size:
            return jsonify({
                "success": False,
                "error": f"파일 크기가 너무 큽니다. 최대 300MB까지 업로드 가능합니다. (현재: {file_size / (1024*1024):.1f}MB)"
            }), 413
        
        # 데이터 검증
        try:
            df = pd.read_csv(file_path)
            
            # 필수 컬럼 확인 및 자동 생성
            required_columns = ['employee_id']
            
            # employee_id 컬럼 확인
            if 'employee_id' not in df.columns:
                return jsonify({
                    "success": False,
                    "error": "필수 컬럼이 누락되었습니다: employee_id",
                    "required_columns": ['employee_id', 'week 또는 date'],
                    "found_columns": list(df.columns)
                }), 400
            
            # week 컬럼이 없으면 date 컬럼에서 생성
            if 'week' not in df.columns:
                if 'date' in df.columns:
                    print("📅 date 컬럼에서 week 컬럼 자동 생성 중...")
                    df['date'] = pd.to_datetime(df['date'])
                    df['year'] = df['date'].dt.year
                    df['week'] = df['date'].dt.isocalendar().week
                    df['week'] = df['year'].astype(str) + '-W' + df['week'].astype(str).str.zfill(2)
                    print(f"✅ week 컬럼 생성 완료")
                else:
                    return jsonify({
                        "success": False,
                        "error": "week 또는 date 컬럼이 필요합니다",
                        "required_columns": ['employee_id', 'week 또는 date'],
                        "found_columns": list(df.columns)
                    }), 400
            
            # 시계열 데이터 형식 확인
            unique_employees = df['employee_id'].nunique()
            weeks_per_employee = df.groupby('employee_id')['week'].count()
            min_weeks = weeks_per_employee.min()
            max_weeks = weeks_per_employee.max()
            
            # 기존 시스템 초기화 (새 데이터로 재처리 필요)
            global processor, model
            processor = None
            model = None
            
            return jsonify({
                "success": True,
                "message": "시계열 데이터가 성공적으로 업로드되었습니다.",
                "file_info": {
                    "original_filename": filename,
                    "saved_filename": new_filename,
                    "upload_path": upload_dir,
                    "file_path": file_path,
                    "latest_link": latest_link,
                    "total_rows": len(df),
                    "unique_employees": unique_employees,
                    "weeks_range": f"{min_weeks}-{max_weeks}주",
                    "columns": len(df.columns),
                    "column_names": list(df.columns)
                },
                "note": "새로운 데이터로 시스템을 재초기화하고 모델을 재훈련해주세요."
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

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    모델 학습
    """
    global model
    
    try:
        # 프로세서가 없거나 데이터가 없으면 다시 초기화
        if processor is None or not hasattr(processor, 'ts_data') or processor.ts_data is None:
            print("🔄 데이터가 없어서 시스템을 다시 초기화합니다...")
            success = initialize_system()
            if not success or processor is None or not hasattr(processor, 'ts_data') or processor.ts_data is None:
                return jsonify({'error': '데이터가 로드되지 않았습니다. 먼저 데이터를 업로드해주세요.'}), 400
        
        # 요청 파라미터 파싱
        params = request.get_json() or {}
        sequence_length = params.get('sequence_length', 6)
        epochs = params.get('epochs', 50)
        batch_size = params.get('batch_size', 32)
        learning_rate = params.get('learning_rate', 0.001)
        optimize_hyperparameters_flag = params.get('optimize_hyperparameters', True)  # 기본적으로 최적화 사용
        
        print(f"🚀 모델 학습 시작 - Epochs: {epochs}, Batch Size: {batch_size}")
        if optimize_hyperparameters_flag and OPTUNA_AVAILABLE:
            print("🔧 베이지안 하이퍼파라미터 최적화 활성화")
        else:
            print("⚙️ 고정 하이퍼파라미터 사용")
        
        # 시퀀스 길이 업데이트
        processor.sequence_length = sequence_length
        
        # 개선된 시퀀스 생성
        X, y, employee_ids = processor.create_proper_sequences()
        
        # 직원 기반 분할 (데이터 누수 방지 - 시계열 특성 고려)
        X_train, X_test, y_train, y_test = employee_based_train_test_split(
            X, y, employee_ids, test_ratio=0.2
        )
        
        print(f"📊 시계열 검증 방식:")
        print(f"   - 직원별 분할: 동일 직원 데이터가 train/test 동시 포함 방지")
        print(f"   - 시간 순서 유지: 각 직원의 과거→현재 시퀀스 보존")
        print(f"   - 예측 방향: 과거 6주 데이터로 미래 퇴사 여부 예측")
        
        # 텐서 변환
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)
        
        # 데이터로더 생성
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 모델 생성 (최적화 여부에 따라)
        input_size = len(processor.feature_columns)
        
        if optimize_hyperparameters_flag and OPTUNA_AVAILABLE:
            # 베이지안 최적화를 통한 하이퍼파라미터 탐색
            print("🔧 베이지안 최적화로 최적 하이퍼파라미터 탐색 중...")
            
            def objective(trial):
                # 하이퍼파라미터 제안
                gru_hidden = trial.suggest_categorical('gru_hidden', [16, 32, 64, 128])
                cnn_filters = trial.suggest_categorical('cnn_filters', [8, 16, 32, 64])
                dropout = trial.suggest_float('dropout', 0.1, 0.5)
                trial_lr = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
                trial_batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
                
                try:
                    # 모델 생성
                    trial_model = create_hybrid_model(
                        input_size=input_size,
                        gru_hidden=gru_hidden,
                        cnn_filters=cnn_filters,
                        kernel_sizes=[2, 3],
                        dropout=dropout
                    )
                    trial_model.to(device)
                    
                    # 데이터 로더 (배치 크기 적용)
                    trial_train_loader = DataLoader(train_dataset, batch_size=trial_batch_size, shuffle=True)
                    trial_test_loader = DataLoader(test_dataset, batch_size=trial_batch_size, shuffle=False)
                    
                    # 트레이너 설정
                    trial_trainer = ChronosModelTrainer(trial_model, device)
                    trial_optimizer = optim.Adam(trial_model.parameters(), lr=trial_lr)
                    trial_criterion = nn.CrossEntropyLoss()
                    
                    # 짧은 학습 (최적화용)
                    best_val_acc = 0
                    patience = 5
                    patience_counter = 0
                    
                    for epoch in range(min(epochs, 30)):  # 최대 30 에포크로 제한
                        train_loss, train_acc = trial_trainer.train_epoch(trial_train_loader, trial_optimizer, trial_criterion)
                        test_results = trial_trainer.evaluate(trial_test_loader, trial_criterion)
                        val_acc = test_results['accuracy']
                        
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                break
                        
                        # Optuna pruning
                        trial.report(val_acc, epoch)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    
                    return best_val_acc
                    
                except Exception as e:
                    print(f"Trial 실패: {str(e)}")
                    return 0.0
            
            # Optuna Study 실행
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
            )
            
            n_optimization_trials = 50  # 베이지안 최적화 50회로 설정
            print(f"🚀 {n_optimization_trials}회 베이지안 최적화 시행...")
            study.optimize(objective, n_trials=n_optimization_trials, timeout=1800)  # 30분 타임아웃
            
            # 최적 하이퍼파라미터로 모델 생성
            best_params = study.best_params
            print(f"✅ 최적 하이퍼파라미터: {best_params}")
            
            model = create_hybrid_model(
                input_size=input_size,
                gru_hidden=best_params['gru_hidden'],
                cnn_filters=best_params['cnn_filters'],
                kernel_sizes=[2, 3],
                dropout=best_params['dropout']
            )
            
            # 최적 설정 적용
            learning_rate = best_params['learning_rate']
            batch_size = best_params['batch_size']
            
            # 데이터로더 재생성 (최적 배치 크기로)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
        else:
            # 기존 고정 하이퍼파라미터 사용
            print("⚙️ 고정 하이퍼파라미터로 모델 생성...")
            model = create_hybrid_model(
                input_size=input_size,
                gru_hidden=32,
                cnn_filters=16,
                kernel_sizes=[2, 3],
                dropout=0.2
            )
        
        model.to(device)
        
        # 트레이너 설정
        trainer = ChronosModelTrainer(model, device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # 학습 진행
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # 학습
            train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # 평가
            test_results = trainer.evaluate(test_loader, criterion)
            test_losses.append(test_results['loss'])
            test_accuracies.append(test_results['accuracy'])
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f}, Test Acc: {test_results['accuracy']:.4f}")
        
        # 최종 평가
        final_results = trainer.evaluate(test_loader, criterion)
        
        # 모델 저장
        training_info = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'final_accuracy': final_results['accuracy'],
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }
        
        save_model(model, processor, training_info)
        
        return jsonify({
            'message': '모델 학습이 완료되었습니다.',
            'results': {
                'final_accuracy': final_results['accuracy'],
                'final_loss': final_results['loss'],
                'training_epochs': epochs,
                'data_size': len(X),
                'feature_count': input_size
            }
        })
        
    except Exception as e:
        print(f"❌ 학습 오류: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize_hyperparameters', methods=['POST'])
def optimize_hyperparameters():
    """
    Optuna를 사용한 베이지안 하이퍼파라미터 최적화
    """
    global model, processor
    
    if not OPTUNA_AVAILABLE:
        return jsonify({
            'error': 'Optuna가 설치되지 않았습니다. pip install optuna로 설치해주세요.'
        }), 400
    
    try:
        if processor is None or processor.ts_data is None:
            return jsonify({'error': '데이터가 로드되지 않았습니다.'}), 400
        
        # 요청 파라미터 파싱
        params = request.get_json() or {}
        n_trials = params.get('n_trials', 50)  # 최적화 시행 횟수
        timeout = params.get('timeout', 1800)  # 30분 타임아웃
        
        print(f"🔧 Chronos 베이지안 하이퍼파라미터 최적화 시작")
        print(f"   시행 횟수: {n_trials}")
        print(f"   타임아웃: {timeout}초")
        
        # 시퀀스 생성 (시간 순서 고려)
        X, y, employee_ids = processor.create_proper_sequences()
        
        # 직원 기반 분할 (데이터 누수 방지 - 같은 직원의 데이터가 train/test에 동시 포함되지 않음)
        X_train, X_test, y_train, y_test = employee_based_train_test_split(
            X, y, employee_ids, test_ratio=0.2
        )
        
        print(f"📊 시계열 데이터 검증 방식:")
        print(f"   - 직원별 분할: 같은 직원 데이터가 train/test에 동시 포함되지 않음")
        print(f"   - 시간 순서 보존: 각 직원의 시계열 순서 유지")
        print(f"   - 미래 예측 방식: 과거 시퀀스로 미래 퇴사 여부 예측")
        
        # GPU 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_size = len(processor.feature_columns)
        
        # 최적화 목적 함수 정의
        def objective(trial):
            # 하이퍼파라미터 제안
            gru_hidden = trial.suggest_categorical('gru_hidden', [16, 32, 64, 128])
            cnn_filters = trial.suggest_categorical('cnn_filters', [8, 16, 32, 64])
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            epochs = trial.suggest_int('epochs', 20, 100)
            
            try:
                # 모델 생성
                trial_model = create_hybrid_model(
                    input_size=input_size,
                    gru_hidden=gru_hidden,
                    cnn_filters=cnn_filters,
                    kernel_sizes=[2, 3],  # 고정
                    dropout=dropout
                )
                trial_model.to(device)
                
                # 데이터 로더 생성
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train), 
                    torch.LongTensor(y_train)
                )
                test_dataset = TensorDataset(
                    torch.FloatTensor(X_test), 
                    torch.LongTensor(y_test)
                )
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                # 트레이너 및 옵티마이저 설정
                trainer = ChronosModelTrainer(trial_model, device)
                optimizer = optim.Adam(trial_model.parameters(), lr=learning_rate)
                criterion = nn.CrossEntropyLoss()
                
                # 조기 종료를 위한 변수
                best_val_acc = 0
                patience = 10
                patience_counter = 0
                
                # 학습 진행
                for epoch in range(epochs):
                    train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
                    test_results = trainer.evaluate(test_loader, criterion)
                    val_acc = test_results['accuracy']
                    
                    # 조기 종료 체크
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break
                    
                    # Optuna pruning (중간 결과 기반 조기 종료)
                    trial.report(val_acc, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                return best_val_acc
                
            except Exception as e:
                print(f"Trial 실패: {str(e)}")
                return 0.0  # 실패 시 최저 점수 반환
        
        # Optuna Study 생성 및 실행
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        print("🚀 베이지안 최적화 시작...")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # 최적 하이퍼파라미터로 최종 모델 학습
        best_params = study.best_params
        print(f"✅ 최적 하이퍼파라미터: {best_params}")
        
        # 최적 모델 생성 및 학습
        optimized_model = create_hybrid_model(
            input_size=input_size,
            gru_hidden=best_params['gru_hidden'],
            cnn_filters=best_params['cnn_filters'],
            kernel_sizes=[2, 3],
            dropout=best_params['dropout']
        )
        optimized_model.to(device)
        
        # 최적 설정으로 최종 학습
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        trainer = ChronosModelTrainer(optimized_model, device)
        optimizer = optim.Adam(optimized_model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        # 최종 학습
        final_epochs = min(best_params['epochs'], 50)  # 최대 50 에포크로 제한
        for epoch in range(final_epochs):
            train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
            if epoch % 10 == 0:
                test_results = trainer.evaluate(test_loader, criterion)
                print(f"Final Training Epoch {epoch+1}/{final_epochs} - Val Acc: {test_results['accuracy']:.4f}")
        
        # 최종 평가
        final_results = trainer.evaluate(test_loader, criterion)
        
        # 최적화된 모델 저장
        model = optimized_model
        
        optimization_info = {
            'best_params': best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'optimization_time': datetime.now().isoformat(),
            'final_accuracy': final_results['accuracy'],
            'final_loss': final_results['loss']
        }
        
        save_model(model, processor, optimization_info)
        
        return jsonify({
            'message': 'Chronos 베이지안 하이퍼파라미터 최적화 완료',
            'optimization_results': {
                'best_hyperparameters': best_params,
                'best_validation_score': study.best_value,
                'final_test_accuracy': final_results['accuracy'],
                'final_test_loss': final_results['loss'],
                'total_trials': len(study.trials),
                'optimization_method': 'Optuna TPESampler',
                'trials_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'trials_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            },
            'model_info': {
                'input_size': input_size,
                'feature_count': len(processor.feature_columns),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
        })
        
    except Exception as e:
        print(f"❌ 하이퍼파라미터 최적화 오류: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    예측 수행
    """
    global DATA_PATH, processor, model
    
    try:
        if model is None:
            return jsonify({'error': '모델이 로드되지 않았습니다. 먼저 모델을 학습시켜주세요.'}), 400
        
        if processor is None or processor.processed_data is None:
            return jsonify({'error': '데이터가 로드되지 않았습니다.'}), 400
        
        # 요청 파라미터 파싱
        params = request.get_json() or {}
        logger.info(f"🔮 Chronos 예측 요청 받음: {type(params)} 타입")
        
        # employee_id 단일 요청 처리
        if 'employee_id' in params:
            employee_ids = [params['employee_id']]
            logger.info(f"👤 단일 직원 ID로 예측 요청: {params['employee_id']}")
        else:
            employee_ids = params.get('employee_ids', [])
            if employee_ids:
                logger.info(f"📊 복수 직원 ID로 예측 요청: {len(employee_ids)}명")
            
        analysis_type = params.get('analysis_type', 'batch')
        logger.info(f"📋 분석 타입: {analysis_type}")
        
        # 분석 타입에 따른 데이터 경로 확인 및 재로드
        new_data_paths = get_chronos_data_paths(analysis_type)
        current_data_paths = get_chronos_data_paths()
        
        if new_data_paths != current_data_paths:
            logger.info(f"🔄 Chronos: {analysis_type} 분석을 위한 데이터 재로드")
            DATA_PATH = new_data_paths
            
            # 데이터 재로드 필요 시 processor 재초기화
            if processor and (new_data_paths['timeseries'] != current_data_paths['timeseries'] or 
                            new_data_paths['personas'] != current_data_paths['personas']):
                try:
                    processor.load_data(new_data_paths['timeseries'], new_data_paths['personas'])
                    processor.detect_columns()
                    processor.preprocess_data()
                    processor.identify_features()
                    logger.info(f"✅ Chronos {analysis_type} 데이터 재로드 완료")
                except Exception as e:
                    logger.warning(f"⚠️ Chronos 데이터 재로드 실패: {e}")
        
        logger.info(f"📊 Chronos {analysis_type} 분석 시작")
        
        if not employee_ids:
            # 모든 직원 예측
            if processor.processed_data is not None:
                employee_ids = processor.processed_data['employee_id'].unique().tolist()
                logger.info(f"🏢 전체 직원 예측 모드: {len(employee_ids)}명")
            else:
                return jsonify({'error': '처리된 데이터가 없습니다. 먼저 데이터를 로드해주세요.'}), 400
        else:
            logger.info(f"🎯 지정된 직원 예측: {len(employee_ids)}명")
        
        # 예측 수행
        predictions = []
        
        for idx, emp_id in enumerate(employee_ids):
            logger.info(f"🔮 직원 {emp_id} 예측 시작 ({idx+1}/{len(employee_ids)})")
            
            emp_data = processor.processed_data[
                processor.processed_data['employee_id'] == emp_id
            ].sort_values('period')
            
            if len(emp_data) >= processor.sequence_length:
                logger.info(f"📊 직원 {emp_id}: {len(emp_data)}개 시계열 데이터 발견 (최소 {processor.sequence_length}개 필요)")
                # 피처 데이터 추출 및 정규화
                feature_data = emp_data[processor.feature_columns].values
                feature_data_scaled = processor.scaler.transform(feature_data)
                
                # 시퀀스 생성
                sequence = feature_data_scaled[-processor.sequence_length:]
                X_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
                
                # 예측 수행 (사후 분석에서는 Feature Importance 불필요)
                model.eval()
                with torch.no_grad():
                    output = model(X_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    
                    pred_class = torch.argmax(output, dim=1).item()
                    pred_prob = probabilities[0][1].item()  # 퇴사 확률
                    
                    # NaN 값 처리
                    if np.isnan(pred_prob) or np.isinf(pred_prob):
                        pred_prob = 0.0
                    
                    # 안전한 float 변환
                    safe_pred_prob = float(pred_prob)
                    
                    # Attention weights 추출 (간단한 시각화용)
                    attention_weights = None
                    try:
                        attention_weights = model.get_attention_weights(X_tensor)
                        if attention_weights is not None:
                            attention_weights = attention_weights[0].cpu().numpy().tolist()
                    except:
                        attention_weights = []
                    
                    risk_level = 'High' if safe_pred_prob > 0.7 else 'Medium' if safe_pred_prob > 0.3 else 'Low'
                    
                    predictions.append({
                        'employee_id': int(emp_id),
                        'attrition_probability': safe_pred_prob,
                        'predicted_class': int(pred_class),
                        'risk_level': risk_level,
                        'temporal_attention': attention_weights or []
                    })
                    
                    logger.info(f"✅ 직원 {emp_id} 예측 완료: 위험도 {safe_pred_prob:.3f} ({risk_level})")
            else:
                logger.warning(f"⚠️ 직원 {emp_id}: 데이터 부족 ({len(emp_data)}개 < {processor.sequence_length}개)")
        
        # 결과 정렬 (퇴사 확률 높은 순)
        predictions.sort(key=lambda x: x['attrition_probability'], reverse=True)
        
        logger.info(f"🎉 Chronos {analysis_type} 분석 완료 - 총 {len(predictions)}명 예측 완료")
        
        # 안전한 평균 계산 (빈 리스트 처리)
        avg_prob = np.mean([p['attrition_probability'] for p in predictions]) if predictions else 0.0
        
        response_data = {
            'predictions': predictions,
            'summary': {
                'total_employees': len(predictions),
                'high_risk_count': sum(1 for p in predictions if p['risk_level'] == 'High'),
                'medium_risk_count': sum(1 for p in predictions if p['risk_level'] == 'Medium'),
                'low_risk_count': sum(1 for p in predictions if p['risk_level'] == 'Low'),
                'average_attrition_probability': avg_prob
            }
        }
        
        # NaN 값 안전 처리
        safe_response = safe_json_serialize(response_data)
        return jsonify(safe_response)
        
    except Exception as e:
        print(f"❌ 예측 오류: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature_importance')
def get_feature_importance():
    """
    Feature importance 시각화 (전체 평균)
    """
    try:
        if model is None:
            return jsonify({'error': '모델이 로드되지 않았습니다.'}), 400
        
        # 전체 데이터에 대한 평균 feature importance 계산
        X, y, employee_ids = processor.create_proper_sequences()
        X_tensor = torch.FloatTensor(X).to(device)
        
        model.eval()
        feature_importances = []
        
        # gradient 계산을 위해 torch.no_grad() 제거
        for i in range(0, len(X_tensor), 32):  # 배치 처리
            batch = X_tensor[i:i+32]
            interpretation = model.interpret_prediction(batch, processor.feature_columns)
            feature_importances.append(interpretation['feature_importance'])
        
        # 평균 계산
        avg_importance = np.mean(feature_importances, axis=0)
        
        # 시각화 생성
        html_plot = visualizer.plot_feature_importance(
            avg_importance, 
            processor.feature_columns,
            "Average Feature Importance Across All Employees"
        )
        
        return html_plot, 200, {'Content-Type': 'text/html'}
        
    except Exception as e:
        print(f"❌ Feature importance 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/employee_feature_importance')
def get_employee_feature_importance():
    """
    직원별 Attention & Gradient 기반 Feature Importance 시각화
    """
    try:
        if model is None:
            return jsonify({'error': '모델이 로드되지 않았습니다.'}), 400
        
        if processor is None or processor.processed_data is None:
            return jsonify({'error': '데이터가 로드되지 않았습니다.'}), 400
        
        print("🚀 직원별 Feature Importance 분석 시작")
        
        # 시퀀스 데이터 생성
        X, y, employee_ids = processor.create_proper_sequences()
        
        # 직원별 Attention 중요도 계산
        attention_results = visualizer.calculate_attention_importance_per_employee(
            model, X, y, employee_ids, processor.feature_columns, device
        )
        
        # 직원별 Gradient 중요도 계산
        gradient_results = visualizer.calculate_gradient_importance_per_employee(
            model, X, y, employee_ids, processor.feature_columns, device, max_samples_per_employee=5
        )
        
        # 직원별 비교 시각화
        comparison_html = visualizer.plot_employee_feature_importance_comparison(
            attention_results, gradient_results, processor.feature_columns,
            top_employees=10, top_features=10
        )
        
        return comparison_html, 200, {'Content-Type': 'text/html'}
        
    except Exception as e:
        print(f"❌ 직원별 Feature importance 오류: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/employee_attention_heatmap')
def get_employee_attention_heatmap():
    """
    직원별 Attention Weights 히트맵
    """
    try:
        if model is None:
            return jsonify({'error': '모델이 로드되지 않았습니다.'}), 400
        
        if processor is None or processor.processed_data is None:
            return jsonify({'error': '데이터가 로드되지 않았습니다.'}), 400
        
        print("🚀 직원별 Attention Heatmap 생성 시작")
        
        # 시퀀스 데이터 생성
        X, y, employee_ids = processor.create_proper_sequences()
        
        # 직원별 Attention 중요도 계산
        attention_results = visualizer.calculate_attention_importance_per_employee(
            model, X, y, employee_ids, processor.feature_columns, device
        )
        
        # Attention 히트맵 생성
        heatmap_html = visualizer.plot_employee_attention_heatmap(
            attention_results, processor.feature_columns, employee_ids_to_show=25
        )
        
        return heatmap_html, 200, {'Content-Type': 'text/html'}
        
    except Exception as e:
        print(f"❌ Attention Heatmap 오류: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature_importance_distribution')
def get_feature_importance_distribution():
    """
    Feature Importance 분포 분석
    """
    try:
        if model is None:
            return jsonify({'error': '모델이 로드되지 않았습니다.'}), 400
        
        if processor is None or processor.processed_data is None:
            return jsonify({'error': '데이터가 로드되지 않았습니다.'}), 400
        
        print("🚀 Feature Importance 분포 분석 시작")
        
        # 시퀀스 데이터 생성
        X, y, employee_ids = processor.create_proper_sequences()
        
        # 직원별 Attention 중요도 계산
        attention_results = visualizer.calculate_attention_importance_per_employee(
            model, X, y, employee_ids, processor.feature_columns, device
        )
        
        # 직원별 Gradient 중요도 계산
        gradient_results = visualizer.calculate_gradient_importance_per_employee(
            model, X, y, employee_ids, processor.feature_columns, device, max_samples_per_employee=3
        )
        
        # 분포 분석 시각화
        distribution_html = visualizer.plot_feature_importance_distribution(
            attention_results, gradient_results, processor.feature_columns
        )
        
        return distribution_html, 200, {'Content-Type': 'text/html'}
        
    except Exception as e:
        print(f"❌ Feature Importance 분포 분석 오류: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_analysis')
def get_model_analysis():
    """
    모델 분석 대시보드
    """
    try:
        if model is None:
            return jsonify({'error': '모델이 로드되지 않았습니다.'}), 400
        
        # 전체 데이터 예측
        X, y, employee_ids = processor.create_proper_sequences()
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y)
        
        model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), 32):
                batch_x = X_tensor[i:i+32]
                outputs = model(batch_x)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # 분석 시각화 생성
        html_plot = visualizer.plot_prediction_analysis(
            all_predictions, all_probabilities, y
        )
        
        return html_plot, 200, {'Content-Type': 'text/html'}
        
    except Exception as e:
        print(f"❌ 모델 분석 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/employee_timeline/<int:employee_id>')
def get_employee_timeline(employee_id):
    """
    개별 직원 시계열 분석
    """
    try:
        if processor is None or processor.processed_data is None:
            return jsonify({'error': '데이터가 로드되지 않았습니다.'}), 400
        
        # 직원 데이터 추출
        emp_data = processor.processed_data[
            processor.processed_data['employee_id'] == employee_id
        ].sort_values('period')
        
        if len(emp_data) == 0:
            return jsonify({'error': f'직원 ID {employee_id}를 찾을 수 없습니다.'}), 404
        
        attention_weights = None
        
        # 모델이 있으면 attention weights 계산
        if model is not None and len(emp_data) >= processor.sequence_length:
            # 시간별 집계 처리
            emp_data['year'] = emp_data[processor.date_column].dt.year
            emp_data['week'] = emp_data[processor.date_column].dt.isocalendar().week
            emp_data['time_period'] = emp_data['year'].astype(str) + '-W' + emp_data['week'].astype(str).str.zfill(2)
            
            agg_data = emp_data.groupby('time_period')[processor.feature_columns].mean().reset_index()
            agg_data = agg_data.sort_values('time_period')
            
            # 최근 시퀀스 생성
            if len(agg_data) >= processor.sequence_length:
                sequence_data = agg_data[processor.feature_columns].values[-processor.sequence_length:]
                
                # 정규화
                sequence_scaled = processor.scaler.transform(sequence_data.reshape(-1, len(processor.feature_columns)))
                sequence_scaled = sequence_scaled.reshape(1, processor.sequence_length, -1)
                
                X_tensor = torch.FloatTensor(sequence_scaled).to(device)
                
                model.eval()
                with torch.no_grad():
                    interpretation = model.interpret_prediction(X_tensor, processor.feature_columns)
                    attention_weights = interpretation['temporal_attention'][0] if interpretation['temporal_attention'].ndim > 1 else interpretation['temporal_attention']
        
        # 시각화 생성
        html_plot = visualizer.create_employee_timeline(emp_data, attention_weights)
        
        return html_plot, 200, {'Content-Type': 'text/html'}
        
    except Exception as e:
        print(f"❌ 직원 타임라인 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-analysis', methods=['POST'])
def batch_analysis():
    """배치 분석: Post 데이터로 학습 → Batch 데이터로 예측 (개별 직원 지원, 모델 캐싱 적용)"""
    global DATA_PATH, processor, model, model_cache
    
    try:
        # 요청 데이터 파싱
        request_data = request.get_json() or {}
        requested_employee_id = request_data.get('employee_id')
        
        if requested_employee_id:
            logger.info(f"🚀 Chronos 개별 직원 배치 분석 시작: 직원 {requested_employee_id}")
        else:
            logger.info("🚀 Chronos 전체 배치 분석 시작: Post 데이터 학습 → Batch 데이터 예측")
        
        # 1단계: Post 데이터 확인
        post_data_paths = get_chronos_data_paths('post')
        if not post_data_paths['timeseries'] or not os.path.exists(post_data_paths['timeseries']):
            return jsonify({"error": "Post 분석 데이터가 없습니다. 먼저 사후 분석을 수행해주세요."}), 400
        
        # 데이터 해시 계산 (변경 감지용)
        import hashlib
        with open(post_data_paths['timeseries'], 'rb') as f:
            data_hash = hashlib.md5(f.read()).hexdigest()
        
        # 캐시된 모델 확인
        if (model_cache['trained_model'] is not None and 
            model_cache['trained_processor'] is not None and
            model_cache['data_hash'] == data_hash):
            logger.info("💾 캐시된 최적화 모델 사용 (재학습 생략)")
            model = model_cache['trained_model']
            processor = model_cache['trained_processor']
            DATA_PATH = post_data_paths
        else:
            logger.info(f"📚 Post 데이터로 모델 학습: {post_data_paths['timeseries']}")
            
            # Post 데이터로 시스템 초기화
            DATA_PATH = post_data_paths
            processor = ProperTimeSeriesProcessor(sequence_length=50, aggregation_unit='week')
            
            # 데이터 로드 및 전처리
            processor.load_data(post_data_paths['timeseries'], post_data_paths.get('personas'))
            processor.detect_columns()
            processor.preprocess_data()
            processor.identify_features()
            
            # Post 데이터로 모델 학습 (하이퍼파라미터 최적화 포함)
            logger.info("🔧 Post 데이터로 모델 학습 및 하이퍼파라미터 최적화 시작")
        
        # 시퀀스 생성
        X, y, employee_ids = processor.create_proper_sequences()
        
        # 직원 기반 분할
        from chronos_processor_fixed import employee_based_train_test_split
        X_train, X_test, y_train, y_test = employee_based_train_test_split(
            X, y, employee_ids, test_ratio=0.2
        )
        
        # 하이퍼파라미터 최적화 수행 (Post 데이터로 한 번만 최적화하여 최고 성능 모델 생성)
        if OPTUNA_AVAILABLE:
            print("🚀 Post 데이터 기반 베이지안 하이퍼파라미터 최적화 시작")
            
            def objective(trial):
                # 하이퍼파라미터 제안
                gru_hidden = trial.suggest_categorical('gru_hidden', [16, 32, 64, 128])
                cnn_filters = trial.suggest_categorical('cnn_filters', [8, 16, 32, 64])
                dropout = trial.suggest_float('dropout', 0.1, 0.5)
                trial_lr = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
                trial_batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
                
                try:
                    # 모델 생성
                    trial_model = create_hybrid_model(
                        input_size=len(processor.feature_columns),
                        gru_hidden=gru_hidden,
                        cnn_filters=cnn_filters,
                        kernel_sizes=[2, 3],
                        dropout=dropout
                    )
                    trial_model.to(device)
                    
                    # 데이터 로더 (배치 크기 적용)
                    from torch.utils.data import TensorDataset, DataLoader
                    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
                    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
                    
                    trial_train_loader = DataLoader(train_dataset, batch_size=trial_batch_size, shuffle=True)
                    trial_test_loader = DataLoader(test_dataset, batch_size=trial_batch_size, shuffle=False)
                    
                    # 트레이너 설정
                    from chronos_models import ChronosModelTrainer
                    import torch.optim as optim
                    import torch.nn as nn
                    
                    trial_trainer = ChronosModelTrainer(trial_model, device)
                    trial_optimizer = optim.Adam(trial_model.parameters(), lr=trial_lr)
                    trial_criterion = nn.CrossEntropyLoss()
                    
                    # 학습 (30 에포크로 제한)
                    best_val_acc = 0
                    patience = 5
                    patience_counter = 0
                    
                    for epoch in range(30):
                        train_loss, train_acc = trial_trainer.train_epoch(trial_train_loader, trial_optimizer, trial_criterion)
                        test_results = trial_trainer.evaluate(trial_test_loader, trial_criterion)
                        val_acc = test_results['accuracy']
                        
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                break
                        
                        # Optuna pruning
                        trial.report(val_acc, epoch)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    
                    return best_val_acc
                    
                except Exception as e:
                    print(f"Trial 실패: {str(e)}")
                    return 0.0
            
            # Optuna Study 실행
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
            )
            
            n_optimization_trials = 30  # 배치 분석용으로 30회로 제한
            print(f"🚀 {n_optimization_trials}회 베이지안 최적화 시행...")
            study.optimize(objective, n_trials=n_optimization_trials, timeout=1200)  # 20분 타임아웃
            
            # 최적 하이퍼파라미터로 모델 생성
            best_params = study.best_params
            print(f"✅ 최적 하이퍼파라미터: {best_params}")
            
            model = create_hybrid_model(
                input_size=len(processor.feature_columns),
                gru_hidden=best_params['gru_hidden'],
                cnn_filters=best_params['cnn_filters'],
                kernel_sizes=[2, 3],
                dropout=best_params['dropout']
            )
            model.to(device)
            
            # 최적 설정으로 최종 학습
            from torch.utils.data import TensorDataset, DataLoader
            from chronos_models import ChronosModelTrainer
            import torch.optim as optim
            import torch.nn as nn
            
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
            
            trainer = ChronosModelTrainer(model, device)
            optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
            criterion = nn.CrossEntropyLoss()
            
            # 최종 학습
            for epoch in range(50):
                train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
                if epoch % 10 == 0:
                    print(f"최종 학습 Epoch {epoch+1}/50 - Acc: {train_acc:.4f}")
            
            training_result = {
                "message": "Post 데이터 기반 하이퍼파라미터 최적화 완료", 
                "best_params": best_params,
                "best_score": study.best_value,
                "epochs": 50
            }
            
            # 모델 캐시에 저장
            from datetime import datetime
            model_cache['trained_model'] = model
            model_cache['trained_processor'] = processor
            model_cache['model_metadata'] = training_result
            model_cache['training_timestamp'] = datetime.now().isoformat()
            model_cache['data_hash'] = data_hash
        else:
            # Optuna 없는 경우 고정 파라미터로 학습
            print("⚙️ 고정 하이퍼파라미터로 Post 데이터 학습")
            model = create_hybrid_model(
                input_size=len(processor.feature_columns),
                gru_hidden=32,
                cnn_filters=16,
                dropout=0.2
            )
            model.to(device)
            
            from torch.utils.data import TensorDataset, DataLoader
            from chronos_models import ChronosModelTrainer
            import torch.optim as optim
            import torch.nn as nn
            
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            trainer = ChronosModelTrainer(model, device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(50):
                train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
                if epoch % 10 == 0:
                    logger.info(f"학습 Epoch {epoch+1}/50 - Acc: {train_acc:.4f}")
            
            training_result = {"message": "Post 데이터 기반 고정 파라미터 학습 완료", "epochs": 50}
        
        # 모델 캐시에 저장
        from datetime import datetime
        model_cache['trained_model'] = model
        model_cache['trained_processor'] = processor
        model_cache['model_metadata'] = training_result
        model_cache['training_timestamp'] = datetime.now().isoformat()
        model_cache['data_hash'] = data_hash
        
        logger.info(f"✅ Chronos 모델 학습 완료 및 캐시 저장: {training_result}")
        
        # 2단계: Batch 데이터로 예측
        batch_data_paths = get_chronos_data_paths('batch')
        if not batch_data_paths['timeseries'] or not os.path.exists(batch_data_paths['timeseries']):
            return jsonify({"error": "Batch 분석 데이터가 없습니다. 먼저 배치 데이터를 업로드해주세요."}), 400
        
        logger.info(f"🔮 Batch 데이터로 예측 수행: {batch_data_paths['timeseries']}")
        
        # Batch 데이터 로드 및 전처리
        batch_processor = ProperTimeSeriesProcessor(sequence_length=50, aggregation_unit='week')
        batch_processor.load_data(batch_data_paths['timeseries'], batch_data_paths.get('personas'))
        batch_processor.detect_columns()
        batch_processor.preprocess_data()
        batch_processor.identify_features()
        
        # Batch 데이터 시퀀스 생성 (한 번만 처리)
        X_batch, y_batch, batch_employee_ids = batch_processor.create_proper_sequences()
        
        # 예측 수행 (학습된 모델로 전체 배치 한 번에 처리)
        model.eval()
        X_batch_tensor = torch.FloatTensor(X_batch).to(device)
        
        with torch.no_grad():
            outputs = model(X_batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = probabilities[:, 1].cpu().numpy()  # 퇴사 확률
        
        # 결과 포맷팅
        results = []
        logger.info(f"📊 Batch 데이터 로드 완료: {len(batch_employee_ids)}명")
        
        for i, (emp_id, pred) in enumerate(zip(batch_employee_ids, predictions)):
            # 개별 직원 요청인 경우 해당 직원만 처리
            if requested_employee_id and str(emp_id) != str(requested_employee_id):
                continue
                
            try:
                logger.info(f"🔮 직원 {emp_id} 예측 결과 처리 중... ({i+1}/{len(batch_employee_ids)})")
                
                # XAI 정보 생성 (Attention weights + Gradient-based)
                xai_info = {}
                try:
                    # 개별 샘플에 대한 attention weights 추출
                    sample_tensor = X_batch_tensor[i:i+1]
                    attention_weights = model.get_attention_weights(sample_tensor)
                    xai_info['attention_weights'] = attention_weights.cpu().numpy().tolist()
                    
                    # Gradient-based feature importance 계산
                    sample_tensor_grad = X_batch_tensor[i:i+1].clone().detach().requires_grad_(True)
                    output = model(sample_tensor_grad)
                    attrition_prob = torch.softmax(output, dim=1)[:, 1]
                    
                    # Gradient 계산
                    attrition_prob.backward()
                    gradients = torch.abs(sample_tensor_grad.grad).detach().cpu().numpy()
                    
                    # Feature importance (시간축 평균)
                    feature_importance = np.mean(gradients[0], axis=0)  # (sequence_length, features) -> (features,)
                    xai_info['gradient_importance'] = feature_importance.tolist()
                    
                    # 시간별 중요도 (피처축 평균)
                    temporal_importance = np.mean(gradients[0], axis=1)  # (sequence_length, features) -> (sequence_length,)
                    xai_info['temporal_importance'] = temporal_importance.tolist()
                    
                except Exception as e:
                    logger.warning(f"직원 {emp_id} XAI 정보 추출 실패: {e}")
                    xai_info['attention_weights'] = []
                    xai_info['gradient_importance'] = []
                    xai_info['temporal_importance'] = []
                
                # 시퀀스 중요도 계산 (개선된 방식)
                sequence_importance = {}
                try:
                    sequence_length = X_batch.shape[1]
                    if 'temporal_importance' in xai_info and len(xai_info['temporal_importance']) > 0:
                        # Gradient 기반 시간별 중요도 사용
                        for t in range(min(sequence_length, len(xai_info['temporal_importance']))):
                            sequence_importance[f'timestep_{t}'] = float(xai_info['temporal_importance'][t])
                    else:
                        # 기본 시간 가중치 방식
                        for t in range(sequence_length):
                            time_weight = (t + 1) / sequence_length
                            sequence_importance[f'timestep_{t}'] = float(pred * time_weight)
                except Exception as e:
                    logger.warning(f"직원 {emp_id} Sequence importance 계산 실패: {e}")
                    sequence_importance = {}
                
                # NaN 값 처리
                safe_pred = float(pred) if not (np.isnan(pred) or np.isinf(pred)) else 0.0
                
                results.append({
                    'employee_id': str(emp_id),
                    'risk_score': safe_pred,
                    'predicted_attrition': int(safe_pred > 0.5),
                    'confidence': 0.8,  # Chronos 기본 신뢰도
                    'sequence_length': batch_processor.sequence_length,
                    'features_used': len(batch_processor.feature_columns) if hasattr(batch_processor, 'feature_columns') else 0,
                    'xai_explanation': {
                        'attention_weights': xai_info.get('attention_weights', []),
                        'gradient_importance': xai_info.get('gradient_importance', []),
                        'temporal_importance': xai_info.get('temporal_importance', []),
                        'sequence_importance': sequence_importance,
                        'feature_names': batch_processor.feature_columns if hasattr(batch_processor, 'feature_columns') else [],
                        'trend_analysis': {
                            'prediction_trend': 'increasing' if safe_pred > 0.5 else 'stable',
                            'confidence_level': 'high' if abs(safe_pred - 0.5) > 0.3 else 'medium'
                        }
                    },
                    'model_metadata': {
                        'model_type': 'GRU_CNN_Hybrid',
                        'sequence_length': batch_processor.sequence_length,
                        'feature_dimensions': len(batch_processor.feature_columns) if hasattr(batch_processor, 'feature_columns') else 0,
                        'training_optimized': True  # Post 데이터로 하이퍼파라미터 최적화 수행
                    }
                })
                
                logger.info(f"✅ 직원 {emp_id} 예측 완료: 위험도 {safe_pred:.3f}")
                
            except Exception as e:
                logger.error(f"❌ 직원 {emp_id} 예측 실패: {str(e)}")
                # 실패한 직원도 결과에 포함 (Structura와 동일한 방식)
                results.append({
                    'employee_id': str(emp_id),
                    'risk_score': 0.5,  # 기본값
                    'predicted_attrition': 0,
                    'confidence': 0.1,  # 낮은 신뢰도
                    'error': str(e),
                    'xai_explanation': {
                        'attention_weights': [],
                        'gradient_importance': [],
                        'temporal_importance': [],
                        'sequence_importance': {},
                        'feature_names': [],
                        'trend_analysis': {
                            'prediction_trend': 'unknown',
                            'confidence_level': 'low'
                        }
                    },
                    'model_metadata': {
                        'model_type': 'GRU_CNN_Hybrid',
                        'sequence_length': 50,
                        'feature_dimensions': 0,
                        'training_optimized': True
                    }
                })
        
        # 개별 직원 요청이지만 결과가 없는 경우 에러 처리
        if requested_employee_id and len(results) == 0:
            return jsonify({
                "success": False,
                "error": f"직원 ID {requested_employee_id}를 배치 데이터에서 찾을 수 없습니다.",
                "available_employees": batch_employee_ids[:10].tolist() if len(batch_employee_ids) > 10 else batch_employee_ids.tolist()  # 처음 10명만 표시
            }), 404
        
        if requested_employee_id:
            logger.info(f"✅ Chronos 개별 직원 배치 분석 완료: 직원 {requested_employee_id}")
            message = f"Chronos 개별 직원 배치 분석 완료: Post 데이터 학습 → 직원 {requested_employee_id} 예측"
        else:
            logger.info(f"✅ Chronos 전체 배치 분석 완료: {len(results)}명 예측")
            message = f"Chronos 배치 분석 완료: Post 데이터 학습 → {len(results)}명 Batch 예측"
        
        response_data = {
            "success": True,
            "message": message,
            "agent": "chronos",
            "training_data_path": post_data_paths['timeseries'],
            "prediction_data_path": batch_data_paths['timeseries'],
            "total_predictions": len(results),
            "predictions": results,
            "requested_employee_id": requested_employee_id,
            "model_info": {
                "training_samples": len(X_train),
                "sequence_length": processor.sequence_length,
                "feature_dim": len(processor.feature_columns),
                "model_type": "GRU_CNN_Hybrid with Post-Data Hyperparameter Optimization"
            }
        }
        
        # NaN 값 안전 처리
        safe_response = safe_json_serialize(response_data)
        return jsonify(safe_response)
        
    except Exception as e:
        logger.error(f"❌ Chronos 배치 분석 실패: {str(e)}")
        return jsonify({"error": f"Chronos 배치 분석 실패: {str(e)}"}), 500

if __name__ == '__main__':
    print("🚀 Chronos Flask 서버 시작 중...")
    
    # 시스템 초기화
    if initialize_system():
        print("✅ 시스템 초기화 완료")
        print("🌐 서버 주소: http://localhost:5003")
        print("📋 API 문서: http://localhost:5003")
        app.run(host='0.0.0.0', port=5003, debug=True)
    else:
        print("❌ 시스템 초기화 실패")
