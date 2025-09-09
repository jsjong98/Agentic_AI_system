# ============================================================================
# Chronos Flask 백엔드 API
# ============================================================================

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
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
warnings.filterwarnings('ignore')

# 로컬 모듈 import
from chronos_models import GRU_CNN_AttentionModel, ChronosModelTrainer, create_attention_model
from chronos_processor import ChronosDataProcessor, ChronosVisualizer

app = Flask(__name__)
CORS(app)

# 전역 변수
processor = None
model = None
visualizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 경로 설정
DATA_PATH = {
    'timeseries': 'data/IBM_HR_timeseries.csv',
    'personas': 'data/IBM_HR_personas_assigned.csv'
}

MODEL_PATH = 'app/Chronos/models'
os.makedirs(MODEL_PATH, exist_ok=True)

def initialize_system():
    """
    시스템 초기화
    """
    global processor, model, visualizer
    
    try:
        print("🚀 Chronos 시스템 초기화 중...")
        
        # 프로세서 및 시각화 도구 초기화
        processor = ChronosDataProcessor(sequence_length=6, aggregation_unit='week')
        visualizer = ChronosVisualizer()
        
        # 데이터 로드 및 전처리
        if os.path.exists(DATA_PATH['timeseries']) and os.path.exists(DATA_PATH['personas']):
            processor.load_data(DATA_PATH['timeseries'], DATA_PATH['personas'])
            processor.preprocess_data()
            print("✅ 데이터 로드 및 전처리 완료")
        else:
            print("⚠️ 데이터 파일을 찾을 수 없습니다. 학습된 모델만 사용 가능합니다.")
        
        # 기존 모델 로드 시도
        model_file = os.path.join(MODEL_PATH, 'chronos_attention_model.pth')
        if os.path.exists(model_file):
            load_model()
            print("✅ 기존 모델 로드 완료")
        else:
            print("ℹ️ 기존 모델이 없습니다. 새로운 모델을 학습해주세요.")
            
        print("🎉 Chronos 시스템 초기화 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {str(e)}")
        traceback.print_exc()
        return False

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
            model = create_attention_model(
                input_size=checkpoint['input_size'],
                gru_hidden=checkpoint.get('gru_hidden', 64),
                cnn_filters=checkpoint.get('cnn_filters', 32),
                dropout=checkpoint.get('dropout', 0.3)
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
        joblib.save(processor.scaler, scaler_file)
        
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
                    <p>Feature importance 시각화</p>
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
                    <li><strong>결과 분석:</strong> GET /api/feature_importance로 중요한 피처들을 확인하세요</li>
                    <li><strong>개별 분석:</strong> GET /api/employee_timeline/{id}로 개별 직원을 분석하세요</li>
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
            'timestamp': datetime.now().isoformat()
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

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    모델 학습
    """
    global model
    
    try:
        if processor is None or processor.ts_data is None:
            return jsonify({'error': '데이터가 로드되지 않았습니다.'}), 400
        
        # 요청 파라미터 파싱
        params = request.get_json() or {}
        sequence_length = params.get('sequence_length', 6)
        epochs = params.get('epochs', 50)
        batch_size = params.get('batch_size', 32)
        learning_rate = params.get('learning_rate', 0.001)
        
        print(f"🚀 모델 학습 시작 - Epochs: {epochs}, Batch Size: {batch_size}")
        
        # 시퀀스 길이 업데이트
        processor.sequence_length = sequence_length
        
        # 시퀀스 생성
        X, y, employee_ids = processor.create_sequences()
        
        # 텐서 변환
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # 데이터셋 분할
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor
        )
        
        # 데이터로더 생성
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 모델 생성
        input_size = len(processor.feature_columns)
        model = create_attention_model(input_size=input_size)
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

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    예측 수행
    """
    try:
        if model is None:
            return jsonify({'error': '모델이 로드되지 않았습니다. 먼저 모델을 학습시켜주세요.'}), 400
        
        if processor is None or processor.processed_data is None:
            return jsonify({'error': '데이터가 로드되지 않았습니다.'}), 400
        
        # 요청 파라미터 파싱
        params = request.get_json() or {}
        employee_ids = params.get('employee_ids', [])
        
        if not employee_ids:
            # 모든 직원 예측
            employee_ids = processor.processed_data['employee_id'].unique().tolist()
        
        # 예측 수행
        predictions = []
        
        for emp_id in employee_ids:
            emp_data = processor.processed_data[
                processor.processed_data['employee_id'] == emp_id
            ].sort_values('period')
            
            if len(emp_data) >= processor.sequence_length:
                # 피처 데이터 추출 및 정규화
                feature_data = emp_data[processor.feature_columns].values
                feature_data_scaled = processor.scaler.transform(feature_data)
                
                # 시퀀스 생성
                sequence = feature_data_scaled[-processor.sequence_length:]
                X_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
                
                # 예측 및 해석
                model.eval()
                with torch.no_grad():
                    interpretation = model.interpret_prediction(X_tensor, processor.feature_columns)
                    
                    pred_class = np.argmax(interpretation['predictions'][0])
                    pred_prob = interpretation['probabilities'][0][1]  # 퇴사 확률
                    
                    predictions.append({
                        'employee_id': int(emp_id),
                        'attrition_probability': float(pred_prob),
                        'predicted_class': int(pred_class),
                        'risk_level': 'High' if pred_prob > 0.7 else 'Medium' if pred_prob > 0.3 else 'Low',
                        'feature_importance': {
                            name: float(importance) 
                            for name, importance in zip(
                                processor.feature_columns, 
                                interpretation['feature_importance']
                            )
                        },
                        'temporal_attention': interpretation['temporal_attention'][0].tolist() if interpretation['temporal_attention'].ndim > 1 else interpretation['temporal_attention'].tolist()
                    })
        
        # 결과 정렬 (퇴사 확률 높은 순)
        predictions.sort(key=lambda x: x['attrition_probability'], reverse=True)
        
        return jsonify({
            'predictions': predictions,
            'summary': {
                'total_employees': len(predictions),
                'high_risk_count': sum(1 for p in predictions if p['risk_level'] == 'High'),
                'medium_risk_count': sum(1 for p in predictions if p['risk_level'] == 'Medium'),
                'low_risk_count': sum(1 for p in predictions if p['risk_level'] == 'Low'),
                'average_attrition_probability': np.mean([p['attrition_probability'] for p in predictions])
            }
        })
        
    except Exception as e:
        print(f"❌ 예측 오류: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature_importance')
def get_feature_importance():
    """
    Feature importance 시각화
    """
    try:
        if model is None:
            return jsonify({'error': '모델이 로드되지 않았습니다.'}), 400
        
        # 전체 데이터에 대한 평균 feature importance 계산
        X, y, employee_ids = processor.create_sequences()
        X_tensor = torch.FloatTensor(X).to(device)
        
        model.eval()
        feature_importances = []
        
        with torch.no_grad():
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

@app.route('/api/model_analysis')
def get_model_analysis():
    """
    모델 분석 대시보드
    """
    try:
        if model is None:
            return jsonify({'error': '모델이 로드되지 않았습니다.'}), 400
        
        # 전체 데이터 예측
        X, y, employee_ids = processor.create_sequences()
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
            feature_data = emp_data[processor.feature_columns].values
            feature_data_scaled = processor.scaler.transform(feature_data)
            sequence = feature_data_scaled[-processor.sequence_length:]
            X_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            
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

if __name__ == '__main__':
    print("🚀 Chronos Flask 서버 시작 중...")
    
    # 시스템 초기화
    if initialize_system():
        print("✅ 시스템 초기화 완료")
        print("🌐 서버 주소: http://localhost:5002")
        print("📋 API 문서: http://localhost:5002")
        app.run(host='0.0.0.0', port=5002, debug=True)
    else:
        print("❌ 시스템 초기화 실패")
