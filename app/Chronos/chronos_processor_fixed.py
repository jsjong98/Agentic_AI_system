# ============================================================================
# 개선된 Chronos 데이터 처리 모듈 (Chronos_analysis_fixed.py에서 추출)
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
# sklearn import with fallback
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  sklearn import 실패: {e}")
    print("   일부 기능이 제한됩니다.")
    SKLEARN_AVAILABLE = False
    
    # 기본 대체 클래스들
    class StandardScaler:
        def fit(self, X): return self
        def transform(self, X): return X
        def fit_transform(self, X): return X
    
    def train_test_split(*arrays, **kwargs):
        # 간단한 분할 구현
        test_size = kwargs.get('test_size', 0.2)
        split_idx = int(len(arrays[0]) * (1 - test_size))
        result = []
        for arr in arrays:
            result.extend([arr[:split_idx], arr[split_idx:]])
        return result
    
    def classification_report(*args, **kwargs):
        return "sklearn not available"
    
    def confusion_matrix(*args, **kwargs):
        return [[0, 0], [0, 0]]
    
    def roc_auc_score(*args, **kwargs):
        return 0.5
    
    def roc_curve(*args, **kwargs):
        return [0, 1], [0, 1], [0.5]
import joblib
import io
import base64
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ProperTimeSeriesProcessor:
    def __init__(self, sequence_length=6, prediction_horizon=4, aggregation_unit='week'):
        """
        올바른 시계열 예측을 위한 프로세서
        
        Args:
            sequence_length (int): 예측에 사용할 과거 시퀀스 길이
            prediction_horizon (int): 예측 시점 (N주 후 퇴사 여부 예측)
            aggregation_unit (str): 집계 단위 ('day', 'week', 'month')
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.aggregation_unit = aggregation_unit
        self.scaler = StandardScaler()
        self.excluded_ratio_columns = [
            'work_focused_ratio', 'meeting_collaboration_ratio', 
            'social_dining_ratio', 'break_relaxation_ratio', 'shared_work_ratio'
        ]
        self.feature_columns = []
        
        print(f"🔧 올바른 시계열 설정:")
        print(f"   시퀀스 길이: {sequence_length}{aggregation_unit[0]}")
        print(f"   예측 시점: {prediction_horizon}{aggregation_unit[0]} 후")
        print(f"   집계 단위: {aggregation_unit}")
        
    def load_data(self, timeseries_path, personas_path=None):
        """데이터 로딩"""
        print("=" * 50)
        print("데이터 로딩 중...")
        print("=" * 50)
        
        # 시계열 데이터 로드
        self.ts_data = pd.read_csv(timeseries_path)
        print(f"✅ 시계열 데이터 로드 완료: {self.ts_data.shape}")
        
        # 직원 속성 데이터 로드
        if personas_path and os.path.exists(personas_path):
            self.personas_data = pd.read_csv(personas_path)
            print(f"✅ 직원 속성 데이터 로드 완료: {self.personas_data.shape}")
        else:
            # 시계열 데이터에서 직원 정보 추출
            print("📋 시계열 데이터에서 직원 정보 추출 중...")
            unique_employees = self.ts_data[['employee_id']].drop_duplicates()
            
            # attrition_status가 있으면 사용, 없으면 기본값
            if 'attrition_status' in self.ts_data.columns:
                # 각 직원의 attrition_status 추출 (첫 번째 값 사용)
                attrition_info = self.ts_data.groupby('employee_id')['attrition_status'].first().reset_index()
                self.personas_data = pd.merge(unique_employees, attrition_info, on='employee_id')
                # 컬럼명을 표준화
                self.personas_data = self.personas_data.rename(columns={'attrition_status': 'Attrition'})
            else:
                # attrition 정보가 없으면 모두 0으로 설정
                self.personas_data = unique_employees.copy()
                self.personas_data['Attrition'] = 0
                
            print(f"✅ 직원 정보 추출 완료: {self.personas_data.shape}")
        
        return self.ts_data.head(), self.personas_data.head()

    def detect_columns(self):
        """컬럼 자동 감지 및 매칭"""
        print("=" * 50)
        print("컬럼 자동 감지 중...")
        print("=" * 50)
        
        # 날짜 컬럼 자동 감지
        date_columns = [col for col in self.ts_data.columns 
                       if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp'])]
        
        if date_columns:
            self.date_column = date_columns[0]
            print(f"🗓️  감지된 날짜 컬럼: {self.date_column}")
        else:
            raise ValueError("❌ 날짜 컬럼을 찾을 수 없습니다.")
            
        # 직원 ID 컬럼 자동 감지
        employee_id_candidates = [col for col in self.ts_data.columns 
                                 if any(keyword in col.lower() for keyword in ['employee', 'id', 'number'])]
        
        if employee_id_candidates:
            self.employee_id_col = employee_id_candidates[0]
            print(f"👤 감지된 시계열 직원 ID 컬럼: {self.employee_id_col}")
        else:
            raise ValueError("❌ 직원 ID 컬럼을 찾을 수 없습니다.")
            
        # 직원 속성 데이터에서 매칭되는 ID 컬럼 찾기
        personas_id_candidates = [col for col in self.personas_data.columns 
                                 if any(keyword in col.lower() for keyword in ['employee', 'id', 'number'])]
        
        max_overlap = 0
        best_match_col = None
        
        for col in personas_id_candidates:
            try:
                overlap = len(set(self.ts_data[self.employee_id_col]).intersection(
                             set(self.personas_data[col])))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_match_col = col
            except:
                continue
                
        if best_match_col:
            self.personas_id_col = best_match_col
            print(f"✅ 최적 매칭 컬럼: {self.personas_id_col} ({max_overlap}명 겹침)")
        else:
            raise ValueError("❌ 직원 속성 데이터에서 매칭되는 ID 컬럼을 찾을 수 없습니다.")
        
        # Attrition 컬럼 찾기
        attrition_cols = [col for col in self.personas_data.columns 
                         if 'attrition' in col.lower()]
        
        if attrition_cols:
            self.attrition_col = attrition_cols[0]
            print(f"🎯 감지된 Attrition 컬럼: {self.attrition_col}")
        else:
            raise ValueError("❌ Attrition 컬럼을 찾을 수 없습니다.")

    def preprocess_data(self):
        """데이터 전처리 및 시간 범위 분석"""
        print("=" * 50)
        print("올바른 시계열 전처리 중...")
        print("=" * 50)
        
        # 날짜 변환
        self.ts_data[self.date_column] = pd.to_datetime(self.ts_data[self.date_column])
        
        # 2023-2024년 데이터만 필터링
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        original_shape = self.ts_data.shape
        self.ts_data = self.ts_data[
            (self.ts_data[self.date_column] >= start_date) & 
            (self.ts_data[self.date_column] <= end_date)
        ].copy()
        
        print(f"📅 2023-2024년 필터링: {original_shape} → {self.ts_data.shape}")
        
        # Attrition 라벨을 0/1로 변환
        def convert_attrition(value):
            if pd.isna(value):
                return 0
            value_str = str(value).lower().strip()
            if value_str in ['yes', 'y', 'true', '1', '1.0']:
                return 1
            else:
                return 0
                
        self.personas_data['attrition_binary'] = self.personas_data[self.attrition_col].apply(convert_attrition)
        
        attrition_dist = self.personas_data['attrition_binary'].value_counts()
        print(f"\n🎯 Attrition 분포:")
        print(f"   재직(0): {attrition_dist.get(0, 0)}명")
        print(f"   퇴사(1): {attrition_dist.get(1, 0)}명")
        print(f"   퇴사율: {attrition_dist.get(1, 0) / len(self.personas_data) * 100:.1f}%")

    def identify_features(self):
        """피처 식별"""
        print("=" * 50)
        print("피처 식별 중...")
        print("=" * 50)
        
        # 수치형 컬럼 식별
        numeric_columns = self.ts_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 제외할 컬럼들 제거
        exclude_columns = [self.employee_id_col] + self.excluded_ratio_columns
        exclude_columns = [col for col in exclude_columns if col in numeric_columns]
        
        self.feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        print(f"🔍 전체 수치형 컬럼: {len(numeric_columns)}개")
        print(f"❌ 제외된 컬럼: {exclude_columns}")
        print(f"✅ 사용할 피처: {len(self.feature_columns)}개")

    def create_proper_sequences(self):
        """올바른 시계열 시퀀스 생성 - 시간 순서 고려"""
        print("=" * 50)
        print("올바른 시계열 시퀀스 생성 중...")
        print("=" * 50)
        
        # 시간별 집계
        if self.aggregation_unit == 'week':
            self.ts_data['year'] = self.ts_data[self.date_column].dt.year
            self.ts_data['week'] = self.ts_data[self.date_column].dt.isocalendar().week
            self.ts_data['time_period'] = self.ts_data['year'].astype(str) + '-W' + self.ts_data['week'].astype(str).str.zfill(2)
        
        # 시간별 집계
        agg_data = self.ts_data.groupby([self.employee_id_col, 'time_period'])[self.feature_columns].mean().reset_index()
        
        # 직원 속성 데이터와 병합
        merged_data = pd.merge(
            agg_data,
            self.personas_data[[self.personas_id_col, 'attrition_binary']],
            left_on=self.employee_id_col,
            right_on=self.personas_id_col,
            how='inner'
        )
        
        # Flask 백엔드 호환성을 위한 컬럼명 표준화
        merged_data = merged_data.rename(columns={
            self.employee_id_col: 'employee_id',
            'time_period': 'period'
        })
        
        # 시간 순서 정렬
        merged_data = merged_data.sort_values(['employee_id', 'period'])
        
        sequences = []
        labels = []
        employee_ids = []
        time_points = []
        
        print("🔄 직원별 올바른 시퀀스 생성 중...")
        
        for employee_id in tqdm(merged_data['employee_id'].unique(), desc="직원별 처리"):
            employee_data = merged_data[
                merged_data['employee_id'] == employee_id
            ].sort_values('period').reset_index(drop=True)
            
            attrition_label = employee_data['attrition_binary'].iloc[0]
            
            # 충분한 데이터가 있는 경우만 처리
            min_required_length = self.sequence_length + self.prediction_horizon
            if len(employee_data) >= min_required_length:
                
                # 전체 시계열 데이터 사용 (샘플링 제거)
                if len(employee_data) >= self.sequence_length:
                    # 최신 데이터를 사용하여 시퀀스 생성 (샘플링 대신 최근 데이터 사용)
                    sequence_data = employee_data[self.feature_columns].values[-self.sequence_length:]
                    
                    # 각 직원당 하나의 시퀀스만 생성
                    sequences.append(sequence_data)
                    labels.append(attrition_label)
                    employee_ids.append(employee_id)
                    time_points.append(employee_data.iloc[0]['period'])  # 시작 시점
        
        self.X = np.array(sequences, dtype=np.float32)
        self.y = np.array(labels, dtype=np.int64)
        self.employee_ids_seq = np.array(employee_ids)
        self.time_points = np.array(time_points)
        
        # Scaler를 피처 데이터에 fit
        if len(sequences) > 0:
            # 모든 시퀀스의 피처 데이터를 하나로 합쳐서 scaler fit
            all_features = np.concatenate([seq for seq in sequences], axis=0)
            self.scaler.fit(all_features)
            print(f"✅ StandardScaler fit 완료: {all_features.shape}")
        
        # Flask 백엔드 호환성을 위한 processed_data 속성 추가
        self.processed_data = merged_data
        
        print(f"✅ 올바른 시퀀스 생성 완료:")
        print(f"   총 시퀀스: {len(self.X)}개")
        print(f"   시퀀스 형태: {self.X.shape}")
        print(f"   퇴사 라벨 비율: {np.mean(self.y) * 100:.1f}%")
        
        return self.X, self.y, self.employee_ids_seq

def employee_based_train_test_split(X, y, employee_ids, test_ratio=0.2):
    """직원 기반 train/test 분할 (시간 순서 고려하면서 클래스 균형 유지)"""
    
    # 직원별 라벨 확인
    unique_employees = np.unique(employee_ids)
    employee_labels = []
    
    for emp_id in unique_employees:
        emp_sequences = y[employee_ids == emp_id]
        # 해당 직원의 시퀀스 중 하나라도 positive면 퇴사 직원으로 분류
        emp_label = 1 if np.any(emp_sequences == 1) else 0
        employee_labels.append(emp_label)
    
    employee_labels = np.array(employee_labels)
    
    print(f"📊 직원별 라벨 분포:")
    print(f"   총 직원 수: {len(unique_employees)}명")
    print(f"   퇴사 직원: {np.sum(employee_labels)}명")
    print(f"   재직 직원: {np.sum(employee_labels == 0)}명")
    
    # 직원 레벨에서 stratified split
    try:
        train_employees, test_employees = train_test_split(
            unique_employees, test_size=test_ratio, random_state=42, 
            stratify=employee_labels
        )
    except ValueError:
        # stratify가 불가능한 경우 일반 분할
        train_employees, test_employees = train_test_split(
            unique_employees, test_size=test_ratio, random_state=42
        )
    
    # 직원 기반으로 시퀀스 분할
    train_mask = np.isin(employee_ids, train_employees)
    test_mask = np.isin(employee_ids, test_employees)
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"👥 직원 기반 분할 완료:")
    print(f"   훈련 직원: {len(train_employees)}명")
    print(f"   테스트 직원: {len(test_employees)}명")
    print(f"   훈련 시퀀스: {len(X_train)}개 (퇴사율: {np.mean(y_train)*100:.1f}%)")
    print(f"   테스트 시퀀스: {len(X_test)}개 (퇴사율: {np.mean(y_test)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

# 시각화 클래스는 기존 chronos_processor.py에서 가져옴
class ChronosVisualizer:
    """
    Chronos 시각화 클래스
    """
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_feature_importance(self, feature_importance: np.ndarray, feature_names: List[str], 
                              title: str = "Feature Importance") -> str:
        """
        Feature importance 시각화
        """
        # Plotly를 사용한 인터랙티브 차트
        fig = go.Figure(data=[
            go.Bar(
                x=feature_importance,
                y=feature_names,
                orientation='h',
                marker=dict(
                    color=feature_importance,
                    colorscale='Viridis',
                    showscale=True
                )
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(feature_names) * 25),
            template="plotly_white"
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def plot_temporal_attention(self, attention_weights: np.ndarray, 
                              sequence_length: int, title: str = "Temporal Attention") -> str:
        """
        시계열 attention 시각화
        """
        # 평균 attention weights 계산
        avg_attention = np.mean(attention_weights, axis=0) if attention_weights.ndim > 1 else attention_weights
        
        time_steps = [f"Week -{sequence_length-i}" for i in range(sequence_length)]
        
        fig = go.Figure(data=[
            go.Scatter(
                x=time_steps,
                y=avg_attention,
                mode='lines+markers',
                line=dict(width=3, color='#1f77b4'),
                marker=dict(size=8, color='#ff7f0e')
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Time Steps",
            yaxis_title="Attention Weight",
            template="plotly_white",
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def plot_prediction_analysis(self, predictions: np.ndarray, probabilities: np.ndarray, 
                               labels: np.ndarray) -> str:
        """
        예측 결과 분석 시각화
        """
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'ROC Curve', 
                          'Prediction Distribution', 'Probability Distribution'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # 1. Confusion Matrix
        cm = confusion_matrix(labels, predictions)
        fig.add_trace(
            go.Heatmap(z=cm, colorscale='Blues', showscale=False),
            row=1, col=1
        )
        
        # 2. ROC Curve
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
            auc_score = roc_auc_score(labels, probabilities[:, 1])
            
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines', 
                          name=f'ROC (AUC = {auc_score:.3f})'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                          line=dict(dash='dash'), name='Random'),
                row=1, col=2
            )
        
        # 3. Prediction Distribution
        fig.add_trace(
            go.Histogram(x=predictions, name='Predictions', nbinsx=10),
            row=2, col=1
        )
        
        # 4. Probability Distribution
        fig.add_trace(
            go.Histogram(x=probabilities[:, 1], name='Attrition Probability', nbinsx=20),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Prediction Analysis Dashboard",
            height=800,
            template="plotly_white",
            showlegend=False
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def create_employee_timeline(self, employee_data: pd.DataFrame, 
                               attention_weights: np.ndarray = None) -> str:
        """
        개별 직원의 시계열 데이터 시각화
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Feature Timeline', 'Attention Weights'),
            vertical_spacing=0.1
        )
        
        # Feature timeline
        for i, col in enumerate(employee_data.columns[2:]):  # Skip employee_id and period
            fig.add_trace(
                go.Scatter(
                    x=employee_data['period'],
                    y=employee_data[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Attention weights
        if attention_weights is not None:
            fig.add_trace(
                go.Bar(
                    x=employee_data['period'][-len(attention_weights):],
                    y=attention_weights,
                    name='Attention',
                    marker_color='red',
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Employee Timeline Analysis",
            height=600,
            template="plotly_white"
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def calculate_attention_importance_per_employee(self, model, X_test, y_test, employee_ids, feature_names, device):
        """직원별 Attention 기반 Feature Importance 계산"""
        print("🔍 직원별 Attention Feature Importance 계산 중...")
        
        employee_attention_results = {}
        
        # PyTorch 텐서로 변환
        if isinstance(X_test, np.ndarray):
            X_tensor = torch.FloatTensor(X_test).to(device)
        else:
            X_tensor = X_test.to(device)
        
        model.eval()
        with torch.no_grad():
            # 전체 배치에 대한 attention weights 추출
            attention_weights = model.get_attention_weights(X_tensor)  # (batch_size, sequence_length)
            attention_weights_np = attention_weights.cpu().numpy()
        
        # 직원별로 결과 정리
        unique_employees = np.unique(employee_ids)
        
        for emp_id in unique_employees:
            emp_mask = employee_ids == emp_id
            emp_attention = attention_weights_np[emp_mask]
            
            if len(emp_attention) > 0:
                # 해당 직원의 평균 attention
                mean_attention = np.mean(emp_attention, axis=0)  # (sequence_length,)
                
                # 각 피처에 대한 attention 중요도 (모든 시간 단계에서 동일하게 적용)
                feature_attention_importance = np.tile(mean_attention, (len(feature_names), 1)).T
                overall_attention_importance = np.mean(feature_attention_importance, axis=0)
                
                employee_attention_results[emp_id] = {
                    'attention_weights': mean_attention,
                    'feature_importance': overall_attention_importance,
                    'attrition_label': y_test[emp_mask][0] if len(y_test[emp_mask]) > 0 else 0
                }
        
        print(f"✅ {len(employee_attention_results)}명의 직원별 Attention Importance 계산 완료")
        return employee_attention_results
    
    def calculate_gradient_importance_per_employee(self, model, X_test, y_test, employee_ids, feature_names, device, max_samples_per_employee=10):
        """직원별 Gradient 기반 Feature Importance 계산"""
        print("🔍 직원별 Gradient Feature Importance 계산 중...")
        
        employee_gradient_results = {}
        unique_employees = np.unique(employee_ids)
        
        model.eval()
        
        for emp_id in unique_employees:
            emp_mask = employee_ids == emp_id
            emp_X = X_test[emp_mask]
            emp_y = y_test[emp_mask]
            
            if len(emp_X) == 0:
                continue
            
            # 샘플 수 제한
            if len(emp_X) > max_samples_per_employee:
                sample_indices = np.random.choice(len(emp_X), max_samples_per_employee, replace=False)
                emp_X_sample = emp_X[sample_indices]
            else:
                emp_X_sample = emp_X
            
            # PyTorch 텐서로 변환 (gradient 계산을 위해 requires_grad=True)
            X_tensor = torch.FloatTensor(emp_X_sample).to(device)
            X_tensor.requires_grad_(True)
            
            try:
                # Forward pass
                outputs = model(X_tensor)
                
                # 퇴사 클래스(1)에 대한 확률
                probs = torch.softmax(outputs, dim=1)
                attrition_probs = probs[:, 1]
                
                # 각 샘플에 대한 gradient 계산
                gradients_list = []
                
                for i in range(len(attrition_probs)):
                    if X_tensor.grad is not None:
                        X_tensor.grad.zero_()
                    
                    attrition_probs[i].backward(retain_graph=True)
                    
                    # Gradient의 절댓값을 중요도로 사용
                    sample_gradient = torch.abs(X_tensor.grad[i]).detach().cpu().numpy()
                    gradients_list.append(sample_gradient)
                
                # 모든 샘플의 gradient 평균
                mean_gradients = np.mean(gradients_list, axis=0)  # (sequence_length, n_features)
                
                # 각 피처의 전체 시간에 대한 평균 gradient 중요도
                feature_gradient_importance = np.mean(mean_gradients, axis=0)
                
                employee_gradient_results[emp_id] = {
                    'gradient_by_timestep': mean_gradients,
                    'feature_importance': feature_gradient_importance,
                    'attrition_label': emp_y[0] if len(emp_y) > 0 else 0
                }
                
            except Exception as e:
                # 로그 스팸 방지 - 첫 번째 오류만 출력
                if emp_id == list(unique_employees)[0]:
                    print(f"⚠️ Gradient 계산 실패 (대표 오류): {e}")
                    print("⚠️ 모든 직원에 대해 동일한 오류가 발생할 수 있습니다. 로그 스팸 방지를 위해 추가 오류는 출력하지 않습니다.")
                continue
        
        print(f"✅ {len(employee_gradient_results)}명의 직원별 Gradient Importance 계산 완료")
        return employee_gradient_results
    
    def plot_employee_feature_importance_comparison(self, attention_results, gradient_results, feature_names, 
                                                  top_employees=10, top_features=15, save_path=None):
        """직원별 Feature Importance 비교 시각화"""
        print("📊 직원별 Feature Importance 비교 시각화 중...")
        
        # 퇴사 확률이 높은 상위 직원들 선택
        employee_scores = {}
        for emp_id in attention_results.keys():
            if emp_id in gradient_results:
                # Attention과 Gradient 중요도의 평균으로 종합 점수 계산
                att_score = np.max(attention_results[emp_id]['feature_importance'])
                grad_score = np.max(gradient_results[emp_id]['feature_importance'])
                employee_scores[emp_id] = (att_score + grad_score) / 2
        
        # 상위 직원들 선택
        top_employee_ids = sorted(employee_scores.keys(), key=lambda x: employee_scores[x], reverse=True)[:top_employees]
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=len(top_employee_ids), cols=2,
            subplot_titles=[f'Employee {emp_id} - Attention' if i % 2 == 0 else f'Employee {emp_id} - Gradient' 
                          for emp_id in top_employee_ids for i in range(2)],
            vertical_spacing=0.02,
            horizontal_spacing=0.1,
            specs=[[{"type": "bar"}, {"type": "bar"}] for _ in range(len(top_employee_ids))]
        )
        
        colors = px.colors.qualitative.Set3
        
        for idx, emp_id in enumerate(top_employee_ids):
            # Attention 중요도
            att_importance = attention_results[emp_id]['feature_importance']
            att_sorted_indices = np.argsort(att_importance)[-top_features:]
            att_top_features = [feature_names[i] for i in att_sorted_indices]
            att_top_values = att_importance[att_sorted_indices]
            
            fig.add_trace(
                go.Bar(
                    x=att_top_values,
                    y=att_top_features,
                    orientation='h',
                    name=f'Emp {emp_id} Attention',
                    marker_color=colors[idx % len(colors)],
                    showlegend=False
                ),
                row=idx+1, col=1
            )
            
            # Gradient 중요도
            grad_importance = gradient_results[emp_id]['feature_importance']
            grad_sorted_indices = np.argsort(grad_importance)[-top_features:]
            grad_top_features = [feature_names[i] for i in grad_sorted_indices]
            grad_top_values = grad_importance[grad_sorted_indices]
            
            fig.add_trace(
                go.Bar(
                    x=grad_top_values,
                    y=grad_top_features,
                    orientation='h',
                    name=f'Emp {emp_id} Gradient',
                    marker_color=colors[idx % len(colors)],
                    showlegend=False
                ),
                row=idx+1, col=2
            )
        
        fig.update_layout(
            title=f"Top {top_employees} Employees - Feature Importance Comparison (Attention vs Gradient)",
            height=200 * len(top_employee_ids),
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"💾 직원별 Feature Importance 비교 그래프 저장: {save_path}")
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def plot_employee_attention_heatmap(self, attention_results, feature_names, employee_ids_to_show=20, save_path=None):
        """직원별 Attention Weights 히트맵"""
        print("📊 직원별 Attention Heatmap 시각화 중...")
        
        # 상위 직원들 선택 (attention 점수 기준)
        employee_scores = {emp_id: np.max(results['feature_importance']) 
                          for emp_id, results in attention_results.items()}
        top_employees = sorted(employee_scores.keys(), key=lambda x: employee_scores[x], reverse=True)[:employee_ids_to_show]
        
        # 히트맵 데이터 준비
        attention_matrix = []
        employee_labels = []
        
        for emp_id in top_employees:
            attention_weights = attention_results[emp_id]['attention_weights']
            attention_matrix.append(attention_weights)
            attrition_status = "퇴사" if attention_results[emp_id]['attrition_label'] == 1 else "재직"
            employee_labels.append(f"직원 {emp_id} ({attrition_status})")
        
        attention_matrix = np.array(attention_matrix)
        
        # 시간 단계 라벨
        time_labels = [f"Week -{len(attention_weights)-i}" for i in range(len(attention_weights))]
        
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=time_labels,
            y=employee_labels,
            colorscale='YlOrRd',
            colorbar=dict(title="Attention Weight")
        ))
        
        fig.update_layout(
            title=f"Top {employee_ids_to_show} Employees - Attention Weights Heatmap",
            xaxis_title="Time Steps",
            yaxis_title="Employees",
            height=max(600, employee_ids_to_show * 25),
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"💾 Attention Heatmap 저장: {save_path}")
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def plot_feature_importance_distribution(self, attention_results, gradient_results, feature_names, save_path=None):
        """Feature Importance 분포 분석"""
        print("📊 Feature Importance 분포 분석 중...")
        
        # 모든 직원의 feature importance 수집
        all_attention_importance = []
        all_gradient_importance = []
        
        for emp_id in attention_results.keys():
            if emp_id in gradient_results:
                all_attention_importance.append(attention_results[emp_id]['feature_importance'])
                all_gradient_importance.append(gradient_results[emp_id]['feature_importance'])
        
        all_attention_importance = np.array(all_attention_importance)
        all_gradient_importance = np.array(all_gradient_importance)
        
        # 각 피처별 평균 중요도 계산
        mean_attention = np.mean(all_attention_importance, axis=0)
        mean_gradient = np.mean(all_gradient_importance, axis=0)
        std_attention = np.std(all_attention_importance, axis=0)
        std_gradient = np.std(all_gradient_importance, axis=0)
        
        # 상위 피처들 선택
        top_features_idx = np.argsort(mean_attention + mean_gradient)[-20:]
        top_feature_names = [feature_names[i] for i in top_features_idx]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Attention Importance Distribution', 'Gradient Importance Distribution',
                          'Feature Importance Correlation', 'Top Features Comparison'),
            specs=[[{"type": "box"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Attention Importance Box Plot
        for i, feature_idx in enumerate(top_features_idx[-10:]):  # 상위 10개만
            fig.add_trace(
                go.Box(y=all_attention_importance[:, feature_idx], 
                      name=feature_names[feature_idx],
                      showlegend=False),
                row=1, col=1
            )
        
        # 2. Gradient Importance Box Plot  
        for i, feature_idx in enumerate(top_features_idx[-10:]):  # 상위 10개만
            fig.add_trace(
                go.Box(y=all_gradient_importance[:, feature_idx], 
                      name=feature_names[feature_idx],
                      showlegend=False),
                row=1, col=2
            )
        
        # 3. Correlation Scatter Plot
        fig.add_trace(
            go.Scatter(
                x=mean_attention[top_features_idx],
                y=mean_gradient[top_features_idx],
                mode='markers+text',
                text=[f'F{i}' for i in range(len(top_features_idx))],
                textposition="top center",
                marker=dict(size=8, color='blue'),
                name='Features',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Top Features Comparison
        fig.add_trace(
            go.Bar(
                x=top_feature_names,
                y=mean_attention[top_features_idx],
                name='Attention',
                marker_color='lightblue',
                error_y=dict(type='data', array=std_attention[top_features_idx])
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=top_feature_names,
                y=mean_gradient[top_features_idx],
                name='Gradient',
                marker_color='lightcoral',
                error_y=dict(type='data', array=std_gradient[top_features_idx])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Feature Importance Distribution Analysis",
            height=800,
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"💾 Feature Importance 분포 분석 저장: {save_path}")
        
        return fig.to_html(include_plotlyjs='cdn')