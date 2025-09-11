# ============================================================================
# Chronos 데이터 처리 및 시각화 모듈
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import io
import base64
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ChronosDataProcessor:
    """
    Chronos 시계열 데이터 처리 클래스
    """
    def __init__(self, sequence_length: int = 6, aggregation_unit: str = 'week'):
        self.sequence_length = sequence_length
        self.aggregation_unit = aggregation_unit
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.ts_data = None
        self.hr_data = None
        self.processed_data = None
        
    def load_data(self, timeseries_path: str, hr_data_path: str):
        """
        데이터 로딩
        """
        print("🔄 데이터 로딩 중...")
        
        # 시계열 데이터 로드
        self.ts_data = pd.read_csv(timeseries_path)
        print(f"✅ 시계열 데이터 로드 완료: {self.ts_data.shape}")
        
        # 기본 HR 데이터 로드 (페르소나 정보 없이)
        self.hr_data = pd.read_csv(hr_data_path)
        print(f"✅ HR 데이터 로드 완룼: {self.hr_data.shape}")
        
        return self.ts_data.head(), self.hr_data.head()
    
    def preprocess_data(self):
        """
        데이터 전처리 및 집계
        """
        print("🔄 데이터 전처리 중...")
        
        # 날짜 컬럼 처리
        self.ts_data['date'] = pd.to_datetime(self.ts_data['date'])
        
        # 2023-2024년 필터링
        mask = (self.ts_data['date'] >= '2023-01-01') & (self.ts_data['date'] <= '2024-12-31')
        self.ts_data = self.ts_data[mask]
        
        # Attrition 정보 매핑
        attrition_map = dict(zip(self.hr_data['EmployeeNumber'], 
                               self.hr_data['Attrition'].map({'Yes': 1, 'No': 0})))
        self.ts_data['attrition'] = self.ts_data['employee_id'].map(attrition_map)
        
        # 피처 선택
        exclude_cols = ['employee_id', 'date', 'day_of_week', 'attrition',
                       'work_focused_ratio', 'meeting_collaboration_ratio', 
                       'social_dining_ratio', 'break_relaxation_ratio', 'shared_work_ratio']
        
        self.feature_columns = [col for col in self.ts_data.columns if col not in exclude_cols]
        print(f"✅ 선택된 피처: {len(self.feature_columns)}개")
        
        # 집계 처리
        if self.aggregation_unit == 'week':
            self.ts_data['period'] = self.ts_data['date'].dt.isocalendar().week + \
                                   (self.ts_data['date'].dt.year - 2023) * 52
        elif self.aggregation_unit == 'month':
            self.ts_data['period'] = self.ts_data['date'].dt.month + \
                                   (self.ts_data['date'].dt.year - 2023) * 12
        
        # 집계 수행
        agg_dict = {col: 'mean' for col in self.feature_columns}
        agg_dict['attrition'] = 'first'
        
        self.processed_data = self.ts_data.groupby(['employee_id', 'period']).agg(agg_dict).reset_index()
        
        print(f"✅ 집계 완료: {self.processed_data.shape}")
        return self.processed_data
    
    def create_sequences(self):
        """
        시퀀스 데이터 생성
        """
        print("🔄 시퀀스 생성 중...")
        
        sequences = []
        labels = []
        employee_ids = []
        
        for emp_id in self.processed_data['employee_id'].unique():
            emp_data = self.processed_data[self.processed_data['employee_id'] == emp_id].sort_values('period')
            
            if len(emp_data) >= self.sequence_length:
                # 피처 데이터 추출
                feature_data = emp_data[self.feature_columns].values
                
                # 정규화
                if len(sequences) == 0:  # 첫 번째 직원 데이터로 scaler 학습
                    feature_data_scaled = self.scaler.fit_transform(feature_data)
                else:
                    feature_data_scaled = self.scaler.transform(feature_data)
                
                # 시퀀스 생성 (마지막 sequence_length개 사용)
                sequence = feature_data_scaled[-self.sequence_length:]
                label = emp_data['attrition'].iloc[-1]
                
                sequences.append(sequence)
                labels.append(label)
                employee_ids.append(emp_id)
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        print(f"✅ 시퀀스 생성 완료: {sequences.shape}")
        print(f"   퇴사율: {np.mean(labels):.1%}")
        
        return sequences, labels, employee_ids

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

def create_matplotlib_plot_base64(plot_func, *args, **kwargs) -> str:
    """
    Matplotlib 플롯을 base64 문자열로 변환
    """
    plt.figure(figsize=(10, 6))
    plot_func(*args, **kwargs)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    
    plot_data = buffer.getvalue()
    buffer.close()
    plt.close()
    
    encoded_plot = base64.b64encode(plot_data).decode('utf-8')
    return f"data:image/png;base64,{encoded_plot}"
