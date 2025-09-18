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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import io
import base64
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
        
    def load_data(self, timeseries_path, personas_path):
        """데이터 로딩"""
        print("=" * 50)
        print("데이터 로딩 중...")
        print("=" * 50)
        
        # 시계열 데이터 로드
        self.ts_data = pd.read_csv(timeseries_path)
        print(f"✅ 시계열 데이터 로드 완료: {self.ts_data.shape}")
        
        # 직원 속성 데이터 로드
        self.personas_data = pd.read_csv(personas_path)
        print(f"✅ 직원 속성 데이터 로드 완료: {self.personas_data.shape}")
        
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
        
        # 시간 순서 정렬
        merged_data = merged_data.sort_values(['employee_id', 'time_period'])
        
        sequences = []
        labels = []
        employee_ids = []
        time_points = []
        
        print("🔄 직원별 올바른 시퀀스 생성 중...")
        
        for employee_id in tqdm(merged_data[self.employee_id_col].unique(), desc="직원별 처리"):
            employee_data = merged_data[
                merged_data[self.employee_id_col] == employee_id
            ].sort_values('time_period').reset_index(drop=True)
            
            attrition_label = employee_data['attrition_binary'].iloc[0]
            
            # 충분한 데이터가 있는 경우만 처리
            min_required_length = self.sequence_length + self.prediction_horizon
            if len(employee_data) >= min_required_length:
                
                # 전체 시계열 데이터를 고정 길이로 맞춤 (패딩 또는 샘플링)
                if len(employee_data) >= self.sequence_length:
                    if len(employee_data) > self.sequence_length:
                        # 데이터가 더 길면 균등하게 샘플링
                        indices = np.linspace(0, len(employee_data)-1, self.sequence_length, dtype=int)
                        sequence_data = employee_data.iloc[indices][self.feature_columns].values
                    else:
                        # 데이터가 정확히 맞으면 그대로 사용
                        sequence_data = employee_data[self.feature_columns].values
                    
                    # 각 직원당 하나의 시퀀스만 생성
                    sequences.append(sequence_data)
                    labels.append(attrition_label)
                    employee_ids.append(employee_id)
                    time_points.append(employee_data.iloc[0]['time_period'])  # 시작 시점
        
        self.X = np.array(sequences, dtype=np.float32)
        self.y = np.array(labels, dtype=np.int64)
        self.employee_ids_seq = np.array(employee_ids)
        self.time_points = np.array(time_points)
        
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