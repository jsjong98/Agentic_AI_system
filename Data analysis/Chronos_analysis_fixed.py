# ============================================================================
# 올바른 시계열 예측을 위한 수정된 Chronos 분석 코드
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# PyTorch 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

# 하이퍼파라미터 최적화
import optuna
from optuna.trial import TrialState

# 기타 필요한 라이브러리
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import os
import json

# 시드 설정
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

# 플롯 설정
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Optuna 로깅 설정
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# 올바른 시계열 데이터 처리 클래스
# ============================================================================

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
                
                # 올바른 접근: 각 직원당 하나의 전체 시계열 패턴
                # 1470명 각각의 완전한 시계열 데이터를 학습
                
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

# ============================================================================
# 모델 정의 (기존과 동일)
# ============================================================================

class GRU_CNN_HybridModel(nn.Module):
    """파라미터화된 GRU+CNN 하이브리드 모델"""
    
    def __init__(self, input_size, gru_hidden=32, cnn_filters=16, kernel_sizes=[2, 3], dropout=0.2):
        super(GRU_CNN_HybridModel, self).__init__()
        
        # GRU 브랜치
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=1,
            dropout=dropout,
            batch_first=True
        )
        self.gru_dropout = nn.Dropout(dropout)
        
        # CNN 브랜치
        self.conv_layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv_layer = nn.Sequential(
                nn.Conv1d(input_size, cnn_filters, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(cnn_filters),
                nn.AdaptiveMaxPool1d(1)
            )
            self.conv_layers.append(conv_layer)
        
        # 분류기
        combined_features = gru_hidden + len(kernel_sizes) * cnn_filters
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
        
        # 어텐션
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # GRU + 어텐션
        gru_out, _ = self.gru(x)
        attention_weights = self.attention(gru_out)
        gru_features = torch.sum(gru_out * attention_weights, dim=1)
        gru_features = self.gru_dropout(gru_features)
        
        # CNN
        x_cnn = x.transpose(1, 2)
        cnn_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x_cnn).squeeze(-1)
            cnn_outputs.append(conv_out)
        
        cnn_features = torch.cat(cnn_outputs, dim=1)
        
        # 결합 및 분류
        combined_features = torch.cat([gru_features, cnn_features], dim=1)
        output = self.classifier(combined_features)
        return output
    
    def get_attention_weights(self, x):
        """Attention weights 추출 (Feature Importance용)"""
        self.eval()
        with torch.no_grad():
            gru_out, _ = self.gru(x)
            attention_weights = self.attention(gru_out)
            return attention_weights.squeeze(-1)  # (batch_size, sequence_length)

# ============================================================================
# 시간 기반 데이터 분할 함수
# ============================================================================

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

# ============================================================================
# 메인 실행 코드
# ============================================================================

if __name__ == "__main__":
    print("🚀 올바른 시계열 예측 시작")
    print("=" * 70)
    
    # 프로세서 초기화 (더 긴 시퀀스로 전체 패턴 캡처)
    processor = ProperTimeSeriesProcessor(
        sequence_length=50,  # 50주 = 약 1년간의 패턴
        prediction_horizon=4,  # 사용하지 않지만 호환성 유지
        aggregation_unit='week'
    )
    
    # 데이터 로딩 (실제 파일명으로 수정)
    ts_sample, personas_sample = processor.load_data(
        '../data/IBM_HR_timeseries.csv', 
        '../data/IBM_HR_personas_assigned.csv'
    )
    
    # 전처리
    processor.detect_columns()
    processor.preprocess_data()
    processor.identify_features()
    
    # 올바른 시퀀스 생성
    X, y, employee_ids = processor.create_proper_sequences()
    
    # 직원 기반 분할 (클래스 균형 유지)
    X_train, X_test, y_train, y_test = employee_based_train_test_split(
        X, y, employee_ids, test_ratio=0.2
    )
    
    print(f"\n🎯 올바른 시계열 예측 준비 완료!")
    print(f"   이제 현실적인 성능 지표를 확인할 수 있습니다.")
    print(f"   예상 성능: AUC 0.6-0.8 (현실적인 범위)")
    
    # 전체 파이프라인 실행은 함수들이 정의된 후에 수행됩니다.

# ============================================================================
# 하이퍼파라미터 최적화 함수들
# ============================================================================

def prepare_data_for_training(X_train, y_train, X_test, y_test, val_size=0.2):
    """훈련용 데이터 준비"""
    # 시간 기반으로 이미 분할된 데이터를 사용하므로 추가 분할 없이 진행
    # Validation은 train에서 시간 순서를 고려하여 분할
    split_idx = int(len(X_train) * (1 - val_size))
    
    X_train_split = X_train[:split_idx]
    X_val = X_train[split_idx:]
    y_train_split = y_train[:split_idx]
    y_val = y_train[split_idx:]
    
    # 클래스 분포 확인
    print(f"📊 시간 기반 분할 후 클래스 분포:")
    print(f"   Train: {np.bincount(y_train_split)} (퇴사율: {np.mean(y_train_split)*100:.1f}%)")
    print(f"   Val: {np.bincount(y_val)} (퇴사율: {np.mean(y_val)*100:.1f}%)")
    print(f"   Test: {np.bincount(y_test)} (퇴사율: {np.mean(y_test)*100:.1f}%)")
    
    # 정규화
    scaler = StandardScaler()
    n_samples, n_timesteps, n_features = X_train_split.shape
    X_train_reshaped = X_train_split.reshape(-1, n_features)
    
    scaler.fit(X_train_reshaped)
    
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(n_samples, n_timesteps, n_features)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    
    # PyTorch 텐서로 변환
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train_split)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 클래스 가중치 (안전한 처리)
    unique_classes = np.unique(y_train_split)
    if len(unique_classes) == 2:
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_split)
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    else:
        # 한 클래스만 있는 경우 가중치 없이 처리
        class_weights_tensor = None
        print(f"⚠️  경고: 훈련 데이터에 {len(unique_classes)}개 클래스만 존재합니다.")
    
    return (X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, 
            X_test_tensor, y_test_tensor, class_weights_tensor, scaler)

def train_model(model, X_train, y_train, X_val, y_val, class_weights, 
                lr=0.001, epochs=30, batch_size=64, patience=10):
    """모델 학습 함수"""
    
    # 데이터 로더 생성
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 학습 설정
    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # 학습
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                val_predictions.extend(probs[:, 1].cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    # 최고 모델 로드
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 최종 검증 점수 계산
    model.eval()
    val_predictions = []
    val_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            probs = F.softmax(outputs, dim=1)
            val_predictions.extend(probs[:, 1].cpu().numpy())
            val_labels.extend(batch_y.cpu().numpy())
    
    try:
        auc_score = roc_auc_score(val_labels, val_predictions)
    except:
        auc_score = 0.5
    
    return auc_score

def objective_hybrid(trial, X_train, y_train, feature_count):
    """하이퍼파라미터 최적화 목적함수"""
    
    # 하이퍼파라미터 샘플링
    gru_hidden = trial.suggest_int('gru_hidden', 16, 128, step=16)
    cnn_filters = trial.suggest_int('cnn_filters', 8, 64, step=8)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # 데이터 준비
    try:
        (X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, 
         _, _, class_weights, _) = prepare_data_for_training(X_train, y_train, X_train, y_train)
    except:
        return 0.5
    
    # 모델 생성
    model = GRU_CNN_HybridModel(
        input_size=feature_count,
        gru_hidden=gru_hidden,
        cnn_filters=cnn_filters,
        kernel_sizes=[2, 3],
        dropout=dropout
    ).to(device)
    
    # 학습 및 평가
    auc_score = train_model(
        model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, class_weights,
        lr=lr, epochs=25, batch_size=batch_size, patience=8
    )
    
    return auc_score

def run_hyperparameter_optimization(X_train, y_train, feature_count, n_trials=30):
    """하이퍼파라미터 최적화 실행"""
    print("🚀 하이퍼파라미터 최적화 시작")
    print("=" * 50)
    print(f"총 {n_trials}회 시도로 최적 파라미터 탐색 중...")
    
    # Optuna 스터디 생성
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # 최적화 실행
    study.optimize(lambda trial: objective_hybrid(trial, X_train, y_train, feature_count), 
                   n_trials=n_trials, show_progress_bar=True)
    
    # 결과 출력
    print("\n🏆 최적화 완료!")
    print("=" * 50)
    print(f"✅ 최고 AUC 스코어: {study.best_value:.4f}")
    print(f"🎯 최적 파라미터:")
    
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    return study

def evaluate_final_model(model, X_test, y_test, model_name="Fixed Hybrid Model"):
    """최종 모델 평가"""
    model.eval()
    
    dataset = TensorDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_predictions == all_labels)
    try:
        auc_score = roc_auc_score(all_labels, all_probabilities)
    except:
        auc_score = 0.5
    
    print(f"\n🎯 {model_name} 최종 평가")
    print("=" * 50)
    print(f"📊 테스트 결과:")
    print(f"   정확도: {accuracy:.4f}")
    print(f"   AUC: {auc_score:.4f}")
    
    print(f"\n📋 테스트 세트 분류 리포트:")
    print(classification_report(all_labels, all_predictions, target_names=['재직', '퇴사']))
    
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"\n🔍 혼동 행렬:")
    print(cm)
    
    return {
        'accuracy': accuracy,
        'auc': auc_score,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'labels': all_labels,
        'confusion_matrix': cm
    }

def predict_employee_with_fixed_model(employee_id, processor, model, scaler, sequence_length):
    """수정된 모델로 개별 직원 예측"""
    
    # 해당 직원의 데이터 찾기
    employee_data = processor.ts_data[
        processor.ts_data[processor.employee_id_col] == employee_id
    ].sort_values(processor.date_column)
    
    if len(employee_data) < sequence_length:
        raise ValueError(f"직원 {employee_id}의 데이터가 충분하지 않습니다.")
    
    # 시간별 집계
    employee_data['year'] = employee_data[processor.date_column].dt.year
    employee_data['week'] = employee_data[processor.date_column].dt.isocalendar().week
    employee_data['time_period'] = employee_data['year'].astype(str) + '-W' + employee_data['week'].astype(str).str.zfill(2)
    
    agg_data = employee_data.groupby('time_period')[processor.feature_columns].mean().reset_index()
    agg_data = agg_data.sort_values('time_period')
    
    # 최근 시퀀스 생성
    sequence_data = agg_data[processor.feature_columns].values[-sequence_length:]
    
    # 정규화
    sequence_scaled = scaler.transform(sequence_data.reshape(-1, len(processor.feature_columns)))
    sequence_scaled = sequence_scaled.reshape(1, sequence_length, -1)
    
    # PyTorch 텐서로 변환
    sequence_tensor = torch.FloatTensor(sequence_scaled).to(device)
    
    # 예측
    model.eval()
    with torch.no_grad():
        outputs = model(sequence_tensor)
        probabilities = F.softmax(outputs, dim=1)
        attrition_prob = probabilities[0, 1].item()
        prediction = torch.argmax(outputs, dim=1).item()
    
    return {
        'employee_id': employee_id,
        'attrition_probability': attrition_prob,
        'prediction': prediction,
        'risk_level': 'High' if attrition_prob > 0.7 else 'Medium' if attrition_prob > 0.3 else 'Low',
        'sequence_length_used': sequence_length
    }

def predict_all_employees_and_save_fixed(processor, model, scaler, sequence_length, 
                                        save_path='Data analysis/data/employee_attrition_predictions_fixed.csv'):
    """전체 직원에 대한 수정된 예측 수행 및 CSV 저장"""
    print("🔮 전체 직원 수정된 시계열 예측 수행")
    print("=" * 50)
    
    # 모든 고유 직원 ID 가져오기
    all_employees = processor.ts_data[processor.employee_id_col].unique()
    print(f"📊 총 직원 수: {len(all_employees)}명")
    
    # 예측 결과 저장할 리스트
    all_predictions = []
    successful_predictions = 0
    failed_predictions = 0
    
    print("🚀 예측 진행 중...")
    
    # 진행률 표시를 위한 tqdm 사용
    for emp_id in tqdm(all_employees, desc="직원별 예측"):
        try:
            result = predict_employee_with_fixed_model(emp_id, processor, model, scaler, sequence_length)
            result['prediction_method'] = 'Fixed Timeseries Model'
            all_predictions.append(result)
            successful_predictions += 1
            
        except Exception as e:
            # 예측 실패한 경우 기본값으로 저장
            failed_result = {
                'employee_id': emp_id,
                'attrition_probability': 0.5,
                'prediction': 0,
                'risk_level': 'Unknown',
                'sequence_length_used': sequence_length,
                'prediction_method': 'Failed',
                'prediction_status': 'Failed',
                'error_message': str(e)
            }
            all_predictions.append(failed_result)
            failed_predictions += 1
    
    print(f"\n📊 예측 완료 요약:")
    print(f"   ✅ 성공: {successful_predictions}명")
    print(f"   ❌ 실패: {failed_predictions}명")
    print(f"   📈 성공률: {successful_predictions/len(all_employees)*100:.1f}%")
    
    # DataFrame으로 변환
    predictions_df = pd.DataFrame(all_predictions)
    
    # 퇴사 확률 기준으로 정렬 (내림차순)
    predictions_df = predictions_df.sort_values('attrition_probability', ascending=False)
    
    # CSV 파일로 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    predictions_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    print(f"\n💾 예측 결과 저장 완료: {save_path}")
    print(f"   📄 저장된 레코드 수: {len(predictions_df)}개")
    
    # 결과 요약 출력
    successful_df = predictions_df[predictions_df.get('prediction_method', '') == 'Fixed Timeseries Model']
    
    if len(successful_df) > 0:
        print(f"\n📈 예측 결과 요약:")
        print(f"   평균 퇴사 확률: {successful_df['attrition_probability'].mean():.3f}")
        print(f"   최고 퇴사 확률: {successful_df['attrition_probability'].max():.3f}")
        print(f"   최저 퇴사 확률: {successful_df['attrition_probability'].min():.3f}")
        
        # 위험도별 분포
        risk_dist = successful_df['risk_level'].value_counts()
        print(f"\n🎯 위험도별 분포:")
        for risk, count in risk_dist.items():
            if risk != 'Unknown':
                print(f"   {risk}: {count}명 ({count/len(successful_df)*100:.1f}%)")
        
        # 상위 10명 출력
        print(f"\n🚨 퇴사 위험 상위 10명:")
        top_10 = successful_df.head(10)[['employee_id', 'attrition_probability', 'risk_level']]
        for idx, row in top_10.iterrows():
            print(f"   직원 {row['employee_id']}: {row['attrition_probability']:.3f} ({row['risk_level']})")
    
    return predictions_df

# ============================================================================
# Feature Importance 분석 함수들 (Attention & Gradient 기반)
# ============================================================================

def calculate_attention_importance(model, X_test, feature_names):
    """Attention 기반 Feature Importance 계산"""
    print("🔍 Attention 기반 Feature Importance 계산 중...")
    print("=" * 50)
    
    # PyTorch 텐서로 변환
    if isinstance(X_test, np.ndarray):
        X_tensor = torch.FloatTensor(X_test).to(device)
    else:
        X_tensor = X_test.to(device)
    
    # Attention weights 추출
    attention_weights = model.get_attention_weights(X_tensor)  # (batch_size, sequence_length)
    attention_weights_np = attention_weights.cpu().numpy()
    
    # 시간 단계별 평균 attention 계산
    mean_attention_by_timestep = np.mean(attention_weights_np, axis=0)  # (sequence_length,)
    
    # 각 피처에 대한 전체 attention 중요도 (모든 시간 단계에서 동일하게 적용)
    # 실제로는 GRU의 hidden state에 기반하므로, 모든 피처가 각 시점의 attention에 기여
    feature_attention_importance = np.tile(mean_attention_by_timestep, (len(feature_names), 1)).T
    
    # 각 피처의 전체 시간에 대한 평균 attention 중요도
    overall_attention_importance = np.mean(feature_attention_importance, axis=0)
    
    # 결과 정리
    attention_importance_df = pd.DataFrame({
        'feature': feature_names,
        'attention_importance': overall_attention_importance
    }).sort_values('attention_importance', ascending=False)
    
    print(f"✅ Attention Importance 계산 완료")
    print(f"   상위 10개 중요 피처 (Attention 기준):")
    for idx, row in attention_importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['attention_importance']:.4f}")
    
    # 시간별 attention weights도 반환
    return attention_importance_df, attention_weights_np, mean_attention_by_timestep

def calculate_gradient_based_importance(model, X_test, feature_names, max_samples=100):
    """Gradient 기반 Feature Importance 계산 (더 정확한 방법)"""
    print("🔍 Gradient 기반 Feature Importance 계산 중...")
    print("=" * 50)
    
    # 샘플 수 제한
    if len(X_test) > max_samples:
        sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
        X_sample = X_test[sample_indices]
    else:
        X_sample = X_test
    
    # PyTorch 텐서로 변환 (gradient 계산을 위해 requires_grad=True)
    if isinstance(X_sample, np.ndarray):
        X_tensor = torch.FloatTensor(X_sample).to(device)
    else:
        X_tensor = X_sample.to(device)
    
    X_tensor.requires_grad_(True)
    
    model.eval()
    
    # Forward pass
    outputs = model(X_tensor)
    
    # 퇴사 클래스(1)에 대한 확률
    probs = F.softmax(outputs, dim=1)
    attrition_probs = probs[:, 1]
    
    # 각 샘플에 대한 gradient 계산
    gradients_list = []
    
    for i in range(len(attrition_probs)):
        # 개별 샘플에 대한 gradient 계산
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
    
    # 결과 정리
    gradient_importance_df = pd.DataFrame({
        'feature': feature_names,
        'gradient_importance': feature_gradient_importance
    }).sort_values('gradient_importance', ascending=False)
    
    print(f"✅ Gradient Importance 계산 완료")
    print(f"   상위 10개 중요 피처 (Gradient 기준):")
    for idx, row in gradient_importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['gradient_importance']:.4f}")
    
    return gradient_importance_df, mean_gradients


def plot_feature_importance(attention_importance_df, gradient_importance_df=None, top_n=15, save_path=None):
    """Feature Importance 시각화 (Attention & Gradient 기반)"""
    print("📊 Feature Importance 시각화 중...")
    
    # 사용 가능한 방법 수 계산
    methods = [attention_importance_df]
    method_names = ['Attention']
    
    if gradient_importance_df is not None:
        methods.append(gradient_importance_df)
        method_names.append('Gradient')
    
    n_methods = len(methods)
    
    # 그래프 설정
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 8))
    if n_methods == 1:
        axes = [axes]
    
    # 각 방법별 플롯
    for i, (method_df, method_name) in enumerate(zip(methods, method_names)):
        if method_name == 'Attention':
            top_features = method_df.head(top_n)
            y_values = top_features['attention_importance']
            xlabel = 'Attention Importance'
        else:  # Gradient
            top_features = method_df.head(top_n)
            y_values = top_features['gradient_importance']
            xlabel = 'Gradient Importance'
        
        axes[i].barh(range(len(top_features)), y_values)
        axes[i].set_yticks(range(len(top_features)))
        axes[i].set_yticklabels(top_features['feature'])
        axes[i].set_xlabel(xlabel)
        axes[i].set_title(f'Top {top_n} Features - {method_name}')
        axes[i].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Feature Importance 그래프 저장: {save_path}")
    
    plt.show()
    
    return fig

def plot_attention_heatmap(attention_weights, feature_names, top_samples=20, save_path=None):
    """Attention weights 히트맵 시각화"""
    print("📊 Attention Heatmap 시각화 중...")
    
    # 상위 샘플만 선택
    sample_attention = attention_weights[:top_samples]
    
    # 히트맵 생성
    fig, ax = plt.subplots(figsize=(15, 8))
    
    im = ax.imshow(sample_attention, cmap='YlOrRd', aspect='auto')
    
    # 축 설정
    ax.set_xlabel('Time Step (Week)')
    ax.set_ylabel('Sample')
    ax.set_title(f'Attention Weights Heatmap (Top {top_samples} Samples)')
    
    # 컬러바 추가
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Attention Heatmap 저장: {save_path}")
    
    plt.show()
    
    return fig

def save_feature_importance_results(attention_importance_df, gradient_importance_df=None,
                                  save_path='data/feature_importance_results.csv'):
    """Feature Importance 결과를 CSV로 저장 (Attention & Gradient 기반)"""
    print("💾 Feature Importance 결과 저장 중...")
    
    # 기본 데이터프레임으로 시작
    merged_df = attention_importance_df.copy()
    
    # Gradient 결과 병합
    if gradient_importance_df is not None:
        merged_df = pd.merge(merged_df, gradient_importance_df[['feature', 'gradient_importance']], 
                           on='feature', how='outer')
    
    # 순위 추가
    merged_df['attention_rank'] = merged_df['attention_importance'].rank(ascending=False)
    if 'gradient_importance' in merged_df.columns:
        merged_df['gradient_rank'] = merged_df['gradient_importance'].rank(ascending=False)
        
        # 평균 순위 계산
        merged_df['average_rank'] = (merged_df['attention_rank'] + merged_df['gradient_rank']) / 2
        merged_df = merged_df.sort_values('average_rank')
    else:
        merged_df = merged_df.sort_values('attention_rank')
    
    # 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ Feature Importance 결과 저장 완료: {save_path}")
    print(f"   저장된 피처 수: {len(merged_df)}개")
    
    return merged_df

# ============================================================================
# 전체 파이프라인 실행
# ============================================================================

if __name__ == "__main__":
    # 메인 실행 부분에서 전체 파이프라인 실행
    
    # 데이터 준비
    (X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, 
     X_test_tensor, y_test_tensor, class_weights_tensor, scaler) = prepare_data_for_training(
        X_train, y_train, X_test, y_test
    )
    
    # 하이퍼파라미터 최적화
    study = run_hyperparameter_optimization(X_train, y_train, len(processor.feature_columns), n_trials=20)
    
    # 최적 모델 생성 및 학습
    print("\n🏆 최적 파라미터로 최종 모델 학습")
    print("=" * 50)
    
    best_params = study.best_params
    best_model = GRU_CNN_HybridModel(
        input_size=len(processor.feature_columns),
        gru_hidden=best_params['gru_hidden'],
        cnn_filters=best_params['cnn_filters'],
        kernel_sizes=[2, 3],
        dropout=best_params['dropout']
    ).to(device)
    
    # 최종 학습
    final_auc = train_model(
        best_model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, class_weights_tensor,
        lr=best_params['lr'], epochs=50, batch_size=best_params['batch_size'], patience=15
    )
    
    print(f"✅ 최종 검증 AUC: {final_auc:.4f}")
    
    # 최종 평가
    final_results = evaluate_final_model(best_model, X_test_tensor, y_test_tensor)
    
    # Feature Importance 분석 (Attention & Gradient 기반)
    print("\n" + "="*70)
    print("🔍 효율적인 Feature Importance 분석 시작")
    print("="*70)
    
    # 1. Attention 기반 Feature Importance (가장 효율적)
    attention_importance_df, attention_weights, mean_attention = calculate_attention_importance(
        best_model, X_test_tensor.cpu().numpy(), processor.feature_columns
    )
    
    # 2. Gradient 기반 Feature Importance (정확한 방법)
    gradient_importance_df, gradient_by_timestep = calculate_gradient_based_importance(
        best_model, X_test_tensor.cpu().numpy(), processor.feature_columns, max_samples=50
    )
    
    # Feature Importance 시각화
    plot_feature_importance(
        attention_importance_df, gradient_importance_df,
        top_n=15, save_path='data/feature_importance_analysis.png'
    )
    
    # Attention Heatmap 시각화
    plot_attention_heatmap(
        attention_weights, processor.feature_columns, top_samples=20,
        save_path='data/attention_heatmap.png'
    )
    
    # Feature Importance 결과 저장
    feature_importance_results = save_feature_importance_results(
        attention_importance_df, gradient_importance_df,
        save_path='data/feature_importance_results.csv'
    )
    
    print(f"\n✅ Feature Importance 분석 완료!")
    print(f"\n📊 상위 5개 중요 피처:")
    
    print(f"\n🎯 Attention 기반:")
    for idx, row in attention_importance_df.head(5).iterrows():
        print(f"   {idx+1}. {row['feature']}: {row['attention_importance']:.4f}")
    
    print(f"\n🔍 Gradient 기반:")
    for idx, row in gradient_importance_df.head(5).iterrows():
        print(f"   {idx+1}. {row['feature']}: {row['gradient_importance']:.4f}")
    
    # 전체 직원 예측 및 저장
    print("\n" + "="*70)
    print("🎯 전체 직원 수정된 시계열 예측 및 결과 저장 시작")
    print("="*70)
    
    final_predictions_df = predict_all_employees_and_save_fixed(
        processor, 
        best_model, 
        scaler, 
        processor.sequence_length,
        save_path='data/employee_attrition_predictions_fixed.csv'
    )
    
    print(f"\n🎉 수정된 전체 프로세스 완료!")
    print(f"   📊 총 {len(final_predictions_df)}명의 직원 예측 결과가 저장되었습니다.")
    print(f"   📁 저장 위치: data/employee_attrition_predictions_fixed.csv")
    print(f"   🎯 현실적인 성능: AUC {final_results['auc']:.3f} (데이터 누수 제거 후)")
