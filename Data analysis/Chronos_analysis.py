# ============================================================================
# 셀 1: 라이브러리 import 및 환경 설정
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
# 셀 2: 데이터 로딩 및 기본 정보 확인
# ============================================================================

class EmployeeAttritionDataProcessor:
    def __init__(self, sequence_length=6, aggregation_unit='week'):
        """
        Args:
            sequence_length (int): 시계열 시퀀스 길이 (최적화 대상)
            aggregation_unit (str): 집계 단위 ('day', 'week', 'month')
        """
        self.sequence_length = sequence_length
        self.aggregation_unit = aggregation_unit
        self.scaler = StandardScaler()
        self.excluded_ratio_columns = [
            'work_focused_ratio', 'meeting_collaboration_ratio', 
            'social_dining_ratio', 'break_relaxation_ratio', 'shared_work_ratio'
        ]
        self.feature_columns = []
        
        print(f"🔧 기본 설정: {sequence_length}{aggregation_unit[0]} 시퀀스, {aggregation_unit} 단위 집계")
        
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

# 데이터 프로세서 초기화
processor = EmployeeAttritionDataProcessor(
    sequence_length=6,  # 기본값 (최적화 대상)
    aggregation_unit='week'
)

# 데이터 로딩 (파일 경로를 실제 경로로 변경하세요)
ts_sample, personas_sample = processor.load_data(
    'data/IBM_HR_timeseries.csv', 
    'data/IBM_HR_personas_assigned.csv'
)

print("\n📋 시계열 데이터 샘플:")
print(ts_sample)

# ============================================================================
# 셀 3: 자동 컬럼 감지 및 데이터 매칭
# ============================================================================

def detect_columns(processor):
    """컬럼 자동 감지 및 매칭"""
    print("=" * 50)
    print("컬럼 자동 감지 중...")
    print("=" * 50)
    
    # 날짜 컬럼 자동 감지
    date_columns = [col for col in processor.ts_data.columns 
                   if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp'])]
    
    if date_columns:
        processor.date_column = date_columns[0]
        print(f"🗓️  감지된 날짜 컬럼: {processor.date_column}")
    else:
        raise ValueError("❌ 날짜 컬럼을 찾을 수 없습니다.")
        
    # 직원 ID 컬럼 자동 감지
    employee_id_candidates = [col for col in processor.ts_data.columns 
                             if any(keyword in col.lower() for keyword in ['employee', 'id', 'number'])]
    
    if employee_id_candidates:
        processor.employee_id_col = employee_id_candidates[0]
        print(f"👤 감지된 시계열 직원 ID 컬럼: {processor.employee_id_col}")
    else:
        raise ValueError("❌ 직원 ID 컬럼을 찾을 수 없습니다.")
        
    # 직원 속성 데이터에서 매칭되는 ID 컬럼 찾기
    personas_id_candidates = [col for col in processor.personas_data.columns 
                             if any(keyword in col.lower() for keyword in ['employee', 'id', 'number'])]
    
    max_overlap = 0
    best_match_col = None
    
    for col in personas_id_candidates:
        try:
            overlap = len(set(processor.ts_data[processor.employee_id_col]).intersection(
                         set(processor.personas_data[col])))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match_col = col
        except:
            continue
            
    if best_match_col:
        processor.personas_id_col = best_match_col
        print(f"✅ 최적 매칭 컬럼: {processor.personas_id_col} ({max_overlap}명 겹침)")
    else:
        raise ValueError("❌ 직원 속성 데이터에서 매칭되는 ID 컬럼을 찾을 수 없습니다.")
    
    # Attrition 컬럼 찾기
    attrition_cols = [col for col in processor.personas_data.columns 
                     if 'attrition' in col.lower()]
    
    if attrition_cols:
        processor.attrition_col = attrition_cols[0]
        print(f"🎯 감지된 Attrition 컬럼: {processor.attrition_col}")
    else:
        raise ValueError("❌ Attrition 컬럼을 찾을 수 없습니다.")
    
    return processor

# 컬럼 감지 실행
processor = detect_columns(processor)

# ============================================================================
# 셀 4: 데이터 전처리 및 시간 범위 분석
# ============================================================================

def preprocess_data(processor):
    """데이터 전처리 및 시간 범위 분석"""
    print("=" * 50)
    print("데이터 전처리 중...")
    print("=" * 50)
    
    # 날짜 변환
    processor.ts_data[processor.date_column] = pd.to_datetime(processor.ts_data[processor.date_column])
    
    # 2023-2024년 데이터만 필터링
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    original_shape = processor.ts_data.shape
    processor.ts_data = processor.ts_data[
        (processor.ts_data[processor.date_column] >= start_date) & 
        (processor.ts_data[processor.date_column] <= end_date)
    ].copy()
    
    print(f"📅 2023-2024년 필터링: {original_shape} → {processor.ts_data.shape}")
    
    # Attrition 라벨을 0/1로 변환
    def convert_attrition(value):
        if pd.isna(value):
            return 0
        value_str = str(value).lower().strip()
        if value_str in ['yes', 'y', 'true', '1', '1.0']:
            return 1
        else:
            return 0
            
    processor.personas_data['attrition_binary'] = processor.personas_data[processor.attrition_col].apply(convert_attrition)
    
    attrition_dist = processor.personas_data['attrition_binary'].value_counts()
    print(f"\n🎯 Attrition 분포:")
    print(f"   재직(0): {attrition_dist.get(0, 0)}명")
    print(f"   퇴사(1): {attrition_dist.get(1, 0)}명")
    print(f"   퇴사율: {attrition_dist.get(1, 0) / len(processor.personas_data) * 100:.1f}%")
    
    return processor

# 전처리 실행
processor = preprocess_data(processor)

# ============================================================================
# 셀 5: 피처 선택 및 단순 집계
# ============================================================================

def identify_and_prepare_features(processor):
    """피처 식별 및 단순 집계 (통계량 확장 없음)"""
    print("=" * 50)
    print("피처 식별 및 단순 집계 중...")
    print("=" * 50)
    
    # 수치형 컬럼 식별
    numeric_columns = processor.ts_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # 제외할 컬럼들 제거
    exclude_columns = [processor.employee_id_col] + processor.excluded_ratio_columns
    exclude_columns = [col for col in exclude_columns if col in numeric_columns]
    
    processor.feature_columns = [col for col in numeric_columns if col not in exclude_columns]
    
    print(f"🔍 전체 수치형 컬럼: {len(numeric_columns)}개")
    print(f"❌ 제외된 컬럼: {exclude_columns}")
    print(f"✅ 사용할 피처: {len(processor.feature_columns)}개")
    
    return processor

def create_simple_aggregated_data(processor):
    """단순 시간 집계 (평균만 사용, 통계량 확장 없음)"""
    print("=" * 50)
    print(f"{processor.aggregation_unit} 단위 단순 집계 중...")
    print("=" * 50)
    
    # 집계 단위별 처리
    if processor.aggregation_unit == 'day':
        processor.ts_data['time_period'] = processor.ts_data[processor.date_column].dt.date
        period_range = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        period_format = lambda x: x.date()
        
    elif processor.aggregation_unit == 'week':
        processor.ts_data['year'] = processor.ts_data[processor.date_column].dt.year
        processor.ts_data['week'] = processor.ts_data[processor.date_column].dt.isocalendar().week
        processor.ts_data['time_period'] = processor.ts_data['year'].astype(str) + '-W' + processor.ts_data['week'].astype(str).str.zfill(2)
        
        period_range = pd.date_range('2023-01-02', '2024-12-30', freq='W-MON')
        period_format = lambda x: f"{x.year}-W{x.isocalendar().week:02d}"
        
    else:  # month
        processor.ts_data['time_period'] = processor.ts_data[processor.date_column].dt.to_period('M').astype(str)
        period_range = pd.period_range('2023-01', '2024-12', freq='M')
        period_format = lambda x: str(x)
    
    # 시간별 단순 집계 (평균만)
    agg_data = processor.ts_data.groupby([processor.employee_id_col, 'time_period'])[processor.feature_columns].mean().reset_index()
    
    print(f"📊 집계 전 데이터: {processor.ts_data.shape}")
    print(f"📊 집계 후 데이터: {agg_data.shape}")
    print(f"📊 피처 수: {len(processor.feature_columns)}개 (단순 평균)")
    
    # 모든 직원-시간 조합 생성
    all_periods = [period_format(p) for p in period_range]
    all_employees = agg_data[processor.employee_id_col].unique()
    
    print(f"👥 고유 직원 수: {len(all_employees)}")
    print(f"📅 총 {processor.aggregation_unit} 수: {len(all_periods)}")
    
    # 완전한 시간 인덱스 생성
    complete_index = pd.MultiIndex.from_product(
        [all_employees, all_periods], 
        names=[processor.employee_id_col, 'time_period']
    ).to_frame(index=False)
    
    # 병합 및 결측치 처리
    processor.aggregated_data = pd.merge(
        complete_index, 
        agg_data, 
        on=[processor.employee_id_col, 'time_period'], 
        how='left'
    )
    
    # 결측치를 직원별 평균으로 채우기
    missing_before = processor.aggregated_data.isnull().sum().sum()
    
    for col in processor.feature_columns:
        processor.aggregated_data[col] = processor.aggregated_data.groupby(processor.employee_id_col)[col].transform(
            lambda x: x.fillna(x.mean()) if not x.isna().all() else x.fillna(0)
        )
    
    missing_after = processor.aggregated_data.isnull().sum().sum()
    
    print(f"🔧 결측치 처리: {missing_before} → {missing_after}")
    print(f"✅ 최종 집계 데이터: {processor.aggregated_data.shape}")
    
    return processor

# 피처 식별 및 집계 실행
processor = identify_and_prepare_features(processor)
processor = create_simple_aggregated_data(processor)

# ============================================================================
# 셀 6: 시계열 시퀀스 생성 (파라미터화)
# ============================================================================

def create_sequences_with_params(processor, sequence_length=None):
    """파라미터화된 시계열 시퀀스 생성"""
    if sequence_length is not None:
        processor.sequence_length = sequence_length
    
    # 직원 속성 데이터와 병합
    merged_data = pd.merge(
        processor.aggregated_data,
        processor.personas_data[[processor.personas_id_col, 'attrition_binary']],
        left_on=processor.employee_id_col,
        right_on=processor.personas_id_col,
        how='inner'
    )
    
    sequences = []
    labels = []
    employee_ids = []
    
    valid_employees = 0
    total_sequences = 0
    
    for employee_id in merged_data[processor.employee_id_col].unique():
        employee_data = merged_data[
            merged_data[processor.employee_id_col] == employee_id
        ].sort_values('time_period')
        
        available_periods = len(employee_data)
        
        if available_periods >= processor.sequence_length:
            valid_employees += 1
            
            sequence_data = employee_data[processor.feature_columns].values
            label = employee_data['attrition_binary'].iloc[0]
            
            # 최근 데이터 위주로 시퀀스 생성
            sequences.append(sequence_data[-processor.sequence_length:])
            labels.append(label)
            employee_ids.append(employee_id)
            total_sequences += 1
    
    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    employee_ids_seq = np.array(employee_ids)
    
    return X, y, employee_ids_seq, valid_employees, total_sequences

# 기본 시퀀스 생성
processor.X, processor.y, processor.employee_ids_seq, valid_emp, total_seq = create_sequences_with_params(processor)

print(f"✅ 기본 시퀀스 생성 완료:")
print(f"   유효 직원: {valid_emp}명")
print(f"   시퀀스 수: {total_seq}개")
print(f"   시퀀스 형태: {processor.X.shape}")
print(f"   퇴사율: {np.mean(processor.y) * 100:.1f}%")

# ============================================================================
# 셀 7: 모델 정의 (파라미터화)
# ============================================================================

class GRUModel(nn.Module):
    """파라미터화된 GRU 모델"""
    
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.classifier(last_output)
        return output

class CNN1DModel(nn.Module):
    """파라미터화된 1D CNN 모델"""
    
    def __init__(self, input_size, num_filters=32, kernel_sizes=[2, 3, 4], dropout=0.2):
        super(CNN1DModel, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv_layer = nn.Sequential(
                nn.Conv1d(input_size, num_filters, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(num_filters),
                nn.Conv1d(num_filters, num_filters, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(num_filters),
                nn.AdaptiveMaxPool1d(1)
            )
            self.conv_layers.append(conv_layer)
        
        combined_features = len(kernel_sizes) * num_filters
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x).squeeze(-1)
            conv_outputs.append(conv_out)
        
        combined = torch.cat(conv_outputs, dim=1)
        output = self.classifier(combined)
        return output

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

print("🧠 파라미터화된 모델 정의 완료")

# ============================================================================
# 셀 8: 하이퍼파라미터 최적화 목적 함수
# ============================================================================

def prepare_data_for_optimization(X, y, employee_ids, test_size=0.2, val_size=0.2):
    """최적화용 데이터 분할"""
    unique_employees = np.unique(employee_ids)
    employee_labels = [y[employee_ids == emp][0] for emp in unique_employees]
    
    train_employees, test_employees = train_test_split(
        unique_employees, test_size=test_size, random_state=42, stratify=employee_labels
    )
    
    train_emp_labels = [y[employee_ids == emp][0] for emp in train_employees]
    train_employees, val_employees = train_test_split(
        train_employees, test_size=val_size/(1-test_size), random_state=42, stratify=train_emp_labels
    )
    
    # 인덱스 생성
    train_idx = np.isin(employee_ids, train_employees)
    val_idx = np.isin(employee_ids, val_employees)
    test_idx = np.isin(employee_ids, test_employees)
    
    # 데이터 분할
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # 정규화
    scaler = StandardScaler()
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    
    scaler.fit(X_train_reshaped)
    
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(n_samples, n_timesteps, n_features)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    
    # PyTorch 텐서로 변환
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 클래스 가중치
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    return (X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, 
            X_test_tensor, y_test_tensor, class_weights_tensor, scaler)

def train_model_for_optimization(model, X_train, y_train, X_val, y_val, class_weights, 
                                lr=0.001, epochs=30, batch_size=64, patience=10):
    """최적화용 모델 학습 함수"""
    
    # 데이터 로더 생성
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 학습 설정
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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

def objective_hybrid(trial):
    """GRU+CNN 하이브리드 모델 최적화 목적함수"""
    
    # 하이퍼파라미터 샘플링
    sequence_length = trial.suggest_int('sequence_length', 4, 12)
    gru_hidden = trial.suggest_int('gru_hidden', 16, 128, step=16)
    cnn_filters = trial.suggest_int('cnn_filters', 8, 64, step=8)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # 시퀀스 생성 (새로운 프로세서 사용)
    X, y, employee_ids, valid_emp, total_seq = create_sequences_with_params(processor, sequence_length)
    
    if total_seq < 50:  # 시퀀스가 너무 적으면 스킵
        return 0.5
    
    # 데이터 준비
    try:
        (X_train, y_train, X_val, y_val, 
         X_test, y_test, class_weights, scaler) = prepare_data_for_optimization(X, y, employee_ids)
    except:
        return 0.5
    
    # 모델 생성
    input_size = len(processor.feature_columns)
    model = GRU_CNN_HybridModel(
        input_size=input_size,
        gru_hidden=gru_hidden,
        cnn_filters=cnn_filters,
        kernel_sizes=[2, 3],  # 고정
        dropout=dropout
    ).to(device)
    
    # 학습 및 평가
    auc_score = train_model_for_optimization(
        model, X_train, y_train, X_val, y_val, class_weights,
        lr=lr, epochs=25, batch_size=batch_size, patience=8
    )
    
    return auc_score

print("🎯 하이퍼파라미터 최적화 목적함수 정의 완료")

# ============================================================================
# 셀 9: 하이퍼파라미터 최적화 실행
# ============================================================================

def run_hyperparameter_optimization(n_trials=50):
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
    study.optimize(objective_hybrid, n_trials=n_trials, show_progress_bar=True)
    
    # 결과 출력
    print("\n🏆 최적화 완료!")
    print("=" * 50)
    print(f"✅ 최고 AUC 스코어: {study.best_value:.4f}")
    print(f"🎯 최적 파라미터:")
    
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    # 시도별 결과 시각화
    plot_optimization_results(study)
    
    return study

def plot_optimization_results(study):
    """최적화 결과 시각화"""
    
    # 시도별 AUC 점수
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 시도별 점수 변화
    trial_numbers = [trial.number for trial in study.trials if trial.value is not None]
    trial_values = [trial.value for trial in study.trials if trial.value is not None]
    
    axes[0, 0].plot(trial_numbers, trial_values, 'o-', alpha=0.7)
    axes[0, 0].axhline(y=study.best_value, color='r', linestyle='--', alpha=0.7, label=f'Best: {study.best_value:.4f}')
    axes[0, 0].set_title('Optimization Progress')
    axes[0, 0].set_xlabel('Trial')
    axes[0, 0].set_ylabel('AUC Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 파라미터 중요도 (상위 파라미터들)
    if len(study.trials) > 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())[:6]  # 상위 6개
            values = [importance[param] for param in params]
            
            axes[0, 1].barh(params, values)
            axes[0, 1].set_title('Parameter Importance')
            axes[0, 1].set_xlabel('Importance')
        except:
            axes[0, 1].text(0.5, 0.5, 'Importance calculation failed', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # 3. sequence_length vs AUC
    seq_lengths = []
    seq_aucs = []
    for trial in study.trials:
        if trial.value is not None and 'sequence_length' in trial.params:
            seq_lengths.append(trial.params['sequence_length'])
            seq_aucs.append(trial.value)
    
    if seq_lengths:
        axes[1, 0].scatter(seq_lengths, seq_aucs, alpha=0.6)
        axes[1, 0].set_title('Sequence Length vs AUC')
        axes[1, 0].set_xlabel('Sequence Length')
        axes[1, 0].set_ylabel('AUC Score')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. learning_rate vs AUC
    lrs = []
    lr_aucs = []
    for trial in study.trials:
        if trial.value is not None and 'lr' in trial.params:
            lrs.append(trial.params['lr'])
            lr_aucs.append(trial.value)
    
    if lrs:
        axes[1, 1].scatter(lrs, lr_aucs, alpha=0.6)
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_title('Learning Rate vs AUC')
        axes[1, 1].set_xlabel('Learning Rate')
        axes[1, 1].set_ylabel('AUC Score')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 최적화 실행 (시간이 오래 걸릴 수 있으므로 적절히 조정)
study = run_hyperparameter_optimization(n_trials=30)  # 30회 시도

# ============================================================================
# 셀 10: 최적 모델 생성 및 전체 학습
# ============================================================================

def create_and_train_best_model(study, processor):
    """최적 파라미터로 모델 생성 및 전체 데이터로 학습"""
    print("🏆 최적 파라미터로 최종 모델 학습")
    print("=" * 50)
    
    best_params = study.best_params
    print("📋 사용할 최적 파라미터:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    # 최적 파라미터로 시퀀스 생성
    X, y, employee_ids, valid_emp, total_seq = create_sequences_with_params(
        processor, best_params['sequence_length']
    )
    
    print(f"\n📊 최적 시퀀스 정보:")
    print(f"   시퀀스 길이: {best_params['sequence_length']}")
    print(f"   총 시퀀스: {total_seq}개")
    print(f"   시퀀스 형태: {X.shape}")
    
    # 데이터 분할
    (X_train, y_train, X_val, y_val, 
     X_test, y_test, class_weights, scaler) = prepare_data_for_optimization(X, y, employee_ids)
    
    # 최적 모델 생성
    input_size = len(processor.feature_columns)
    best_model = GRU_CNN_HybridModel(
        input_size=input_size,
        gru_hidden=best_params['gru_hidden'],
        cnn_filters=best_params['cnn_filters'],
        kernel_sizes=[2, 3],
        dropout=best_params['dropout']
    ).to(device)
    
    print(f"\n🧠 최적 모델 정보:")
    print(f"   GRU Hidden: {best_params['gru_hidden']}")
    print(f"   CNN Filters: {best_params['cnn_filters']}")
    print(f"   Dropout: {best_params['dropout']}")
    print(f"   파라미터 수: {sum(p.numel() for p in best_model.parameters()):,}")
    
    # 더 긴 학습
    print(f"\n🚀 최종 모델 학습 시작...")
    final_auc = train_model_for_optimization(
        best_model, X_train, y_train, X_val, y_val, class_weights,
        lr=best_params['lr'], 
        epochs=50,  # 더 긴 학습
        batch_size=best_params['batch_size'], 
        patience=15
    )
    
    print(f"✅ 최종 검증 AUC: {final_auc:.4f}")
    
    return best_model, (X_train, y_train, X_val, y_val, X_test, y_test), scaler, best_params

# 최적 모델 생성 및 학습
best_model, data_splits, best_scaler, best_params = create_and_train_best_model(study, processor)

# ============================================================================
# 셀 11: 최종 모델 평가
# ============================================================================

def evaluate_final_model(model, data_splits, model_name="Best Hybrid Model"):
    """최종 모델 평가"""
    X_train, y_train, X_val, y_val, X_test, y_test = data_splits
    
    model.eval()
    
    def evaluate_split(X, y, split_name):
        dataset = TensorDataset(X, y)
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
        
        print(f"\n📊 {split_name} 결과:")
        print(f"   정확도: {accuracy:.4f}")
        print(f"   AUC: {auc_score:.4f}")
        
        if split_name == "Test":
            print(f"\n📋 테스트 세트 분류 리포트:")
            print(classification_report(all_labels, all_predictions, target_names=['재직', '퇴사']))
            
            cm = confusion_matrix(all_labels, all_probabilities > 0.5)
            print(f"\n🔍 혼동 행렬:")
            print(cm)
        
        return {
            'accuracy': accuracy,
            'auc': auc_score,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'labels': all_labels
        }
    
    print(f"🎯 {model_name} 최종 평가")
    print("=" * 50)
    
    train_results = evaluate_split(X_train, y_train, "Train")
    val_results = evaluate_split(X_val, y_val, "Validation") 
    test_results = evaluate_split(X_test, y_test, "Test")
    
    return {
        'train': train_results,
        'val': val_results,
        'test': test_results
    }

# 최종 평가 실행
final_results = evaluate_final_model(best_model, data_splits)

# ROC 곡선 그리기
def plot_final_roc_curve(results):
    """최종 ROC 곡선 시각화"""
    plt.figure(figsize=(10, 8))
    
    for split_name, result in results.items():
        fpr, tpr, _ = roc_curve(result['labels'], result['probabilities'])
        auc_score = result['auc']
        
        plt.plot(fpr, tpr, label=f'{split_name.title()} (AUC = {auc_score:.3f})', marker='o', markersize=3)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Final Model ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

plot_final_roc_curve(final_results)

# ============================================================================
# 셀 12: 개별 직원 예측 함수 (최적화 버전)
# ============================================================================

def predict_employee_with_best_model(employee_id, processor, model, scaler, best_params):
    """최적화된 모델로 개별 직원 예측"""
    
    # 해당 직원의 데이터 찾기
    employee_data = processor.aggregated_data[
        processor.aggregated_data[processor.employee_id_col] == employee_id
    ].sort_values('time_period')
    
    sequence_length = best_params['sequence_length']
    
    if len(employee_data) < sequence_length:
        raise ValueError(f"직원 {employee_id}의 데이터가 충분하지 않습니다. "
                        f"최소 {sequence_length}{processor.aggregation_unit[0]} 필요 "
                        f"(현재: {len(employee_data)}{processor.aggregation_unit[0]})")
    
    # 시퀀스 생성 (최근 데이터 사용)
    sequence_data = employee_data[processor.feature_columns].values[-sequence_length:]
    
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

# 샘플 예측 수행
print("\n🔮 최적화된 모델로 샘플 직원 예측")
print("=" * 50)

sample_employees = processor.aggregated_data[processor.employee_id_col].unique()[:10]
sample_predictions = []

for emp_id in sample_employees:
    try:
        result = predict_employee_with_best_model(emp_id, processor, best_model, best_scaler, best_params)
        sample_predictions.append(result)
    except Exception as e:
        print(f"⚠️  직원 {emp_id} 예측 실패: {e}")

# 결과 출력
if sample_predictions:
    predictions_df = pd.DataFrame(sample_predictions)
    print("\n📋 샘플 직원 예측 결과:")
    print(predictions_df.sort_values('attrition_probability', ascending=False))
    
    risk_dist = predictions_df['risk_level'].value_counts()
    print(f"\n📊 위험도 분포:")
    for risk, count in risk_dist.items():
        print(f"   {risk}: {count}명 ({count/len(predictions_df)*100:.1f}%)")

# ============================================================================
# 셀 13: 상세 성능 분석 및 시각화
# ============================================================================

from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score, matthews_corrcoef

def detailed_performance_analysis(results, model_name="Best Hybrid Model"):
    """상세 성능 분석 및 시각화"""
    
    print(f"🎯 {model_name} 상세 성능 분석")
    print("=" * 60)
    
    # 테스트 데이터 결과 사용
    test_results = results['test']
    y_true = test_results['labels']
    y_pred = test_results['predictions'] 
    y_prob = test_results['probabilities']
    
    # 1. 기본 분류 메트릭 계산
    accuracy = test_results['accuracy']
    auc_score = test_results['auc']
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # 특이도 (Specificity) 계산
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Average Precision Score
    avg_precision = average_precision_score(y_true, y_prob)
    
    print("📊 종합 성능 지표:")
    print(f"   🎯 Accuracy:     {accuracy:.4f}")
    print(f"   🚀 AUC Score:    {auc_score:.4f}")
    print(f"   ⚖️  F1-Score:     {f1:.4f}")
    print(f"   📍 Precision:    {precision:.4f}")
    print(f"   🔍 Recall:       {recall:.4f}")
    print(f"   🛡️  Specificity:  {specificity:.4f}")
    print(f"   🔗 MCC:          {mcc:.4f}")
    print(f"   📈 Avg Precision: {avg_precision:.4f}")
    
    # 2. 혼동 행렬 상세 분석
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🔍 혼동 행렬 상세 분석:")
    print(f"   True Negative (TN):  {tn:4d} - 재직 예측이 맞은 경우")
    print(f"   False Positive (FP): {fp:4d} - 퇴사 예측했지만 실제 재직")
    print(f"   False Negative (FN): {fn:4d} - 재직 예측했지만 실제 퇴사")
    print(f"   True Positive (TP):  {tp:4d} - 퇴사 예측이 맞은 경우")
    
    # 3. 비즈니스 관점 해석
    total_employees = len(y_true)
    actual_attrition = np.sum(y_true)
    predicted_attrition = np.sum(y_pred)
    
    print(f"\n💼 비즈니스 관점 해석:")
    print(f"   📊 전체 직원 수:        {total_employees:4d}명")
    print(f"   📉 실제 퇴사자:         {actual_attrition:4d}명 ({actual_attrition/total_employees*100:.1f}%)")
    print(f"   🎯 예측 퇴사자:         {predicted_attrition:4d}명 ({predicted_attrition/total_employees*100:.1f}%)")
    print(f"   ✅ 퇴사 정확 예측:      {tp:4d}명 (실제 퇴사자 중 {tp/actual_attrition*100:.1f}%)")
    print(f"   ❌ 퇴사 놓친 경우:      {fn:4d}명 (실제 퇴사자 중 {fn/actual_attrition*100:.1f}%)")
    print(f"   ⚠️  잘못된 퇴사 예측:   {fp:4d}명 (재직자 중 {fp/(total_employees-actual_attrition)*100:.1f}%)")
    
    return {
        'accuracy': accuracy, 'auc': auc_score, 'f1': f1, 'precision': precision,
        'recall': recall, 'specificity': specificity, 'mcc': mcc, 'avg_precision': avg_precision,
        'confusion_matrix': cm, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

def plot_detailed_performance(results, metrics_dict):
    """상세 성능 시각화"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    test_results = results['test']
    y_true = test_results['labels']
    y_pred = test_results['predictions']
    y_prob = test_results['probabilities']
    cm = metrics_dict['confusion_matrix']
    
    # 1. 혼동 행렬 히트맵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['재직', '퇴사'], yticklabels=['재직', '퇴사'],
                ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')
    
    # 2. 정규화된 혼동 행렬
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=['재직', '퇴사'], yticklabels=['재직', '퇴사'],
                ax=axes[0, 1])
    axes[0, 1].set_title('Normalized Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted Label')
    axes[0, 1].set_ylabel('True Label')
    
    # 3. ROC 곡선 (모든 split)
    for split_name, result in results.items():
        fpr, tpr, _ = roc_curve(result['labels'], result['probabilities'])
        auc_score = result['auc']
        axes[0, 2].plot(fpr, tpr, label=f'{split_name.title()} (AUC={auc_score:.3f})', marker='o', markersize=3)
    
    axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    axes[0, 2].set_xlabel('False Positive Rate')
    axes[0, 2].set_ylabel('True Positive Rate')
    axes[0, 2].set_title('ROC Curves')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Precision-Recall 곡선
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    axes[1, 0].plot(recall, precision, marker='o', markersize=3, 
                    label=f'AP={avg_precision:.3f}')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 임계값별 성능 변화
    thresholds_range = np.arange(0.1, 1.0, 0.05)
    f1_scores = []
    precisions = []
    recalls = []
    
    for threshold in thresholds_range:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_thresh))
        precisions.append(precision_score(y_true, y_pred_thresh))
        recalls.append(recall_score(y_true, y_pred_thresh))
    
    axes[1, 1].plot(thresholds_range, f1_scores, 'o-', label='F1-Score', markersize=3)
    axes[1, 1].plot(thresholds_range, precisions, 's-', label='Precision', markersize=3)
    axes[1, 1].plot(thresholds_range, recalls, '^-', label='Recall', markersize=3)
    axes[1, 1].axvline(x=0.5, color='r', linestyle='--', alpha=0.7, label='Default (0.5)')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance vs Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 메트릭 요약 바 차트
    metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Specificity', 'AUC']
    metric_values = [
        metrics_dict['accuracy'], metrics_dict['f1'], metrics_dict['precision'],
        metrics_dict['recall'], metrics_dict['specificity'], metrics_dict['auc']
    ]
    
    bars = axes[1, 2].bar(metric_names, metric_values, alpha=0.7, 
                         color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral', 'gold'])
    axes[1, 2].set_title('Performance Metrics Summary')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_ylim(0, 1)
    
    # 막대 위에 값 표시
    for bar, value in zip(bars, metric_values):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    # x축 레이블 회전
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def find_optimal_threshold(y_true, y_prob):
    """최적 임계값 찾기"""
    print("\n🎯 최적 임계값 분석")
    print("=" * 40)
    
    thresholds_range = np.arange(0.1, 0.95, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    print("Threshold  F1-Score  Precision  Recall   Specificity")
    print("-" * 55)
    
    for threshold in thresholds_range:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        
        f1 = f1_score(y_true, y_pred_thresh)
        precision = precision_score(y_true, y_pred_thresh)
        recall = recall_score(y_true, y_pred_thresh)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"  {threshold:.2f}      {f1:.3f}     {precision:.3f}      {recall:.3f}      {specificity:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print("-" * 55)
    print(f"🏆 최적 임계값: {best_threshold:.2f} (F1-Score: {best_f1:.3f})")
    
    return best_threshold, best_f1

def business_impact_analysis(metrics_dict):
    """비즈니스 임팩트 분석"""
    print("\n💼 비즈니스 임팩트 분석")
    print("=" * 40)
    
    tp, fp, fn, tn = metrics_dict['tp'], metrics_dict['fp'], metrics_dict['fn'], metrics_dict['tn']
    
    # 가정: 퇴사자 1명당 대체 비용 (임의 설정)
    replacement_cost = 50000  # 50,000 달러
    intervention_cost = 5000   # 5,000 달러 (예방 조치 비용)
    
    # 시나리오 분석
    print("📊 비즈니스 시나리오 분석:")
    print(f"   가정 - 퇴사자 대체 비용: ${replacement_cost:,}")
    print(f"   가정 - 예방 조치 비용: ${intervention_cost:,}")
    print()
    
    # 모델 없이 (아무 조치 없음)
    cost_without_model = (tp + fn) * replacement_cost
    
    # 모델 사용시
    cost_with_model = (
        fn * replacement_cost +  # 놓친 퇴사자들
        (tp + fp) * intervention_cost  # 예방 조치 대상들
    )
    
    savings = cost_without_model - cost_with_model
    roi = (savings / (cost_with_model if cost_with_model > 0 else 1)) * 100
    
    print(f"💰 비용 분석:")
    print(f"   모델 미사용시 비용:  ${cost_without_model:,}")
    print(f"   모델 사용시 비용:    ${cost_with_model:,}")
    print(f"   절약 비용:          ${savings:,}")
    print(f"   ROI:               {roi:.1f}%")
    print()
    
    print(f"📈 효과성 분석:")
    print(f"   예방 가능한 퇴사:    {tp}명 / {tp + fn}명 ({tp/(tp+fn)*100:.1f}%)")
    print(f"   불필요한 개입:       {fp}명 (비용 손실: ${fp * intervention_cost:,})")
    print(f"   놓친 퇴사자:        {fn}명 (비용 손실: ${fn * replacement_cost:,})")

# 상세 성능 분석 실행
detailed_metrics = detailed_performance_analysis(final_results)

# 시각화 실행
plot_detailed_performance(final_results, detailed_metrics)

# 최적 임계값 찾기
optimal_threshold, optimal_f1 = find_optimal_threshold(
    final_results['test']['labels'], 
    final_results['test']['probabilities']
)

# 비즈니스 임팩트 분석
business_impact_analysis(detailed_metrics)

# ============================================================================
# 셀 14: 최적화 결과 저장 및 최종 요약
# ============================================================================

def save_optimization_results(study, best_model, best_params, processor, scaler, detailed_metrics, save_dir='optimized_models'):
    """최적화 결과 및 상세 성능 저장"""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"💾 최적화 결과 저장 중... (경로: {save_dir})")
    
    # 모델 저장
    model_path = os.path.join(save_dir, 'best_hybrid_model.pth')
    torch.save(best_model.state_dict(), model_path)
    print(f"   ✅ 최적 모델 저장: {model_path}")
    
    # 파라미터 저장
    params_path = os.path.join(save_dir, 'best_params.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"   ✅ 최적 파라미터 저장: {params_path}")
    
    # 상세 성능 지표 저장
    performance_path = os.path.join(save_dir, 'performance_metrics.json')
    # numpy 타입을 일반 타입으로 변환
    metrics_to_save = {}
    for key, value in detailed_metrics.items():
        if isinstance(value, np.ndarray):
            metrics_to_save[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            metrics_to_save[key] = float(value)
        else:
            metrics_to_save[key] = value
    
    with open(performance_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"   ✅ 상세 성능 지표 저장: {performance_path}")
    
    # 스터디 저장
    study_path = os.path.join(save_dir, 'optimization_study.pkl')
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"   ✅ 최적화 스터디 저장: {study_path}")
    
    # 프로세서 정보 저장
    processor_info = {
        'scaler': scaler,
        'feature_columns': processor.feature_columns,
        'sequence_length': best_params['sequence_length'],
        'aggregation_unit': best_params.get('aggregation_unit', processor.aggregation_unit),
        'employee_id_col': processor.employee_id_col,
        'date_column': processor.date_column,
        'excluded_ratio_columns': processor.excluded_ratio_columns
    }
    
    processor_path = os.path.join(save_dir, 'processor_info.pkl')
    with open(processor_path, 'wb') as f:
        pickle.dump(processor_info, f)
    print(f"   ✅ 프로세서 정보 저장: {processor_path}")
    
    # 모델 사용 가이드 생성
    guide_path = os.path.join(save_dir, 'model_usage_guide.md')
    create_usage_guide(guide_path, best_params, detailed_metrics)
    print(f"   ✅ 사용 가이드 저장: {guide_path}")
    
    print("💾 저장 완료!")

def create_usage_guide(file_path, best_params, metrics):
    """모델 사용 가이드 생성"""
    guide_content = f"""# 직원 퇴사 예측 모델 사용 가이드

## 📊 모델 성능 요약
- **AUC Score**: {metrics['auc']:.4f}
- **F1-Score**: {metrics['f1']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **Accuracy**: {metrics['accuracy']:.4f}

## 🎯 최적 하이퍼파라미터
- **Sequence Length**: {best_params['sequence_length']}
- **Aggregation Unit**: {best_params.get('aggregation_unit', 'week')}
- **GRU Hidden**: {best_params['gru_hidden']}
- **CNN Filters**: {best_params['cnn_filters']}
- **Dropout**: {best_params['dropout']:.3f}
- **Learning Rate**: {best_params['lr']:.2e}
- **Batch Size**: {best_params['batch_size']}

## 🔍 모델 해석
### Confusion Matrix
- True Negative: {metrics['tn']} (재직 정확 예측)
- False Positive: {metrics['fp']} (퇴사 오예측)
- False Negative: {metrics['fn']} (퇴사 놓침)
- True Positive: {metrics['tp']} (퇴사 정확 예측)

### 비즈니스 의미
- **퇴사 감지율**: {metrics['tp']/(metrics['tp']+metrics['fn'])*100:.1f}% (실제 퇴사자 중 모델이 감지한 비율)
- **오탐률**: {metrics['fp']/(metrics['fp']+metrics['tn'])*100:.1f}% (재직자 중 퇴사로 잘못 예측한 비율)

## 💼 실전 활용 방법

### 1. 위험도 기준
- **High Risk (0.7~1.0)**: 즉시 면담 및 개입 필요
- **Medium Risk (0.3~0.7)**: 주의 관찰 및 모니터링
- **Low Risk (0.0~0.3)**: 안정 상태

### 2. 주기적 모니터링
- 매주 모든 직원의 퇴사 확률 계산
- 위험도 변화 추이 관찰
- 임계값 이상 직원에 대한 선제적 조치

### 3. 모델 활용 팁
- 단일 예측보다는 추세 변화에 주목
- 다른 HR 지표와 함께 종합 판단
- 정기적인 모델 재학습 권장 (분기별)

## 🚀 모델 로딩 및 예측 코드 예시
```python
import torch
import pickle

# 모델 로딩
model = GRU_CNN_HybridModel(input_size, **best_params)
model.load_state_dict(torch.load('best_hybrid_model.pth'))
model.eval()

# 프로세서 로딩
with open('processor_info.pkl', 'rb') as f:
    processor_info = pickle.load(f)

# 예측 수행
result = predict_employee_with_best_model(
    employee_id, processor, model, scaler, best_params
)
print(f"퇴사 확률: {{result['attrition_probability']:.3f}}")"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)

print("🎯 최적화된 모델로 샘플 직원 예측")
print("=" * 50)

sample_employees = processor.aggregated_data[processor.employee_id_col].unique()[:10]
sample_predictions = []

for emp_id in sample_employees:
    try:
        result = predict_employee_with_best_model(emp_id, processor, best_model, best_scaler, best_params)
        sample_predictions.append(result)
    except Exception as e:
        print(f"⚠️  직원 {emp_id} 예측 실패: {e}")

# 결과 출력
if sample_predictions:
    predictions_df = pd.DataFrame(sample_predictions)
    print("\n📋 샘플 직원 예측 결과:")
    print(predictions_df.sort_values('attrition_probability', ascending=False))
    
    risk_dist = predictions_df['risk_level'].value_counts()
    print(f"\n📊 위험도 분포:")
    for risk, count in risk_dist.items():
        print(f"   {risk}: {count}명 ({count/len(predictions_df)*100:.1f}%)")

# ============================================================================
# 셀 15: 전체 직원 시계열 예측 및 결과 저장
# ============================================================================

def predict_all_employees_and_save(processor, model, scaler, best_params, save_path='Data analysis/data/employee_attrition_predictions.csv'):
    """전체 직원에 대한 시계열 예측 수행 및 CSV 저장"""
    print("🔮 전체 1470명 직원 시계열 예측 수행")
    print("=" * 50)
    
    # 전체 직원 ID 가져오기 (시계열 데이터와 직원 속성 데이터 모두에서)
    timeseries_employees = set(processor.aggregated_data[processor.employee_id_col].unique())
    personas_employees = set(processor.personas_data[processor.personas_id_col].unique())
    
    # 두 데이터셋에 모두 존재하는 직원들
    common_employees = timeseries_employees.intersection(personas_employees)
    
    # 시계열 데이터에만 있는 직원들 (속성 정보 없음)
    timeseries_only = timeseries_employees - personas_employees
    
    # 속성 데이터에만 있는 직원들 (시계열 정보 없음)
    personas_only = personas_employees - timeseries_employees
    
    print(f"📊 데이터 현황:")
    print(f"   시계열 데이터 직원 수: {len(timeseries_employees)}명")
    print(f"   속성 데이터 직원 수: {len(personas_employees)}명")
    print(f"   공통 직원 수: {len(common_employees)}명")
    print(f"   시계열만 있는 직원: {len(timeseries_only)}명")
    print(f"   속성만 있는 직원: {len(personas_only)}명")
    
    # 전체 직원 리스트 (모든 직원 포함 - 1470명 목표)
    all_employees = list(timeseries_employees.union(personas_employees))
    print(f"📊 예측 대상 총 직원 수: {len(all_employees)}명")
    
    if len(all_employees) >= 1470:
        print(f"✅ 목표 1470명 달성! (실제: {len(all_employees)}명)")
    else:
        print(f"⚠️  목표 1470명 미달 (실제: {len(all_employees)}명)")
    
    # 예측 결과 저장할 리스트
    all_predictions = []
    successful_predictions = 0
    failed_predictions = 0
    
    print("🚀 예측 진행 중...")
    
    # 진행률 표시를 위한 tqdm 사용
    for i, emp_id in enumerate(tqdm(all_employees, desc="직원별 예측")):
        try:
            # 시계열 데이터가 있는 직원인지 확인
            if emp_id in timeseries_employees:
                # 개별 직원 예측 수행 (시계열 기반 예측)
                result = predict_employee_with_best_model(emp_id, processor, model, scaler, best_params)
                result['prediction_method'] = 'Timeseries Model'
                all_predictions.append(result)
                successful_predictions += 1
            else:
                # 시계열 데이터가 없는 경우 속성 데이터만으로 처리
                raise ValueError(f"시계열 데이터가 없어 속성 기반 예측으로 전환합니다.")
            
        except Exception as e:
            # 예측 실패한 경우 기본값으로 저장
            # 속성 데이터에서 실제 attrition 정보가 있다면 사용
            actual_attrition = None
            if emp_id in personas_employees:
                try:
                    persona_row = processor.personas_data[processor.personas_data[processor.personas_id_col] == emp_id]
                    if not persona_row.empty and 'attrition_binary' in persona_row.columns:
                        actual_attrition = persona_row['attrition_binary'].iloc[0]
                except:
                    pass
            
            # 예측 방법 결정
            if emp_id not in timeseries_employees and emp_id in personas_employees:
                prediction_method = 'Attribute-based (No timeseries)'
                prediction_status = 'Success - Attribute only'
            else:
                prediction_method = 'Failed'
                prediction_status = 'Failed - Prediction error'
            
            failed_result = {
                'employee_id': emp_id,
                'attrition_probability': 0.5 if actual_attrition is None else float(actual_attrition),  # 불확실한 경우 0.5
                'prediction': 0 if actual_attrition is None else int(actual_attrition),  # 실제 값이 있으면 사용
                'risk_level': 'Unknown' if actual_attrition is None else ('High' if actual_attrition == 1 else 'Low'),
                'sequence_length_used': best_params['sequence_length'] if emp_id in timeseries_employees else 0,
                'prediction_method': prediction_method,
                'prediction_status': prediction_status,
                'error_message': str(e) if prediction_status.startswith('Failed') else 'No timeseries data available'
            }
            all_predictions.append(failed_result)
            failed_predictions += 1
    
    print(f"\n📊 예측 완료 요약:")
    print(f"   ✅ 성공: {successful_predictions}명")
    print(f"   ❌ 실패: {failed_predictions}명")
    print(f"   📈 성공률: {successful_predictions/len(all_employees)*100:.1f}%")
    
    # DataFrame으로 변환
    predictions_df = pd.DataFrame(all_predictions)
    
    # 컬럼 순서 정리
    column_order = ['employee_id', 'attrition_probability', 'prediction', 'risk_level', 
                   'sequence_length_used', 'prediction_method']
    
    # 추가 정보 컬럼 포함
    column_order.extend(['prediction_status', 'error_message'])
    predictions_df['prediction_method'] = predictions_df.get('prediction_method', 'Timeseries Model')
    predictions_df['prediction_status'] = predictions_df.get('prediction_status', 'Success')
    predictions_df['error_message'] = predictions_df.get('error_message', '')
    
    predictions_df = predictions_df[column_order]
    
    # 퇴사 확률 기준으로 정렬 (내림차순)
    predictions_df = predictions_df.sort_values('attrition_probability', ascending=False)
    
    # CSV 파일로 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    predictions_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    print(f"\n💾 예측 결과 저장 완료: {save_path}")
    print(f"   📄 저장된 레코드 수: {len(predictions_df)}개")
    
    # 결과 요약 출력
    print(f"\n📈 예측 결과 요약:")
    
    # 성공한 예측만으로 통계 계산
    successful_df = predictions_df[predictions_df.get('prediction_status', 'Success') == 'Success']
    
    if len(successful_df) > 0:
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

# 전체 직원 예측 실행 및 저장
print("\n" + "="*70)
print("🎯 전체 직원 시계열 예측 및 결과 저장 시작")
print("="*70)

# 변수 존재 확인
print("🔍 변수 존재 확인:")
print(f"   processor 존재: {'processor' in locals()}")
print(f"   best_model 존재: {'best_model' in locals()}")
print(f"   best_scaler 존재: {'best_scaler' in locals()}")
print(f"   best_params 존재: {'best_params' in locals()}")

try:
    final_predictions_df = predict_all_employees_and_save(
        processor, 
        best_model, 
        best_scaler, 
        best_params,
        save_path='Data analysis/data/employee_attrition_predictions.csv'
    )
except Exception as e:
    print(f"❌ 예측 함수 실행 중 오류 발생: {e}")
    print(f"   오류 타입: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    final_predictions_df = None

if final_predictions_df is not None:
    print(f"\n🎉 전체 프로세스 완료!")
    print(f"   📊 총 {len(final_predictions_df)}명의 직원 예측 결과가 저장되었습니다.")
    print(f"   📁 저장 위치: Data analysis/data/employee_attrition_predictions.csv")
else:
    print(f"\n❌ 프로세스 실행 실패!")
    print(f"   위의 오류 메시지를 확인해주세요.")