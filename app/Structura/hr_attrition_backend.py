"""
HR Attrition Prediction Backend
XGBoost + XAI 기반 설명 가능한 이직 예측 시스템

주요 기능:
- 데이터 전처리 자동화 (노트북 기반 최신 버전)
- XGBoost 모델 훈련 및 하이퍼파라미터 최적화
- SHAP 기반 XAI 설명 (EmployeeNumber별)
- 모델 저장/로딩
- 예측 서비스 (Probability 중심)
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix, precision_recall_curve,
    balanced_accuracy_score
)
import xgboost as xgb
from xgboost import XGBClassifier

# 클래스 불균형 처리
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter

# XAI 라이브러리
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

# Optional: Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Using default hyperparameters.")

warnings.filterwarnings('ignore')


class HRAttritionPredictor:
    """HR 이탈 예측 시스템 (XAI 포함)"""
    
    def __init__(self, data_path: str = "../data/IBM_HR_personas_assigned.csv", random_state: int = 42):
        self.data_path = data_path
        self.random_state = random_state
        self.model = None
        self.feature_columns = None
        self.optimal_threshold = 0.018  # 노트북에서 최적화된 임계값
        self.scale_pos_weight = 1.0
        
        # XAI 관련
        self.shap_explainer = None
        self.X_train_sample = None  # SHAP 배경 데이터
        
        # 전처리 설정
        self.setup_preprocessing_config()
        
    def setup_preprocessing_config(self):
        """전처리 설정 초기화 (노트북 기반)"""
        # 순서형 변수들
        self.ordinal_cols = ['RelationshipSatisfaction', 'Education', 'PerformanceRating', 
                            'JobInvolvement', 'EnvironmentSatisfaction', 'JobLevel', 
                            'JobSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']

        # 명목형 변수들
        self.nominal_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 
                            'JobRole', 'MaritalStatus', 'OverTime']

        # 수치형 변수들
        self.numerical_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 
                              'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
                              'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 
                              'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
                              'YearsWithCurrManager']
        
        # 상수 컬럼들 (제거 예정)
        self.constant_cols = ['Over18', 'EmployeeCount', 'StandardHours']
        
        # 저상관 임계값
        self.low_corr_threshold = 0.03
        
        # 순서형 변수 매핑
        self.ORDINAL_MAPS = {
            "Education": {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"},
            "JobLevel": {1: "Entry", 2: "Junior", 3: "Mid", 4: "Senior", 5: "Executive"},
            "JobInvolvement": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
            "JobSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
            "EnvironmentSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
            "RelationshipSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
            "WorkLifeBalance": {1: "Bad", 2: "Good", 3: "Better", 4: "Best"},
            "PerformanceRating": {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"},
            "StockOptionLevel": {0: "None", 1: "Level 1", 2: "Level 2", 3: "Level 3"},
        }
        
        # 명목형 변수들
        self.NOMINAL_COLS = [
            "BusinessTravel", "Department", "EducationField", "Gender",
            "JobRole", "MaritalStatus", "OverTime"
        ]
        
        # 순서형 라벨 순서 (문자열 매핑용)
        self.ORDINAL_LABEL_ORDERS = {
            "JobInvolvement": ["Low", "Medium", "High", "Very High"],
            "JobSatisfaction": ["Low", "Medium", "High", "Very High"],
            "EnvironmentSatisfaction": ["Low", "Medium", "High", "Very High"],
            "RelationshipSatisfaction": ["Low", "Medium", "High", "Very High"],
            "WorkLifeBalance": ["Bad", "Good", "Better", "Best"],
            "PerformanceRating": ["Low", "Good", "Excellent", "Outstanding"],
            "Education": ["Below College", "College", "Bachelor", "Master", "Doctor"],
            "JobLevel": ["Entry", "Junior", "Mid", "Senior", "Executive"],
            "StockOptionLevel": ["None", "Level 1", "Level 2", "Level 3"],
        }
    
    def load_data(self) -> pd.DataFrame:
        """데이터 로딩"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        print(f"데이터 로딩 완료: {df.shape}")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """데이터 전처리 (노트북 기반 최신 버전)"""
        print("데이터 전처리 시작...")
        
        # 1. 변수 타입별 분류
        all_feature_columns = self.ordinal_cols + self.nominal_cols + self.numerical_cols
        
        # 2. 필요한 컬럼만 선택 + 타겟
        df_processed = df[all_feature_columns + ['Attrition']].copy()
        print(f"선택된 특성 변수 개수: {len(all_feature_columns)}")
        
        # 3. 상수 컬럼 제거
        constant_cols_found = []
        for col in df_processed.columns:
            if col != 'Attrition':
                if df_processed[col].nunique() <= 1:
                    constant_cols_found.append(col)
        
        if constant_cols_found:
            df_processed = df_processed.drop(columns=constant_cols_found)
            print(f"상수 컬럼 제거: {constant_cols_found}")
            # 제거된 컬럼들을 각 타입 리스트에서도 제거
            self.ordinal_cols = [col for col in self.ordinal_cols if col not in constant_cols_found]
            self.nominal_cols = [col for col in self.nominal_cols if col not in constant_cols_found]
            self.numerical_cols = [col for col in self.numerical_cols if col not in constant_cols_found]
        
        # 4. 명목형 범주형 변수만 라벨 인코딩
        print(f"명목형 범주형 변수 인코딩: {self.nominal_cols}")
        self.encoders = {}
        for col in self.nominal_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.encoders[col] = le
        
        # 5. 타겟 변수 인코딩
        df_processed['Attrition'] = (df_processed['Attrition'] == 'Yes').astype(int)
        
        # 6. 상관관계 분석 및 저상관 변수 제거
        correlation_with_target = df_processed.corr()['Attrition'].abs().sort_values(ascending=False)
        low_corr_features = correlation_with_target[correlation_with_target < self.low_corr_threshold].index.tolist()
        if 'Attrition' in low_corr_features:
            low_corr_features.remove('Attrition')
        
        if low_corr_features:
            df_processed = df_processed.drop(columns=low_corr_features)
            print(f"저상관 변수 제거 (< {self.low_corr_threshold}): {low_corr_features}")
        
        # 7. X, y 분리
        y = df_processed['Attrition']
        X = df_processed.drop(columns=['Attrition'])
        
        # 8. 최종 특성 변수 리스트 업데이트
        self.final_features = X.columns.tolist()
        print(f"최종 특성 변수 개수: {len(self.final_features)}")
        print("데이터 전처리 완료")
        
        return X, y
    
    def _coerce_ordinal_to_numeric(self, df: pd.DataFrame, ordinal_cols: List[str]) -> pd.DataFrame:
        """순서형 변수를 수치형으로 변환"""
        df = df.copy()
        
        for c in ordinal_cols:
            s = df[c]
            
            # 이미 수치형인 경우
            if pd.api.types.is_numeric_dtype(s):
                df[c] = pd.to_numeric(s, errors="coerce")
                continue
            
            # 카테고리 타입인 경우
            if pd.api.types.is_categorical_dtype(s):
                if getattr(s.dtype, "ordered", False):
                    codes = s.cat.codes.replace(-1, np.nan)
                    df[c] = codes.astype(float)
                    continue
                
                # 수치형 문자열 시도
                try_num = pd.to_numeric(s.astype(str), errors="coerce")
                if try_num.notna().any():
                    df[c] = try_num
                    continue
                
                # 라벨 매핑 시도
                if c in self.ORDINAL_LABEL_ORDERS:
                    mapped = self._map_labels_to_codes(s, self.ORDINAL_LABEL_ORDERS[c])
                    if mapped.notna().any():
                        df[c] = mapped.astype(float)
                        continue
                
                # 마지막 수단: 카테고리 코드
                df[c] = s.cat.codes.replace(-1, np.nan).astype(float)
                continue
            
            # 객체/문자열 타입인 경우
            try_num = pd.to_numeric(s, errors="coerce")
            if try_num.notna().any():
                df[c] = try_num
                continue
            
            if c in self.ORDINAL_LABEL_ORDERS:
                mapped = self._map_labels_to_codes(s, self.ORDINAL_LABEL_ORDERS[c])
                df[c] = mapped.astype(float)
            else:
                # 마지막 수단: factorize
                codes, _ = pd.factorize(s, na_sentinel=-1)
                df[c] = pd.Series(codes, index=s.index).replace(-1, np.nan).astype(float)
        
        return df
    
    def _map_labels_to_codes(self, series: pd.Series, labels_in_order: List[str]) -> pd.Series:
        """라벨을 코드로 매핑"""
        s = series.astype(str).str.strip()
        start_code = 0 if labels_in_order[0] == "None" else 1
        mapping = {lab.lower(): code for code, lab in enumerate(labels_in_order, start=start_code)}
        return s.str.lower().map(mapping)
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> Tuple:
        """데이터 분할"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        
        print(f"데이터 분할 완료 - 훈련: {X_train.shape}, 테스트: {X_test.shape}")
        print(f"클래스 균형 (훈련): {y_train.value_counts(normalize=True).round(3).to_dict()}")
        print(f"클래스 균형 (테스트): {y_test.value_counts(normalize=True).round(3).to_dict()}")
        
        # 클래스 불균형 가중치 계산
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        self.scale_pos_weight = neg / max(pos, 1)
        print(f"클래스 불균형 가중치: {self.scale_pos_weight:.2f}")
        
        return X_train, X_test, y_train, y_test
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                n_trials: int = 50) -> Dict:
        """하이퍼파라미터 최적화"""
        if not OPTUNA_AVAILABLE:
            print("Optuna가 설치되지 않아 기본 하이퍼파라미터를 사용합니다.")
            return self._get_default_params()
        
        print(f"하이퍼파라미터 최적화 시작 (시행 횟수: {n_trials})...")
        
        def objective(trial):
            params = {
                'n_estimators': 3000,
                'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int("max_depth", 3, 10),
                'subsample': trial.suggest_float("subsample", 0.5, 1.0),
                'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
                'min_child_weight': trial.suggest_float("min_child_weight", 1.0, 20.0),
                'gamma': trial.suggest_float("gamma", 0.0, 5.0),
                'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 2.0, log=True),
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist',
                'enable_categorical': True,
                'scale_pos_weight': self.scale_pos_weight,
                'n_jobs': -1,
                'random_state': self.random_state,
                'verbosity': 0,
            }
            
            # 교차 검증
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model = XGBClassifier(**params)
                model.fit(X_tr, y_tr, 
                         eval_set=[(X_val, y_val)],
                         early_stopping_rounds=100,
                         verbose=False)
                
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred_proba)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction="maximize", 
                                   pruner=MedianPruner(n_warmup_steps=5))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_trial.params
        print(f"최적화 완료 - 최고 AUC: {study.best_value:.4f}")
        print(f"최적 파라미터: {best_params}")
        
        return best_params
    
    def _get_default_params(self) -> Dict:
        """기본 하이퍼파라미터"""
        return {
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_lambda': 1,
            'reg_alpha': 0,
        }
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   hyperparams: Optional[Dict] = None, use_sampling: bool = True) -> XGBClassifier:
        """모델 훈련 (클래스 불균형 해결 + XAI 설정 포함)"""
        print("모델 훈련 시작...")
        
        # 1. 클래스 불균형 해결
        if use_sampling:
            print("클래스 불균형 해결 중...")
            # 여러 샘플링 기법 비교
            sampling_methods = {
                'SMOTE': SMOTE(random_state=self.random_state),
                'SMOTETomek': SMOTETomek(random_state=self.random_state)
            }
            
            best_sampler = None
            best_f1 = 0
            
            for name, sampler in sampling_methods.items():
                try:
                    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                    # 간단한 테스트로 성능 확인
                    quick_model = XGBClassifier(n_estimators=100, random_state=self.random_state)
                    from sklearn.model_selection import cross_val_score
                    cv_f1 = cross_val_score(quick_model, X_resampled, y_resampled, 
                                           cv=3, scoring='f1', n_jobs=-1).mean()
                    print(f"  {name}: F1={cv_f1:.3f}")
                    
                    if cv_f1 > best_f1:
                        best_f1 = cv_f1
                        best_sampler = sampler
                except Exception as e:
                    print(f"  {name}: 실패 ({str(e)[:30]})")
                    continue
            
            if best_sampler:
                X_train_balanced, y_train_balanced = best_sampler.fit_resample(X_train, y_train)
                print(f"선택된 샘플링: {type(best_sampler).__name__}")
                print(f"데이터 크기: {len(y_train)} → {len(y_train_balanced)}")
            else:
                X_train_balanced, y_train_balanced = X_train, y_train
                print("샘플링 실패, 원본 데이터 사용")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # 2. 하이퍼파라미터 설정
        if hyperparams is None:
            # 노트북에서 최적화된 파라미터 사용
            hyperparams = {
                'n_estimators': 800,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 5,
                'reg_lambda': 10,
                'min_child_weight': 5,
                'scale_pos_weight': 8.0
            }
        
        # 3. 모델 파라미터 설정
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'enable_categorical': True,
            'n_jobs': -1,
            'random_state': self.random_state,
            'verbosity': 0,
            **hyperparams
        }
        
        # 4. 모델 훈련
        self.model = XGBClassifier(**params)
        self.model.fit(X_train_balanced, y_train_balanced)
        
        self.feature_columns = X_train.columns.tolist()
        
        # 5. XAI 설정
        self._setup_shap_explainer(X_train_balanced)
        
        print("모델 훈련 및 XAI 설정 완료")
        return self.model
    
    def _setup_shap_explainer(self, X_train: pd.DataFrame):
        """SHAP 설명기 설정"""
        if SHAP_AVAILABLE:
            try:
                print("SHAP 설명기 설정 중...")
                self.shap_explainer = shap.TreeExplainer(self.model)
                # 배경 데이터 샘플링 (메모리 효율성)
                sample_size = min(500, len(X_train))
                sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
                self.X_train_sample = X_train.iloc[sample_indices]
                print("SHAP 설정 완료")
            except Exception as e:
                print(f"SHAP 설정 실패: {str(e)}")
                self.shap_explainer = None
    
    def optimize_threshold(self, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """최적 임계값 찾기 (F1 점수 기준)"""
        print("최적 임계값 탐색 중...")
        
        # 교차 검증으로 OOF 예측 확률 생성
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof_proba = np.zeros(len(y_train))
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # 임시 모델 훈련
            temp_model = XGBClassifier(**self.model.get_params())
            temp_model.fit(X_tr, y_tr)
            
            oof_proba[val_idx] = temp_model.predict_proba(X_val)[:, 1]
        
        # F1 점수가 최대가 되는 임계값 찾기
        precision, recall, thresholds = precision_recall_curve(y_train, oof_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
        
        best_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[max(0, best_idx - 1)] if best_idx < len(thresholds) else 0.5
        
        print(f"최적 임계값: {self.optimal_threshold:.3f} (F1: {f1_scores[best_idx]:.4f})")
        return self.optimal_threshold
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """모델 평가"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        print("모델 평가 중...")
        
        # 예측
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)
        
        # 메트릭 계산
        metrics = {
            'ROC_AUC': roc_auc_score(y_test, y_pred_proba),
            'PR_AUC': average_precision_score(y_test, y_pred_proba),
            'F1': f1_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Threshold': self.optimal_threshold
        }
        
        print("\n=== 모델 성능 평가 ===")
        for metric, value in metrics.items():
            if metric != 'Threshold':
                print(f"{metric:>12}: {value:.4f}")
            else:
                print(f"{metric:>12}: {value:.3f}")
        
        print("\n=== 분류 리포트 ===")
        print(classification_report(y_test, y_pred, digits=3))
        
        print("\n=== 혼동 행렬 ===")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return metrics
    
    def predict(self, X: pd.DataFrame, return_proba: bool = True) -> np.ndarray:
        """예측 (기본적으로 확률 반환)"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        if self.feature_columns is not None:
            # 훈련 시 사용된 피처만 선택
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                raise ValueError(f"필요한 피처가 누락되었습니다: {missing_cols}")
            X = X[self.feature_columns]
        
        if return_proba:
            return self.model.predict_proba(X)[:, 1]
        else:
            proba = self.model.predict_proba(X)[:, 1]
            return (proba >= self.optimal_threshold).astype(int)
    
    def predict_single_employee(self, employee_data: Dict, employee_number: Optional[str] = None) -> Dict:
        """단일 직원 예측 및 XAI 설명"""
        # DataFrame으로 변환
        df = pd.DataFrame([employee_data])
        
        # 예측 확률
        probability = self.predict(df, return_proba=True)[0]
        
        # 위험도 카테고리
        if probability >= 0.7:
            risk_category = "HIGH"
        elif probability >= 0.4:
            risk_category = "MEDIUM"
        else:
            risk_category = "LOW"
        
        # XAI 설명 생성
        explanation = self.explain_prediction(df, employee_number)
        
        result = {
            'employee_number': employee_number,
            'attrition_probability': float(probability),
            'risk_category': risk_category,
            'explanation': explanation
        }
        
        return result
    
    def explain_prediction(self, X: pd.DataFrame, employee_number: Optional[str] = None) -> Dict:
        """예측 설명 (SHAP 기반 XAI)"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 피처 중요도 (전역)
        feature_importance = {}
        if self.feature_columns:
            for feature, importance in zip(self.feature_columns, self.model.feature_importances_):
                feature_importance[feature] = float(importance)
        
        # SHAP 값 계산 (개별 예측 설명)
        shap_explanation = {}
        if self.shap_explainer and SHAP_AVAILABLE:
            try:
                # 피처 순서 맞추기
                X_aligned = X[self.feature_columns] if self.feature_columns else X
                shap_values = self.shap_explainer.shap_values(X_aligned)
                
                # 단일 샘플인 경우
                if len(X_aligned) == 1:
                    shap_values_single = shap_values[0]
                    
                    # 변수별 SHAP 값
                    variable_importance = {}
                    for feature, shap_val in zip(self.feature_columns, shap_values_single):
                        variable_importance[feature] = float(shap_val)
                    
                    # 상위 위험 요인 (양수 SHAP 값)
                    risk_factors = [(feat, val) for feat, val in variable_importance.items() if val > 0]
                    risk_factors.sort(key=lambda x: x[1], reverse=True)
                    
                    # 상위 보호 요인 (음수 SHAP 값)
                    protective_factors = [(feat, abs(val)) for feat, val in variable_importance.items() if val < 0]
                    protective_factors.sort(key=lambda x: x[1], reverse=True)
                    
                    shap_explanation = {
                        'variable_importance': variable_importance,
                        'top_risk_factors': [{'feature': f, 'impact': v} for f, v in risk_factors[:5]],
                        'top_protective_factors': [{'feature': f, 'impact': v} for f, v in protective_factors[:5]],
                        'base_value': float(self.shap_explainer.expected_value),
                        'prediction_explanation': f"기준값 {self.shap_explainer.expected_value:.3f}에서 각 변수의 기여도를 합하여 최종 예측"
                    }
                
            except Exception as e:
                print(f"SHAP 분석 실패: {str(e)}")
                shap_explanation = {'error': f'SHAP 분석 실패: {str(e)}'}
        
        explanation = {
            'employee_number': employee_number,
            'global_feature_importance': feature_importance,
            'individual_explanation': shap_explanation,
            'explanation_method': 'SHAP TreeExplainer' if self.shap_explainer else 'Feature Importance Only'
        }
        
        return explanation
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """피처 중요도 반환"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath: str):
        """모델 저장 (XAI 포함)"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'optimal_threshold': self.optimal_threshold,
            'scale_pos_weight': self.scale_pos_weight,
            'final_features': getattr(self, 'final_features', self.feature_columns),
            'preprocessing_config': {
                'ordinal_cols': self.ordinal_cols,
                'nominal_cols': self.nominal_cols,
                'numerical_cols': self.numerical_cols,
                'constant_cols': self.constant_cols,
                'low_corr_threshold': self.low_corr_threshold,
                'encoders': getattr(self, 'encoders', {})
            },
            'X_train_sample': self.X_train_sample
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"모델이 저장되었습니다: {filepath}")
    
    def load_model(self, filepath: str):
        """모델 로딩 (XAI 포함)"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.optimal_threshold = model_data['optimal_threshold']
        self.scale_pos_weight = model_data['scale_pos_weight']
        self.final_features = model_data.get('final_features', self.feature_columns)
        self.X_train_sample = model_data.get('X_train_sample')
        
        # 전처리 설정 복원
        config = model_data['preprocessing_config']
        self.ordinal_cols = config.get('ordinal_cols', [])
        self.nominal_cols = config.get('nominal_cols', [])
        self.numerical_cols = config.get('numerical_cols', [])
        self.constant_cols = config.get('constant_cols', [])
        self.low_corr_threshold = config.get('low_corr_threshold', 0.03)
        self.encoders = config.get('encoders', {})
        
        # XAI 재설정
        if self.model and SHAP_AVAILABLE:
            try:
                self.shap_explainer = shap.TreeExplainer(self.model)
                print("SHAP 설명기 재설정 완료")
            except Exception as e:
                print(f"SHAP 재설정 실패: {str(e)}")
                self.shap_explainer = None
        
        print(f"모델이 로딩되었습니다: {filepath}")
    
    def run_full_pipeline(self, optimize_hp: bool = False, use_sampling: bool = True) -> Dict:
        """전체 파이프라인 실행 (노트북 기반 최신 버전)"""
        print("=== HR 이탈 예측 모델 파이프라인 시작 (XAI 포함) ===\n")
        
        # 1. 데이터 로딩
        df = self.load_data()
        
        # 2. 전처리
        X, y = self.preprocess_data(df)
        
        # 3. 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=self.random_state
        )
        
        print(f"데이터 분할 완료 - 훈련: {X_train.shape}, 테스트: {X_test.shape}")
        print(f"클래스 균형 (훈련): {y_train.value_counts(normalize=True).round(3).to_dict()}")
        
        # 4. 하이퍼파라미터 설정 (노트북 최적화 결과 사용)
        if optimize_hp and OPTUNA_AVAILABLE:
            best_params = self.optimize_hyperparameters(X_train, y_train, 30)
        else:
            # 노트북에서 최적화된 파라미터 사용
            best_params = {
                'n_estimators': 800,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 5,
                'reg_lambda': 10,
                'min_child_weight': 5,
                'scale_pos_weight': 8.0
            }
        
        # 5. 모델 훈련 (클래스 불균형 해결 + XAI 설정 포함)
        self.train_model(X_train, y_train, hyperparams=best_params, use_sampling=use_sampling)
        
        # 6. 모델 평가 (노트북 기반 임계값 사용)
        metrics = self.evaluate_model(X_test, y_test)
        
        # 7. 피처 중요도
        print("\n=== 상위 피처 중요도 ===")
        importance_df = self.get_feature_importance()
        print(importance_df.head(10))
        
        # 8. XAI 테스트 (첫 번째 테스트 샘플)
        if len(X_test) > 0:
            print("\n=== XAI 설명 테스트 ===")
            test_sample = X_test.iloc[:1]
            explanation = self.explain_prediction(test_sample, "TEST_001")
            print(f"테스트 샘플 예측 확률: {self.predict(test_sample)[0]:.3f}")
            if 'individual_explanation' in explanation and 'top_risk_factors' in explanation['individual_explanation']:
                print("상위 위험 요인:")
                for factor in explanation['individual_explanation']['top_risk_factors'][:3]:
                    print(f"  - {factor['feature']}: {factor['impact']:.3f}")
        
        print("\n=== 파이프라인 완료 (XAI 포함) ===")
        
        return metrics


def main():
    """메인 실행 함수 (XAI 포함)"""
    # 예측기 초기화
    predictor = HRAttritionPredictor(data_path="../data/IBM_HR_personas_assigned.csv")
    
    # 전체 파이프라인 실행
    metrics = predictor.run_full_pipeline(
        optimize_hp=False,  # 노트북 최적화 결과 사용
        use_sampling=True   # 클래스 불균형 해결 사용
    )
    
    # 모델 저장
    predictor.save_model("hr_attrition_model_xai.pkl")
    
    # 사용 예시: 새로운 데이터로 예측 및 XAI 설명
    print("\n=== 예측 및 XAI 설명 예시 ===")
    
    # 예시 데이터 (실제 사용 시에는 새로운 데이터를 로드)
    df = predictor.load_data()
    X, y = predictor.preprocess_data(df)
    
    # 처음 3개 샘플로 예측 및 설명
    sample_data = X.head(3)
    
    for i in range(len(sample_data)):
        employee_data = sample_data.iloc[i:i+1]
        employee_number = f"EMP_{i+1:03d}"
        
        # 예측 확률
        probability = predictor.predict(employee_data)[0]
        
        # XAI 설명
        explanation = predictor.explain_prediction(employee_data, employee_number)
        
        print(f"\n직원 {employee_number}:")
        print(f"  이직 확률: {probability:.3f}")
        
        if 'individual_explanation' in explanation:
            ind_exp = explanation['individual_explanation']
            if 'top_risk_factors' in ind_exp and len(ind_exp['top_risk_factors']) > 0:
                print("  상위 위험 요인:")
                for factor in ind_exp['top_risk_factors'][:3]:
                    print(f"    - {factor['feature']}: {factor['impact']:.3f}")
            
            if 'top_protective_factors' in ind_exp and len(ind_exp['top_protective_factors']) > 0:
                print("  상위 보호 요인:")
                for factor in ind_exp['top_protective_factors'][:3]:
                    print(f"    - {factor['feature']}: {factor['impact']:.3f}")
    
    print("\n=== XAI 기반 HR 이탈 예측 시스템 완료 ===")


if __name__ == "__main__":
    main()
