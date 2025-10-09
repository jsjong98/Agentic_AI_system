# -*- coding: utf-8 -*-
"""
Structura HR 이직 예측 Flask 백엔드 서비스
XGBoost + xAI (SHAP, LIME) 기반 설명 가능한 AI 시스템
React 연동에 최적화된 REST API 서버
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
import logging
import os
import json
import pickle
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# ML 관련 imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix, precision_recall_curve
)
import xgboost as xgb
from xgboost import XGBClassifier

# xAI 관련 imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available. Install with: pip install lime")

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Using default hyperparameters.")

import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------
# 데이터 모델 정의
# ------------------------------------------------------

@dataclass
class PredictionResult:
    """예측 결과 데이터 클래스"""
    employee_id: Optional[str]
    attrition_probability: float
    attrition_prediction: int
    risk_category: str
    confidence_score: float
    prediction_timestamp: str = None
    
    def __post_init__(self):
        if self.prediction_timestamp is None:
            self.prediction_timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return asdict(self)

@dataclass
class ExplainabilityResult:
    """설명 가능성 분석 결과"""
    employee_id: Optional[str]
    feature_importance: Dict[str, float]
    shap_values: Optional[Dict[str, float]]
    lime_explanation: Optional[Dict[str, Any]]
    top_risk_factors: List[Dict[str, Any]]
    top_protective_factors: List[Dict[str, Any]]
    explanation_timestamp: str = None
    
    def __post_init__(self):
        if self.explanation_timestamp is None:
            self.explanation_timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return asdict(self)

# ------------------------------------------------------
# HR 이직 예측기 클래스 (Flask 최적화)
# ------------------------------------------------------

class StructuraHRPredictor:
    """Structura HR 이탈 예측 시스템 (xAI 포함)"""
    
    def __init__(self, data_path: str = None, random_state: int = 42):
        self.data_path = data_path
        self.random_state = random_state
        self.model = None
        self.data = None  # 데이터 저장용 속성 추가
        self.feature_columns = None
        self.optimal_threshold = 0.018  # 노트북에서 최적화된 임계값 (재현율 70% 목표)
        self.scale_pos_weight = 1.0
        
        # xAI 관련
        self.shap_explainer = None
        self.lime_explainer = None
        self.X_train_sample = None  # LIME용 배경 데이터
        
        # 전처리 설정
        self.setup_preprocessing_config()
        
    def setup_preprocessing_config(self):
        """전처리 설정 초기화 (노트북 기반)"""
        # 순서형 변수들 (노트북과 동일)
        self.ordinal_cols = ['RelationshipSatisfaction', 'Education', 'PerformanceRating', 
                            'JobInvolvement', 'EnvironmentSatisfaction', 'JobLevel', 
                            'JobSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']

        # 명목형 변수들 (노트북과 동일)
        self.nominal_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 
                            'JobRole', 'MaritalStatus', 'OverTime']

        # 수치형 변수들 (노트북과 동일)
        self.numerical_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 
                              'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
                              'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 
                              'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
                              'YearsWithCurrManager', 'EmployeeCount', 'EmployeeNumber', 'StandardHours']
        
        # 저상관 임계값 (노트북과 동일)
        self.low_corr_threshold = 0.03
        
        # 레거시 호환성을 위한 매핑들
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
        
        # 명목형 변수들 (레거시 호환성)
        self.NOMINAL_COLS = self.nominal_cols
        
        # 순서형 라벨 순서
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
        self.data = df  # 데이터를 인스턴스 변수에 저장
        logger.info(f"데이터 로딩 완료: {df.shape}")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """데이터 전처리 (노트북 기반 최신 버전)"""
        logger.info("데이터 전처리 시작...")
        
        # 1. 변수 타입별 분류 (노트북과 동일)
        all_feature_columns = self.ordinal_cols + self.nominal_cols + self.numerical_cols
        
        # 2. 실제 존재하는 컬럼만 필터링
        available_columns = df.columns.tolist()
        existing_feature_columns = [col for col in all_feature_columns if col in available_columns]
        missing_columns = [col for col in all_feature_columns if col not in available_columns]
        
        if missing_columns:
            logger.warning(f"누락된 컬럼들: {missing_columns}")
            logger.info(f"사용 가능한 컬럼들: {available_columns}")
        
        # 3. 필요한 컬럼만 선택 + 타겟
        target_columns = existing_feature_columns + ['Attrition']
        df_processed = df[target_columns].copy()
        logger.info(f"선택된 특성 변수 개수: {len(existing_feature_columns)} (전체 {len(all_feature_columns)}개 중)")
        
        # 3. 상수 컬럼 제거 (노트북과 동일)
        constant_cols_found = []
        for col in df_processed.columns:
            if col != 'Attrition':
                if df_processed[col].nunique() <= 1:
                    constant_cols_found.append(col)
        
        if constant_cols_found:
            df_processed = df_processed.drop(columns=constant_cols_found)
            logger.info(f"상수 컬럼 제거: {constant_cols_found}")
        
        # 4. 컬럼 타입 리스트 업데이트 (존재하는 컬럼만 유지)
        self.ordinal_cols = [col for col in self.ordinal_cols if col in df_processed.columns]
        self.nominal_cols = [col for col in self.nominal_cols if col in df_processed.columns]
        self.numerical_cols = [col for col in self.numerical_cols if col in df_processed.columns]
        
        # 5. 명목형 범주형 변수만 라벨 인코딩 (노트북과 동일)
        logger.info(f"명목형 범주형 변수 인코딩: {self.nominal_cols}")
        self.encoders = {}
        for col in self.nominal_cols:
            if col in df_processed.columns:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.encoders[col] = le
        
        # 6. 타겟 변수 인코딩 (노트북과 동일)
        df_processed['Attrition'] = (df_processed['Attrition'] == 'Yes').astype(int)
        
        # 7. 상관관계 분석 및 저상관 변수 제거 (노트북과 동일)
        correlation_with_target = df_processed.corr()['Attrition'].abs().sort_values(ascending=False)
        low_corr_features = correlation_with_target[correlation_with_target < self.low_corr_threshold].index.tolist()
        if 'Attrition' in low_corr_features:
            low_corr_features.remove('Attrition')
        
        if low_corr_features:
            df_processed = df_processed.drop(columns=low_corr_features)
            logger.info(f"저상관 변수 제거 (< {self.low_corr_threshold}): {low_corr_features}")
        
        # 8. X, y 분리
        y = df_processed['Attrition']
        X = df_processed.drop(columns=['Attrition'])
        
        # 9. 데이터 타입 변환 (XGBoost 호환성)
        logger.info("데이터 타입 변환 중...")
        for col in X.columns:
            if X[col].dtype == 'object':
                # object 타입을 수치형으로 변환
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    # NaN 값이 있으면 0으로 채움
                    if X[col].isna().any():
                        X[col] = X[col].fillna(0)
                        logger.info(f"  {col}: object → numeric (NaN → 0)")
                    else:
                        logger.info(f"  {col}: object → numeric")
                except:
                    # 변환 실패 시 라벨 인코딩
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    logger.info(f"  {col}: object → label encoded")
        
        # 10. 최종 특성 변수 리스트 업데이트
        self.final_features = X.columns.tolist()
        logger.info(f"최종 특성 변수 개수: {len(self.final_features)}")
        logger.info("데이터 전처리 완료")
        
        return X, y
    
    def _preprocess_single_employee(self, df: pd.DataFrame) -> pd.DataFrame:
        """개별 직원 데이터 전처리 (예측용)"""
        df_processed = df.copy()
        
        # Attrition 컬럼 제거 (있다면)
        if 'Attrition' in df_processed.columns:
            df_processed = df_processed.drop(columns=['Attrition'])
        
        # 훈련 시 사용된 피처만 선택
        if hasattr(self, 'final_features') and self.final_features:
            # 누락된 피처는 0으로 채움
            for feature in self.final_features:
                if feature not in df_processed.columns:
                    df_processed[feature] = 0
            
            # 필요한 피처만 선택하고 순서 맞춤
            df_processed = df_processed[self.final_features]
        else:
            # 기본 전처리 적용
            # 1. 필요한 컬럼만 선택
            available_cols = [col for col in (self.ordinal_cols + self.nominal_cols + self.numerical_cols) 
                            if col in df_processed.columns]
            df_processed = df_processed[available_cols]
            
            # 2. 명목형 변수 인코딩 (훈련 시 사용된 인코더 적용)
            if hasattr(self, 'encoders'):
                for col, encoder in self.encoders.items():
                    if col in df_processed.columns:
                        try:
                            # 새로운 카테고리는 가장 빈번한 값으로 대체
                            unknown_mask = ~df_processed[col].astype(str).isin(encoder.classes_)
                            if unknown_mask.any():
                                most_frequent = encoder.classes_[0]  # 첫 번째 클래스 사용
                                df_processed.loc[unknown_mask, col] = most_frequent
                            
                            df_processed[col] = encoder.transform(df_processed[col].astype(str))
                        except Exception as e:
                            logger.warning(f"인코딩 실패 {col}: {str(e)}, 기본값 0 사용")
                            df_processed[col] = 0
        
        # 3. 데이터 타입 변환 (XGBoost 호환성)
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                try:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    if df_processed[col].isna().any():
                        df_processed[col] = df_processed[col].fillna(0)
                except:
                    # 변환 실패 시 0으로 설정
                    df_processed[col] = 0
        
        return df_processed
    
    def _coerce_ordinal_to_numeric(self, df: pd.DataFrame, ordinal_cols: List[str]) -> pd.DataFrame:
        """순서형 변수를 수치형으로 변환"""
        df = df.copy()
        
        for c in ordinal_cols:
            s = df[c]
            
            if pd.api.types.is_numeric_dtype(s):
                df[c] = pd.to_numeric(s, errors="coerce")
                continue
            
            if pd.api.types.is_categorical_dtype(s):
                if getattr(s.dtype, "ordered", False):
                    codes = s.cat.codes.replace(-1, np.nan)
                    df[c] = codes.astype(float)
                    continue
                
                try_num = pd.to_numeric(s.astype(str), errors="coerce")
                if try_num.notna().any():
                    df[c] = try_num
                    continue
                
                if c in self.ORDINAL_LABEL_ORDERS:
                    mapped = self._map_labels_to_codes(s, self.ORDINAL_LABEL_ORDERS[c])
                    if mapped.notna().any():
                        df[c] = mapped.astype(float)
                        continue
                
                df[c] = s.cat.codes.replace(-1, np.nan).astype(float)
                continue
            
            try_num = pd.to_numeric(s, errors="coerce")
            if try_num.notna().any():
                df[c] = try_num
                continue
            
            if c in self.ORDINAL_LABEL_ORDERS:
                mapped = self._map_labels_to_codes(s, self.ORDINAL_LABEL_ORDERS[c])
                df[c] = mapped.astype(float)
            else:
                codes, _ = pd.factorize(s, na_sentinel=-1)
                df[c] = pd.Series(codes, index=s.index).replace(-1, np.nan).astype(float)
        
        return df
    
    def _map_labels_to_codes(self, series: pd.Series, labels_in_order: List[str]) -> pd.Series:
        """라벨을 코드로 매핑"""
        s = series.astype(str).str.strip()
        start_code = 0 if labels_in_order[0] == "None" else 1
        mapping = {lab.lower(): code for code, lab in enumerate(labels_in_order, start=start_code)}
        return s.str.lower().map(mapping)

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   hyperparams: Optional[Dict] = None) -> XGBClassifier:
        """모델 훈련 및 xAI 설정"""
        import time
        start_time = time.time()
        
        logger.info(f"🚀 XGBoost 모델 훈련 시작... (데이터: {len(X_train)}행 x {len(X_train.columns)}열)")
        
        if hyperparams is None:
            hyperparams = self._get_default_params()
        
        # 클래스 불균형 가중치 계산
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        self.scale_pos_weight = neg / max(pos, 1)
        
        logger.info(f"📊 클래스 분포: Negative={neg}, Positive={pos}, Scale_pos_weight={self.scale_pos_weight:.3f}")
        
        # 모델 파라미터 설정
        params = {
            'n_estimators': 1000,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'enable_categorical': True,
            'scale_pos_weight': self.scale_pos_weight,
            'n_jobs': -1,
            'random_state': self.random_state,
            'verbosity': 1,  # 학습 과정 표시
            **hyperparams
        }
        
        logger.info(f"🔧 XGBoost 파라미터: n_estimators={params['n_estimators']}, tree_method={params['tree_method']}")
        
        self.model = XGBClassifier(**params)
        
        # 실제 학습 시간 측정
        fit_start_time = time.time()
        self.model.fit(X_train, y_train)
        fit_end_time = time.time()
        
        fit_duration = fit_end_time - fit_start_time
        logger.info(f"⏱️ XGBoost 학습 완료: {fit_duration:.2f}초 ({fit_duration/60:.1f}분)")
        
        self.feature_columns = X_train.columns.tolist()
        
        # xAI 설정
        logger.info("🔍 xAI 설명기 설정 중...")
        xai_start_time = time.time()
        self._setup_explainers(X_train, y_train)
        xai_end_time = time.time()
        
        xai_duration = xai_end_time - xai_start_time
        logger.info(f"🔍 xAI 설정 완료: {xai_duration:.2f}초")
        
        total_duration = time.time() - start_time
        logger.info(f"✅ 전체 모델 훈련 및 xAI 설정 완료: {total_duration:.2f}초 ({total_duration/60:.1f}분)")
        return self.model
    
    def _setup_explainers(self, X_train: pd.DataFrame, y_train: pd.Series):
        """SHAP 및 LIME 설명기 설정"""
        
        # SHAP 설정
        if SHAP_AVAILABLE:
            try:
                logger.info("🔍 SHAP 설명기 설정 중...")
                self.shap_explainer = shap.TreeExplainer(self.model)
                logger.info("✅ SHAP 설정 완료")
            except Exception as e:
                logger.error(f"❌ SHAP 설정 실패: {str(e)}")
                logger.error(f"SHAP 오류 상세: {traceback.format_exc()}")
                self.shap_explainer = None
        else:
            logger.warning("⚠️ SHAP 라이브러리가 설치되지 않았습니다")
            self.shap_explainer = None
        
        # LIME 설정
        if LIME_AVAILABLE:
            try:
                logger.info("🔍 LIME 설명기 설정 중...")
                # 샘플링된 훈련 데이터 (LIME 배경용)
                sample_size = min(1000, len(X_train))
                sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
                self.X_train_sample = X_train.iloc[sample_indices].values
                
                # 범주형 피처 인덱스 찾기
                categorical_features = []
                for i, col in enumerate(X_train.columns):
                    if col in self.nominal_cols or pd.api.types.is_categorical_dtype(X_train[col]):
                        categorical_features.append(i)
                
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    self.X_train_sample,
                    feature_names=self.feature_columns,
                    categorical_features=categorical_features,
                    mode='classification',
                    random_state=self.random_state
                )
                logger.info("✅ LIME 설정 완료")
            except Exception as e:
                logger.error(f"❌ LIME 설정 실패: {str(e)}")
                logger.error(f"LIME 오류 상세: {traceback.format_exc()}")
                self.lime_explainer = None
        else:
            logger.warning("⚠️ LIME 라이브러리가 설치되지 않았습니다")
            self.lime_explainer = None
        
        # XAI 설정 상태 로깅
        logger.info(f"🔍 XAI 설정 상태:")
        logger.info(f"  - SHAP explainer: {'✅ 활성화' if self.shap_explainer else '❌ 비활성화'}")
        logger.info(f"  - LIME explainer: {'✅ 활성화' if self.lime_explainer else '❌ 비활성화'}")
        logger.info(f"  - Feature columns: {len(self.feature_columns)}개")
    
    def _get_default_params(self) -> Dict:
        """기본 하이퍼파라미터 (노트북에서 최적화된 값)"""
        return {
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

    def predict(self, X: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """예측"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        if self.feature_columns is not None:
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                raise ValueError(f"필요한 피처가 누락되었습니다: {missing_cols}")
            X = X[self.feature_columns]
        
        if return_proba:
            return self.model.predict_proba(X)[:, 1]
        else:
            proba = self.model.predict_proba(X)[:, 1]
            return (proba >= self.optimal_threshold).astype(int)

    def predict_single(self, employee_data: Dict) -> PredictionResult:
        """단일 직원 예측"""
        # 딕셔너리를 DataFrame으로 변환
        df = pd.DataFrame([employee_data])
        
        # 전처리 적용 (개별 직원용)
        df_processed = self._preprocess_single_employee(df)
        
        # 예측
        probability = self.predict(df_processed, return_proba=True)[0]
        prediction = int(probability >= self.optimal_threshold)
        
        # 위험 범주 결정
        if probability >= 0.7:
            risk_category = "HIGH"
        elif probability >= 0.4:
            risk_category = "MEDIUM"
        else:
            risk_category = "LOW"
        
        # 신뢰도 점수 (확률과 임계값의 거리 기반)
        confidence_score = abs(probability - 0.5) * 2
        
        return PredictionResult(
            employee_id=employee_data.get('EmployeeNumber', None),
            attrition_probability=float(probability),
            attrition_prediction=prediction,
            risk_category=risk_category,
            confidence_score=float(confidence_score)
        )

    def explain_prediction(self, employee_data: Dict) -> ExplainabilityResult:
        """예측 설명 (xAI) - 실제 XAI 데이터 생성"""
        # 딕셔너리를 DataFrame으로 변환
        df = pd.DataFrame([employee_data])
        
        # 전처리 적용 (개별 직원용)
        df_processed = self._preprocess_single_employee(df)
        
        # 피처 중요도 (모델 기본) - 실제 데이터만 사용
        feature_importance = {}
        if self.model and hasattr(self.model, 'feature_importances_') and self.feature_columns:
            for feature, importance in zip(self.feature_columns, self.model.feature_importances_):
                feature_importance[feature] = float(importance)
            logger.info(f"실제 Feature Importance 계산 완료: {len(feature_importance)}개 피처")
        else:
            logger.warning("모델이 훈련되지 않았거나 피처 중요도를 계산할 수 없습니다")
        
        # SHAP 값 - 실제 계산만 수행
        shap_values = {}
        if self.shap_explainer and len(df_processed) > 0:
            try:
                # 실제 SHAP 계산
                shap_vals = self.shap_explainer.shap_values(df_processed[self.feature_columns])
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                
                for feature, shap_val in zip(self.feature_columns, shap_vals[0]):
                    shap_values[feature] = float(shap_val)
                logger.info(f"실제 SHAP 값 계산 완료: {len(shap_values)}개 피처")
            except Exception as e:
                logger.warning(f"SHAP 분석 실패: {str(e)}")
                shap_values = {}
        else:
            logger.warning("SHAP explainer가 초기화되지 않았습니다")
            if not SHAP_AVAILABLE:
                logger.warning("SHAP 라이브러리가 설치되지 않았습니다. 설치 명령: pip install shap>=0.42.0")
            elif not self.shap_explainer:
                logger.warning("SHAP explainer 초기화에 실패했습니다")
        
        # LIME 설명 - 실제 계산만 수행
        lime_explanation = {}
        if self.lime_explainer and len(df_processed) > 0:
            try:
                def predict_fn(x):
                    return self.model.predict_proba(pd.DataFrame(x, columns=self.feature_columns))
                
                explanation = self.lime_explainer.explain_instance(
                    df_processed[self.feature_columns].values[0],
                    predict_fn,
                    num_features=8
                )
                
                lime_explanation = {
                    'features': [str(item[0]) for item in explanation.as_list()],
                    'values': [float(item[1]) for item in explanation.as_list()],
                    'intercept': float(explanation.intercept[1]) if hasattr(explanation, 'intercept') else 0.0
                }
                logger.info(f"실제 LIME 설명 생성 완료: {len(lime_explanation['features'])}개 피처")
            except Exception as e:
                logger.warning(f"LIME 분석 실패: {str(e)}")
                lime_explanation = {}
        else:
            logger.warning("LIME explainer가 초기화되지 않았습니다")
            if not LIME_AVAILABLE:
                logger.warning("LIME 라이브러리가 설치되지 않았습니다. 설치 명령: pip install lime>=0.2.0.1")
            elif not self.lime_explainer:
                logger.warning("LIME explainer 초기화에 실패했습니다")
        
        # 상위 위험/보호 요인 추출
        importance_items = list(feature_importance.items()) if feature_importance else []
        importance_items.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # SHAP 값이 있으면 우선 사용
        if shap_values:
            shap_items = list(shap_values.items())
            shap_items.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_risk_factors = [
                {"feature": feat, "impact": float(val), "type": "risk"}
                for feat, val in shap_items if val > 0
            ][:5]
            top_protective_factors = [
                {"feature": feat, "impact": float(abs(val)), "type": "protective"}
                for feat, val in shap_items if val < 0
            ][:5]
        else:
            # 피처 중요도 기반
            top_risk_factors = [
                {"feature": feat, "impact": float(val), "type": "risk"}
                for feat, val in importance_items[:5]
            ]
            top_protective_factors = []
        
        return ExplainabilityResult(
            employee_id=employee_data.get('EmployeeNumber', None),
            feature_importance=feature_importance,
            shap_values=shap_values,
            lime_explanation=lime_explanation,
            top_risk_factors=top_risk_factors,
            top_protective_factors=top_protective_factors
        )

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
        """모델 및 설정 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'optimal_threshold': self.optimal_threshold,
            'scale_pos_weight': self.scale_pos_weight,
            'preprocessing_config': {
                'ordinal_cols': getattr(self, 'ordinal_cols', []),
                'nominal_cols': getattr(self, 'nominal_cols', []),
                'numerical_cols': getattr(self, 'numerical_cols', []),
                'low_corr_threshold': getattr(self, 'low_corr_threshold', 0.03),
                'ORDINAL_LABEL_ORDERS': getattr(self, 'ORDINAL_LABEL_ORDERS', {}),
                'encoders': getattr(self, 'encoders', {})
            },
            'X_train_sample': self.X_train_sample
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"모델이 저장되었습니다: {filepath}")
    
    def load_model(self, filepath: str):
        """모델 및 설정 로딩"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.optimal_threshold = model_data['optimal_threshold']
        self.scale_pos_weight = model_data['scale_pos_weight']
        
        # 전처리 설정 복원
        config = model_data['preprocessing_config']
        self.ordinal_cols = config.get('ordinal_cols', [])
        self.nominal_cols = config.get('nominal_cols', [])
        self.numerical_cols = config.get('numerical_cols', [])
        self.low_corr_threshold = config.get('low_corr_threshold', 0.03)
        self.ORDINAL_LABEL_ORDERS = config.get('ORDINAL_LABEL_ORDERS', {})
        self.encoders = config.get('encoders', {})
        
        # xAI 데이터 복원
        self.X_train_sample = model_data.get('X_train_sample')
        
        # xAI 설명기 재설정
        if self.model and self.X_train_sample is not None:
            self._setup_explainers_from_saved_data()
        
        logger.info(f"모델이 로딩되었습니다: {filepath}")
    
    def _setup_explainers_from_saved_data(self):
        """저장된 데이터로부터 xAI 설명기 재설정"""
        # SHAP 설정
        if SHAP_AVAILABLE:
            try:
                self.shap_explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                logger.warning(f"SHAP 재설정 실패: {str(e)}")
        
        # LIME 설정
        if LIME_AVAILABLE and self.X_train_sample is not None:
            try:
                categorical_features = []
                for i, col in enumerate(self.feature_columns):
                    if col in self.nominal_cols:
                        categorical_features.append(i)
                
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    self.X_train_sample,
                    feature_names=self.feature_columns,
                    categorical_features=categorical_features,
                    mode='classification',
                    random_state=self.random_state
                )
            except Exception as e:
                logger.warning(f"LIME 재설정 실패: {str(e)}")

    def run_full_pipeline(self, optimize_hp: bool = False, use_sampling: bool = True) -> Dict:
        """전체 파이프라인 실행 (노트북 기반 최신 버전)"""
        logger.info("Structura 전체 파이프라인 시작 (노트북 기반)...")
        
        try:
            # 1. 데이터 로딩
            logger.info("1. 데이터 로딩 중...")
            df = self.load_data()
            
            # 2. 전처리
            logger.info("2. 데이터 전처리 중...")
            X, y = self.preprocess_data(df)
            
            # 3. 훈련/테스트 분할 (노트북과 동일하게 30%)
            logger.info("3. 데이터 분할 중...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=self.random_state, stratify=y
            )
            
            logger.info(f"데이터 분할 완료 - 훈련: {X_train.shape}, 테스트: {X_test.shape}")
            logger.info(f"클래스 균형 (훈련): {y_train.value_counts(normalize=True).round(3).to_dict()}")
            
            # 4. 클래스 불균형 해결 (노트북 기반)
            if use_sampling:
                logger.info("4. 클래스 불균형 해결 중...")
                X_train_balanced, y_train_balanced = self._apply_sampling(X_train, y_train)
                logger.info(f"샘플링 완료: {len(X_train)} → {len(X_train_balanced)}")
            else:
                X_train_balanced, y_train_balanced = X_train, y_train
            
            # 5. 하이퍼파라미터 최적화 (노트북 기반)
            logger.info("5. 하이퍼파라미터 설정 중...")
            if optimize_hp and OPTUNA_AVAILABLE:
                hyperparams = self._optimize_hyperparameters_enhanced(X_train_balanced, y_train_balanced)
            else:
                # 노트북에서 최적화된 파라미터 사용
                hyperparams = self._get_default_params()
            
            # 6. 모델 훈련
            logger.info("6. 모델 훈련 중...")
            self.train_model(X_train_balanced, y_train_balanced, hyperparams)
            
            # 7. 임계값 최적화 (재현율 70% 목표)
            logger.info("7. 임계값 최적화 중...")
            self.optimal_threshold = self._optimize_threshold(X_test, y_test, target_recall=0.7)
            logger.info(f"최적 임계값: {self.optimal_threshold:.3f}")
            
            # 8. 성능 평가
            logger.info("8. 성능 평가 중...")
            y_pred_proba = self.predict(X_test, return_proba=True)
            y_pred = self.predict(X_test, return_proba=False)
            
            # 9. 메트릭 계산
            metrics = {
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'avg_precision': average_precision_score(y_test, y_pred_proba),
                'optimal_threshold': self.optimal_threshold,
                'training_samples': len(X_train_balanced),
                'test_samples': len(X_test)
            }
            
            logger.info(f"✅ 모델 훈련 완료!")
            logger.info(f"성능 지표 - AUC: {metrics['roc_auc']:.3f}, F1: {metrics['f1_score']:.3f}, 재현율: {metrics['recall']:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"파이프라인 실행 실패: {str(e)}")
            raise

    def predict_single_employee(self, employee_data: Dict, employee_number: str) -> Dict:
        """단일 직원 예측 (마스터 서버 호환) - 실제 XAI 데이터 포함"""
        try:
            logger.info(f"🔍 직원 {employee_number} 예측 시작...")
            
            # 예측 수행
            prediction_result = self.predict_single(employee_data)
            explanation_result = self.explain_prediction(employee_data)
            
            logger.info(f"📊 직원 {employee_number} 예측 결과: 확률={prediction_result.attrition_probability:.3f}, 위험도={prediction_result.risk_category}")
            
            # XAI 데이터 로깅
            logger.info(f"🔍 XAI 데이터 생성 결과:")
            logger.info(f"  - Feature Importance: {len(explanation_result.feature_importance)}개 피처")
            logger.info(f"  - SHAP Values: {len(explanation_result.shap_values) if explanation_result.shap_values else 0}개 피처")
            logger.info(f"  - LIME Explanation: {'있음' if explanation_result.lime_explanation else '없음'}")
            
            # 통합 결과 구성 - XAI 데이터를 최상위 레벨에도 포함
            result = {
                'employee_number': employee_number,
                'attrition_probability': prediction_result.attrition_probability,
                'attrition_prediction': prediction_result.attrition_prediction,
                'risk_category': prediction_result.risk_category,
                'confidence_score': prediction_result.confidence_score,
                'prediction_timestamp': prediction_result.prediction_timestamp,
                
                # XAI 데이터를 최상위 레벨에 포함 (기존 호환성)
                'feature_importance': explanation_result.feature_importance or {},
                'shap_values': explanation_result.shap_values or {},
                'lime_explanation': explanation_result.lime_explanation or {},
                'model_interpretation': {
                    'top_risk_factors': explanation_result.top_risk_factors,
                    'top_protective_factors': explanation_result.top_protective_factors
                },
                'xai_explanation': {
                    'feature_importance': explanation_result.feature_importance or {},
                    'shap_values': explanation_result.shap_values or {},
                    'lime_explanation': explanation_result.lime_explanation or {},
                    'interpretation_method': 'XGBoost + SHAP + LIME'
                },
                
                # explanation 구조 (새로운 구조)
                'explanation': {
                    'feature_importance': explanation_result.feature_importance or {},
                    'shap_values': explanation_result.shap_values or {},
                    'lime_explanation': explanation_result.lime_explanation or {},
                    'top_risk_factors': explanation_result.top_risk_factors,
                    'top_protective_factors': explanation_result.top_protective_factors,
                    'individual_explanation': {
                        'top_risk_factors': explanation_result.top_risk_factors,
                        'top_protective_factors': explanation_result.top_protective_factors
                    }
                },
                'recommendations': self._generate_recommendations({
                    'risk_category': prediction_result.risk_category,
                    'attrition_probability': prediction_result.attrition_probability,
                    'explanation': {
                        'individual_explanation': {
                            'top_risk_factors': explanation_result.top_risk_factors
                        }
                    }
                })
            }
            
            return result
            
        except Exception as e:
            logger.error(f"직원 {employee_number} 예측 실패: {str(e)}")
            raise

    def _apply_sampling(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """클래스 불균형 해결 (노트북 기반)"""
        try:
            # 여러 샘플링 기법 비교 테스트
            from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
            from imblearn.combine import SMOTETomek, SMOTEENN
            from collections import Counter
            
            logger.info("클래스 불균형 해결 기법 비교 중...")
            
            sampling_methods = {
                'SMOTE': SMOTE(random_state=self.random_state),
                'ADASYN': ADASYN(random_state=self.random_state),
                'BorderlineSMOTE': BorderlineSMOTE(random_state=self.random_state),
                'SMOTETomek': SMOTETomek(random_state=self.random_state)
            }
            
            sampling_results = {}
            for name, sampler in sampling_methods.items():
                try:
                    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                    
                    # 간단한 테스트 모델로 성능 확인
                    quick_model = XGBClassifier(
                        n_estimators=100, 
                        random_state=self.random_state,
                        scale_pos_weight=5,
                        verbosity=0
                    )
                    
                    cv_f1 = cross_val_score(quick_model, X_resampled, y_resampled, 
                                           cv=3, scoring='f1', n_jobs=-1).mean()
                    
                    sampling_results[name] = {
                        'cv_f1': cv_f1,
                        'size': len(y_resampled),
                        'class_dist': Counter(y_resampled)
                    }
                    
                    logger.info(f"  {name}: F1={cv_f1:.3f}, Size={len(y_resampled)}")
                    
                except Exception as e:
                    logger.warning(f"  {name}: 실패 ({str(e)[:50]})")
                    continue
            
            # 가장 좋은 샘플링 방법 선택
            if sampling_results:
                best_sampling_name = max(sampling_results.items(), key=lambda x: x[1]['cv_f1'])[0]
                best_sampler = sampling_methods[best_sampling_name]
                X_train_balanced, y_train_balanced = best_sampler.fit_resample(X_train, y_train)
                
                logger.info(f"선택된 샘플링: {best_sampling_name}")
                logger.info(f"CV F1 Score: {sampling_results[best_sampling_name]['cv_f1']:.3f}")
                
                balanced_dist = pd.Series(y_train_balanced).value_counts()
                logger.info(f"균형 후 분포: No={balanced_dist[0]}, Yes={balanced_dist[1]}")
                
                return pd.DataFrame(X_train_balanced, columns=X_train.columns), pd.Series(y_train_balanced)
            else:
                # 백업: 기본 SMOTE 사용
                logger.warning("모든 샘플링 실패, 기본 SMOTE 사용")
                smote = SMOTE(random_state=self.random_state)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                return pd.DataFrame(X_train_balanced, columns=X_train.columns), pd.Series(y_train_balanced)
                
        except ImportError:
            logger.warning("imblearn 라이브러리가 없습니다. 샘플링 생략")
            return X_train, y_train
        except Exception as e:
            logger.error(f"샘플링 실패: {str(e)}")
            return X_train, y_train
    
    def _optimize_threshold(self, X_test: pd.DataFrame, y_test: pd.Series, target_recall: float = 0.7) -> float:
        """임계값 최적화 (재현율 목표 기반)"""
        try:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Precision-Recall 곡선 계산
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            
            # 배열 길이 조정
            if len(precision) > len(thresholds):
                precision = precision[:-1]
                recall = recall[:-1]
            
            # F1 점수 계산
            with np.errstate(divide='ignore', invalid='ignore'):
                f1_scores = 2 * (precision * recall) / (precision + recall)
                f1_scores = np.nan_to_num(f1_scores)
            
            # 목표 재현율 이상인 인덱스 찾기
            valid_indices = recall >= target_recall
            
            if np.any(valid_indices):
                valid_f1 = f1_scores[valid_indices]
                valid_thresholds = thresholds[valid_indices]
                
                if len(valid_f1) > 0 and len(valid_thresholds) > 0:
                    best_idx = np.argmax(valid_f1)
                    optimal_threshold = valid_thresholds[best_idx]
                    logger.info(f"목표 재현율 {target_recall:.1%} 달성 임계값: {optimal_threshold:.3f}")
                    return float(optimal_threshold)
            
            # 목표 달성 불가능한 경우, F1 최적화 임계값 사용
            if len(f1_scores) > 0:
                best_f1_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[best_f1_idx]
                logger.warning(f"목표 재현율 미달성, F1 최적화 임계값 사용: {optimal_threshold:.3f}")
                return float(optimal_threshold)
            
            # 최종 백업
            logger.warning("임계값 최적화 실패, 기본값 0.5 사용")
            return 0.5
            
        except Exception as e:
            logger.error(f"임계값 최적화 실패: {str(e)}")
            return 0.5
    
    def _optimize_hyperparameters_enhanced(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """향상된 하이퍼파라미터 최적화 (F1 Score 중심)"""
        if not OPTUNA_AVAILABLE:
            return self._get_default_params()
        
        def objective(trial):
            # 클래스 가중치 범위 확대
            pos_weight = trial.suggest_float('scale_pos_weight', 3.0, 15.0)
            
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.25),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 20),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 20),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'scale_pos_weight': pos_weight
            }
            
            # F1 Score 기반 교차검증
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                tree_method='hist',
                enable_categorical=True,
                n_jobs=-1,
                random_state=self.random_state,
                verbosity=0,
                **params
            )
            
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=kfold,
                scoring='f1',
                n_jobs=-1
            )
            
            return cv_scores.mean()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        logger.info("향상된 하이퍼파라미터 최적화 시작 (150회 시행)...")
        study.optimize(objective, n_trials=150, show_progress_bar=False)
        
        logger.info(f"최적 F1 점수: {study.best_value:.4f}")
        logger.info(f"최적 하이퍼파라미터: {study.best_params}")
        return study.best_params

    def _generate_recommendations(self, prediction_result):
        """예측 결과 기반 권장사항 생성"""
        recommendations = []
        
        risk_category = prediction_result['risk_category']
        probability = prediction_result['attrition_probability']
        
        if risk_category == 'HIGH':
            recommendations.extend([
                "즉시 면담을 통한 이직 의도 파악 필요",
                "업무 환경 및 만족도 개선 방안 논의",
                "경력 개발 및 승진 기회 제공 검토",
                "보상 체계 재검토 및 조정 고려"
            ])
        elif risk_category == 'MEDIUM':
            recommendations.extend([
                "정기적인 1:1 면담을 통한 상태 모니터링",
                "업무 만족도 향상을 위한 개선 방안 모색",
                "교육 및 개발 기회 제공",
                "팀 내 역할 및 책임 재조정 검토"
            ])
        else:  # LOW
            recommendations.extend([
                "현재 상태 유지를 위한 지속적 관리",
                "성과 인정 및 피드백 제공",
                "장기 경력 개발 계획 수립 지원"
            ])
        
        # XAI 설명 기반 추가 권장사항
        if 'explanation' in prediction_result and 'individual_explanation' in prediction_result['explanation']:
            exp = prediction_result['explanation']['individual_explanation']
            if 'top_risk_factors' in exp:
                for factor in exp['top_risk_factors'][:2]:  # 상위 2개 위험 요인
                    feature = factor['feature']
                    if 'Satisfaction' in feature:
                        recommendations.append(f"{feature} 개선을 위한 구체적 액션 플랜 수립")
                    elif 'WorkLifeBalance' in feature:
                        recommendations.append("워라밸 개선을 위한 유연근무제 도입 검토")
                    elif 'OverTime' in feature:
                        recommendations.append("업무량 조정 및 초과근무 최소화 방안 마련")
        
        return recommendations

# ------------------------------------------------------
# Flask 애플리케이션 생성 및 설정
# ------------------------------------------------------

# 유틸리티 함수들
def get_data_path_by_analysis_type(analysis_type='batch'):
    """분석 타입에 따른 데이터 경로 반환"""
    # 여러 가능한 경로 시도
    possible_paths = [
        f"../uploads/Structura/{analysis_type}",  # 대문자 S
        f"../uploads/structura/{analysis_type}",  # 소문자 s
        f"app/uploads/Structura/{analysis_type}",  # 절대 경로 스타일
        f"app/uploads/structura/{analysis_type}"   # 절대 경로 스타일 소문자
    ]
    
    for uploads_dir in possible_paths:
        if os.path.exists(uploads_dir):
            files = [f for f in os.listdir(uploads_dir) if f.endswith('.csv')]
            if files:
                files.sort(reverse=True)  # 최신 파일 우선
                logger.info(f"✅ 데이터 파일 발견: {uploads_dir}/{files[0]}")
                return os.path.join(uploads_dir, files[0])
    
    logger.error(f"❌ 데이터 파일을 찾을 수 없습니다. 시도한 경로들: {possible_paths}")
    return None

def create_app():
    """Flask 애플리케이션 팩토리"""
    
    app = Flask(__name__)
    
    # CORS 설정 (React 연동) - Chronos/Sentio와 동일한 단순 설정
    CORS(app)
    
    # 설정
    app.config['JSON_AS_ASCII'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    
    # 모델 캐시 (최적화된 모델 재사용)
    app.model_cache = {
        'trained_model': None,
        'model_metadata': None,
        'training_timestamp': None,
        'data_hash': None  # 데이터 변경 감지용
    }
    
    # ------------------------------------------------------
    # 애플리케이션 초기화
    # ------------------------------------------------------
    
    def initialize_services():
        """서비스 초기화"""
        try:
            logger.info("Structura HR 예측 서비스 초기화 중...")
            
            # uploads 디렉토리에서 Structura 파일 찾기 (post 파일 우선)
            data_path = None
            
            # post 분석용 파일 확인 (사후 분석과 배치 분석의 학습용)
            post_dir = "../uploads/structura/post"
            if os.path.exists(post_dir):
                files = [f for f in os.listdir(post_dir) if f.endswith('.csv')]
                if files:
                    files.sort(reverse=True)  # 최신 파일 우선
                    data_path = os.path.join(post_dir, files[0])
                    logger.info(f"Structura post 데이터 파일 발견: {data_path}")
            
            # post 파일이 없으면 기본 데이터 사용
            if data_path is None:
                # 기본 데이터 경로 (fallback)
                default_data_path = "../data/IBM_HR.csv"
                if os.path.exists(default_data_path):
                    data_path = default_data_path
                    logger.info(f"기본 데이터 파일 사용: {data_path}")
                else:
                    logger.warning("사용 가능한 데이터 파일이 없습니다.")
            
            # 예측기 초기화
            predictor = StructuraHRPredictor(data_path=data_path)
            
            # 기존 모델이 있으면 로딩, 없으면 즉시 훈련
            model_path = "hr_attrition_model.pkl"
            if os.path.exists(model_path):
                try:
                    predictor.load_model(model_path)
                    logger.info("기존 모델 로딩 완료")
                except Exception as e:
                    logger.warning(f"기존 모델 로딩 실패: {e}, 새로 훈련합니다")
                    # 모델 로딩 실패 시 새로 훈련
                    if data_path:
                        logger.info("🚀 서버 시작 시 모델 훈련 시작...")
                        training_result = predictor.run_full_pipeline(optimize_hp=True, use_sampling=True)
                        predictor.save_model(model_path)
                        logger.info(f"✅ 초기 모델 훈련 완료: {training_result}")
            else:
                # 모델이 없으면 즉시 훈련
                if data_path:
                    logger.info("🚀 서버 시작 시 모델 훈련 시작...")
                    training_result = predictor.run_full_pipeline(optimize_hp=True, use_sampling=True)
                    predictor.save_model(model_path)
                    logger.info(f"✅ 초기 모델 훈련 완료: {training_result}")
                else:
                    logger.warning("훈련할 데이터가 없어서 모델을 생성할 수 없습니다")
            
            # Flask 앱에 저장
            app.predictor = predictor
            
            logger.info("Structura HR 예측 서비스 준비 완료")
            
        except Exception as e:
            logger.error(f"서비스 초기화 실패: {str(e)}")
            raise
    
    # 앱 생성 시 즉시 초기화
    initialize_services()
    
    # ------------------------------------------------------
    # 유틸리티 함수
    # ------------------------------------------------------
    
    def get_predictor():
        """예측기 가져오기"""
        if not hasattr(app, 'predictor') or app.predictor is None:
            return None
        return app.predictor
    
    # ------------------------------------------------------
    # 에러 핸들러
    # ------------------------------------------------------
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Not Found",
            "message": "요청한 리소스를 찾을 수 없습니다",
            "status_code": 404
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "error": "Internal Server Error",
            "message": "서버 내부 오류가 발생했습니다",
            "status_code": 500
        }), 500
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        if isinstance(e, HTTPException):
            return jsonify({
                "error": e.name,
                "message": e.description,
                "status_code": e.code
            }), e.code
        
        logger.error(f"예상치 못한 오류: {str(e)}")
        return jsonify({
            "error": "Unexpected Error",
            "message": str(e),
            "status_code": 500
        }), 500
    
    # ------------------------------------------------------
    # API 라우트
    # ------------------------------------------------------
    
    @app.route('/')
    def index():
        """루트 엔드포인트"""
        return jsonify({
            "service": "Structura HR 이직 예측 Flask API",
            "version": "1.0.0",
            "status": "running",
            "description": "XGBoost + xAI 기반 설명 가능한 HR 이직 예측 서비스",
            "features": {
                "machine_learning": "XGBoost",
                "explainable_ai": ["SHAP", "LIME", "Feature Importance"],
                "react_integration": True
            },
            "endpoints": {
                "health": "/api/health",
                "train_model": "/api/train",
                "predict": "/api/predict",
                "explain": "/api/explain",
                "feature_importance": "/api/feature-importance",
                "model_info": "/api/model/info",
                "predict_batch": "/api/predict/batch",
                "employee_analysis": "/api/employee/analysis"
            }
        })
    
    @app.route('/api/health')
    def health_check():
        """헬스체크 엔드포인트"""
        
        predictor = get_predictor()
        
        model_status = "loaded" if (predictor and predictor.model) else "not_loaded"
        xai_status = {
            "shap_available": SHAP_AVAILABLE and (predictor and predictor.shap_explainer is not None),
            "lime_available": LIME_AVAILABLE and (predictor and predictor.lime_explainer is not None)
        }
        
        return jsonify({
            "status": "healthy",
            "model_status": model_status,
            "xai_status": xai_status,
            "dependencies": {
                "shap": SHAP_AVAILABLE,
                "lime": LIME_AVAILABLE,
                "optuna": OPTUNA_AVAILABLE
            },
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route('/api/upload/data', methods=['POST'])
    def upload_data():
        """HR 데이터 CSV 파일 업로드"""
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
            upload_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'Structura')
            os.makedirs(upload_dir, exist_ok=True)
            
            # 타임스탬프를 포함한 파일명으로 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(filename)[0]
            new_filename = f"{base_name}_{timestamp}.csv"
            file_path = os.path.join(upload_dir, new_filename)
            
            # 최신 파일 링크도 생성 (기존 코드 호환성을 위해)
            latest_link = os.path.join(upload_dir, 'latest_hr_data.csv')
            file.save(file_path)
            
            # 최신 파일 링크 생성 (심볼릭 링크 또는 복사)
            try:
                if os.path.exists(latest_link):
                    os.remove(latest_link)
                # Windows에서는 심볼릭 링크 대신 복사 사용
                import shutil
                shutil.copy2(file_path, latest_link)
            except Exception as e:
                logger.warning(f"최신 파일 링크 생성 실패: {e}")
            
            # 데이터 검증
            try:
                df = pd.read_csv(file_path)
                
                # 필수 컬럼 확인
                required_columns = ['Age', 'JobSatisfaction', 'OverTime', 'MonthlyIncome', 'WorkLifeBalance']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    return jsonify({
                        "success": False,
                        "error": f"필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}",
                        "required_columns": required_columns,
                        "found_columns": list(df.columns)
                    }), 400
                
                # 기존 predictor 초기화 (새 데이터로 재훈련 필요)
                global predictor_instance
                predictor_instance = None
                
                return jsonify({
                    "success": True,
                    "message": "데이터가 성공적으로 업로드되었습니다.",
                    "file_info": {
                        "original_filename": filename,
                        "saved_filename": new_filename,
                        "upload_path": upload_dir,
                        "file_path": file_path,
                        "latest_link": latest_link,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "column_names": list(df.columns)
                    },
                    "note": "새로운 데이터로 모델을 재훈련해주세요."
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
        """모델 훈련 엔드포인트 (노트북 기반 최신 버전)"""
        
        predictor = get_predictor()
        if not predictor:
            return jsonify({"error": "예측기가 초기화되지 않았습니다"}), 503
        
        try:
            # 요청 데이터 파싱
            data = request.get_json() or {}
            optimize_hp = data.get('optimize_hyperparameters', True)
            use_sampling = data.get('use_sampling', True)
            
            logger.info(f"모델 훈련 시작 - 하이퍼파라미터 최적화: {optimize_hp}, 샘플링: {use_sampling}")
            
            # 전체 파이프라인 실행
            metrics = predictor.run_full_pipeline(
                optimize_hp=optimize_hp,
                use_sampling=use_sampling
            )
            
            # 모델 저장
            predictor.save_model("hr_attrition_model_xai.pkl")
            
            # 피처 중요도 가져오기
            importance_df = predictor.get_feature_importance(10)
            feature_importance = [
                {"feature": row['feature'], "importance": float(row['importance'])}
                for _, row in importance_df.iterrows()
            ]
            
            return jsonify({
                "status": "success",
                "message": "모델 훈련 완료 (XAI 포함)",
                "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
                "feature_importance": feature_importance,
                "xai_enabled": {
                    "shap": predictor.shap_explainer is not None,
                    "feature_importance": True
                },
                "model_config": {
                    "optimal_threshold": predictor.optimal_threshold,
                    "features_count": len(predictor.feature_columns) if predictor.feature_columns else 0,
                    "sampling_used": use_sampling
                },
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"모델 훈련 실패: {str(e)}")
            return jsonify({"error": f"모델 훈련 실패: {str(e)}"}), 500
    
    @app.route('/api/predict', methods=['POST'])
    def predict_attrition():
        """이직 예측 엔드포인트 (Probability 중심 + XAI)"""
        
        predictor = get_predictor()
        if not predictor or not predictor.model:
            return jsonify({"error": "모델이 로딩되지 않았습니다. 서버를 재시작하거나 /api/train 엔드포인트로 모델을 훈련해주세요."}), 503
        
        try:
            # 요청 데이터 파싱
            data = request.get_json()
            if not data:
                return jsonify({"error": "예측할 직원 데이터가 필요합니다"}), 400
            
            logger.info(f"🔮 Structura 예측 요청 받음: {type(data)} 타입")
            
            # employee_id만 있는 경우 batch 데이터에서 찾기
            if 'employee_id' in data and len(data) == 1:
                employee_id = data['employee_id']
                logger.info(f"👤 단일 직원 ID로 예측 요청: {employee_id}")
                
                # batch 데이터에서 해당 직원 찾기
                batch_dir = "../uploads/structura/batch"
                if os.path.exists(batch_dir):
                    files = [f for f in os.listdir(batch_dir) if f.endswith('.csv')]
                    if files:
                        files.sort(reverse=True)  # 최신 파일 우선
                        batch_file = os.path.join(batch_dir, files[0])
                        logger.info(f"📂 배치 파일에서 직원 검색: {batch_file}")
                        
                        import pandas as pd
                        df = pd.read_csv(batch_file)
                        
                        # employee_id 또는 EmployeeNumber로 찾기
                        employee_row = None
                        if 'employee_id' in df.columns:
                            employee_row = df[df['employee_id'] == int(employee_id)]
                        elif 'EmployeeNumber' in df.columns:
                            employee_row = df[df['EmployeeNumber'] == int(employee_id)]
                        
                        if employee_row is not None and len(employee_row) > 0:
                            # 첫 번째 매칭 행을 딕셔너리로 변환
                            employee_data = employee_row.iloc[0].to_dict()
                            employee_number = employee_data.get('EmployeeNumber', employee_id)
                            logger.info(f"✅ 직원 {employee_number} 데이터 발견, 예측 수행 중...")
                            result = predictor.predict_single_employee(employee_data, str(employee_number))
                            logger.info(f"🎯 직원 {employee_number} 예측 완료: 위험도 {result.get('attrition_probability', 0):.3f}")
                            return jsonify(result)
                        else:
                            logger.warning(f"❌ 직원 ID {employee_id}를 batch 데이터에서 찾을 수 없음")
                            return jsonify({"error": f"직원 ID {employee_id}를 batch 데이터에서 찾을 수 없습니다"}), 404
                    else:
                        logger.error("❌ batch 데이터 파일이 없음")
                        return jsonify({"error": "batch 데이터 파일이 없습니다"}), 404
                else:
                    logger.error("❌ batch 데이터 디렉토리가 없음")
                    return jsonify({"error": "batch 데이터 디렉토리가 없습니다"}), 404
            
            # 단일 직원 데이터인지 확인
            elif isinstance(data, list):
                # 배치 예측
                logger.info(f"📊 배치 예측 요청: {len(data)}명")
                results = []
                for i, employee_data in enumerate(data):
                    employee_number = employee_data.get('EmployeeNumber', str(i+1))
                    logger.info(f"🔮 배치 예측 {i+1}/{len(data)}: 직원 {employee_number}")
                    result = predictor.predict_single_employee(employee_data, employee_number)
                    logger.info(f"✅ 직원 {employee_number} 예측 완료: 위험도 {result.get('attrition_probability', 0):.3f}")
                    results.append(result)
                
                logger.info(f"🎯 배치 예측 전체 완료: {len(results)}명")
                return jsonify({
                    "predictions": results,
                    "batch_size": len(results),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                # 단일 예측
                employee_number = data.get('EmployeeNumber', 'SINGLE_001')
                logger.info(f"👤 단일 직원 예측 요청: {employee_number}")
                result = predictor.predict_single_employee(data, employee_number)
                logger.info(f"🎯 직원 {employee_number} 예측 완료: 위험도 {result.get('attrition_probability', 0):.3f}")
                return jsonify(result)
                
        except Exception as e:
            logger.error(f"예측 실패: {str(e)}")
            return jsonify({"error": f"예측 실패: {str(e)}"}), 500
    
    @app.route('/api/explain', methods=['POST'])
    def explain_prediction():
        """예측 설명 엔드포인트 (SHAP 기반 XAI)"""
        
        predictor = get_predictor()
        if not predictor or not predictor.model:
            return jsonify({"error": "모델이 로딩되지 않았습니다"}), 503
        
        try:
            # 요청 데이터 파싱
            data = request.get_json()
            if not data:
                return jsonify({"error": "설명할 직원 데이터가 필요합니다"}), 400
            
            # EmployeeNumber 추출
            employee_number = data.get('EmployeeNumber', 'EXPLAIN_001')
            
            # DataFrame으로 변환
            df = pd.DataFrame([data])
            
            # 예측 설명 생성
            explanation = predictor.explain_prediction(df, employee_number)
            
            return jsonify(explanation)
            
        except Exception as e:
            logger.error(f"예측 설명 실패: {str(e)}")
            return jsonify({"error": f"예측 설명 실패: {str(e)}"}), 500
    
    @app.route('/api/feature-importance')
    def get_feature_importance():
        """피처 중요도 조회 엔드포인트"""
        
        predictor = get_predictor()
        if not predictor or not predictor.model:
            return jsonify({"error": "모델이 로딩되지 않았습니다"}), 503
        
        try:
            top_n = request.args.get('top_n', 20, type=int)
            
            importance_df = predictor.get_feature_importance(top_n)
            
            # DataFrame을 딕셔너리로 변환
            importance_data = [
                {
                    "feature": row['feature'],
                    "importance": float(row['importance']),
                    "rank": i + 1
                }
                for i, (_, row) in enumerate(importance_df.iterrows())
            ]
            
            return jsonify({
                "feature_importance": importance_data,
                "total_features": len(predictor.feature_columns) if predictor.feature_columns else 0,
                "top_n": top_n,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"피처 중요도 조회 실패: {str(e)}")
            return jsonify({"error": f"피처 중요도 조회 실패: {str(e)}"}), 500
    
    @app.route('/api/model/info')
    def get_model_info():
        """모델 정보 조회 엔드포인트"""
        
        predictor = get_predictor()
        if not predictor:
            return jsonify({"error": "예측기가 초기화되지 않았습니다"}), 503
        
        model_info = {
            "model_loaded": predictor.model is not None,
            "feature_count": len(predictor.feature_columns) if predictor.feature_columns else 0,
            "feature_columns": predictor.feature_columns,
            "optimal_threshold": predictor.optimal_threshold,
            "scale_pos_weight": predictor.scale_pos_weight,
            "xai_capabilities": {
                "shap_available": predictor.shap_explainer is not None,
                "lime_available": predictor.lime_explainer is not None,
                "feature_importance": True
            },
            "preprocessing_config": {
                "dropped_columns": predictor.DROP_COLS,
                "ordinal_columns": list(predictor.ORDINAL_MAPS.keys()),
                "nominal_columns": predictor.NOMINAL_COLS
            }
        }
        
        return jsonify(model_info)
    
    @app.route('/api/predict/batch', methods=['POST', 'OPTIONS'])
    def predict_batch():
        """배치 예측 엔드포인트 (여러 직원 동시 처리)"""
        
        # OPTIONS 요청 처리
        if request.method == 'OPTIONS':
            response = jsonify({})
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add('Access-Control-Allow-Headers', "*")
            response.headers.add('Access-Control-Allow-Methods', "*")
            return response
        
        logger.info("🚀 Structura 배치 예측 API 호출됨")
        
        predictor = get_predictor()
        if not predictor:
            logger.error("❌ 예측기가 초기화되지 않았습니다")
            return jsonify({"error": "예측기가 초기화되지 않았습니다"}), 503
        
        try:
            # 요청 데이터 파싱
            request_data = request.get_json()
            if not request_data:
                logger.error("❌ 요청 데이터가 없습니다")
                return jsonify({"error": "요청 데이터가 없습니다"}), 400
            
            # 분석 타입 확인 (기본값: batch)
            analysis_type = request_data.get('analysis_type', 'batch')
            logger.info(f"📊 분석 타입: {analysis_type}")
            
            # 사후 분석(post)인 경우: post 데이터만 사용
            if analysis_type == 'post':
                logger.info("🔍 사후 분석 모드: post 데이터 확인 중...")
                post_data_path = get_data_path_by_analysis_type('post')
                if not post_data_path:
                    logger.error("❌ Post 분석 데이터가 없습니다")
                    return jsonify({"error": "Post 분석 데이터가 없습니다. 먼저 사후 분석용 데이터를 업로드해주세요."}), 400
                
                if post_data_path != predictor.data_path:
                    logger.info(f"🔄 사후 분석용 데이터로 예측기 업데이트: {post_data_path}")
                    predictor.data_path = post_data_path
                    try:
                        predictor.load_data()
                        logger.info(f"✅ Post 데이터 재로드 완료")
                    except Exception as e:
                        logger.error(f"❌ Post 데이터 재로드 실패: {e}")
                        return jsonify({"error": f"Post 데이터 로드 실패: {str(e)}"}), 500
                else:
                    logger.info(f"✅ 기존 Post 데이터 경로 사용: {predictor.data_path}")
            
            # 배치 분석(batch)인 경우: 별도 함수에서 처리 (post로 학습 → batch로 예측)
            elif analysis_type == 'batch':
                logger.info("🔀 배치 분석은 별도 엔드포인트에서 처리됩니다.")
                return jsonify({"error": "배치 분석은 /api/batch-analysis 엔드포인트를 사용해주세요."}), 400
            
            # employees 키가 있는 경우와 직접 배열인 경우 모두 처리
            if isinstance(request_data, dict) and 'employees' in request_data:
                data = request_data['employees']
                logger.info(f"📋 employees 키에서 {len(data)}명 데이터 추출")
            elif isinstance(request_data, list):
                data = request_data
                logger.info(f"📋 직접 배열에서 {len(data)}명 데이터 추출")
            else:
                logger.error("❌ 배치 예측을 위한 직원 데이터 리스트가 필요합니다")
                return jsonify({"error": "배치 예측을 위한 직원 데이터 리스트가 필요합니다"}), 400
            
            if not isinstance(data, list):
                logger.error("❌ 직원 데이터는 배열 형태여야 합니다")
                return jsonify({"error": "직원 데이터는 배열 형태여야 합니다"}), 400
            
            # 모델이 없으면 받은 데이터로 즉시 훈련
            if not predictor.model:
                logger.info("모델이 없어서 받은 데이터로 즉시 훈련을 시작합니다")
                
                # 받은 데이터를 DataFrame으로 변환
                import pandas as pd
                df = pd.DataFrame(data)
                
                # Attrition 컬럼이 있는지 확인
                if 'Attrition' not in df.columns:
                    return jsonify({"error": "훈련을 위해서는 Attrition 컬럼이 필요합니다"}), 400
                
                # 임시로 데이터를 저장하고 훈련
                temp_data_path = "temp_training_data.csv"
                df.to_csv(temp_data_path, index=False)
                
                # 예측기의 데이터 경로를 임시 파일로 변경
                predictor.data_path = temp_data_path
                
                try:
                    # 모델 훈련 실행
                    metrics = predictor.run_full_pipeline(optimize_hp=True, use_sampling=True)
                    logger.info(f"즉시 훈련 완료: {metrics}")
                except Exception as train_error:
                    logger.error(f"즉시 훈련 실패: {str(train_error)}")
                    return jsonify({"error": f"모델 훈련 실패: {str(train_error)}"}), 500
                finally:
                    # 임시 파일 정리
                    import os
                    if os.path.exists(temp_data_path):
                        os.remove(temp_data_path)
            
            results = []
            for i, employee_data in enumerate(data):
                try:
                    employee_number = employee_data.get('EmployeeNumber', str(i+1))
                    result = predictor.predict_single_employee(employee_data, employee_number)
                    results.append(result)
                except Exception as e:
                    # 개별 예측 실패 시 오류 정보 포함
                    results.append({
                        'employee_number': employee_data.get('EmployeeNumber', str(i+1)),
                        'error': str(e),
                        'attrition_probability': None,
                        'risk_category': 'ERROR'
                    })
            
            # 통계 정보 계산
            successful_predictions = [r for r in results if 'error' not in r]
            if successful_predictions:
                probabilities = [r['attrition_probability'] for r in successful_predictions]
                risk_distribution = {}
                for r in successful_predictions:
                    risk_cat = r['risk_category']
                    risk_distribution[risk_cat] = risk_distribution.get(risk_cat, 0) + 1
                
                stats = {
                    'total_employees': len(data),
                    'successful_predictions': len(successful_predictions),
                    'failed_predictions': len(data) - len(successful_predictions),
                    'average_probability': sum(probabilities) / len(probabilities),
                    'risk_distribution': risk_distribution,
                    'high_risk_count': risk_distribution.get('HIGH', 0)
                }
            else:
                stats = {
                    'total_employees': len(data),
                    'successful_predictions': 0,
                    'failed_predictions': len(data),
                    'average_probability': None,
                    'risk_distribution': {},
                    'high_risk_count': 0
                }
            
            return jsonify({
                "predictions": results,
                "statistics": stats,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ 배치 예측 실패: {str(e)}")
            logger.error(f"❌ 오류 상세: {type(e).__name__}")
            import traceback
            logger.error(f"❌ 스택 트레이스: {traceback.format_exc()}")
            return jsonify({
                "error": f"배치 예측 실패: {str(e)}",
                "error_type": type(e).__name__,
                "details": "서버 로그를 확인해주세요."
            }), 500
    
    @app.route('/api/employee/analysis/<employee_number>', methods=['POST'])
    def employee_analysis(employee_number):
        """개별 직원 심층 분석 엔드포인트"""
        
        predictor = get_predictor()
        if not predictor or not predictor.model:
            return jsonify({"error": "모델이 로딩되지 않았습니다"}), 503
        
        try:
            # 요청 데이터 파싱
            data = request.get_json()
            if not data:
                return jsonify({"error": "직원 데이터가 필요합니다"}), 400
            
            # 예측 및 설명
            result = predictor.predict_single_employee(data, employee_number)
            
            # 추가 분석 정보
            df = pd.DataFrame([data])
            probability = predictor.predict(df, return_proba=True)[0]
            
            # 위험도 레벨별 임계값 정보
            risk_thresholds = {
                'LOW': 0.4,
                'MEDIUM': 0.7,
                'HIGH': 1.0
            }
            
            # 현재 위험도와 다음 단계까지의 거리
            current_risk = result['risk_category']
            if current_risk == 'LOW':
                next_threshold = risk_thresholds['LOW']
                distance_to_next = next_threshold - probability
            elif current_risk == 'MEDIUM':
                next_threshold = risk_thresholds['MEDIUM']
                distance_to_next = next_threshold - probability
            else:
                next_threshold = None
                distance_to_next = None
            
            # 심층 분석 결과
            analysis = {
                **result,
                'detailed_analysis': {
                    'probability_score': float(probability),
                    'risk_thresholds': risk_thresholds,
                    'current_risk_level': current_risk,
                    'distance_to_next_level': float(distance_to_next) if distance_to_next else None,
                    'percentile_rank': None  # 전체 직원 대비 순위 (추후 구현)
                },
                'recommendations': predictor._generate_recommendations(result),
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(analysis)
            
        except Exception as e:
            logger.error(f"직원 분석 실패: {str(e)}")
            return jsonify({"error": f"직원 분석 실패: {str(e)}"}), 500
    
    # 배치 분석 라우트 추가
    @app.route('/api/batch-analysis', methods=['POST'])
    def batch_analysis_route():
        return batch_analysis()
    
    return app

# ------------------------------------------------------
# 배치 분석 전용 엔드포인트
# ------------------------------------------------------

def batch_analysis():
    """배치 분석: Post 데이터로 학습 → Batch 데이터로 예측 (모델 캐싱 적용)"""
    from flask import current_app
    
    # 모델 캐시 (함수 내부에서 정의)
    if not hasattr(current_app, 'model_cache'):
        current_app.model_cache = {
            'trained_model': None,
            'model_metadata': None,
            'training_timestamp': None,
            'data_hash': None  # 데이터 변경 감지용
        }
    model_cache = current_app.model_cache
    
    try:
        logger.info("🚀 Structura 배치 분석 시작: Post 데이터 학습 → Batch 데이터 예측")
        
        # 1단계: Post 데이터 확인
        post_data_path = get_data_path_by_analysis_type('post')
        if not post_data_path:
            return jsonify({"error": "Post 분석 데이터가 없습니다. 먼저 사후 분석을 수행해주세요."}), 400
        
        # 데이터 해시 계산 (변경 감지용)
        import hashlib
        import os
        with open(post_data_path, 'rb') as f:
            data_hash = hashlib.md5(f.read()).hexdigest()
        
        # Flask app의 predictor 가져오기
        predictor = getattr(current_app, 'predictor', None)
        
        # 캐시된 모델 확인
        if (model_cache['trained_model'] is not None and 
            model_cache['data_hash'] == data_hash and
            predictor is not None and predictor.model is not None):
            logger.info("💾 캐시된 최적화 모델 사용 (재학습 생략)")
            training_result = model_cache['model_metadata']
        else:
            logger.info(f"📚 Post 데이터로 모델 학습: {post_data_path}")
            
            # 예측기 초기화 (Post 데이터로)
            predictor = StructuraHRPredictor(data_path=post_data_path)
            predictor.load_data()
            
            # 모델 학습 (Post 데이터로 한 번만 최적화 수행하여 최고 성능 모델 생성)
            training_result = predictor.run_full_pipeline(optimize_hp=True, use_sampling=True)
            
            # Flask app에 predictor 저장
            current_app.predictor = predictor
            
            # 모델 캐시에 저장
            from datetime import datetime
            model_cache['trained_model'] = predictor.model
            model_cache['model_metadata'] = training_result
            model_cache['training_timestamp'] = datetime.now().isoformat()
            model_cache['data_hash'] = data_hash
            
            logger.info(f"✅ 모델 학습 완료 및 캐시 저장: {training_result}")
        
        
        # 2단계: Batch 데이터로 예측
        batch_data_path = get_data_path_by_analysis_type('batch')
        if not batch_data_path:
            return jsonify({"error": "Batch 분석 데이터가 없습니다. 먼저 배치 데이터를 업로드해주세요."}), 400
        
        logger.info(f"🔮 Batch 데이터로 예측 수행: {batch_data_path}")
        
        # Batch 데이터 로드
        import pandas as pd
        batch_data = pd.read_csv(batch_data_path)
        logger.info(f"📊 Batch 데이터 로드 완료: {len(batch_data)}명")
        
        # 예측 수행
        predictions = []
        for idx, row in batch_data.iterrows():
            try:
                # 개별 직원 예측
                employee_data = row.to_dict()
                employee_number = employee_data.get('EmployeeNumber', str(idx+1))
                prediction = predictor.predict_single_employee(employee_data, str(employee_number))
                
                predictions.append({
                    'employee_id': employee_data.get('EmployeeNumber', idx),
                    'risk_score': prediction.get('attrition_probability', 0.5),
                    'predicted_attrition': prediction.get('predicted_attrition', 0),
                    'confidence': prediction.get('confidence', 0.7),
                    'feature_importance': prediction.get('feature_importance', {}),
                    'xai_explanation': prediction.get('xai_explanation', {}),
                    'shap_values': prediction.get('shap_values', {}),
                    'lime_explanation': prediction.get('lime_explanation', {}),
                    'model_interpretation': prediction.get('model_interpretation', {}),
                    'employee_data': employee_data
                })
                
            except Exception as e:
                logger.warning(f"직원 {idx} 예측 실패: {str(e)}")
                predictions.append({
                    'employee_id': employee_data.get('EmployeeNumber', idx),
                    'risk_score': 0.5,
                    'predicted_attrition': 0,
                    'confidence': 0.1,
                    'error': str(e),
                    'employee_data': employee_data
                })
        
        logger.info(f"✅ Structura 배치 분석 완료: {len(predictions)}명 예측")
        
        return jsonify({
            "success": True,
            "message": f"Structura 배치 분석 완료: Post 데이터 학습 → {len(predictions)}명 Batch 예측",
            "agent": "structura",
            "training_data_path": post_data_path,
            "prediction_data_path": batch_data_path,
            "total_predictions": len(predictions),
            "predictions": predictions,
            "model_info": {
                "training_samples": len(predictor.data) if predictor.data is not None else 0,
                "features_used": list(predictor.feature_columns) if hasattr(predictor, 'feature_columns') else [],
                "model_type": "XGBoost with Hyperparameter Optimization"
            }
        })
        
    except Exception as e:
        logger.error(f"Structura 배치 분석 실패: {str(e)}")
        return jsonify({"error": f"Structura 배치 분석 실패: {str(e)}"}), 500

# ------------------------------------------------------
# 서버 실행 함수
# ------------------------------------------------------

def run_server(host='0.0.0.0', port=5001, debug=False):
    """Flask 서버 실행 (XAI 포함 최신 버전)"""
    try:
        app = create_app()
        
        print("=" * 70)
        print("🚀 Structura HR 예측 Flask 백엔드 서버 시작 (XAI 포함)")
        print("=" * 70)
        print(f"📡 서버 주소: http://{host}:{port}")
        print(f"🔗 React 연동: http://localhost:3000에서 접근 가능")
        print(f"🤖 XAI 기능: SHAP 기반 설명 가능한 AI")
        print(f"🔄 디버그 모드: {'활성화' if debug else '비활성화'}")
        print()
        print("주요 엔드포인트:")
        print(f"  • 헬스체크: http://{host}:{port}/api/health")
        print(f"  • 데이터 업로드: http://{host}:{port}/api/upload/data")
        print(f"  • 모델 훈련: http://{host}:{port}/api/train")
        print(f"  • 이직 예측: http://{host}:{port}/api/predict")
        print(f"  • 배치 예측: http://{host}:{port}/api/predict/batch")
        print(f"  • 예측 설명: http://{host}:{port}/api/explain")
        print(f"  • 직원 분석: http://{host}:{port}/api/employee/analysis/<employee_number>")
        print(f"  • 피처 중요도: http://{host}:{port}/api/feature-importance")
        print(f"  • 모델 정보: http://{host}:{port}/api/model/info")
        print()
        print("XAI 기능:")
        print(f"  • SHAP: {'✅' if SHAP_AVAILABLE else '❌'}")
        print(f"  • Feature Importance: ✅")
        print(f"  • Optuna: {'✅' if OPTUNA_AVAILABLE else '❌'}")
        print()
        print("새로운 기능:")
        print("  • EmployeeNumber별 개별 XAI 설명")
        print("  • Probability 중심 예측 결과")
        print("  • 배치 처리 및 통계 분석")
        print("  • 위험도 기반 권장사항 제공")
        print()
        print("서버를 중지하려면 Ctrl+C를 누르세요.")
        print("=" * 70)
        
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        print(f"❌ 서버 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    run_server()
