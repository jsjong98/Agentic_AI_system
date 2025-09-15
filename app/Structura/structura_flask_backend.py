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
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# ML 관련 imports
from sklearn.model_selection import train_test_split, StratifiedKFold
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
    
    def __init__(self, data_path: str = "../data/IBM_HR.csv", random_state: int = 42):
        self.data_path = data_path
        self.random_state = random_state
        self.model = None
        self.feature_columns = None
        self.optimal_threshold = 0.5
        self.scale_pos_weight = 1.0
        
        # xAI 관련
        self.shap_explainer = None
        self.lime_explainer = None
        self.X_train_sample = None  # LIME용 배경 데이터
        
        # 전처리 설정
        self.setup_preprocessing_config()
        
    def setup_preprocessing_config(self):
        """전처리 설정 초기화"""
        # 제거할 컬럼들
        self.DROP_COLS = [
            "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours",
            "DailyRate", "HourlyRate", "MonthlyRate",
            "PercentSalaryHike", "YearsSinceLastPromotion", "NumCompaniesWorked",
            "TotalWorkingYears"
        ]
        
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
        logger.info(f"데이터 로딩 완료: {df.shape}")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """데이터 전처리"""
        logger.info("데이터 전처리 시작...")
        
        # 1. 불필요한 컬럼 제거
        drop_present = [c for c in self.DROP_COLS if c in df.columns]
        df = df.drop(columns=drop_present)
        
        # 2. 타겟 변수 처리
        TARGET = "Attrition"
        if TARGET not in df.columns:
            raise ValueError(f"Target column '{TARGET}' not found in data.")
        
        y = df[TARGET].map({"Yes": 1, "No": 0})
        if y.isna().any():
            y = pd.Series(pd.factorize(df[TARGET])[0], index=df.index)
        
        X = df.drop(columns=[TARGET])
        
        # 3. 컬럼 그룹 분류
        ordinal_cols = [c for c in self.ORDINAL_MAPS.keys() if c in X.columns]
        nominal_cols = [c for c in self.NOMINAL_COLS if c in X.columns]
        numeric_cols = [c for c in X.columns 
                       if (c not in ordinal_cols + nominal_cols) 
                       and pd.api.types.is_numeric_dtype(X[c])]
        
        # 4. 순서형 변수를 수치형으로 변환
        X = self._coerce_ordinal_to_numeric(X, ordinal_cols)
        
        # 5. 결측값 처리
        for c in numeric_cols + ordinal_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            X[c] = X[c].fillna(X[c].median())
        
        for c in nominal_cols:
            X[c] = X[c].astype("category")
            if "__UNK__" not in X[c].cat.categories:
                X[c] = X[c].cat.add_categories(["__UNK__"])
            X[c] = X[c].fillna("__UNK__")
        
        logger.info("데이터 전처리 완료")
        return X, y
    
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
        logger.info("모델 훈련 시작...")
        
        if hyperparams is None:
            hyperparams = self._get_default_params()
        
        # 클래스 불균형 가중치 계산
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        self.scale_pos_weight = neg / max(pos, 1)
        
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
            'verbosity': 0,
            **hyperparams
        }
        
        self.model = XGBClassifier(**params)
        self.model.fit(X_train, y_train)
        self.feature_columns = X_train.columns.tolist()
        
        # xAI 설정
        self._setup_explainers(X_train, y_train)
        
        logger.info("모델 훈련 및 xAI 설정 완료")
        return self.model
    
    def _setup_explainers(self, X_train: pd.DataFrame, y_train: pd.Series):
        """SHAP 및 LIME 설명기 설정"""
        
        # SHAP 설정
        if SHAP_AVAILABLE:
            try:
                logger.info("SHAP 설명기 설정 중...")
                self.shap_explainer = shap.TreeExplainer(self.model)
                logger.info("SHAP 설정 완료")
            except Exception as e:
                logger.warning(f"SHAP 설정 실패: {str(e)}")
                self.shap_explainer = None
        
        # LIME 설정
        if LIME_AVAILABLE:
            try:
                logger.info("LIME 설명기 설정 중...")
                # 샘플링된 훈련 데이터 (LIME 배경용)
                sample_size = min(1000, len(X_train))
                sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
                self.X_train_sample = X_train.iloc[sample_indices].values
                
                # 범주형 피처 인덱스 찾기
                categorical_features = []
                for i, col in enumerate(X_train.columns):
                    if col in self.NOMINAL_COLS or pd.api.types.is_categorical_dtype(X_train[col]):
                        categorical_features.append(i)
                
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    self.X_train_sample,
                    feature_names=self.feature_columns,
                    categorical_features=categorical_features,
                    mode='classification',
                    random_state=self.random_state
                )
                logger.info("LIME 설정 완료")
            except Exception as e:
                logger.warning(f"LIME 설정 실패: {str(e)}")
                self.lime_explainer = None
    
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
        
        # 전처리 (타겟 제외)
        if 'Attrition' in df.columns:
            df = df.drop(columns=['Attrition'])
        
        # 예측
        probability = self.predict(df, return_proba=True)[0]
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
        """예측 설명 (xAI)"""
        # 딕셔너리를 DataFrame으로 변환
        df = pd.DataFrame([employee_data])
        
        if 'Attrition' in df.columns:
            df = df.drop(columns=['Attrition'])
        
        # 피처 중요도 (모델 기본)
        feature_importance = {}
        if self.model and self.feature_columns:
            for feature, importance in zip(self.feature_columns, self.model.feature_importances_):
                feature_importance[feature] = float(importance)
        
        # SHAP 값
        shap_values = None
        if self.shap_explainer:
            try:
                shap_vals = self.shap_explainer.shap_values(df[self.feature_columns])
                shap_values = {}
                for feature, shap_val in zip(self.feature_columns, shap_vals[0]):
                    shap_values[feature] = float(shap_val)
            except Exception as e:
                logger.warning(f"SHAP 분석 실패: {str(e)}")
        
        # LIME 설명
        lime_explanation = None
        if self.lime_explainer:
            try:
                def predict_fn(x):
                    return self.model.predict_proba(pd.DataFrame(x, columns=self.feature_columns))
                
                explanation = self.lime_explainer.explain_instance(
                    df[self.feature_columns].values[0],
                    predict_fn,
                    num_features=10
                )
                
                lime_explanation = {
                    'features': [item[0] for item in explanation.as_list()],
                    'values': [item[1] for item in explanation.as_list()],
                    'intercept': float(explanation.intercept[1])
                }
            except Exception as e:
                logger.warning(f"LIME 분석 실패: {str(e)}")
        
        # 상위 위험/보호 요인 추출
        importance_items = list(feature_importance.items()) if feature_importance else []
        importance_items.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # SHAP 값이 있으면 우선 사용
        if shap_values:
            shap_items = list(shap_values.items())
            shap_items.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_risk_factors = [
                {"feature": feat, "impact": val, "type": "risk"}
                for feat, val in shap_items[:5] if val > 0
            ]
            top_protective_factors = [
                {"feature": feat, "impact": abs(val), "type": "protective"}
                for feat, val in shap_items[:5] if val < 0
            ]
        else:
            # 피처 중요도 기반
            top_risk_factors = [
                {"feature": feat, "impact": val, "type": "risk"}
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
                'DROP_COLS': self.DROP_COLS,
                'ORDINAL_MAPS': self.ORDINAL_MAPS,
                'NOMINAL_COLS': self.NOMINAL_COLS,
                'ORDINAL_LABEL_ORDERS': self.ORDINAL_LABEL_ORDERS
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
        self.DROP_COLS = config['DROP_COLS']
        self.ORDINAL_MAPS = config['ORDINAL_MAPS']
        self.NOMINAL_COLS = config['NOMINAL_COLS']
        self.ORDINAL_LABEL_ORDERS = config['ORDINAL_LABEL_ORDERS']
        
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
                    if col in self.NOMINAL_COLS:
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

# ------------------------------------------------------
# Flask 애플리케이션 생성 및 설정
# ------------------------------------------------------

def create_app():
    """Flask 애플리케이션 팩토리"""
    
    app = Flask(__name__)
    
    # CORS 설정 (React 연동)
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # 설정
    app.config['JSON_AS_ASCII'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    
    # 전역 변수
    predictor = None
    
    # ------------------------------------------------------
    # 애플리케이션 초기화
    # ------------------------------------------------------
    
    @app.before_first_request
    def initialize_services():
        """첫 요청 전 서비스 초기화"""
        nonlocal predictor
        
        try:
            logger.info("Structura HR 예측 서비스 초기화 중...")
            
            # 예측기 초기화
            predictor = StructuraHRPredictor()
            
            # 기존 모델이 있으면 로딩
            model_path = "hr_attrition_model.pkl"
            if os.path.exists(model_path):
                predictor.load_model(model_path)
                logger.info("기존 모델 로딩 완료")
            else:
                logger.info("새 모델 훈련이 필요합니다")
            
            # Flask 앱에 저장
            app.predictor = predictor
            
            logger.info("Structura HR 예측 서비스 준비 완료")
            
        except Exception as e:
            logger.error(f"서비스 초기화 실패: {str(e)}")
            raise
    
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
            optimize_hp = data.get('optimize_hyperparameters', False)
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
            return jsonify({"error": "모델이 로딩되지 않았습니다"}), 503
        
        try:
            # 요청 데이터 파싱
            data = request.get_json()
            if not data:
                return jsonify({"error": "예측할 직원 데이터가 필요합니다"}), 400
            
            # 단일 직원 데이터인지 확인
            if isinstance(data, list):
                # 배치 예측
                results = []
                for i, employee_data in enumerate(data):
                    employee_number = employee_data.get('EmployeeNumber', f'BATCH_{i+1:03d}')
                    result = predictor.predict_single_employee(employee_data, employee_number)
                    results.append(result)
                
                return jsonify({
                    "predictions": results,
                    "batch_size": len(results),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                # 단일 예측
                employee_number = data.get('EmployeeNumber', 'SINGLE_001')
                result = predictor.predict_single_employee(data, employee_number)
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
    
    @app.route('/api/predict/batch', methods=['POST'])
    def predict_batch():
        """배치 예측 엔드포인트 (여러 직원 동시 처리)"""
        
        predictor = get_predictor()
        if not predictor or not predictor.model:
            return jsonify({"error": "모델이 로딩되지 않았습니다"}), 503
        
        try:
            # 요청 데이터 파싱
            data = request.get_json()
            if not data or not isinstance(data, list):
                return jsonify({"error": "배치 예측을 위한 직원 데이터 리스트가 필요합니다"}), 400
            
            results = []
            for i, employee_data in enumerate(data):
                try:
                    employee_number = employee_data.get('EmployeeNumber', f'BATCH_{i+1:03d}')
                    result = predictor.predict_single_employee(employee_data, employee_number)
                    results.append(result)
                except Exception as e:
                    # 개별 예측 실패 시 오류 정보 포함
                    results.append({
                        'employee_number': employee_data.get('EmployeeNumber', f'BATCH_{i+1:03d}'),
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
            logger.error(f"배치 예측 실패: {str(e)}")
            return jsonify({"error": f"배치 예측 실패: {str(e)}"}), 500
    
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
                'recommendations': self._generate_recommendations(result),
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(analysis)
            
        except Exception as e:
            logger.error(f"직원 분석 실패: {str(e)}")
            return jsonify({"error": f"직원 분석 실패: {str(e)}"}), 500
    
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
    
    return app

# ------------------------------------------------------
# 서버 실행 함수
# ------------------------------------------------------

def run_server(host='0.0.0.0', port=5001, debug=True):
    """Flask 서버 실행 (XAI 포함 최신 버전)"""
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

if __name__ == '__main__':
    run_server()
