"""
HR Attrition Prediction Backend
XGBoost 기반 자동화된 백엔드 시스템

주요 기능:
- 데이터 전처리 자동화
- XGBoost 모델 훈련 및 하이퍼파라미터 최적화
- 모델 저장/로딩
- 예측 서비스
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix, precision_recall_curve
)
import xgboost as xgb
from xgboost import XGBClassifier

# Optional: Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Using default hyperparameters.")

warnings.filterwarnings('ignore')


class HRAttritionPredictor:
    """HR 이탈 예측 시스템"""
    
    def __init__(self, data_path: str = "data/IBM_HR.csv", random_state: int = 42):
        self.data_path = data_path
        self.random_state = random_state
        self.model = None
        self.feature_columns = None
        self.optimal_threshold = 0.5
        self.scale_pos_weight = 1.0
        
        # 전처리 설정
        self.setup_preprocessing_config()
        
    def setup_preprocessing_config(self):
        """전처리 설정 초기화"""
        # 제거할 컬럼들
        self.DROP_COLS = [
            "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours",
            "DailyRate", "HourlyRate", "MonthlyRate",  # 강력 제거 (노이즈)
            "PercentSalaryHike", "YearsSinceLastPromotion", "NumCompaniesWorked",  # 유력 제거
            "TotalWorkingYears"  # 공선성 (YearsAtCompany와 중복)
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
        """데이터 전처리"""
        print("데이터 전처리 시작...")
        
        # 1. 불필요한 컬럼 제거
        drop_present = [c for c in self.DROP_COLS if c in df.columns]
        df = df.drop(columns=drop_present)
        print(f"제거된 컬럼: {drop_present}")
        
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
        
        print(f"수치형 변수 ({len(numeric_cols)}): {numeric_cols}")
        print(f"순서형 변수 ({len(ordinal_cols)}): {ordinal_cols}")
        print(f"명목형 변수 ({len(nominal_cols)}): {nominal_cols}")
        
        # 4. 순서형 변수를 수치형으로 변환
        X = self._coerce_ordinal_to_numeric(X, ordinal_cols)
        
        # 5. 결측값 처리
        # 수치형 & 순서형 -> 중앙값
        for c in numeric_cols + ordinal_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            X[c] = X[c].fillna(X[c].median())
        
        # 명목형 -> 카테고리 타입 + '__UNK__'
        for c in nominal_cols:
            X[c] = X[c].astype("category")
            if "__UNK__" not in X[c].cat.categories:
                X[c] = X[c].cat.add_categories(["__UNK__"])
            X[c] = X[c].fillna("__UNK__")
        
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
                   X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                   hyperparams: Optional[Dict] = None) -> XGBClassifier:
        """모델 훈련"""
        print("모델 훈련 시작...")
        
        if hyperparams is None:
            hyperparams = self._get_default_params()
        
        # 모델 파라미터 설정
        params = {
            'n_estimators': 3000,
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
        
        # 검증 데이터가 있으면 조기 종료 사용
        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          early_stopping_rounds=100,
                          verbose=False)
        else:
            self.model.fit(X_train, y_train)
        
        self.feature_columns = X_train.columns.tolist()
        print("모델 훈련 완료")
        
        return self.model
    
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
    
    def predict(self, X: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """예측"""
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
        """모델 저장"""
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
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"모델이 저장되었습니다: {filepath}")
    
    def load_model(self, filepath: str):
        """모델 로딩"""
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
        
        print(f"모델이 로딩되었습니다: {filepath}")
    
    def run_full_pipeline(self, optimize_hp: bool = True, n_trials: int = 50) -> Dict:
        """전체 파이프라인 실행"""
        print("=== HR 이탈 예측 모델 파이프라인 시작 ===\n")
        
        # 1. 데이터 로딩
        df = self.load_data()
        
        # 2. 전처리
        X, y = self.preprocess_data(df)
        
        # 3. 데이터 분할
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # 4. 하이퍼파라미터 최적화 (선택적)
        if optimize_hp:
            best_params = self.optimize_hyperparameters(X_train, y_train, n_trials)
        else:
            best_params = self._get_default_params()
        
        # 5. 모델 훈련
        self.train_model(X_train, y_train, hyperparams=best_params)
        
        # 6. 최적 임계값 찾기
        self.optimize_threshold(X_train, y_train)
        
        # 7. 모델 평가
        metrics = self.evaluate_model(X_test, y_test)
        
        # 8. 피처 중요도
        print("\n=== 상위 피처 중요도 ===")
        importance_df = self.get_feature_importance()
        print(importance_df.head(10))
        
        print("\n=== 파이프라인 완료 ===")
        
        return metrics


def main():
    """메인 실행 함수"""
    # 예측기 초기화
    predictor = HRAttritionPredictor(data_path="data/IBM_HR.csv")
    
    # 전체 파이프라인 실행
    metrics = predictor.run_full_pipeline(
        optimize_hp=True,  # 하이퍼파라미터 최적화 사용
        n_trials=30        # 최적화 시행 횟수
    )
    
    # 모델 저장
    predictor.save_model("hr_attrition_model.pkl")
    
    # 사용 예시: 새로운 데이터로 예측
    print("\n=== 예측 예시 ===")
    
    # 예시 데이터 (실제 사용 시에는 새로운 데이터를 로드)
    df = predictor.load_data()
    X, y = predictor.preprocess_data(df)
    
    # 처음 5개 샘플로 예측
    sample_data = X.head(5)
    predictions = predictor.predict(sample_data)
    probabilities = predictor.predict(sample_data, return_proba=True)
    
    print("예측 결과:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"샘플 {i+1}: 이탈 예측 = {pred}, 확률 = {prob:.3f}")


if __name__ == "__main__":
    main()
