"""
임계값 계산 모듈
Threshold_setting.ipynb 기반으로 구현
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Bayesian Optimization 라이브러리 (선택적)
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


class ThresholdCalculator:
    """
    각 Score별 최적 임계값을 계산하는 클래스
    """
    
    def __init__(self):
        self.optimal_thresholds = {}
        self.performance_results = {}
        
    def find_optimal_threshold(self, y_true: np.ndarray, y_score: np.ndarray, 
                             thresholds: Optional[np.ndarray] = None, 
                             method: str = 'grid') -> Tuple[float, float, pd.DataFrame]:
        """
        F1-score가 최대가 되는 임계값을 찾는 함수
        
        Parameters:
        y_true: 실제 라벨 (0, 1)
        y_score: 예측 점수
        thresholds: 테스트할 임계값 리스트 (None이면 자동 생성)
        method: 최적화 방법 ('grid' 또는 'bayesian')
        
        Returns:
        best_threshold: 최적 임계값
        best_f1: 최대 F1-score
        results: 모든 임계값에 대한 결과
        """
        
        if method == 'bayesian' and SKOPT_AVAILABLE:
            return self._find_optimal_threshold_bayesian(y_true, y_score)
        else:
            return self._find_optimal_threshold_grid(y_true, y_score, thresholds)
    
    def _find_optimal_threshold_grid(self, y_true: np.ndarray, y_score: np.ndarray, 
                                   thresholds: Optional[np.ndarray] = None) -> Tuple[float, float, pd.DataFrame]:
        """Grid Search를 통한 임계값 최적화"""
        
        if thresholds is None:
            # 점수의 범위를 기반으로 임계값 생성
            min_score = np.min(y_score)
            max_score = np.max(y_score)
            thresholds = np.linspace(min_score, max_score, 100)
        
        results = []
        
        for threshold in thresholds:
            # 임계값을 기준으로 예측값 생성
            y_pred = (y_score >= threshold).astype(int)
            
            # 메트릭 계산
            if len(np.unique(y_pred)) > 1:  # 예측값이 모두 같은 클래스가 아닌 경우에만
                f1 = f1_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
            else:
                f1 = precision = recall = 0
            
            results.append({
                'threshold': threshold,
                'f1_score': f1,
                'precision': precision,
                'recall': recall
            })
        
        results_df = pd.DataFrame(results)
        
        # 최적 임계값 찾기
        best_idx = results_df['f1_score'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_f1 = results_df.loc[best_idx, 'f1_score']
        
        return best_threshold, best_f1, results_df
    
    def _find_optimal_threshold_bayesian(self, y_true: np.ndarray, y_score: np.ndarray,
                                       n_calls: int = 50) -> Tuple[float, float, pd.DataFrame]:
        """Bayesian Optimization을 통한 임계값 최적화"""
        
        min_score = np.min(y_score)
        max_score = np.max(y_score)
        
        # 임계값 범위 설정
        dimensions = [Real(min_score, max_score, name='threshold')]
        
        @use_named_args(dimensions)
        def objective(threshold):
            """목적 함수: F1-score의 음수를 반환"""
            y_pred = (y_score >= threshold).astype(int)
            
            if len(np.unique(y_pred)) > 1:
                f1 = f1_score(y_true, y_pred)
            else:
                f1 = 0
            
            return -f1  # 최소화 문제로 변환
        
        # Bayesian Optimization 실행
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=42,
            acq_func='EI',
            n_initial_points=10
        )
        
        best_threshold = result.x[0]
        best_f1 = -result.fun
        
        # 결과 DataFrame 생성 (시각화용)
        thresholds = np.linspace(min_score, max_score, 100)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            
            if len(np.unique(y_pred)) > 1:
                f1 = f1_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
            else:
                f1 = precision = recall = 0
            
            results.append({
                'threshold': threshold,
                'f1_score': f1,
                'precision': precision,
                'recall': recall
            })
        
        results_df = pd.DataFrame(results)
        
        return best_threshold, best_f1, results_df
    
    def calculate_thresholds_for_scores(self, data: pd.DataFrame, 
                                      score_columns: List[str], 
                                      method: str = 'bayesian') -> Dict:
        """
        여러 Score에 대해 최적 임계값을 계산
        
        Parameters:
        data: 데이터프레임 (attrition_binary 컬럼 포함)
        score_columns: Score 컬럼명 리스트
        method: 최적화 방법 ('grid' 또는 'bayesian')
        
        Returns:
        results: 각 Score별 최적 임계값 및 성능 지표
        """
        
        results = {}
        
        for score_col in score_columns:
            print(f"=== {score_col} 분석 중 ({method} 방법) ===")
            
            # 결측값 제거
            mask = ~(data[score_col].isna() | data['attrition_binary'].isna())
            y_true = data.loc[mask, 'attrition_binary']
            y_score = data.loc[mask, score_col]
            
            # 최적 임계값 찾기
            best_threshold, best_f1, results_df = self.find_optimal_threshold(y_true, y_score, method=method)
            
            # 최적 임계값에서의 상세 성능 확인
            y_pred_optimal = (y_score >= best_threshold).astype(int)
            
            precision = precision_score(y_true, y_pred_optimal)
            recall = recall_score(y_true, y_pred_optimal)
            accuracy = np.mean(y_true == y_pred_optimal)
            
            # 혼동 행렬
            cm = confusion_matrix(y_true, y_pred_optimal)
            
            results[score_col] = {
                'best_threshold': best_threshold,
                'best_f1': best_f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'confusion_matrix': cm.tolist(),
                'results_df': results_df
            }
            
            print(f"최적 임계값: {best_threshold:.6f}")
            print(f"최대 F1-score: {best_f1:.4f}")
            print(f"정밀도: {precision:.4f}")
            print(f"재현율: {recall:.4f}")
            print(f"정확도: {accuracy:.4f}")
            print()
        
        self.optimal_thresholds = {col: results[col]['best_threshold'] 
                                 for col in results.keys()}
        self.performance_results = results
        
        return results
    
    def predict_attrition(self, employee_scores: Dict[str, float]) -> Dict:
        """
        새로운 직원 데이터에 대해 attrition 예측
        
        Parameters:
        employee_scores: dict, {'Score명': 점수값} 형태
        
        Returns:
        predictions: dict, 각 Score별 예측 결과
        """
        predictions = {}
        
        for score_name, score_value in employee_scores.items():
            if score_name in self.optimal_thresholds:
                threshold = self.optimal_thresholds[score_name]
                prediction = "위험" if score_value >= threshold else "안전"
                predictions[f"{score_name}_prediction"] = prediction
                predictions[f"{score_name}_threshold"] = threshold
                predictions[f"{score_name}_score"] = score_value
        
        return predictions
    
    def apply_thresholds_to_data(self, data: pd.DataFrame, 
                               score_columns: List[str]) -> pd.DataFrame:
        """
        데이터에 임계값을 적용하여 예측 컬럼 생성
        
        Parameters:
        data: 원본 데이터프레임
        score_columns: Score 컬럼명 리스트
        
        Returns:
        data_with_predictions: 예측 컬럼이 추가된 데이터프레임
        """
        
        data_with_predictions = data.copy()
        
        for score_col in score_columns:
            if score_col in self.optimal_thresholds:
                threshold = self.optimal_thresholds[score_col]
                prediction_col = f"{score_col}_prediction"
                
                # 임계값 기준으로 0/1 라벨링
                data_with_predictions[prediction_col] = (
                    data_with_predictions[score_col] >= threshold
                ).astype(int)
        
        return data_with_predictions
    
    def get_summary_table(self) -> pd.DataFrame:
        """
        결과 요약 테이블 생성
        
        Returns:
        summary_df: 요약 결과 데이터프레임
        """
        
        summary_results = []
        
        for score_col, result in self.performance_results.items():
            summary_results.append({
                'Score': score_col,
                'Optimal_Threshold': result['best_threshold'],
                'F1_Score': result['best_f1'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'Accuracy': result['accuracy']
            })
        
        summary_df = pd.DataFrame(summary_results)
        summary_df = summary_df.round(4)
        
        return summary_df
    
    def get_thresholds_dict(self) -> Dict[str, float]:
        """
        실제 사용을 위한 임계값 딕셔너리 반환
        
        Returns:
        thresholds_dict: 임계값 딕셔너리
        """
        return self.optimal_thresholds.copy()


def load_and_process_data(file_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    데이터 로드 및 전처리
    
    Parameters:
    file_path: 데이터 파일 경로
    
    Returns:
    data: 전처리된 데이터프레임
    score_columns: Score 컬럼명 리스트
    """
    
    # 데이터 로드
    data = pd.read_csv(file_path)
    
    # attrition을 이진 변수로 변환 (Yes=1, No=0)
    if 'attrition_binary' not in data.columns:
        data['attrition_binary'] = data['attrition'].map({'Yes': 1, 'No': 0})
    
    # Score 컬럼들 정의
    score_columns = ['Structura_score', 'Cognita_score', 'Chronos_score', 
                    'Sentio_score', 'Agora_score']
    
    # 존재하는 컬럼만 필터링
    score_columns = [col for col in score_columns if col in data.columns]
    
    return data, score_columns


if __name__ == "__main__":
    # 테스트용 코드
    try:
        # 데이터 로드
        data, score_columns = load_and_process_data('../data/Total_score.csv')
        
        # 임계값 계산기 초기화
        calculator = ThresholdCalculator()
        
        # 임계값 계산
        results = calculator.calculate_thresholds_for_scores(data, score_columns)
        
        # 요약 테이블 출력
        summary_df = calculator.get_summary_table()
        print("=== 최적 임계값 요약 결과 ===")
        print(summary_df.to_string(index=False))
        
        # 임계값 딕셔너리 출력
        thresholds_dict = calculator.get_thresholds_dict()
        print("\n=== 실제 적용용 임계값 딕셔너리 ===")
        print(thresholds_dict)
        
        # 예시 직원 예측
        example_employee = {
            'Structura_score': 0.8,
            'Cognita_score': 0.6,
            'Chronos_score': 0.7,
            'Sentio_score': 0.4,
            'Agora_score': 0.3
        }
        
        prediction = calculator.predict_attrition(example_employee)
        print("\n=== 예시 직원 예측 결과 ===")
        for key, value in prediction.items():
            print(f"{key}: {value}")
            
    except FileNotFoundError:
        print("데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    except Exception as e:
        print(f"오류 발생: {e}")
