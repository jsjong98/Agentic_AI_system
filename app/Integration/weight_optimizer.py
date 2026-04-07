"""
가중치 최적화 모듈
Weight_setting.ipynb 기반으로 구현
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize
from itertools import product
from typing import Dict, Tuple, List, Optional, Any
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


class WeightOptimizer:
    """
    가중치 최적화를 수행하는 클래스
    """
    
    def __init__(self):
        self.optimal_weights = {}
        self.optimal_threshold = 0.5
        self.optimization_results = {}
        
    def calculate_weighted_score(self, data: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
        """
        가중치를 적용한 최종 점수 계산
        
        Parameters:
        data: DataFrame, 예측 데이터
        weights: dict, 각 컬럼별 가중치
        
        Returns:
        weighted_score: array, 0~1 사이의 가중 점수
        """
        weighted_score = np.zeros(len(data))
        
        for col, weight in weights.items():
            if col in data.columns:
                weighted_score += data[col] * weight
        
        return weighted_score
    
    def evaluate_weighted_score(self, y_true: np.ndarray, weighted_score: np.ndarray, 
                              threshold: float = 0.5) -> Dict[str, float]:
        """
        가중 점수의 성능을 평가하는 함수
        
        Parameters:
        y_true: 실제 라벨
        weighted_score: 가중 점수
        threshold: 분류 임계값
        
        Returns:
        metrics: dict, 성능 지표들
        """
        y_pred = (weighted_score >= threshold).astype(int)
        
        # 모든 예측이 같은 클래스인 경우 처리
        if len(np.unique(y_pred)) == 1:
            if y_pred[0] == 1:  # 모두 1로 예측
                precision = np.mean(y_true)
                recall = 1.0
            else:  # 모두 0으로 예측
                precision = 0.0
                recall = 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # ROC AUC는 확률 점수로 계산
        try:
            auc = roc_auc_score(y_true, weighted_score)
        except:
            auc = 0.0
        
        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'auc': auc,
            'threshold': threshold
        }
    
    def find_best_threshold_for_weighted_score(self, y_true: np.ndarray, 
                                             weighted_score: np.ndarray, 
                                             n_thresholds: int = 100) -> Tuple[float, Dict]:
        """
        가중 점수에 대한 최적 임계값 찾기
        """
        min_score = np.min(weighted_score)
        max_score = np.max(weighted_score)
        thresholds = np.linspace(min_score, max_score, n_thresholds)
        
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = None
        
        for threshold in thresholds:
            metrics = self.evaluate_weighted_score(y_true, weighted_score, threshold)
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_threshold = threshold
                best_metrics = metrics
        
        return best_threshold, best_metrics
    
    def grid_search_weights_normalized(self, data: pd.DataFrame, y_true: np.ndarray, 
                                     prediction_cols: List[str], 
                                     n_points_per_dim: int = 5) -> Tuple[Dict, float, float]:
        """
        Grid Search를 통한 가중치 최적화 (가중치 합 = 1 제약조건)
        
        Parameters:
        data: DataFrame
        y_true: 실제 라벨
        prediction_cols: 예측 컬럼 리스트
        n_points_per_dim: 각 차원당 테스트할 점 개수
        
        Returns:
        best_weights: dict, 최적 가중치 (합=1)
        best_f1: float, 최고 F1 점수
        best_threshold: float, 최적 임계값
        """
        n_cols = len(prediction_cols)
        
        print(f"🔍 Grid Search 시작 (가중치 합=1 제약조건)...")
        print(f"   예측 컬럼 수: {n_cols}")
        print(f"   각 차원당 점 수: {n_points_per_dim}")
        
        best_f1 = 0
        best_weights = None
        best_threshold = 0.5
        
        # n-1개 가중치만 변화시키고, 마지막 가중치는 1-sum(others)로 계산
        tested_combinations = 0
        total_combinations = n_points_per_dim ** (n_cols - 1)
        
        print(f"   총 조합 수: {total_combinations:,}")
        
        # 첫 n-1개 가중치의 가능한 값들 생성
        weight_values = np.linspace(0.1, 0.9, n_points_per_dim)
        
        # 모든 조합 생성 (n-1차원)
        for weight_combo in product(weight_values, repeat=n_cols-1):
            tested_combinations += 1
            
            # 첫 n-1개 가중치
            weights_partial = list(weight_combo)
            
            # 마지막 가중치 = 1 - sum(첫 n-1개)
            sum_partial = sum(weights_partial)
            
            # 마지막 가중치가 0~1 범위에 있는지 확인
            if sum_partial <= 1.0:
                last_weight = 1.0 - sum_partial
                weights_full = weights_partial + [last_weight]
                
                # 가중치 딕셔너리 생성
                weights = dict(zip(prediction_cols, weights_full))
                
                # 가중 점수 계산
                weighted_score = self.calculate_weighted_score(data, weights)
                
                # 최적 임계값 및 성능 평가
                threshold, metrics = self.find_best_threshold_for_weighted_score(y_true, weighted_score)
                
                # 최고 성능 업데이트
                if metrics['f1_score'] > best_f1:
                    best_f1 = metrics['f1_score']
                    best_weights = weights.copy()
                    best_threshold = threshold
            
            # 진행 상황 출력
            if tested_combinations % 500 == 0:
                print(f"   진행: {tested_combinations:,} / {total_combinations:,}")
        
        print(f"✅ Grid Search 완료!")
        print(f"   테스트된 조합: {tested_combinations:,}")
        
        return best_weights, best_f1, best_threshold
    
    def bayesian_optimize_weights_normalized(self, data: pd.DataFrame, y_true: np.ndarray,
                                           prediction_cols: List[str], 
                                           n_calls: int = 100) -> Tuple[Dict, float, Any]:
        """
        Bayesian Optimization을 통한 가중치 최적화 (가중치 합=1 제약조건)
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize가 설치되지 않았습니다. pip install scikit-optimize")
        
        n_cols = len(prediction_cols)
        
        # n-1개 가중치만 최적화하고, 마지막은 1-sum(others)로 계산
        dimensions = [Real(0.1, 0.9, name=f'weight_{i}') for i in range(n_cols-1)]
        
        @use_named_args(dimensions)
        def objective(**params):
            """목적 함수: F1-score의 음수를 반환"""
            
            # 첫 n-1개 가중치
            weights_partial = list(params.values())
            sum_partial = sum(weights_partial)
            
            # 가중치 합이 1을 초과하면 penalty
            if sum_partial > 1.0:
                return 1.0  # 최악의 점수
            
            # 마지막 가중치
            last_weight = 1.0 - sum_partial
            weights_full = weights_partial + [last_weight]
            
            # 가중치 딕셔너리 생성
            weights = dict(zip(prediction_cols, weights_full))
            
            # 가중 점수 계산
            weighted_score = self.calculate_weighted_score(data, weights)
            
            # 최적 임계값 및 성능 평가
            _, metrics = self.find_best_threshold_for_weighted_score(y_true, weighted_score)
            
            # F1-score의 음수 반환
            return -metrics['f1_score']
        
        print("🧠 Bayesian Optimization 시작 (가중치 합=1 제약조건)...")
        print(f"   반복 횟수: {n_calls}")
        print(f"   최적화 차원: {len(dimensions)} (마지막 가중치는 자동 계산)")
        
        # Bayesian Optimization 실행
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=42,
            acq_func='EI',
            n_initial_points=10
        )
        
        # 최적 가중치 추출
        weights_partial = result.x
        sum_partial = sum(weights_partial)
        last_weight = 1.0 - sum_partial
        weights_full = weights_partial + [last_weight]
        
        best_weights = dict(zip(prediction_cols, weights_full))
        best_f1 = -result.fun
        
        print("✅ Bayesian Optimization 완료!")
        
        return best_weights, best_f1, result
    
    def scipy_optimize_weights_normalized(self, data: pd.DataFrame, y_true: np.ndarray,
                                        prediction_cols: List[str]) -> Tuple[Dict, float]:
        """
        Scipy optimize를 사용한 가중치 최적화 (가중치 합=1 제약조건)
        """
        n_weights = len(prediction_cols)
        
        def objective_function(weights_array, data, y_true, prediction_cols):
            """
            최적화 목적 함수 (F1-score를 최대화하기 위해 음수 반환)
            """
            # 가중치 딕셔너리 생성
            weights = dict(zip(prediction_cols, weights_array))
            
            # 가중 점수 계산
            weighted_score = self.calculate_weighted_score(data, weights)
            
            # 최적 임계값 및 성능 평가
            _, metrics = self.find_best_threshold_for_weighted_score(y_true, weighted_score)
            
            # F1-score의 음수 반환 (최소화 문제로 변환)
            return -metrics['f1_score']
        
        # 제약 조건: 가중치 합이 정확히 1.0
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
        
        # 경계 조건: 각 가중치는 0.0 ~ 1.0
        bounds = [(0.0, 1.0) for _ in range(n_weights)]
        
        # 초기값: 균등 가중치 (합=1)
        x0 = np.array([1.0/n_weights] * n_weights)
        
        print("🚀 Scipy Optimize 최적화 시작 (가중치 합=1 제약조건)...")
        
        # 최적화 실행
        result = minimize(
            objective_function,
            x0,
            args=(data, y_true, prediction_cols),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False, 'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            # 가중치 정규화 (혹시 모를 수치 오차 보정)
            weights_raw = result.x
            weights_normalized = weights_raw / np.sum(weights_raw)
            
            optimal_weights = dict(zip(prediction_cols, weights_normalized))
            optimal_f1 = -result.fun
            
            print("✅ 최적화 성공!")
            
            return optimal_weights, optimal_f1
        else:
            print("❌ 최적화 실패!")
            print(result.message)
            return None, 0
    
    def optimize_weights(self, data: pd.DataFrame, method: str = 'bayesian',
                        **kwargs) -> Dict[str, Any]:
        """
        가중치 최적화 메인 함수
        
        Parameters:
        data: DataFrame (예측 컬럼들과 attrition_binary 포함)
        method: 최적화 방법 ('grid', 'bayesian', 'scipy')
        **kwargs: 각 방법별 추가 파라미터
        
        Returns:
        results: 최적화 결과
        """
        
        # 예측 컬럼들 찾기
        prediction_cols = [col for col in data.columns if col.endswith('_prediction')]
        y_true = data['attrition_binary']
        
        print(f"🎯 가중치 최적화 시작 - 방법: {method}")
        print(f"   예측 컬럼: {prediction_cols}")
        print(f"   데이터 크기: {len(data)}")
        
        results = {
            'method': method,
            'prediction_columns': prediction_cols,
            'data_size': len(data)
        }
        
        if method == 'grid':
            n_points = kwargs.get('n_points_per_dim', 5)
            best_weights, best_f1, best_threshold = self.grid_search_weights_normalized(
                data, y_true, prediction_cols, n_points
            )
            results.update({
                'best_weights': best_weights,
                'best_f1': best_f1,
                'best_threshold': best_threshold
            })
            
        elif method == 'bayesian':
            if not SKOPT_AVAILABLE:
                print("❌ scikit-optimize가 설치되지 않았습니다.")
                print("대신 scipy 방법을 사용합니다.")
                method = 'scipy'
            else:
                n_calls = kwargs.get('n_calls', 100)
                best_weights, best_f1, bayes_result = self.bayesian_optimize_weights_normalized(
                    data, y_true, prediction_cols, n_calls
                )
                
                # 최적 임계값 계산
                weighted_score = self.calculate_weighted_score(data, best_weights)
                best_threshold, _ = self.find_best_threshold_for_weighted_score(y_true, weighted_score)
                
                results.update({
                    'best_weights': best_weights,
                    'best_f1': best_f1,
                    'best_threshold': best_threshold,
                    'bayes_result': bayes_result
                })
        
        if method == 'scipy':
            best_weights, best_f1 = self.scipy_optimize_weights_normalized(
                data, y_true, prediction_cols
            )
            
            if best_weights:
                # 최적 임계값 계산
                weighted_score = self.calculate_weighted_score(data, best_weights)
                best_threshold, _ = self.find_best_threshold_for_weighted_score(y_true, weighted_score)
                
                results.update({
                    'best_weights': best_weights,
                    'best_f1': best_f1,
                    'best_threshold': best_threshold
                })
            else:
                results.update({
                    'best_weights': None,
                    'best_f1': 0,
                    'best_threshold': 0.5,
                    'error': 'Optimization failed'
                })
        
        # 결과 저장
        if results.get('best_weights'):
            self.optimal_weights = results['best_weights']
            self.optimal_threshold = results['best_threshold']
            self.optimization_results = results
        
        return results
    
    def apply_optimal_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        최적 가중치를 데이터에 적용
        
        Parameters:
        data: 원본 데이터프레임
        
        Returns:
        data_with_weighted: 가중 점수가 추가된 데이터프레임
        """
        
        if not self.optimal_weights:
            raise ValueError("최적 가중치가 계산되지 않았습니다. optimize_weights()를 먼저 실행하세요.")
        
        data_with_weighted = data.copy()
        
        # 가중 점수 계산
        data_with_weighted['final_weighted_score'] = self.calculate_weighted_score(
            data_with_weighted, self.optimal_weights
        )
        
        # 최종 예측 생성
        data_with_weighted['final_prediction'] = (
            data_with_weighted['final_weighted_score'] >= self.optimal_threshold
        ).astype(int)
        
        return data_with_weighted
    
    def classify_risk_level(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        위험도 구간 분류 추가
        
        Parameters:
        data: 가중 점수가 포함된 데이터프레임
        
        Returns:
        data_with_risk: 위험도 분류가 추가된 데이터프레임
        """
        
        def get_risk_level(score):
            if score < 0.3:
                return '안전군'
            elif score < 0.7:
                return '주의군'
            else:
                return '고위험군'
        
        def get_risk_level_numeric(score):
            if score < 0.3:
                return 1
            elif score < 0.7:
                return 2
            else:
                return 3
        
        data_with_risk = data.copy()
        
        if 'final_weighted_score' in data_with_risk.columns:
            data_with_risk['risk_level'] = data_with_risk['final_weighted_score'].apply(get_risk_level)
            data_with_risk['risk_level_numeric'] = data_with_risk['final_weighted_score'].apply(get_risk_level_numeric)
        
        return data_with_risk
    
    def get_performance_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        최종 성능 요약 반환
        
        Parameters:
        data: 최종 예측이 포함된 데이터프레임
        
        Returns:
        summary: 성능 요약 딕셔너리
        """
        
        if 'final_prediction' not in data.columns or 'attrition_binary' not in data.columns:
            raise ValueError("final_prediction 또는 attrition_binary 컬럼이 없습니다.")
        
        y_true = data['attrition_binary']
        y_pred = data['final_prediction']
        weighted_score = data.get('final_weighted_score', np.zeros(len(data)))
        
        # 성능 지표 계산
        metrics = self.evaluate_weighted_score(y_true, weighted_score, self.optimal_threshold)
        
        # 혼동 행렬
        cm = confusion_matrix(y_true, y_pred)
        
        # 위험도 구간별 통계
        risk_stats = {}
        if 'risk_level' in data.columns:
            risk_counts = data['risk_level'].value_counts()
            risk_attrition = data.groupby('risk_level')['attrition_binary'].agg(['count', 'sum', 'mean'])
            
            risk_stats = {
                'counts': risk_counts.to_dict(),
                'attrition_rates': risk_attrition['mean'].to_dict()
            }
        
        summary = {
            'optimal_weights': self.optimal_weights,
            'optimal_threshold': self.optimal_threshold,
            'performance_metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'risk_statistics': risk_stats,
            'optimization_method': self.optimization_results.get('method', 'unknown')
        }
        
        return summary


if __name__ == "__main__":
    # 테스트용 코드
    try:
        # 샘플 데이터 생성 (실제로는 threshold_calculator의 결과를 사용)
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'employee_id': range(1, n_samples + 1),
            'Structura_score_prediction': np.random.randint(0, 2, n_samples),
            'Cognita_score_prediction': np.random.randint(0, 2, n_samples),
            'Chronos_score_prediction': np.random.randint(0, 2, n_samples),
            'Sentio_score_prediction': np.random.randint(0, 2, n_samples),
            'Agora_score_prediction': np.random.randint(0, 2, n_samples),
            'attrition_binary': np.random.randint(0, 2, n_samples)
        })
        
        # 가중치 최적화기 초기화
        optimizer = WeightOptimizer()
        
        # Grid Search로 최적화
        print("=== Grid Search 최적화 ===")
        results = optimizer.optimize_weights(data, method='grid', n_points_per_dim=3)
        print(f"최적 가중치: {results['best_weights']}")
        print(f"최고 F1-Score: {results['best_f1']:.4f}")
        
        # 최적 가중치 적용
        data_with_weighted = optimizer.apply_optimal_weights(data)
        data_with_risk = optimizer.classify_risk_level(data_with_weighted)
        
        # 성능 요약
        summary = optimizer.get_performance_summary(data_with_risk)
        print(f"\n=== 최종 성능 요약 ===")
        print(f"최적 임계값: {summary['optimal_threshold']:.4f}")
        print(f"F1-Score: {summary['performance_metrics']['f1_score']:.4f}")
        print(f"정확도: {summary['performance_metrics']['accuracy']:.4f}")
        
        print(f"\n위험도 구간별 분포:")
        for level, count in summary['risk_statistics']['counts'].items():
            print(f"  {level}: {count}명")
            
    except Exception as e:
        print(f"오류 발생: {e}")
