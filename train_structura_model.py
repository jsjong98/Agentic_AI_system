#!/usr/bin/env python3
"""
Structura 모델 훈련 스크립트
"""

import sys
import os
from pathlib import Path

# 경로 설정
sys.path.append(str(Path(__file__).parent / "app" / "Structura"))

try:
    from structura_flask_backend import StructuraHRPredictor
    import logging
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def train_structura_model():
        """Structura 모델 훈련"""
        logger.info("🚀 Structura 모델 훈련 시작")
        
        try:
            # 1. 예측기 초기화
            predictor = StructuraHRPredictor(data_path="data/IBM_HR.csv")
            
            # 2. 전체 파이프라인 실행 (훈련 포함)
            logger.info("📊 데이터 로딩 및 전처리 중...")
            metrics = predictor.run_full_pipeline(optimize_hp=False, use_sampling=True)
            
            # 3. 모델 저장
            model_path = "app/Structura/hr_attrition_model.pkl"
            logger.info(f"💾 모델 저장 중: {model_path}")
            predictor.save_model(model_path)
            
            # 4. 결과 출력
            logger.info("✅ Structura 모델 훈련 완료!")
            logger.info(f"📈 성능 지표:")
            for metric, value in metrics.items():
                logger.info(f"  - {metric}: {value:.4f}")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Structura 모델 훈련 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    if __name__ == "__main__":
        success = train_structura_model()
        sys.exit(0 if success else 1)
        
except ImportError as e:
    print(f"❌ Import 오류: {e}")
    print("필요한 패키지를 설치해주세요:")
    print("pip install xgboost pandas scikit-learn shap lime")
    sys.exit(1)
