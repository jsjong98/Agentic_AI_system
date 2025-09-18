#!/usr/bin/env python3
"""
통합된 Structura 클래스 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 경로 설정
sys.path.append(str(Path(__file__).parent / "app" / "Structura"))

def test_structura_integration():
    """통합된 Structura 클래스 테스트"""
    
    print("=" * 60)
    print("🧪 Structura 통합 테스트 시작")
    print("=" * 60)
    
    try:
        # 1. 모듈 import 테스트
        print("1. 모듈 import 테스트...")
        from app.Structura.structura_flask_backend import StructuraHRPredictor
        print("   ✅ 모듈 import 성공!")
        
        # 2. 클래스 초기화 테스트
        print("\n2. 클래스 초기화 테스트...")
        predictor = StructuraHRPredictor()
        print("   ✅ 클래스 초기화 성공!")
        print(f"   📁 데이터 경로: {predictor.data_path}")
        print(f"   🎯 최적 임계값: {predictor.optimal_threshold}")
        print(f"   📊 순서형 변수: {len(predictor.ordinal_cols)}개")
        print(f"   🏷️ 명목형 변수: {len(predictor.nominal_cols)}개")
        print(f"   🔢 수치형 변수: {len(predictor.numerical_cols)}개")
        
        # 3. 데이터 로딩 테스트
        print("\n3. 데이터 로딩 테스트...")
        if os.path.exists(predictor.data_path):
            df = predictor.load_data()
            print(f"   ✅ 데이터 로딩 성공! 형태: {df.shape}")
            print(f"   📋 컬럼: {list(df.columns)[:5]}... (총 {len(df.columns)}개)")
            
            # 4. 전처리 테스트
            print("\n4. 전처리 테스트...")
            X, y = predictor.preprocess_data(df)
            print(f"   ✅ 전처리 성공!")
            print(f"   📊 특성 데이터: {X.shape}")
            print(f"   🎯 타겟 데이터: {y.shape}")
            print(f"   📈 이직률: {y.mean():.1%}")
            
            # 5. 간단한 모델 훈련 테스트 (소량 데이터)
            print("\n5. 간단한 모델 훈련 테스트...")
            sample_size = min(100, len(X))
            X_sample = X.head(sample_size)
            y_sample = y.head(sample_size)
            
            # 기본 파라미터로 빠른 훈련
            quick_params = {
                'n_estimators': 10,  # 빠른 테스트용
                'max_depth': 3,
                'learning_rate': 0.1
            }
            
            model = predictor.train_model(X_sample, y_sample, quick_params)
            print(f"   ✅ 모델 훈련 성공!")
            print(f"   🤖 모델 타입: {type(model).__name__}")
            
            # 6. 예측 테스트
            print("\n6. 예측 테스트...")
            test_sample = X.head(5)
            predictions = predictor.predict(test_sample, return_proba=True)
            print(f"   ✅ 예측 성공!")
            print(f"   🎯 예측 확률: {predictions[:3].round(3)}")
            
        else:
            print(f"   ❌ 데이터 파일 없음: {predictor.data_path}")
            return False
            
        print("\n" + "=" * 60)
        print("🎉 모든 테스트 통과!")
        print("✅ Structura 통합이 성공적으로 완료되었습니다.")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_structura_integration()
    sys.exit(0 if success else 1)
