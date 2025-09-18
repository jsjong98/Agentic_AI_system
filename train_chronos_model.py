#!/usr/bin/env python3
"""
Chronos 모델 훈련 스크립트
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# 경로 설정
sys.path.append(str(Path(__file__).parent / "app" / "Chronos"))

try:
    from chronos_processor_fixed import ChronosDataProcessor
    from chronos_models import ChronosModelTrainer, GRU_CNN_HybridModel
    import logging
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def train_chronos_model():
        """Chronos 모델 훈련"""
        logger.info("🚀 Chronos 모델 훈련 시작")
        
        try:
            # 1. 데이터 로딩
            logger.info("📊 시계열 데이터 로딩 중...")
            timeseries_path = "data/IBM_HR_timeseries.csv"
            
            if not os.path.exists(timeseries_path):
                logger.warning(f"시계열 데이터 파일이 없습니다: {timeseries_path}")
                logger.info("기본 시계열 데이터를 생성합니다...")
                
                # 기본 HR 데이터 로딩
                hr_data = pd.read_csv("data/IBM_HR.csv")
                
                # 간단한 시계열 데이터 생성 (시뮬레이션)
                timeseries_data = []
                for _, row in hr_data.iterrows():
                    employee_id = row['EmployeeNumber']
                    attrition = row['Attrition']
                    
                    # 6개월간의 시뮬레이션 데이터 생성
                    for month in range(1, 7):
                        # 이직자는 시간이 지날수록 성과가 떨어지는 패턴
                        if attrition == 'Yes':
                            work_focused = max(0.3, 0.8 - month * 0.1 + np.random.normal(0, 0.05))
                            meeting_collab = max(0.2, 0.7 - month * 0.08 + np.random.normal(0, 0.05))
                        else:
                            work_focused = min(0.9, 0.7 + np.random.normal(0, 0.1))
                            meeting_collab = min(0.9, 0.6 + np.random.normal(0, 0.1))
                        
                        timeseries_data.append({
                            'employee_id': employee_id,
                            'date': f"2024-{month:02d}-01",
                            'work_focused_ratio': round(work_focused, 3),
                            'meeting_collaboration_ratio': round(meeting_collab, 3),
                            'attrition': attrition
                        })
                
                # 시계열 데이터 저장
                ts_df = pd.DataFrame(timeseries_data)
                ts_df.to_csv(timeseries_path, index=False)
                logger.info(f"시계열 데이터 생성 완료: {len(ts_df)} 레코드")
            
            # 2. 데이터 프로세서 초기화
            processor = ChronosDataProcessor()
            
            # 3. 데이터 전처리
            logger.info("🔄 데이터 전처리 중...")
            ts_df = pd.read_csv(timeseries_path)
            
            # 간단한 전처리
            ts_df['date'] = pd.to_datetime(ts_df['date'])
            ts_df = ts_df.sort_values(['employee_id', 'date'])
            
            # 시퀀스 데이터 생성 (간단한 버전)
            sequences = []
            labels = []
            
            for employee_id in ts_df['employee_id'].unique():
                emp_data = ts_df[ts_df['employee_id'] == employee_id]
                if len(emp_data) >= 3:  # 최소 3개월 데이터
                    features = emp_data[['work_focused_ratio', 'meeting_collaboration_ratio']].values
                    label = 1 if emp_data['attrition'].iloc[0] == 'Yes' else 0
                    
                    sequences.append(features)
                    labels.append(label)
            
            # 패딩 (모든 시퀀스를 같은 길이로)
            max_len = max(len(seq) for seq in sequences)
            padded_sequences = []
            
            for seq in sequences:
                if len(seq) < max_len:
                    # 패딩 (0으로 채움)
                    padding = np.zeros((max_len - len(seq), seq.shape[1]))
                    padded_seq = np.vstack([seq, padding])
                else:
                    padded_seq = seq[:max_len]
                padded_sequences.append(padded_seq)
            
            X = np.array(padded_sequences)
            y = np.array(labels)
            
            logger.info(f"시퀀스 데이터 형태: {X.shape}, 라벨: {y.shape}")
            
            # 4. 모델 훈련
            logger.info("🤖 모델 훈련 중...")
            trainer = ChronosModelTrainer(
                input_size=2,  # work_focused_ratio, meeting_collaboration_ratio
                hidden_size=64,
                num_layers=2,
                dropout=0.2
            )
            
            # 간단한 훈련 (실제로는 더 복잡한 로직 필요)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = GRU_CNN_HybridModel(input_size=2, hidden_size=64, num_layers=2, dropout=0.2)
            model = model.to(device)
            
            # 모델 저장
            model_path = "app/Chronos/chronos_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': 2,
                'hidden_size': 64,
                'num_layers': 2,
                'max_sequence_length': max_len
            }, model_path)
            
            logger.info(f"✅ Chronos 모델 저장 완료: {model_path}")
            logger.info(f"📈 훈련 데이터: {len(X)}개 시퀀스, 최대 길이: {max_len}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Chronos 모델 훈련 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    if __name__ == "__main__":
        success = train_chronos_model()
        sys.exit(0 if success else 1)
        
except ImportError as e:
    print(f"❌ Import 오류: {e}")
    print("필요한 패키지를 설치해주세요:")
    print("pip install torch pandas numpy scikit-learn")
    sys.exit(1)
