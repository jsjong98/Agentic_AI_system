import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import os

def split_data_with_label_ratio(csv_file_path, test_size=0.4, random_state=42):
    """
    IBM_HR_personas_assigned.csv 파일을 라벨 비율을 유지하면서 Train/Test로 분할
    
    Parameters:
    - csv_file_path: CSV 파일 경로
    - test_size: 테스트 데이터 비율 (기본값: 0.4, 즉 6:4 비율)
    - random_state: 재현 가능한 결과를 위한 시드값
    
    Returns:
    - train_df, test_df: 분할된 데이터프레임
    """
    
    # 데이터 로드
    print(f"데이터 로딩 중: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    print(f"전체 데이터 크기: {df.shape}")
    
    # 라벨 컬럼들 확인 (argmax_Persona와 softmax_Persona 사용)
    label_columns = ['argmax_Persona', 'softmax_Persona']
    
    # 각 라벨의 분포 확인
    print("\n=== 라벨 분포 확인 ===")
    for col in label_columns:
        if col in df.columns:
            print(f"\n{col} 분포:")
            label_counts = df[col].value_counts()
            print(label_counts)
            print(f"비율:")
            print(df[col].value_counts(normalize=True).round(4))
    
    # stratified split을 위해 주요 라벨 선택 (argmax_Persona 사용)
    target_label = 'argmax_Persona'
    
    if target_label not in df.columns:
        raise ValueError(f"라벨 컬럼 '{target_label}'이 데이터에 없습니다.")
    
    # stratified split 수행
    print(f"\n=== {target_label}을 기준으로 stratified split 수행 ===")
    print(f"Train : Test = {1-test_size:.1f} : {test_size:.1f}")
    
    X = df.drop(columns=[target_label])  # 특성 데이터
    y = df[target_label]  # 라벨 데이터
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # 분할된 데이터 재결합
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # 분할 결과 확인
    print(f"\nTrain 데이터 크기: {train_df.shape}")
    print(f"Test 데이터 크기: {test_df.shape}")
    
    print(f"\n=== 분할 후 라벨 분포 확인 ===")
    print("Train 데이터 라벨 분포:")
    train_label_counts = train_df[target_label].value_counts()
    print(train_label_counts)
    print("비율:")
    print(train_df[target_label].value_counts(normalize=True).round(4))
    
    print("\nTest 데이터 라벨 분포:")
    test_label_counts = test_df[target_label].value_counts()
    print(test_label_counts)
    print("비율:")
    print(test_df[target_label].value_counts(normalize=True).round(4))
    
    return train_df, test_df

def save_split_data(train_df, test_df, output_dir="data"):
    """
    분할된 데이터를 CSV 파일로 저장
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    train_path = os.path.join(output_dir, "IBM_HR_personas_train.csv")
    test_path = os.path.join(output_dir, "IBM_HR_personas_test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n=== 파일 저장 완료 ===")
    print(f"Train 데이터: {train_path}")
    print(f"Test 데이터: {test_path}")
    
    return train_path, test_path

if __name__ == "__main__":
    # 파일 경로 설정
    csv_file_path = "data/IBM_HR_personas_assigned.csv"
    
    try:
        # 데이터 분할 수행
        train_df, test_df = split_data_with_label_ratio(
            csv_file_path=csv_file_path,
            test_size=0.4,  # 6:4 비율
            random_state=42
        )
        
        # 분할된 데이터 저장
        train_path, test_path = save_split_data(train_df, test_df)
        
        print(f"\n=== 분할 완료 ===")
        print(f"원본 데이터: {csv_file_path}")
        print(f"Train 데이터: {train_path} ({len(train_df)} rows)")
        print(f"Test 데이터: {test_path} ({len(test_df)} rows)")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
