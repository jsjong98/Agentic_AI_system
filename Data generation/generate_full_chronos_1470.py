#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1470명 전체 Chronos 퇴사 인식 데이터셋 생성 스크립트
"""

import sys
import os
import time
from datetime import datetime

# 현재 디렉토리를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Chronos_data_generation.py의 모든 함수들을 import
chronos_file = os.path.join(current_dir, 'Chronos_data_generation.py')
if not os.path.exists(chronos_file):
    print(f"❌ 오류: {chronos_file} 파일을 찾을 수 없습니다.")
    print(f"현재 디렉토리: {current_dir}")
    print("파일 목록:")
    for f in os.listdir(current_dir):
        if f.endswith('.py'):
            print(f"  - {f}")
    sys.exit(1)

exec(open(chronos_file, 'r', encoding='utf-8').read())

def main():
    """1470명 전체 데이터셋 생성 및 저장"""
    
    print("=" * 60)
    print("🚀 1470명 전체 Chronos 퇴사 인식 데이터셋 생성 시작")
    print("=" * 60)
    
    start_time = time.time()
    
    # 출력 파일 경로 설정
    output_path = "data/IBM_HR_timeseries.csv"
    
    try:
        # 전체 데이터셋 생성 (1470명)
        print("📊 데이터 생성 중... (예상 소요시간: 5-10분)")
        
        full_dataset = generate_attrition_aware_chronos_dataset(
            ibm_hr_path="data/IBM_HR_personas_assigned.csv",
            num_employees=1470, 
            save_path=output_path
        )
        
        if len(full_dataset) == 0:
            print("❌ 데이터 생성 실패!")
            return False
        
        # 생성 완료 정보
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("✅ 데이터셋 생성 완료!")
        print("=" * 60)
        print(f"📁 저장 위치: {output_path}")
        print(f"📊 데이터 크기: {full_dataset.shape}")
        print(f"⏱️  소요시간: {elapsed_time/60:.1f}분")
        print(f"💾 파일 크기: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
        
        # 퇴사자/재직자 분포 확인
        attrition_dist = full_dataset['attrition_status'].value_counts()
        total_records = len(full_dataset)
        
        print(f"\n📈 퇴사자 분포:")
        for status, count in attrition_dist.items():
            status_name = "퇴사자" if status == 'Yes' else "재직자"
            percentage = (count / total_records) * 100
            print(f"  {status_name}: {count:,}개 레코드 ({percentage:.1f}%)")
        
        # 주요 지표 요약
        print(f"\n📊 퇴사자 vs 재직자 주요 지표 비교:")
        summary = full_dataset.groupby('attrition_status')[
            ['work_focused_ratio', 'system_login_hours', 'internal_comm_volume', 'convenience_food_usage']
        ].mean()
        
        for status in ['Yes', 'No']:
            if status in summary.index:
                status_name = "퇴사자" if status == 'Yes' else "재직자"
                print(f"\n  {status_name}:")
                print(f"    업무 집중도: {summary.loc[status, 'work_focused_ratio']:.3f}")
                print(f"    로그인 시간: {summary.loc[status, 'system_login_hours']:.1f}시간")
                print(f"    소통량: {summary.loc[status, 'internal_comm_volume']:.1f}건")
                print(f"    편의점 음식: {summary.loc[status, 'convenience_food_usage']:.3f}회")
        
        # 차이 계산
        if 'Yes' in summary.index and 'No' in summary.index:
            print(f"\n🎯 퇴사자 vs 재직자 차이:")
            work_diff = ((summary.loc['No', 'work_focused_ratio'] - summary.loc['Yes', 'work_focused_ratio']) / summary.loc['Yes', 'work_focused_ratio']) * 100
            comm_diff = ((summary.loc['No', 'internal_comm_volume'] - summary.loc['Yes', 'internal_comm_volume']) / summary.loc['Yes', 'internal_comm_volume']) * 100
            food_diff = ((summary.loc['Yes', 'convenience_food_usage'] - summary.loc['No', 'convenience_food_usage']) / summary.loc['No', 'convenience_food_usage']) * 100
            
            print(f"    재직자가 퇴사자보다 업무 집중도 {work_diff:.1f}% 높음")
            print(f"    재직자가 퇴사자보다 소통량 {comm_diff:.1f}% 많음")
            print(f"    퇴사자가 재직자보다 편의점 음식 {food_diff:.1f}% 많이 섭취")
        
        # 페르소나 분포
        print(f"\n🎭 페르소나 분포 (상위 5개):")
        persona_dist = full_dataset['persona_code'].value_counts().head(5)
        for persona, count in persona_dist.items():
            employees = full_dataset[full_dataset['persona_code'] == persona]['employee_id'].nunique()
            print(f"  {persona}: {employees}명 ({count:,}개 레코드)")
        
        print(f"\n🎉 데이터 생성 완료! 이제 분석에 사용할 수 있습니다.")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 스크립트 실행 완료!")
    else:
        print("\n❌ 스크립트 실행 실패!")
        sys.exit(1)
