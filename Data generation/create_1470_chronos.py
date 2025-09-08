#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1470명 Chronos 데이터 생성 스크립트
"""

import pandas as pd
import numpy as np
import time
import os

print("=== 1470명 Chronos 데이터 생성 시작 ===")

# 1. IBM HR 데이터 로드
print("1. IBM HR 데이터 로드 중...")
try:
    # 절대 경로로 데이터 로드
    data_path = r"C:\Users\jsjon\OneDrive\PwC RA\Agentic_AI_system\Agentic_AI_system\data\IBM_HR_personas_assigned.csv"
    print(f"   데이터 경로: {data_path}")
    print(f"   현재 작업 디렉토리: {os.getcwd()}")
    print(f"   파일 존재 여부: {os.path.exists(data_path)}")
    ibm_data = pd.read_csv(data_path)
    print(f"   IBM HR 데이터 로드 완료: {len(ibm_data)}명")
    
    # 필요한 컬럼만 선택
    employees_df = ibm_data[['EmployeeNumber', 'softmax_Persona_Code']].copy()
    
    # 페르소나 분포 확인
    print("\n   페르소나 분포:")
    persona_counts = employees_df['softmax_Persona_Code'].value_counts()
    for persona, count in persona_counts.items():
        percentage = (count / len(employees_df)) * 100
        risk_tier = "고위험" if persona.startswith('P') else "안정/몰입" if persona.startswith('S') else "중립/관망"
        print(f"   {persona:25} {count:4d}명 ({percentage:5.1f}%) - {risk_tier}")
    
except Exception as e:
    print(f"   오류: {e}")
    exit(1)

# 2. 필요한 함수들 정의 (Chronos_data_generation.py에서 가져옴)
print("\n2. Chronos 생성 함수 정의 중...")

# 시드 설정
np.random.seed(42)

# 영업일 생성
print("   영업일 생성 중...")
business_days = pd.bdate_range(start='2023-01-02', end='2024-12-30').tolist()
print(f"   영업일 생성 완료: {len(business_days)}일")

# 페르소나 패턴 정의
class RealisticPersonaPatterns:
    def __init__(self):
        self.patterns = {
            'P01_burnout': {
                'work_focused_ratio': {'stage1': 0.70, 'stage2': 0.58, 'stage3': 0.42, 'volatility': 0.15},
                'meeting_collaboration_ratio': {'stage1': 0.18, 'stage2': 0.12, 'stage3': 0.08, 'volatility': 0.08},
                'social_dining_ratio': {'stage1': 0.08, 'stage2': 0.05, 'stage3': 0.03, 'volatility': 0.03},
                'break_relaxation_ratio': {'stage1': 0.03, 'stage2': 0.18, 'stage3': 0.37, 'volatility': 0.12},
                'shared_work_ratio': {'stage1': 0.01, 'stage2': 0.07, 'stage3': 0.10, 'volatility': 0.05},
                'system_login_hours': {'stage1': 9.5, 'stage2': 11.2, 'stage3': 7.8, 'volatility': 1.5},
                'internal_comm_volume': {'stage1': 25, 'stage2': 15, 'stage3': 8, 'volatility': 8},
                'cafeteria_usage': {'stage1': 1.2, 'stage2': 0.8, 'stage3': 0.3, 'volatility': 0.4},
                'convenience_food_usage': {'stage1': 1.5, 'stage2': 3.2, 'stage3': 4.8, 'volatility': 1.2}
            },
            'P02_onboarding_failure': {
                'work_focused_ratio': {'base': 0.35, 'volatility': 0.20},
                'meeting_collaboration_ratio': {'base': 0.15, 'volatility': 0.08},
                'social_dining_ratio': {'base': 0.04, 'volatility': 0.02},
                'break_relaxation_ratio': {'base': 0.25, 'volatility': 0.15},
                'shared_work_ratio': {'base': 0.21, 'volatility': 0.12},
                'system_login_hours': {'base': 6.8, 'volatility': 1.8},
                'internal_comm_volume': {'base': 12, 'volatility': 6, 'help_seeking_bursts': True},
                'cafeteria_usage': {'base': 0.4, 'volatility': 0.2},
                'convenience_food_usage': {'base': 2.8, 'volatility': 1.0}
            },
            'P03_career_stagnation': {
                'work_focused_ratio': {'base': 0.62, 'volatility': 0.05},
                'meeting_collaboration_ratio': {'base': 0.12, 'volatility': 0.03},
                'social_dining_ratio': {'base': 0.08, 'volatility': 0.02},
                'break_relaxation_ratio': {'base': 0.12, 'volatility': 0.04},
                'shared_work_ratio': {'base': 0.06, 'volatility': 0.02},
                'system_login_hours': {'base': 8.3, 'volatility': 0.4},
                'internal_comm_volume': {'base': 18, 'trend': -0.002, 'volatility': 4},
                'cafeteria_usage': {'base': 0.9, 'volatility': 0.1},
                'convenience_food_usage': {'base': 1.1, 'volatility': 0.3}
            },
            'P04_under_compensated': {
                'work_focused_ratio': {'base': 0.68, 'volatility': 0.06},
                'meeting_collaboration_ratio': {'base': 0.16, 'volatility': 0.04},
                'social_dining_ratio': {'base': 0.09, 'volatility': 0.02},
                'break_relaxation_ratio': {'base': 0.05, 'volatility': 0.02},
                'shared_work_ratio': {'base': 0.02, 'volatility': 0.01},
                'system_login_hours': {'base': 8.5, 'volatility': 0.3},
                'internal_comm_volume': {'base': 22, 'volatility': 4},
                'cafeteria_usage': {'base': 1.0, 'volatility': 0.2},
                'convenience_food_usage': {'base': 1.3, 'volatility': 0.4}
            },
            'S01_anchor': {
                'work_focused_ratio': {'base': 0.65, 'volatility': 0.03},
                'meeting_collaboration_ratio': {'base': 0.18, 'volatility': 0.02},
                'social_dining_ratio': {'base': 0.12, 'volatility': 0.01},
                'break_relaxation_ratio': {'base': 0.04, 'volatility': 0.01},
                'shared_work_ratio': {'base': 0.01, 'volatility': 0.01},
                'system_login_hours': {'base': 8.5, 'volatility': 0.3},
                'internal_comm_volume': {'base': 22, 'volatility': 3},
                'cafeteria_usage': {'base': 1.0, 'volatility': 0.1},
                'convenience_food_usage': {'base': 0.8, 'volatility': 0.2}
            },
            'S02_rising_star': {
                'work_focused_ratio': {'base': 0.58, 'volatility': 0.06},
                'meeting_collaboration_ratio': {'base': 0.25, 'trend': 0.0002, 'volatility': 0.04},
                'social_dining_ratio': {'base': 0.10, 'volatility': 0.02},
                'break_relaxation_ratio': {'base': 0.03, 'volatility': 0.01},
                'shared_work_ratio': {'base': 0.04, 'trend': 0.0001, 'volatility': 0.03},
                'system_login_hours': {'base': 9.2, 'volatility': 0.6},
                'internal_comm_volume': {'base': 35, 'trend': 0.01, 'volatility': 5},
                'cafeteria_usage': {'base': 1.3, 'volatility': 0.3},
                'convenience_food_usage': {'base': 1.2, 'volatility': 0.3}
            },
            'S03_intrinsically_motivated': {
                'work_focused_ratio': {'base': 0.70, 'volatility': 0.04},
                'meeting_collaboration_ratio': {'base': 0.15, 'volatility': 0.03},
                'social_dining_ratio': {'base': 0.08, 'volatility': 0.02},
                'break_relaxation_ratio': {'base': 0.05, 'volatility': 0.02},
                'shared_work_ratio': {'base': 0.02, 'volatility': 0.01},
                'system_login_hours': {'base': 8.8, 'volatility': 0.4},
                'internal_comm_volume': {'base': 18, 'volatility': 3},
                'cafeteria_usage': {'base': 0.9, 'volatility': 0.15},
                'convenience_food_usage': {'base': 0.9, 'volatility': 0.25}
            },
            'N01_coaster': {
                'work_focused_ratio': {'base': 0.60, 'volatility': 0.02},
                'meeting_collaboration_ratio': {'base': 0.15, 'volatility': 0.02},
                'social_dining_ratio': {'base': 0.10, 'volatility': 0.01},
                'break_relaxation_ratio': {'base': 0.12, 'volatility': 0.02},
                'shared_work_ratio': {'base': 0.03, 'volatility': 0.01},
                'system_login_hours': {'base': 8.0, 'volatility': 0.2},
                'internal_comm_volume': {'base': 15, 'volatility': 2},
                'cafeteria_usage': {'base': 0.9, 'volatility': 0.1},
                'convenience_food_usage': {'base': 1.2, 'volatility': 0.3}
            },
            'N02_competent_malcontent': {
                'work_focused_ratio': {'base': 0.68, 'volatility': 0.08},
                'meeting_collaboration_ratio': {'base': 0.10, 'volatility': 0.05},
                'social_dining_ratio': {'base': 0.06, 'volatility': 0.03},
                'break_relaxation_ratio': {'base': 0.08, 'volatility': 0.04},
                'shared_work_ratio': {'base': 0.08, 'volatility': 0.05},
                'system_login_hours': {'base': 8.2, 'volatility': 0.5},
                'internal_comm_volume': {'base': 12, 'volatility': 6},
                'cafeteria_usage': {'base': 0.7, 'volatility': 0.2},
                'convenience_food_usage': {'base': 1.4, 'volatility': 0.5}
            },
            'N03_new_parent': {
                'work_focused_ratio': {'base': 0.62, 'volatility': 0.06},
                'meeting_collaboration_ratio': {'base': 0.18, 'volatility': 0.04},
                'social_dining_ratio': {'base': 0.08, 'volatility': 0.02},
                'break_relaxation_ratio': {'base': 0.08, 'volatility': 0.03},
                'shared_work_ratio': {'base': 0.04, 'volatility': 0.02},
                'system_login_hours': {'base': 7.5, 'volatility': 0.8},
                'internal_comm_volume': {'base': 20, 'volatility': 4},
                'cafeteria_usage': {'base': 0.8, 'volatility': 0.2},
                'convenience_food_usage': {'base': 1.1, 'volatility': 0.3}
            }
        }
    
    def get_pattern(self, persona_code):
        return self.patterns.get(persona_code, self.patterns['S01_anchor'])

persona_patterns = RealisticPersonaPatterns()

# 간단한 데이터 생성 함수
def generate_simple_chronos_data(employee_id, persona_code, business_days):
    """간단한 Chronos 데이터 생성"""
    
    pattern = persona_patterns.get_pattern(persona_code)
    timeseries_data = []
    
    for day_idx, date in enumerate(business_days):
        # 기본 변수 생성
        if 'stage1' in pattern['work_focused_ratio']:
            # 단계적 패턴 (P01 burnout)
            progress = min(day_idx / 300, 1.0)
            if progress < 0.33:
                work_ratio = pattern['work_focused_ratio']['stage1']
                login_hours = pattern['system_login_hours']['stage1']
                comm_volume = pattern['internal_comm_volume']['stage1']
            elif progress < 0.66:
                work_ratio = pattern['work_focused_ratio']['stage2']
                login_hours = pattern['system_login_hours']['stage2']
                comm_volume = pattern['internal_comm_volume']['stage2']
            else:
                work_ratio = pattern['work_focused_ratio']['stage3']
                login_hours = pattern['system_login_hours']['stage3']
                comm_volume = pattern['internal_comm_volume']['stage3']
        else:
            # 기본 패턴
            work_ratio = pattern['work_focused_ratio']['base']
            login_hours = pattern['system_login_hours']['base']
            comm_volume = pattern['internal_comm_volume']['base']
            
            # 트렌드 적용
            if 'trend' in pattern['internal_comm_volume']:
                comm_volume += pattern['internal_comm_volume']['trend'] * day_idx
        
        # 노이즈 추가
        work_ratio += np.random.normal(0, pattern['work_focused_ratio']['volatility'])
        login_hours += np.random.normal(0, pattern['system_login_hours']['volatility'])
        comm_volume += np.random.normal(0, pattern['internal_comm_volume']['volatility'])
        
        # 다른 비율들 생성
        meeting_ratio = 0.15 + np.random.normal(0, 0.03)
        social_ratio = 0.08 + np.random.normal(0, 0.02)
        break_ratio = 0.10 + np.random.normal(0, 0.03)
        shared_ratio = max(0.01, 1.0 - work_ratio - meeting_ratio - social_ratio - break_ratio)
        
        # 정규화
        total_ratio = work_ratio + meeting_ratio + social_ratio + break_ratio + shared_ratio
        work_ratio /= total_ratio
        meeting_ratio /= total_ratio
        social_ratio /= total_ratio
        break_ratio /= total_ratio
        shared_ratio /= total_ratio
        
        # 범위 제한 (한국 법정 근무시간 기준: 최대 10.4시간)
        login_hours = max(4.0, min(10.4, login_hours))
        comm_volume = max(0, int(round(comm_volume)))  # 정수로 변환 (메시지/이메일 개수)
        
        # 식사 관련 변수들 (하루 3끼 기준)
        # 카페테리아 사용 횟수 (0-3끼)
        if persona_code.startswith('P'):  # 고위험군
            cafe_prob = 0.3  # 카페테리아 사용 확률 낮음
        elif persona_code.startswith('S'):  # 안정/몰입군
            cafe_prob = 0.6  # 카페테리아 사용 확률 높음
        else:  # 중립군
            cafe_prob = 0.5  # 중간 확률
        
        # 금요일은 사회적 식사 증가
        if date.weekday() == 4:
            cafe_prob *= 1.3
        
        # 각 끼니별로 카페테리아 사용 여부 결정
        cafe_usage = 0
        for meal in range(3):  # 아침, 점심, 저녁
            if np.random.random() < cafe_prob:
                cafe_usage += 1
        
        # 편의식품 사용 = 3 - 카페테리아 사용
        conv_usage = 3 - cafe_usage
        
        # 공간별 시간 계산
        work_focused_hours = work_ratio * login_hours
        meeting_collaboration_hours = meeting_ratio * login_hours
        social_dining_hours = social_ratio * login_hours
        break_relaxation_hours = break_ratio * login_hours
        shared_work_hours = shared_ratio * login_hours
        
        # 검증: 공간별 시간 합계가 전체 로그인 시간과 일치하는지 확인
        total_space_hours = (work_focused_hours + meeting_collaboration_hours + 
                           social_dining_hours + break_relaxation_hours + shared_work_hours)
        
        # 부동소수점 오차 보정 (반올림)
        if abs(total_space_hours - login_hours) > 0.01:
            print(f"Warning: 시간 합계 불일치 - 전체: {login_hours:.2f}, 합계: {total_space_hours:.2f}")
        
        daily_data = {
            'employee_id': employee_id,
            'date': date,
            'day_of_week': date.weekday(),
            'day_index': day_idx,
            # persona_code는 제거 - 치팅 방지를 위해 최종 데이터에서 제외
            'work_focused_ratio': work_ratio,
            'meeting_collaboration_ratio': meeting_ratio,
            'social_dining_ratio': social_ratio,
            'break_relaxation_ratio': break_ratio,
            'shared_work_ratio': shared_ratio,
            'system_login_hours': login_hours,
            'work_focused_hours': work_focused_hours,
            'meeting_collaboration_hours': meeting_collaboration_hours,
            'social_dining_hours': social_dining_hours,
            'break_relaxation_hours': break_relaxation_hours,
            'shared_work_hours': shared_work_hours,
            'internal_comm_volume': comm_volume,
            'cafeteria_usage': cafe_usage,
            'convenience_food_usage': conv_usage
        }
        
        timeseries_data.append(daily_data)
    
    return pd.DataFrame(timeseries_data)

print("   함수 정의 완료")

# 3. 1470명 데이터 생성
print("\n3. 1470명 Chronos 데이터 생성 시작...")

all_data = []
start_time = time.time()
total_employees = len(employees_df)

for idx, (_, employee) in enumerate(employees_df.iterrows()):
    if idx % 50 == 0:
        elapsed = time.time() - start_time
        if idx > 0:
            estimated_total = (elapsed / idx) * total_employees
            print(f"   진행률: {idx:4d}/{total_employees} ({idx/total_employees*100:5.1f}%) - 예상 소요시간: {estimated_total/60:.1f}분")
        else:
            print(f"   진행률: {idx:4d}/{total_employees} ({idx/total_employees*100:5.1f}%) - 시작")
    
    emp_id = employee['EmployeeNumber']
    persona = employee['softmax_Persona_Code']
    
    try:
        employee_data = generate_simple_chronos_data(emp_id, persona, business_days)
        all_data.append(employee_data)
    except Exception as e:
        print(f"   직원 {emp_id} 데이터 생성 실패: {e}")
        continue

# 4. 데이터 통합 및 저장
print("\n4. 데이터 통합 및 저장 중...")

if all_data:
    final_dataset = pd.concat(all_data, ignore_index=True)
    
    # CSV 저장 - data 폴더에 IBM_HR_timeseries.csv로 저장
    output_dir = r"C:\Users\jsjon\OneDrive\PwC RA\Agentic_AI_system\Agentic_AI_system\data"
    output_file = os.path.join(output_dir, "IBM_HR_timeseries.csv")
    
    # 디렉토리가 존재하는지 확인
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"   디렉토리 생성: {output_dir}")
    
    final_dataset.to_csv(output_file, index=False)
    
    total_time = time.time() - start_time
    
    print(f"\n=== 생성 완료 ===")
    print(f"파일명: {output_file}")
    print(f"데이터 크기: {final_dataset.shape}")
    print(f"직원 수: {final_dataset['employee_id'].nunique()}명")
    print(f"총 레코드 수: {len(final_dataset):,}개")
    print(f"소요시간: {total_time/60:.1f}분")
    print(f"파일 크기: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    print(f"저장 위치: {output_file}")
    
    # 데이터 분포 확인 (persona_code 제거로 인해 전체 통계만 표시)
    print(f"\n=== 생성된 데이터 분포 ===")
    print(f"총 직원 수: {final_dataset['employee_id'].nunique()}명")
    print(f"총 레코드 수: {len(final_dataset):,}개")
    print(f"직원당 평균 레코드 수: {len(final_dataset) / final_dataset['employee_id'].nunique():.1f}개")
    print("※ persona_code는 치팅 방지를 위해 최종 데이터에서 제외되었습니다.")
    
    # 주요 변수 통계
    print(f"\n=== 주요 변수 통계 ===")
    key_vars = ['work_focused_ratio', 'system_login_hours', 'internal_comm_volume', 'cafeteria_usage']
    for var in key_vars:
        stats = final_dataset[var].describe()
        print(f"{var:25}: 평균 {stats['mean']:6.3f}, 범위 [{stats['min']:6.3f}, {stats['max']:6.3f}]")
    
    # 시간 합계 검증
    print(f"\n=== 시간 합계 검증 ===")
    space_hour_cols = ['work_focused_hours', 'meeting_collaboration_hours', 
                      'social_dining_hours', 'break_relaxation_hours', 'shared_work_hours']
    final_dataset['total_space_hours'] = final_dataset[space_hour_cols].sum(axis=1)
    final_dataset['hour_diff'] = abs(final_dataset['total_space_hours'] - final_dataset['system_login_hours'])
    
    print(f"시간 합계 일치 여부:")
    print(f"  평균 차이: {final_dataset['hour_diff'].mean():.6f}시간")
    print(f"  최대 차이: {final_dataset['hour_diff'].max():.6f}시간")
    print(f"  0.01시간 이상 차이나는 레코드: {(final_dataset['hour_diff'] > 0.01).sum()}개")
    
    # 식사 합계 검증
    print(f"\n=== 식사 합계 검증 ===")
    final_dataset['total_meals'] = final_dataset['cafeteria_usage'] + final_dataset['convenience_food_usage']
    meal_check = final_dataset['total_meals'] == 3
    print(f"식사 합계가 3인 레코드: {meal_check.sum():,}개 / {len(final_dataset):,}개 ({meal_check.mean()*100:.1f}%)")
    if not meal_check.all():
        print(f"WARNING: 식사 합계가 3이 아닌 레코드가 {(~meal_check).sum()}개 있습니다!")
    
    # 정리: 임시 컬럼 제거
    final_dataset = final_dataset.drop(['total_space_hours', 'hour_diff', 'total_meals'], axis=1)
    
else:
    print("ERROR: 생성된 데이터가 없습니다!")

print("\n스크립트 완료!")
