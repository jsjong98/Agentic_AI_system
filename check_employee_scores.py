#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Employee ID별 5개 에이전트 점수 확인 스크립트
각 에이전트에서 0~1 범위의 점수를 조회합니다.
"""

import requests
import json
import pandas as pd
from typing import Dict, List

class EmployeeScoreChecker:
    """직원별 에이전트 점수 확인기"""
    
    def __init__(self):
        self.base_urls = {
            'structura': 'http://localhost:5001',
            'cognita': 'http://localhost:5002', 
            'chronos': 'http://localhost:5005',  # Chronos 포트 변경
            'sentio': 'http://localhost:5003',
            'agora': 'http://localhost:5004'     # Agora는 5004 유지
        }
        
    def check_employee_scores(self, employee_id: str) -> Dict:
        """특정 직원의 모든 에이전트 점수 조회"""
        
        scores = {
            'employee_id': employee_id,
            'structura_score': None,
            'cognita_score': None, 
            'chronos_score': None,
            'sentio_score': None,
            'agora_score': None,
            'errors': []
        }
        
        # 1. Structura - 퇴직 확률
        try:
            response = requests.post(
                f"{self.base_urls['structura']}/api/predict",
                json={"employee_ids": [employee_id]},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                if result.get('predictions'):
                    scores['structura_score'] = result['predictions'][0]['attrition_probability']
        except Exception as e:
            scores['errors'].append(f"Structura: {str(e)}")
            
        # 2. Cognita - 종합 위험도
        try:
            response = requests.post(
                f"{self.base_urls['cognita']}/api/analyze_risk",
                json={"employee_id": employee_id},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                scores['cognita_score'] = result.get('overall_risk_score')
        except Exception as e:
            scores['errors'].append(f"Cognita: {str(e)}")
            
        # 3. Chronos - 시계열 퇴직 확률
        try:
            response = requests.post(
                f"{self.base_urls['chronos']}/api/predict",
                json={"employee_ids": [employee_id]},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                if result.get('predictions'):
                    scores['chronos_score'] = result['predictions'][0]['attrition_probability']
        except Exception as e:
            scores['errors'].append(f"Chronos: {str(e)}")
            
        # 4. Sentio - 심리적 위험도 (텍스트 분석 필요)
        try:
            # 샘플 텍스트로 분석 (실제로는 직원의 실제 텍스트 사용)
            sample_text = "최근 업무가 많아서 스트레스를 받고 있습니다."
            response = requests.post(
                f"{self.base_urls['sentio']}/analyze/text",
                json={
                    "text": sample_text,
                    "employee_id": employee_id,
                    "text_type": "WEEKLY_SURVEY"
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                scores['sentio_score'] = result.get('attrition_risk_score')
        except Exception as e:
            scores['errors'].append(f"Sentio: {str(e)}")
            
        # 5. Agora - 시장 압력 지수
        try:
            # 샘플 직원 데이터로 분석
            sample_employee = {
                "JobRole": "Software Engineer",
                "MonthlyIncome": 5000,
                "YearsAtCompany": 3,
                "Education": 4,
                "JobSatisfaction": 3
            }
            response = requests.post(
                f"{self.base_urls['agora']}/analyze/market", 
                json=sample_employee,
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                scores['agora_score'] = result.get('agora_score')
        except Exception as e:
            scores['errors'].append(f"Agora: {str(e)}")
            
        return scores
    
    def check_multiple_employees(self, employee_ids: List[str]) -> pd.DataFrame:
        """여러 직원의 점수를 일괄 조회"""
        
        results = []
        for emp_id in employee_ids:
            print(f"🔍 직원 {emp_id} 점수 조회 중...")
            scores = self.check_employee_scores(emp_id)
            results.append(scores)
            
        df = pd.DataFrame(results)
        return df
    
    def display_scores(self, scores_df: pd.DataFrame):
        """점수 결과를 보기 좋게 출력"""
        
        print("\n" + "="*80)
        print("📊 Employee ID별 에이전트 점수 (0~1 범위)")
        print("="*80)
        
        for _, row in scores_df.iterrows():
            print(f"\n👤 직원 ID: {row['employee_id']}")
            print("-" * 50)
            
            # 각 에이전트 점수 출력
            agents = [
                ('Structura (퇴직확률)', row['structura_score'], '📈'),
                ('Cognita (관계위험)', row['cognita_score'], '🌐'), 
                ('Chronos (시계열)', row['chronos_score'], '⏰'),
                ('Sentio (심리위험)', row['sentio_score'], '🧠'),
                ('Agora (시장압력)', row['agora_score'], '💼')
            ]
            
            for name, score, icon in agents:
                if score is not None:
                    # 위험도 색상 표시 (모든 점수가 높을수록 위험함)
                    color = "🔴" if score >= 0.7 else "🟡" if score >= 0.4 else "🟢"
                    
                    print(f"  {icon} {name:20} : {score:.3f} {color}")
                else:
                    print(f"  {icon} {name:20} : N/A   ⚪")
            
            # 에러 출력
            if row['errors']:
                print(f"  ⚠️  오류: {', '.join(row['errors'])}")
                
        # 통계 요약
        print("\n" + "="*80)
        print("📈 점수 통계 요약")
        print("="*80)
        
        score_columns = ['structura_score', 'cognita_score', 'chronos_score', 'sentio_score', 'agora_score']
        for col in score_columns:
            valid_scores = scores_df[col].dropna()
            if len(valid_scores) > 0:
                agent_name = col.replace('_score', '').title()
                print(f"{agent_name:12} - 평균: {valid_scores.mean():.3f}, 최고: {valid_scores.max():.3f}, 최저: {valid_scores.min():.3f}")

def main():
    """메인 실행 함수"""
    
    print("🚀 Employee ID별 에이전트 점수 확인 도구")
    print("5개 에이전트에서 0~1 범위의 점수를 조회합니다.")
    
    checker = EmployeeScoreChecker()
    
    # 샘플 직원 ID들 (실제 데이터에 맞게 수정)
    sample_employee_ids = ["1", "2", "3", "4", "5"]
    
    print(f"\n📋 {len(sample_employee_ids)}명의 직원 점수를 확인합니다...")
    
    # 점수 조회
    scores_df = checker.check_multiple_employees(sample_employee_ids)
    
    # 결과 출력
    checker.display_scores(scores_df)
    
    # CSV 저장
    output_file = "employee_scores_summary.csv"
    scores_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과가 {output_file}에 저장되었습니다.")
    
    return scores_df

if __name__ == "__main__":
    main()
