#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentio 텍스트 전달 테스트
"""

import requests
import json
import pandas as pd

def test_sentio_text_processing():
    """Sentio에 실제 텍스트가 제대로 전달되는지 테스트"""
    
    # 1. 실제 데이터 파일에서 샘플 읽기
    try:
        df = pd.read_csv('data/IBM_HR_text.csv')
        print(f"[OK] 데이터 파일 로드 완료: {len(df)}행")
        print(f"[INFO] 컬럼: {list(df.columns)}")
        
        # 첫 번째 직원 데이터 추출
        first_employee = df.iloc[0]
        employee_id = first_employee['EmployeeNumber']
        
        print(f"\n[DEBUG] 직원 {employee_id} 데이터:")
        print(f"  - SELF_REVIEW_text 길이: {len(str(first_employee['SELF_REVIEW_text']))}자")
        print(f"  - PEER_FEEDBACK_text 길이: {len(str(first_employee['PEER_FEEDBACK_text']))}자")
        print(f"  - WEEKLY_SURVEY_text 길이: {len(str(first_employee['WEEKLY_SURVEY_text']))}자")
        
        print(f"\n[TEXT] SELF_REVIEW_text 내용 (처음 200자):")
        print(f"'{str(first_employee['SELF_REVIEW_text'])[:200]}...'")
        
        # 2. Sentio API 테스트 데이터 구성
        test_data = {
            "employees": [
                {
                    "employee_id": int(employee_id),
                    "text_data": {
                        "self_review": str(first_employee['SELF_REVIEW_text']),
                        "peer_feedback": str(first_employee['PEER_FEEDBACK_text']),
                        "weekly_survey": str(first_employee['WEEKLY_SURVEY_text'])
                    }
                }
            ]
        }
        
        print(f"\n[API] Sentio API 호출 시작...")
        
        # 3. Sentio API 호출
        response = requests.post(
            'http://localhost:5004/analyze/batch',
            headers={'Content-Type': 'application/json'},
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Sentio API 호출 성공!")
            print(f"[INFO] 응답 구조: {list(result.keys())}")
            
            if 'analysis_results' in result:
                analysis = result['analysis_results'][0]
                print(f"\n[RESULT] 분석 결과:")
                print(f"  - 직원 ID: {analysis.get('employee_id')}")
                print(f"  - 심리적 위험 점수: {analysis.get('psychological_risk_score', 'N/A')}")
                print(f"  - 위험 수준: {analysis.get('risk_level', 'N/A')}")
                print(f"  - 감정 점수: {analysis.get('sentiment_score', 'N/A')}")
                print(f"  - 퇴직 예측: {analysis.get('attrition_prediction', 'N/A')}")
            else:
                print(f"[WARN] analysis_results가 응답에 없습니다.")
                print(f"전체 응답: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"[ERROR] Sentio API 호출 실패: {response.status_code}")
            print(f"오류 내용: {response.text}")
            
    except Exception as e:
        print(f"[ERROR] 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sentio_text_processing()
