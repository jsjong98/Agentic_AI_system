#!/usr/bin/env python3
"""
Integration /app/results 디렉토리 구조 생성 스크립트
Total score.csv + IBM_HR.csv를 기반으로 employee_ 폴더 및 department_index.json 생성

사용법 (Railway):
  railway run --service integration python /app/scripts/seed_results.py

디렉토리 구조:
  /app/results/{Department}/{JobRole}/Level_{N}/employee_{id}/
      ├── employee_info.json
      ├── comprehensive_report.json
      ├── structura_result.json
      ├── chronos_result.json
      ├── cognita_result.json
      ├── sentio_result.json
      └── agora_result.json
  /app/results/{Department}/department_index.json
"""

import csv
import json
import os
from datetime import datetime
from collections import defaultdict

# ── 경로 설정 ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Railway Integration 컨테이너 내부 경로
RESULTS_DIR = os.environ.get('RESULTS_DIR', '/app/results')
HR_CSV = os.path.join(SCRIPT_DIR, 'data', 'IBM_HR.csv')
SCORE_CSV = os.path.join(SCRIPT_DIR, 'data', 'Total_score.csv')

TIMESTAMP = datetime.now().isoformat()


def classify_risk(score):
    """위험도 점수를 등급으로 분류"""
    if score >= 0.7:
        return 'HIGH'
    elif score >= 0.3:
        return 'MEDIUM'
    return 'LOW'


def load_data():
    """CSV 데이터 로드 및 병합"""
    # IBM HR 데이터
    with open(HR_CSV, 'r', encoding='utf-8-sig') as f:
        hr_rows = list(csv.DictReader(f))
    hr_map = {r['EmployeeNumber']: r for r in hr_rows}

    # Total score 데이터
    with open(SCORE_CSV, 'r', encoding='utf-8-sig') as f:
        score_rows = list(csv.DictReader(f))

    print(f"📊 HR 데이터: {len(hr_rows)}명, Score 데이터: {len(score_rows)}명")
    return hr_map, score_rows


def build_employee_files(hr, scores):
    """직원별 JSON 파일 생성"""
    structura_score = float(scores.get('Structura_score', 0))
    cognita_score = float(scores.get('Cognita_score', 0))
    chronos_score = float(scores.get('Chronos_score', 0))
    sentio_score = float(scores.get('Sentio_score', 0))
    agora_score = float(scores.get('Agora_score', 0))

    # 가중 평균 위험도 (Structura 30%, Chronos 25%, Cognita 20%, Sentio 15%, Agora 10%)
    overall_risk = (
        structura_score * 0.30 +
        chronos_score * 0.25 +
        cognita_score * 0.20 +
        sentio_score * 0.15 +
        agora_score * 0.10
    )
    risk_level = classify_risk(overall_risk)
    attrition = scores.get('attrition', 'No')
    emp_id = scores['employee_id']

    # employee_info.json
    employee_info = {
        "employee_id": emp_id,
        "employee_data": {
            "EmployeeNumber": emp_id,
            "Age": int(hr.get('Age', 0)),
            "Department": hr.get('Department', ''),
            "JobRole": hr.get('JobRole', ''),
            "JobLevel": int(hr.get('JobLevel', 1)),
            "MonthlyIncome": int(hr.get('MonthlyIncome', 0)),
            "YearsAtCompany": int(hr.get('YearsAtCompany', 0)),
            "JobSatisfaction": int(hr.get('JobSatisfaction', 0)),
            "WorkLifeBalance": int(hr.get('WorkLifeBalance', 0)),
            "OverTime": hr.get('OverTime', 'No'),
            "MaritalStatus": hr.get('MaritalStatus', ''),
            "DistanceFromHome": int(hr.get('DistanceFromHome', 0)),
            "EnvironmentSatisfaction": int(hr.get('EnvironmentSatisfaction', 0)),
            "RelationshipSatisfaction": int(hr.get('RelationshipSatisfaction', 0)),
            "Attrition": attrition
        },
        "analysis_timestamp": TIMESTAMP
    }

    # comprehensive_report.json
    comprehensive_report = {
        "employee_id": emp_id,
        "comprehensive_assessment": {
            "overall_risk_score": round(overall_risk, 4),
            "overall_risk_level": risk_level,
            "agent_scores": {
                "structura": round(structura_score, 4),
                "cognita": round(cognita_score, 4),
                "chronos": round(chronos_score, 4),
                "sentio": round(sentio_score, 4),
                "agora": round(agora_score, 4)
            },
            "actual_attrition": attrition
        },
        "generated_at": TIMESTAMP
    }

    # 개별 에이전트 결과 파일
    structura_result = {
        "prediction": {"attrition_probability": round(structura_score, 6)},
        "employee_id": emp_id,
        "timestamp": TIMESTAMP
    }
    chronos_result = {
        "prediction": {"risk_score": round(chronos_score, 6)},
        "employee_id": emp_id,
        "timestamp": TIMESTAMP
    }
    cognita_result = {
        "risk_analysis": {"overall_risk_score": round(cognita_score, 4)},
        "employee_id": emp_id,
        "timestamp": TIMESTAMP
    }
    sentio_result = {
        "sentiment_analysis": {"risk_score": round(sentio_score, 6)},
        "psychological_risk_score": round(sentio_score, 6),
        "employee_id": emp_id,
        "timestamp": TIMESTAMP
    }
    agora_result = {
        "market_analysis": {"risk_score": round(agora_score, 4)},
        "risk_score": round(agora_score, 4),
        "employee_id": emp_id,
        "timestamp": TIMESTAMP
    }

    return {
        'employee_info.json': employee_info,
        'comprehensive_report.json': comprehensive_report,
        'structura_result.json': structura_result,
        'chronos_result.json': chronos_result,
        'cognita_result.json': cognita_result,
        'sentio_result.json': sentio_result,
        'agora_result.json': agora_result,
    }


def build_department_index(dept_name, dept_employees):
    """부서별 department_index.json 생성"""
    job_roles = defaultdict(lambda: defaultdict(list))
    position_count = defaultdict(int)

    for emp in dept_employees:
        job_role = emp['job_role'].replace(' ', '_')
        level = f"Level_{emp['job_level']}"
        job_roles[job_role][level].append(emp['employee_id'])
        position_count[str(emp['job_level'])] += 1

    # 위험도 분포
    high = sum(1 for e in dept_employees if e['risk_level'] == 'HIGH')
    medium = sum(1 for e in dept_employees if e['risk_level'] == 'MEDIUM')
    low = sum(1 for e in dept_employees if e['risk_level'] == 'LOW')

    return {
        "department": dept_name,
        "generated_at": TIMESTAMP,
        "statistics": {
            "total_employees": len(dept_employees),
            "position_count": dict(position_count),
            "risk_distribution": {
                "high": high,
                "medium": medium,
                "low": low
            }
        },
        "job_roles": {k: dict(v) for k, v in job_roles.items()}
    }


def main():
    print("=" * 60)
    print("🚀 Integration /app/results 디렉토리 구조 생성 시작")
    print(f"📁 출력 디렉토리: {RESULTS_DIR}")
    print("=" * 60)

    hr_map, score_rows = load_data()

    # 부서별 직원 그룹핑
    departments = defaultdict(list)
    created = 0
    skipped = 0

    for score in score_rows:
        emp_id = score['employee_id']
        hr = hr_map.get(emp_id)
        if not hr:
            skipped += 1
            continue

        dept = hr['Department']
        job_role = hr['JobRole']
        job_level = hr['JobLevel']

        # 디렉토리 경로: {dept}/{job_role}/Level_{N}/employee_{id}
        dept_dir = dept.replace(' ', '_')
        role_dir = job_role.replace(' ', '_')
        level_dir = f"Level_{job_level}"
        emp_dir = os.path.join(RESULTS_DIR, dept_dir, role_dir, level_dir, f"employee_{emp_id}")

        os.makedirs(emp_dir, exist_ok=True)

        # 파일 생성
        files = build_employee_files(hr, score)
        for filename, data in files.items():
            filepath = os.path.join(emp_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        # 위험도 계산
        structura_score = float(score.get('Structura_score', 0))
        cognita_score = float(score.get('Cognita_score', 0))
        chronos_score = float(score.get('Chronos_score', 0))
        sentio_score = float(score.get('Sentio_score', 0))
        agora_score = float(score.get('Agora_score', 0))
        overall_risk = (
            structura_score * 0.30 +
            chronos_score * 0.25 +
            cognita_score * 0.20 +
            sentio_score * 0.15 +
            agora_score * 0.10
        )

        departments[dept].append({
            'employee_id': emp_id,
            'job_role': job_role,
            'job_level': job_level,
            'risk_level': classify_risk(overall_risk),
            'risk_score': overall_risk,
        })
        created += 1

    # department_index.json 생성
    for dept_name, employees in departments.items():
        dept_dir = dept_name.replace(' ', '_')
        index_path = os.path.join(RESULTS_DIR, dept_dir, 'department_index.json')
        index_data = build_department_index(dept_name, employees)
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        print(f"  📂 {dept_name}: {len(employees)}명, index 생성 완료")

    print()
    print("=" * 60)
    print(f"✅ 완료: {created}명 생성, {skipped}명 스킵")
    print(f"📁 부서: {len(departments)}개")
    for dept, emps in departments.items():
        high = sum(1 for e in emps if e['risk_level'] == 'HIGH')
        medium = sum(1 for e in emps if e['risk_level'] == 'MEDIUM')
        low = sum(1 for e in emps if e['risk_level'] == 'LOW')
        print(f"   {dept}: {len(emps)}명 (🔴{high} 🟡{medium} 🟢{low})")
    print("=" * 60)


if __name__ == '__main__':
    main()
