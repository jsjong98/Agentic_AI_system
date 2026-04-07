"""
Test data로부터 초안 result JSON 파일 생성 스크립트
- Total score.csv: 직원별 에이전트 점수
- Structura test.csv: 직원 HR 정보 (부서, 직무, 나이 등)
→ app/results/{Department}/employee_{id}/batch_analysis_*.json
→ app/results/{Department}/employee_{id}/comprehensive_report.json
→ app/results/batch_analysis/department_summary_*.json
"""

import csv
import json
import os
from datetime import datetime

# ─── 경로 설정 ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(BASE_DIR, 'Test_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'app', 'results')

TOTAL_SCORE_CSV = os.path.join(TEST_DATA_DIR, 'Total score.csv')
STRUCTURA_CSV   = os.path.join(TEST_DATA_DIR, 'Structura test.csv')

TIMESTAMP = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

# ─── 부서명 정규화 ────────────────────────────────────────────
DEPT_MAPPING = {
    'Human Resources':        'Human_Resources',
    'Research & Development': 'Research_&_Development',
    'Research and Development':'Research_&_Development',
    'R&D':                    'Research_&_Development',
    'Sales':                  'Sales',
    'HR':                     'Human_Resources',
    'Information Technology': 'Information_Technology',
    'IT':                     'Information_Technology',
    'Marketing':              'Marketing',
    'Finance':                'Finance',
    'Operations':             'Operations',
    'Manufacturing':          'Manufacturing',
}

def normalize_dept(dept):
    return DEPT_MAPPING.get(dept, dept.replace(' ', '_').replace('&', '_&_'))

# ─── 위험도 계산 ───────────────────────────────────────────────
WEIGHTS = {
    'Structura_score': 0.30,
    'Chronos_score':   0.25,
    'Sentio_score':    0.20,
    'Cognita_score':   0.15,
    'Agora_score':     0.10,
}

def calc_risk(row):
    score = sum(float(row[k]) * w for k, w in WEIGHTS.items())
    if score >= 0.65:
        level = 'HIGH'
    elif score >= 0.35:
        level = 'MEDIUM'
    else:
        level = 'LOW'
    return round(score, 4), level

# ─── CSV 읽기 ─────────────────────────────────────────────────
def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

print("📂 CSV 파일 로드 중...")
scores_rows    = read_csv(TOTAL_SCORE_CSV)   # employee_id 기준
structura_rows = read_csv(STRUCTURA_CSV)     # EmployeeNumber 기준

# Structura → dict (EmployeeNumber → row)
structura_map = {}
for r in structura_rows:
    eid = r.get('EmployeeNumber') or r.get('employee_id') or r.get('EmployeeID')
    if eid:
        structura_map[str(eid).strip()] = r

print(f"  Total score  : {len(scores_rows):,}명")
print(f"  Structura HR : {len(structura_map):,}명")

# ─── 결과 생성 ─────────────────────────────────────────────────
department_results = {}   # normalized_dept → summary
individual_results = []

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'batch_analysis'), exist_ok=True)

for row in scores_rows:
    eid = str(row['employee_id']).strip()
    s   = structura_map.get(eid, {})

    # HR 기본 정보
    dept      = s.get('Department', 'Unknown')
    job_role  = s.get('JobRole', 'Unknown')
    job_level = s.get('JobLevel', 1)
    age       = s.get('Age', '')
    gender    = s.get('Gender', '')
    marital   = s.get('MaritalStatus', '')
    overtime  = s.get('OverTime', 'No')
    monthly_income = s.get('MonthlyIncome', '')
    years_at_co    = s.get('YearsAtCompany', '')
    job_sat        = s.get('JobSatisfaction', '')
    wlb            = s.get('WorkLifeBalance', '')
    perf_rating    = s.get('PerformanceRating', '')
    education_field= s.get('EducationField', '')
    business_travel= s.get('BusinessTravel', '')

    # Persona 정보
    persona_code  = s.get('softmax_Persona_Code', s.get('argmax_Persona_Code', ''))
    persona_name  = s.get('softmax_Persona', s.get('argmax_Persona', ''))
    risk_tier     = s.get('softmax_RiskTier', s.get('argmax_RiskTier', ''))

    # 점수
    structura_score = float(row.get('Structura_score', 0))
    cognita_score   = float(row.get('Cognita_score', 0))
    chronos_score   = float(row.get('Chronos_score', 0))
    sentio_score    = float(row.get('Sentio_score', 0))
    agora_score     = float(row.get('Agora_score', 0))
    attrition_label = row.get('attrition', 'No')

    risk_score, risk_level = calc_risk(row)

    norm_dept = normalize_dept(dept)

    # ─── 에이전트별 세부 결과 구성 ──────────────────────────────
    # Structura
    structura_result = {
        'attrition_probability': round(structura_score, 4),
        'predicted_attrition': 1 if structura_score >= 0.5 else 0,
        'confidence': 0.85,
        'feature_importance': {
            'OverTime':          0.18 if overtime == 'Yes' else 0.05,
            'JobSatisfaction':   round((5 - int(job_sat)) / 4 * 0.20, 3) if job_sat else 0.10,
            'WorkLifeBalance':   round((5 - int(wlb)) / 4 * 0.15, 3) if wlb else 0.10,
            'YearsAtCompany':    round(min(int(years_at_co) / 20, 1.0) * 0.12, 3) if years_at_co else 0.08,
            'MonthlyIncome':     round(max(0, (5000 - int(float(monthly_income))) / 5000) * 0.10, 3) if monthly_income else 0.05,
        },
        'xai_explanation': {
            'top_factors': ['OverTime', 'JobSatisfaction', 'WorkLifeBalance'],
            'interpretation': f"이탈 확률 {structura_score:.1%}로 예측됨. 페르소나: {persona_name}"
        }
    }

    # Chronos
    trend_dir = 'increasing' if chronos_score > 0.5 else 'stable'
    chronos_result = {
        'risk_score': round(chronos_score, 4),
        'anomaly_score': round(min(chronos_score * 1.1, 1.0), 4),
        'trend_analysis': {
            'trend_direction': trend_dir,
            'volatility': 'high' if chronos_score > 0.6 else 'medium' if chronos_score > 0.3 else 'low',
            'seasonal_pattern': 'detected'
        },
        'xai_explanation': {
            'temporal_factors': ['workload_pattern', 'collaboration_trend'],
            'interpretation': f"시계열 위험도 {chronos_score:.1%}로 분석됨"
        },
        'attention_weights': {
            'recent_period': 0.45,
            'mid_period': 0.35,
            'early_period': 0.20
        },
        'sequence_importance': {
            'last_3_months': 0.50,
            'last_6_months': 0.30,
            'last_12_months': 0.20
        }
    }

    # Cognita
    cognita_result = {
        'overall_risk_score': round(cognita_score, 4),
        'network_centrality': round(cognita_score * 0.9, 4),
        'relationship_strength': round(1 - cognita_score * 0.5, 4),
        'influence_score': round(cognita_score * 0.7, 4)
    }

    # Sentio
    sentio_result = {
        'sentiment_score': round(1 - sentio_score, 4),
        'risk_score': round(sentio_score, 4),
        'keyword_analysis': {
            'negative_keywords': ['업무 부담', '번아웃'] if sentio_score > 0.5 else [],
            'positive_keywords': ['성장', '협업'] if sentio_score < 0.4 else []
        },
        'emotion_distribution': {
            'negative': round(sentio_score, 4),
            'neutral': round(max(0, 1 - sentio_score - 0.2), 4),
            'positive': round(max(0, 0.2 - sentio_score * 0.2), 4)
        }
    }

    # Agora
    agora_result = {
        'market_risk_score': round(agora_score, 4),
        'industry_trend': {
            'job_market_demand': 'high' if agora_score > 0.5 else 'moderate',
            'salary_competitiveness': 'low' if agora_score > 0.6 else 'moderate'
        },
        'job_market_analysis': {
            'external_opportunity': round(agora_score, 4)
        },
        'external_factors': {
            'industry_growth': 'stable',
            'market_competition': 'moderate'
        }
    }

    # ─── comprehensive_assessment ──────────────────────────────
    risk_label_kor = {'HIGH': '고위험군', 'MEDIUM': '주의군', 'LOW': '안전군'}.get(risk_level, '미분류')
    comprehensive_assessment = {
        'overall_risk_score': risk_score,
        'overall_risk_level': risk_level,
        'risk_category': risk_label_kor,
        'actual_attrition': attrition_label,
        'persona_code': persona_code,
        'persona_name': persona_name,
        'risk_tier': risk_tier,
        'agent_scores': {
            'structura': round(structura_score, 4),
            'chronos':   round(chronos_score, 4),
            'cognita':   round(cognita_score, 4),
            'sentio':    round(sentio_score, 4),
            'agora':     round(agora_score, 4),
        },
        'employee_profile': {
            'age': age,
            'gender': gender,
            'marital_status': marital,
            'department': dept,
            'job_role': job_role,
            'job_level': job_level,
            'education_field': education_field,
            'business_travel': business_travel,
            'monthly_income': monthly_income,
            'years_at_company': years_at_co,
            'job_satisfaction': job_sat,
            'work_life_balance': wlb,
            'performance_rating': perf_rating,
            'overtime': overtime,
        }
    }

    employee_result = {
        'employee_id': eid,
        'employee_number': eid,
        'department': dept,
        'job_role': job_role,
        'job_level': job_level,
        'risk_score': risk_score,
        'risk_level': risk_level.lower(),
        'analysis_timestamp': TIMESTAMP,
        'agent_results': {
            'structura': structura_result,
            'chronos':   chronos_result,
            'cognita':   cognita_result,
            'sentio':    sentio_result,
            'agora':     agora_result,
        },
        'analysis_result': {
            'comprehensive_assessment': comprehensive_assessment,
            'combined_analysis': {
                'integrated_assessment': comprehensive_assessment
            },
            'employee_data': {k: v for k, v in s.items()}  # raw Structura row
        }
    }

    # ─── 디렉토리 & 파일 저장 ─────────────────────────────────
    employee_dir = os.path.join(RESULTS_DIR, norm_dept, f'employee_{eid}')
    os.makedirs(employee_dir, exist_ok=True)

    # 1) batch_analysis_{timestamp}.json
    batch_file = os.path.join(employee_dir, f'batch_analysis_{TIMESTAMP}.json')
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump({
            'analysis_type': 'batch_analysis',
            'timestamp': TIMESTAMP,
            'employee_result': {
                'employee_id': eid,
                'department': dept,
                'risk_score': risk_score,
                'risk_level': risk_level.lower(),
                'analysis_timestamp': TIMESTAMP,
                'agent_results': employee_result['agent_results']
            },
            'applied_settings': {'source': 'test_data_draft'},
            'xai_included': True,
            'visualizations_generated': False
        }, f, indent=2, ensure_ascii=False)

    # 2) comprehensive_report.json
    comp_file = os.path.join(employee_dir, 'comprehensive_report.json')
    with open(comp_file, 'w', encoding='utf-8') as f:
        json.dump({
            'employee_id': eid,
            'department': dept,
            'job_role': job_role,
            'job_level': job_level,
            'analysis_timestamp': TIMESTAMP,
            'comprehensive_assessment': comprehensive_assessment,
            'rule_based_interpretation': {
                'risk_summary': f"{risk_label_kor} ({risk_score:.2f})",
                'key_drivers': [
                    f"Structura(HR): {structura_score:.2f}",
                    f"Chronos(시계열): {chronos_score:.2f}",
                    f"Sentio(감성): {sentio_score:.2f}",
                ],
                'recommended_action': '즉각적인 면담 필요' if risk_level == 'HIGH' else '모니터링 권장' if risk_level == 'MEDIUM' else '정기 점검'
            },
            'agent_results': employee_result['agent_results'],
            'source': 'test_data_draft'
        }, f, indent=2, ensure_ascii=False)

    # ─── 부서 집계 ─────────────────────────────────────────────
    if norm_dept not in department_results:
        department_results[norm_dept] = {
            'department': norm_dept,
            'total_employees': 0,
            'risk_distribution': {'고위험군': 0, '주의군': 0, '안전군': 0},
            'employees': []
        }
    department_results[norm_dept]['total_employees'] += 1
    department_results[norm_dept]['risk_distribution'][risk_label_kor] += 1
    department_results[norm_dept]['employees'].append({
        'employee_id': eid,
        'risk_score': risk_score,
        'risk_level': risk_level.lower(),
        'job_role': job_role
    })

    individual_results.append(employee_result)

print(f"✅ 개별 직원 파일 생성 완료: {len(individual_results):,}명")

# ─── 배치 요약 파일 저장 ───────────────────────────────────────
batch_summary = {
    'analysis_timestamp': TIMESTAMP,
    'total_employees': len(individual_results),
    'total_departments': len(department_results),
    'department_results': {
        dept: {k: v for k, v in info.items() if k != 'employees'}
        for dept, info in department_results.items()
    },
    'applied_settings': {'source': 'test_data_draft'},
    'summary_statistics': {
        'overall_risk_distribution': {
            '고위험군': sum(d['risk_distribution']['고위험군'] for d in department_results.values()),
            '주의군':   sum(d['risk_distribution']['주의군']   for d in department_results.values()),
            '안전군':   sum(d['risk_distribution']['안전군']   for d in department_results.values()),
        }
    },
    'source': 'test_data_draft'
}

batch_summary_file = os.path.join(RESULTS_DIR, 'batch_analysis', f'department_summary_{TIMESTAMP}.json')
with open(batch_summary_file, 'w', encoding='utf-8') as f:
    json.dump(batch_summary, f, indent=2, ensure_ascii=False)

print(f"✅ 배치 요약 저장: {batch_summary_file}")

# 결과 출력
print("\n📊 부서별 결과:")
for dept, info in sorted(department_results.items()):
    d = info['risk_distribution']
    print(f"  {dept}: 총 {info['total_employees']}명 | "
          f"고위험 {d['고위험군']} / 주의 {d['주의군']} / 안전 {d['안전군']}")

print(f"\n🎉 완료! app/results/ 에 {len(individual_results):,}명 결과 저장됨")
print(f"   타임스탬프: {TIMESTAMP}")
