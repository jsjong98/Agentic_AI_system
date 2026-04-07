# -*- coding: utf-8 -*-
"""
Agentic AI System - 계층적 결과 관리자
부서 → 직무 → 직급 → 직원 계층 구조로 결과를 체계적으로 저장하고 관리
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import base64
import io

logger = logging.getLogger(__name__)

class HierarchicalResultManager:
    """계층적 에이전틱 AI 시스템 결과 관리자"""
    
    def __init__(self, base_output_dir: str = "results"):
        self.base_output_dir = Path(base_output_dir)
        self.setup_directory_structure()
        
    def setup_directory_structure(self):
        """간소화된 디렉토리 구조 생성"""
        # 기본 구조 (간소화)
        directories = [
            "global_reports",      # 전체 종합 보고서
            "models",             # 저장된 모델들
            "temp"                # 임시 파일들
        ]
        
        for directory in directories:
            (self.base_output_dir / directory).mkdir(parents=True, exist_ok=True)
            
        logger.info(f"간소화된 결과 저장 디렉토리 구조 생성 완료: {self.base_output_dir}")
    
    def _sanitize_folder_name(self, name: str) -> str:
        """폴더명으로 사용할 수 있도록 문자열 정리"""
        if not name:
            return "Unknown"
        
        # 특수문자 제거 및 공백을 언더스코어로 변경
        import re
        sanitized = re.sub(r'[<>:"/\\|?*]', '', str(name))
        sanitized = re.sub(r'\s+', '_', sanitized)
        sanitized = sanitized.strip('_')
        
        return sanitized if sanitized else "Unknown"
    
    def _get_hierarchical_path(self, department: str, job_role: str, position: str, employee_id: str) -> Path:
        """완전한 계층적 경로 생성: Department/JobRole/JobLevel/employee_ID"""
        
        dept_clean = self._sanitize_folder_name(department)
        job_role_clean = self._sanitize_folder_name(job_role)
        position_clean = self._sanitize_folder_name(position)
        
        # 완전한 계층적 구조: results/Department/JobRole/JobLevel/employee_ID
        hierarchical_path = (
            self.base_output_dir / 
            dept_clean / 
            job_role_clean /
            position_clean /
            f"employee_{employee_id}"
        )
        
        return hierarchical_path
    
    def _update_department_index(self, department: str, employee_id: str, job_role: str, position: str):
        """계층적 구조의 인덱스 파일들 업데이트"""
        try:
            dept_clean = self._sanitize_folder_name(department)
            job_role_clean = self._sanitize_folder_name(job_role)
            position_clean = self._sanitize_folder_name(position)
            
            # 1. 부서 레벨 인덱스 업데이트
            dept_index_file = self.base_output_dir / dept_clean / "department_index.json"
            dept_dir = self.base_output_dir / dept_clean
            dept_dir.mkdir(parents=True, exist_ok=True)
            
            if dept_index_file.exists():
                with open(dept_index_file, 'r', encoding='utf-8') as f:
                    dept_index = json.load(f)
            else:
                dept_index = {
                    "department": department,
                    "created_at": datetime.now().isoformat(),
                    "structure_version": "hierarchical_v2.0",
                    "job_roles": {},
                    "statistics": {
                        "total_employees": 0,
                        "job_role_count": {},
                        "position_count": {}
                    }
                }
            
            # 2. 직무 레벨 인덱스 업데이트
            job_role_dir = dept_dir / job_role_clean
            job_role_dir.mkdir(parents=True, exist_ok=True)
            job_role_index_file = job_role_dir / "job_role_index.json"
            
            if job_role_index_file.exists():
                with open(job_role_index_file, 'r', encoding='utf-8') as f:
                    job_role_index = json.load(f)
            else:
                job_role_index = {
                    "department": department,
                    "job_role": job_role,
                    "created_at": datetime.now().isoformat(),
                    "job_levels": {},
                    "statistics": {
                        "total_employees": 0,
                        "job_level_count": {}
                    }
                }
            
            # 3. 직급 레벨 인덱스 업데이트
            position_dir = job_role_dir / position_clean
            position_dir.mkdir(parents=True, exist_ok=True)
            position_index_file = position_dir / "job_level_index.json"
            
            if position_index_file.exists():
                with open(position_index_file, 'r', encoding='utf-8') as f:
                    position_index = json.load(f)
            else:
                position_index = {
                    "department": department,
                    "job_role": job_role,
                    "job_level": position,
                    "created_at": datetime.now().isoformat(),
                    "employees": {},
                    "statistics": {
                        "total_employees": 0
                    }
                }
            
            # 직원 정보 업데이트
            employee_info = {
                "job_role": job_role,
                "job_level": position,
                "last_updated": datetime.now().isoformat(),
                "folder_path": f"{job_role_clean}/{position_clean}/employee_{employee_id}"
            }
            
            # 각 레벨별 인덱스에 직원 정보 추가
            position_index["employees"][employee_id] = employee_info
            
            # 직급 레벨 통계 업데이트
            position_index["statistics"]["total_employees"] = len(position_index["employees"])
            position_index["last_updated"] = datetime.now().isoformat()
            
            # 직무 레벨 인덱스 업데이트
            if position not in job_role_index["job_levels"]:
                job_role_index["job_levels"][position] = []
            if employee_id not in job_role_index["job_levels"][position]:
                job_role_index["job_levels"][position].append(employee_id)
            
            job_role_index["statistics"]["total_employees"] = sum(
                len(employees) for employees in job_role_index["job_levels"].values()
            )
            job_role_index["statistics"]["job_level_count"] = {
                level: len(employees) for level, employees in job_role_index["job_levels"].items()
            }
            job_role_index["last_updated"] = datetime.now().isoformat()
            
            # 부서 레벨 인덱스 업데이트
            if job_role not in dept_index["job_roles"]:
                dept_index["job_roles"][job_role] = {}
            if position not in dept_index["job_roles"][job_role]:
                dept_index["job_roles"][job_role][position] = []
            if employee_id not in dept_index["job_roles"][job_role][position]:
                dept_index["job_roles"][job_role][position].append(employee_id)
            
            # 부서 레벨 통계 업데이트
            total_employees = 0
            job_role_count = {}
            position_count = {}
            
            for role, positions in dept_index["job_roles"].items():
                role_total = 0
                for pos, employees in positions.items():
                    role_total += len(employees)
                    position_count[pos] = position_count.get(pos, 0) + len(employees)
                job_role_count[role] = role_total
                total_employees += role_total
            
            dept_index["statistics"]["total_employees"] = total_employees
            dept_index["statistics"]["job_role_count"] = job_role_count
            dept_index["statistics"]["position_count"] = position_count
            dept_index["last_updated"] = datetime.now().isoformat()
            
            # 각 레벨별 인덱스 파일 저장
            with open(position_index_file, 'w', encoding='utf-8') as f:
                json.dump(position_index, f, ensure_ascii=False, indent=2)
            
            with open(job_role_index_file, 'w', encoding='utf-8') as f:
                json.dump(job_role_index, f, ensure_ascii=False, indent=2)
            
            with open(dept_index_file, 'w', encoding='utf-8') as f:
                json.dump(dept_index, f, ensure_ascii=False, indent=2)
            
            logger.info(f"부서 인덱스 업데이트 완료: {department} - {employee_id}")
            
        except Exception as e:
            logger.error(f"부서 인덱스 업데이트 실패: {e}")
    
    def get_employees_by_job_role(self, department: str, job_role: str) -> List[str]:
        """특정 부서의 특정 직무 직원 목록 조회"""
        try:
            dept_clean = self._sanitize_folder_name(department)
            index_file = self.base_output_dir / dept_clean / "department_index.json"
            
            if not index_file.exists():
                return []
            
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            return index_data.get("job_roles", {}).get(job_role, [])
        except Exception as e:
            logger.error(f"직무별 직원 조회 실패: {e}")
            return []
    
    def get_employees_by_position(self, department: str, position: str) -> List[str]:
        """특정 부서의 특정 직급 직원 목록 조회"""
        try:
            dept_clean = self._sanitize_folder_name(department)
            index_file = self.base_output_dir / dept_clean / "department_index.json"
            
            if not index_file.exists():
                return []
            
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            return index_data.get("positions", {}).get(position, [])
        except Exception as e:
            logger.error(f"직급별 직원 조회 실패: {e}")
            return []
    
    def get_department_statistics(self, department: str) -> Dict:
        """부서 통계 조회"""
        try:
            dept_clean = self._sanitize_folder_name(department)
            index_file = self.base_output_dir / dept_clean / "department_index.json"
            
            if not index_file.exists():
                return {}
            
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            return index_data.get("statistics", {})
        except Exception as e:
            logger.error(f"부서 통계 조회 실패: {e}")
            return {}
    
    def save_employee_result(self, employee_id: str, employee_data: Dict, 
                           worker_results: Dict, department: str = None, 
                           job_role: str = None, position: str = None):
        """개별 직원 결과를 계층적 구조로 저장"""
        
        # 직원 데이터에서 정보 추출
        dept = department or employee_data.get('Department', 'Unknown')
        role = job_role or employee_data.get('JobRole', 'Unknown') 
        pos = position or employee_data.get('JobLevel', employee_data.get('Position', 'Unknown'))
        
        # 계층적 경로 생성
        employee_dir = self._get_hierarchical_path(dept, role, pos, employee_id)
        employee_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"직원 {employee_id} 결과 저장 경로: {employee_dir}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 기본 정보 저장 (간소화된 구조에서도 상세 정보 보존)
        employee_info = {
            "employee_id": employee_id,
            "department": dept,
            "job_role": role,
            "position": pos,
            "analysis_timestamp": timestamp,
            "folder_structure": {
                "simplified_path": f"{dept}/employee_{employee_id}",
                "original_hierarchy": {
                    "department": dept,
                    "job_role": role,
                    "position": pos
                }
            },
            "metadata": {
                "structure_version": "simplified_v1.0",
                "migration_date": timestamp,
                "data_preserved": True
            },
            "employee_data": employee_data
        }
        
        with open(employee_dir / "employee_info.json", 'w', encoding='utf-8') as f:
            json.dump(employee_info, f, ensure_ascii=False, indent=2)
        
        # 부서별 인덱스 파일 업데이트
        self._update_department_index(dept, employee_id, role, pos)
        
        # 2. 각 워커 결과 저장
        results_summary = {
            'employee_id': employee_id,
            'department': dept,
            'job_role': role,
            'position': pos,
            'analysis_timestamp': timestamp
        }
        
        # 시각화 폴더 생성
        viz_dir = employee_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Structura 결과
        if 'structura' in worker_results:
            structura_result = worker_results['structura']
            results_summary.update({
                'structura_attrition_prob': structura_result.get('attrition_probability', 0),
                'structura_prediction': structura_result.get('prediction', 'Unknown'),
                'structura_confidence': structura_result.get('confidence', 0)
            })
            
            with open(employee_dir / "structura_result.json", 'w', encoding='utf-8') as f:
                json.dump(structura_result, f, ensure_ascii=False, indent=2)
            
            # XAI 시각화 저장
            self._save_structura_visualizations(employee_id, structura_result, viz_dir)
        
        # Cognita 결과
        if 'cognita' in worker_results:
            cognita_result = worker_results['cognita']
            results_summary.update({
                'cognita_risk_score': cognita_result.get('overall_risk_score', 0),
                'cognita_risk_category': cognita_result.get('risk_category', 'Unknown'),
                'cognita_network_centrality': cognita_result.get('network_centrality', 0)
            })
            
            with open(employee_dir / "cognita_result.json", 'w', encoding='utf-8') as f:
                json.dump(cognita_result, f, ensure_ascii=False, indent=2)
        
        # Chronos 결과
        if 'chronos' in worker_results:
            chronos_result = worker_results['chronos']
            results_summary.update({
                'chronos_prediction': chronos_result.get('prediction', 'Unknown'),
                'chronos_probability': chronos_result.get('probability', 0),
                'chronos_trend': chronos_result.get('trend', 'Stable')
            })
            
            with open(employee_dir / "chronos_result.json", 'w', encoding='utf-8') as f:
                json.dump(chronos_result, f, ensure_ascii=False, indent=2)
            
            # Attention 시각화 저장
            self._save_chronos_visualizations(employee_id, chronos_result, viz_dir)
        
        # Sentio 결과
        if 'sentio' in worker_results:
            sentio_result = worker_results['sentio']
            results_summary.update({
                'sentio_risk_score': sentio_result.get('psychological_risk_score', 0),
                'sentio_risk_level': sentio_result.get('risk_level', 'MEDIUM'),
                'sentio_job_demands': sentio_result.get('jd_r_indicators', {}).get('job_demands_score', 0),
                'sentio_resources_deficiency': sentio_result.get('jd_r_indicators', {}).get('job_resources_deficiency_score', 0)
            })
            
            with open(employee_dir / "sentio_result.json", 'w', encoding='utf-8') as f:
                json.dump(sentio_result, f, ensure_ascii=False, indent=2)
        
        # Agora 결과
        if 'agora' in worker_results:
            agora_result = worker_results['agora']
            results_summary.update({
                'agora_score': agora_result.get('agora_score', 0),
                'agora_market_pressure': agora_result.get('market_pressure_index', 0),
                'agora_compensation_gap': agora_result.get('compensation_gap', 0),
                'agora_market_competitiveness': agora_result.get('market_competitiveness', 'MEDIUM')
            })
            
            with open(employee_dir / "agora_result.json", 'w', encoding='utf-8') as f:
                json.dump(agora_result, f, ensure_ascii=False, indent=2)
        
        # 3. 통합 결과 요약 CSV 저장
        summary_df = pd.DataFrame([results_summary])
        summary_df.to_csv(employee_dir / "analysis_summary.csv", index=False, encoding='utf-8-sig')
        
        # 4. 종합 레포트 생성 (Sentio 활용)
        if len(worker_results) >= 2:  # 최소 2개 워커 결과가 있을 때
            comprehensive_report = self._generate_comprehensive_report(employee_id, worker_results)
            with open(employee_dir / "comprehensive_report.json", 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"직원 {employee_id} 결과가 {employee_dir}에 저장되었습니다.")
        
        # 계층별 요약 업데이트
        self._update_hierarchical_summaries(dept, role, pos, employee_id, results_summary)
        
        return str(employee_dir)
    
    def _generate_comprehensive_report(self, employee_id: str, worker_results: Dict) -> Dict:
        """개별 직원의 종합 레포트 생성 (Sentio 활용)"""
        
        # 종합 위험도 계산 (각 워커의 점수를 가중평균)
        risk_scores = []
        
        # Structura: 퇴직 확률
        if 'structura' in worker_results and worker_results['structura'].get('attrition_probability'):
            risk_scores.append(('structura', worker_results['structura']['attrition_probability'], 0.3))
        
        # Cognita: 전체 위험도
        if 'cognita' in worker_results and worker_results['cognita'].get('overall_risk_score'):
            risk_scores.append(('cognita', worker_results['cognita']['overall_risk_score'], 0.25))
        
        # Chronos: 예측 확률
        if 'chronos' in worker_results and worker_results['chronos'].get('probability'):
            risk_scores.append(('chronos', worker_results['chronos']['probability'], 0.2))
        
        # Sentio: 심리적 위험 점수
        if 'sentio' in worker_results and worker_results['sentio'].get('psychological_risk_score'):
            risk_scores.append(('sentio', worker_results['sentio']['psychological_risk_score'], 0.25))
        
        # Agora: 시장 위험 점수
        if 'agora' in worker_results:
            agora_score = worker_results['agora'].get('agora_score', worker_results['agora'].get('market_risk_score', 0))
            if agora_score:
                risk_scores.append(('agora', agora_score, 0.2))
        
        # 가중평균 계산
        if risk_scores:
            weighted_sum = sum(score * weight for _, score, weight in risk_scores)
            total_weight = sum(weight for _, _, weight in risk_scores)
            comprehensive_risk_score = weighted_sum / total_weight
        else:
            comprehensive_risk_score = 0.5
        
        # 종합 위험 수준 결정
        if comprehensive_risk_score >= 0.7:
            overall_risk_level = "HIGH"
            risk_color = "🔴"
        elif comprehensive_risk_score >= 0.4:
            overall_risk_level = "MEDIUM" 
            risk_color = "🟡"
        else:
            overall_risk_level = "LOW"
            risk_color = "🟢"
        
        # 종합 레포트 구성
        comprehensive_report = {
            'employee_id': employee_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'comprehensive_assessment': {
                'overall_risk_score': round(comprehensive_risk_score, 3),
                'overall_risk_level': overall_risk_level,
                'risk_indicator': risk_color,
                'confidence_level': 'HIGH' if len(risk_scores) >= 3 else 'MEDIUM'
            },
            'worker_contributions': {
                worker: {'score': score, 'weight': weight} 
                for worker, score, weight in risk_scores
            },
            'rule_based_interpretation': self._generate_rule_based_interpretation(
                employee_id, comprehensive_risk_score, overall_risk_level, worker_results
            )
        }
        
        return comprehensive_report
    
    def _generate_rule_based_interpretation(self, employee_id: str, risk_score: float, 
                                          risk_level: str, worker_results: Dict) -> str:
        """규칙 기반 종합 해석 생성"""
        
        interpretation = f"""
=== 직원 {employee_id} 종합 분석 결과 ===

전체 위험도: {risk_level} ({risk_score:.3f}/1.0)

🔍 워커별 상세 분석:
"""
        
        # 각 워커 결과 요약
        if 'structura' in worker_results:
            structura = worker_results['structura']
            prob = structura.get('attrition_probability', 0)
            pred = structura.get('prediction', 'Unknown')
            interpretation += f"📈 구조적 분석 (Structura): 퇴직 확률 {prob:.1%}, 예측 '{pred}'\n"
        
        if 'cognita' in worker_results:
            cognita = worker_results['cognita']
            score = cognita.get('overall_risk_score', 0)
            category = cognita.get('risk_category', 'Unknown')
            interpretation += f"🌐 관계적 분석 (Cognita): 위험도 {score:.3f}, 카테고리 '{category}'\n"
        
        if 'chronos' in worker_results:
            chronos = worker_results['chronos']
            prob = chronos.get('probability', 0)
            trend = chronos.get('trend', 'Stable')
            interpretation += f"⏰ 시계열 분석 (Chronos): 확률 {prob:.1%}, 트렌드 '{trend}'\n"
        
        if 'sentio' in worker_results:
            sentio = worker_results['sentio']
            psych_score = sentio.get('psychological_risk_score', 0)
            level = sentio.get('risk_level', 'MEDIUM')
            interpretation += f"🧠 심리적 분석 (Sentio): 위험도 {psych_score:.3f}, 수준 '{level}'\n"
        
        if 'agora' in worker_results:
            agora = worker_results['agora']
            agora_score = agora.get('agora_score', agora.get('market_risk_score', 0))
            market_comp = agora.get('market_competitiveness', 'MEDIUM')
            interpretation += f"💼 시장 분석 (Agora): 위험도 {agora_score:.3f}, 경쟁력 '{market_comp}'\n"
        
        interpretation += f"\n💡 권장 조치:\n"
        
        # 위험 수준별 권장사항
        if risk_level == 'HIGH':
            interpretation += """🚨 즉시 개입 필요:
- 상급자와의 긴급 면담 실시
- 업무 조정 및 지원 방안 검토
- 정기적 모니터링 체계 구축"""
        elif risk_level == 'MEDIUM':
            interpretation += """⚠️ 예방적 관리 필요:
- 정기적 상담 및 피드백 제공
- 업무 환경 개선 검토
- 스트레스 관리 프로그램 참여 권장"""
        else:
            interpretation += """✅ 현재 상태 유지:
- 정기적 모니터링 지속
- 긍정적 요소 강화
- 성장 기회 제공 검토"""
        
        return interpretation.strip()
    
    def _update_hierarchical_summaries(self, department: str, job_role: str, position: str, 
                                     employee_id: str, summary_data: Dict):
        """계층별 요약 파일들 업데이트"""
        
        # 1. 직급별 요약 업데이트
        self._update_position_summary(department, job_role, position, summary_data)
        
        # 2. 직무별 요약 업데이트  
        self._update_job_role_summary(department, job_role, summary_data)
        
        # 3. 부서별 요약 업데이트
        self._update_department_summary(department, summary_data)
        
        # 4. 전체 요약 업데이트
        self._update_global_summary(summary_data)
    
    def _update_position_summary(self, department: str, job_role: str, position: str, summary_data: Dict):
        """직급별 요약 CSV 업데이트"""
        dept_clean = self._sanitize_folder_name(department)
        role_clean = self._sanitize_folder_name(job_role)
        pos_clean = self._sanitize_folder_name(position)
        
        position_dir = (
            self.base_output_dir / "departments" / dept_clean / 
            "job_roles" / role_clean / "positions" / pos_clean
        )
        position_dir.mkdir(parents=True, exist_ok=True)
        
        summary_path = position_dir / "position_summary.csv"
        self._append_to_csv(summary_path, summary_data)
        
        logger.debug(f"직급 '{position}' 요약 업데이트: {summary_path}")
    
    def _update_job_role_summary(self, department: str, job_role: str, summary_data: Dict):
        """직무별 요약 CSV 업데이트"""
        dept_clean = self._sanitize_folder_name(department)
        role_clean = self._sanitize_folder_name(job_role)
        
        job_role_dir = (
            self.base_output_dir / "departments" / dept_clean / "job_roles" / role_clean
        )
        job_role_dir.mkdir(parents=True, exist_ok=True)
        
        summary_path = job_role_dir / "job_role_summary.csv"
        self._append_to_csv(summary_path, summary_data)
        
        logger.debug(f"직무 '{job_role}' 요약 업데이트: {summary_path}")
    
    def _update_department_summary(self, department: str, summary_data: Dict):
        """부서별 요약 CSV 업데이트"""
        dept_clean = self._sanitize_folder_name(department)
        
        dept_dir = self.base_output_dir / "departments" / dept_clean
        dept_dir.mkdir(parents=True, exist_ok=True)
        
        summary_path = dept_dir / "department_summary.csv"
        self._append_to_csv(summary_path, summary_data)
        
        logger.debug(f"부서 '{department}' 요약 업데이트: {summary_path}")
    
    def _update_global_summary(self, summary_data: Dict):
        """전체 요약 CSV 업데이트"""
        global_dir = self.base_output_dir / "global_reports"
        global_dir.mkdir(parents=True, exist_ok=True)
        
        summary_path = global_dir / "global_summary.csv"
        self._append_to_csv(summary_path, summary_data)
        
        logger.debug(f"전체 요약 업데이트: {summary_path}")
    
    def _append_to_csv(self, csv_path: Path, data: Dict):
        """CSV 파일에 데이터 추가"""
        df_new = pd.DataFrame([data])
        
        if csv_path.exists():
            df_existing = pd.read_csv(csv_path, encoding='utf-8-sig')
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        df_combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.debug(f"CSV 업데이트 완료: {csv_path} ({len(df_combined)}행)")
    
    def _save_structura_visualizations(self, employee_id: str, structura_result: Dict, viz_dir: Path):
        """Structura XAI 시각화 저장"""
        try:
            # Feature Importance
            if 'feature_importance_plot' in structura_result:
                self._save_base64_image(
                    structura_result['feature_importance_plot'], 
                    viz_dir / "structura_feature_importance.png"
                )
            
            # SHAP Analysis
            if 'shap_plot' in structura_result:
                self._save_base64_image(
                    structura_result['shap_plot'], 
                    viz_dir / "structura_shap_analysis.png"
                )
            
            # LIME Explanation
            if 'lime_plot' in structura_result:
                self._save_base64_image(
                    structura_result['lime_plot'], 
                    viz_dir / "structura_lime_explanation.png"
                )
                
            logger.debug(f"Structura 시각화 저장 완료: {viz_dir}")
            
        except Exception as e:
            logger.error(f"Structura 시각화 저장 실패: {e}")
    
    def _save_chronos_visualizations(self, employee_id: str, chronos_result: Dict, viz_dir: Path):
        """Chronos Attention 시각화 저장"""
        try:
            # Temporal Attention
            if 'temporal_attention_plot' in chronos_result:
                self._save_base64_image(
                    chronos_result['temporal_attention_plot'], 
                    viz_dir / "chronos_temporal_attention.png"
                )
            
            # Feature Attention
            if 'feature_attention_plot' in chronos_result:
                self._save_base64_image(
                    chronos_result['feature_attention_plot'], 
                    viz_dir / "chronos_feature_attention.png"
                )
                
            logger.debug(f"Chronos 시각화 저장 완료: {viz_dir}")
            
        except Exception as e:
            logger.error(f"Chronos 시각화 저장 실패: {e}")
    
    def _save_base64_image(self, base64_string: str, output_path: Path):
        """Base64 인코딩된 이미지를 파일로 저장"""
        try:
            # "data:image/png;base64," 접두사 제거
            if "," in base64_string:
                base64_string = base64_string.split(",")[1]
            
            img_data = base64.b64decode(base64_string)
            with open(output_path, 'wb') as f:
                f.write(img_data)
                
            logger.debug(f"이미지 저장 완료: {output_path}")
            
        except Exception as e:
            logger.error(f"이미지 저장 실패 ({output_path}): {e}")
    
    def get_employee_results(self, employee_id: str, department: str = None, 
                           job_role: str = None, position: str = None) -> Dict:
        """특정 직원의 모든 결과 조회 (계층적 경로)"""
        
        if all([department, job_role, position]):
            # 정확한 경로로 조회
            employee_dir = self._get_hierarchical_path(department, job_role, position, employee_id)
        else:
            # 전체 검색 (느림)
            employee_dir = self._find_employee_directory(employee_id)
        
        if not employee_dir or not employee_dir.exists():
            return {"error": f"직원 {employee_id}의 결과가 없습니다."}
        
        results = {"employee_id": employee_id, "path": str(employee_dir), "files": {}}
        
        # 각 워커 결과 파일 읽기
        for worker in ['structura', 'cognita', 'chronos', 'sentio', 'agora']:
            result_file = employee_dir / f"{worker}_result.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    results['files'][f'{worker}_result'] = json.load(f)
        
        # 종합 레포트
        comprehensive_file = employee_dir / "comprehensive_report.json"
        if comprehensive_file.exists():
            with open(comprehensive_file, 'r', encoding='utf-8') as f:
                results['files']['comprehensive_report'] = json.load(f)
        
        # 시각화 파일 목록
        viz_dir = employee_dir / "visualizations"
        if viz_dir.exists():
            results['files']['visualizations'] = [f.name for f in viz_dir.iterdir() if f.is_file()]
        
        return results
    
    def _find_employee_directory(self, employee_id: str) -> Optional[Path]:
        """직원 디렉토리 전체 검색"""
        for dept_dir in (self.base_output_dir / "departments").iterdir():
            if dept_dir.is_dir():
                for role_dir in (dept_dir / "job_roles").iterdir():
                    if role_dir.is_dir():
                        for pos_dir in (role_dir / "positions").iterdir():
                            if pos_dir.is_dir():
                                emp_dir = pos_dir / "employees" / f"employee_{employee_id}"
                                if emp_dir.exists():
                                    return emp_dir
        return None
    
    def get_hierarchy_summary(self) -> Dict:
        """전체 계층 구조 요약 조회"""
        summary = {
            "total_departments": 0,
            "total_job_roles": 0,
            "total_positions": 0,
            "total_employees": 0,
            "hierarchy": {}
        }
        
        departments_dir = self.base_output_dir / "departments"
        if not departments_dir.exists():
            return summary
        
        for dept_dir in departments_dir.iterdir():
            if dept_dir.is_dir():
                dept_name = dept_dir.name
                summary["total_departments"] += 1
                summary["hierarchy"][dept_name] = {"job_roles": {}}
                
                job_roles_dir = dept_dir / "job_roles"
                if job_roles_dir.exists():
                    for role_dir in job_roles_dir.iterdir():
                        if role_dir.is_dir():
                            role_name = role_dir.name
                            summary["total_job_roles"] += 1
                            summary["hierarchy"][dept_name]["job_roles"][role_name] = {"positions": {}}
                            
                            positions_dir = role_dir / "positions"
                            if positions_dir.exists():
                                for pos_dir in positions_dir.iterdir():
                                    if pos_dir.is_dir():
                                        pos_name = pos_dir.name
                                        summary["total_positions"] += 1
                                        
                                        employees_dir = pos_dir / "employees"
                                        employee_count = 0
                                        if employees_dir.exists():
                                            employee_count = len([d for d in employees_dir.iterdir() if d.is_dir()])
                                        
                                        summary["hierarchy"][dept_name]["job_roles"][role_name]["positions"][pos_name] = {
                                            "employee_count": employee_count
                                        }
                                        summary["total_employees"] += employee_count
        
        return summary

# 전역 인스턴스 - app/results 경로로 설정
import os
# 현재 파일: app/Supervisor/hierarchical_result_manager.py
# 프로젝트 루트: app/Supervisor/hierarchical_result_manager.py -> app -> 프로젝트 루트
current_file = os.path.abspath(__file__)
app_dir = os.path.dirname(os.path.dirname(current_file))  # app 폴더
results_path = os.path.join(app_dir, 'results')
hierarchical_result_manager = HierarchicalResultManager(base_output_dir=results_path)
