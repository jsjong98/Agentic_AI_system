# -*- coding: utf-8 -*-
"""
Agentic AI System - 결과 관리자
모든 워커 에이전트의 결과를 체계적으로 저장하고 관리하는 시스템
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# matplotlib과 seaborn을 조건부 import (GUI 백엔드 문제 방지)
try:
    import matplotlib
    matplotlib.use('Agg')  # GUI 없는 백엔드 사용
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: 시각화 라이브러리 import 실패: {e}")
    VISUALIZATION_AVAILABLE = False
    plt = None
    sns = None

logger = logging.getLogger(__name__)

class AgenticResultManager:
    """에이전틱 AI 시스템 결과 관리자"""
    
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
        """간소화된 경로 생성: 부서/직원"""
        
        dept_clean = self._sanitize_folder_name(department)
        
        # 간소화된 구조: results/부서명/employee_ID
        hierarchical_path = (
            self.base_output_dir / 
            dept_clean / 
            f"employee_{employee_id}"
        )
        
        return hierarchical_path
    
    def _update_department_index(self, department: str, employee_id: str, job_role: str, position: str):
        """부서별 직원 인덱스 파일 업데이트"""
        try:
            dept_clean = self._sanitize_folder_name(department)
            index_file = self.base_output_dir / dept_clean / "department_index.json"
            
            # 기존 인덱스 로드
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
            else:
                index_data = {
                    "department": department,
                    "created_at": datetime.now().isoformat(),
                    "structure_version": "simplified_v1.0",
                    "employees": {},
                    "job_roles": {},
                    "positions": {},
                    "statistics": {
                        "total_employees": 0,
                        "job_role_count": {},
                        "position_count": {}
                    }
                }
            
            # 직원 정보 업데이트
            index_data["employees"][employee_id] = {
                "job_role": job_role,
                "position": position,
                "last_updated": datetime.now().isoformat(),
                "folder_path": f"employee_{employee_id}"
            }
            
            # Job Role 인덱스 업데이트
            if job_role not in index_data["job_roles"]:
                index_data["job_roles"][job_role] = []
            if employee_id not in index_data["job_roles"][job_role]:
                index_data["job_roles"][job_role].append(employee_id)
            
            # Position 인덱스 업데이트
            if position not in index_data["positions"]:
                index_data["positions"][position] = []
            if employee_id not in index_data["positions"][position]:
                index_data["positions"][position].append(employee_id)
            
            # 통계 업데이트
            index_data["statistics"]["total_employees"] = len(index_data["employees"])
            index_data["statistics"]["job_role_count"] = {
                role: len(employees) for role, employees in index_data["job_roles"].items()
            }
            index_data["statistics"]["position_count"] = {
                pos: len(employees) for pos, employees in index_data["positions"].items()
            }
            index_data["last_updated"] = datetime.now().isoformat()
            
            # 인덱스 파일 저장
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"부서 인덱스 업데이트 완료: {department} - {employee_id}")
            
        except Exception as e:
            logger.error(f"부서 인덱스 업데이트 실패: {e}")
    
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
        
        # 1. 기본 정보 저장
        employee_info = {
            "employee_id": employee_id,
            "department": department,
            "position": position,
            "analysis_timestamp": timestamp,
            "employee_data": employee_data
        }
        
        with open(employee_dir / "employee_info.json", 'w', encoding='utf-8') as f:
            json.dump(employee_info, f, ensure_ascii=False, indent=2)
        
        # 부서별 인덱스 파일 업데이트
        self._update_department_index(dept, employee_id, role, pos)
        
        # 2. 각 워커 에이전트 결과 저장
        results_summary = {"employee_id": employee_id, "timestamp": timestamp}
        
        # Structura 결과
        if 'structura' in worker_results:
            structura_result = worker_results['structura']
            results_summary.update({
                'attrition_probability': structura_result.get('probability', 0),
                'attrition_prediction': structura_result.get('prediction', 0),
                'risk_level': 'HIGH' if structura_result.get('probability', 0) > 0.7 else 'MEDIUM' if structura_result.get('probability', 0) > 0.3 else 'LOW'
            })
            
            # Structura 상세 결과 저장
            with open(employee_dir / "structura_result.json", 'w', encoding='utf-8') as f:
                json.dump(structura_result, f, ensure_ascii=False, indent=2)
            
            # xAI 시각화 저장
            self._save_structura_visualizations(employee_id, structura_result, employee_dir)
        
        # Cognita 결과
        if 'cognita' in worker_results:
            cognita_result = worker_results['cognita']
            results_summary.update({
                'network_risk_score': cognita_result.get('overall_risk_score', 0),
                'social_isolation': cognita_result.get('social_isolation_index', 0),
                'network_centrality': cognita_result.get('network_centrality_score', 0)
            })
            
            with open(employee_dir / "cognita_result.json", 'w', encoding='utf-8') as f:
                json.dump(cognita_result, f, ensure_ascii=False, indent=2)
        
        # Chronos 결과
        if 'chronos' in worker_results:
            chronos_result = worker_results['chronos']
            results_summary.update({
                'timeseries_prediction': chronos_result.get('prediction', 0),
                'timeseries_probability': chronos_result.get('probability', 0)
            })
            
            with open(employee_dir / "chronos_result.json", 'w', encoding='utf-8') as f:
                json.dump(chronos_result, f, ensure_ascii=False, indent=2)
            
            # Attention 시각화 저장
            self._save_chronos_visualizations(employee_id, chronos_result, employee_dir)
        
        # Sentio 결과
        if 'sentio' in worker_results:
            sentio_result = worker_results['sentio']
            results_summary.update({
                'sentiment_score': sentio_result.get('sentiment_score', 0),
                'risk_keywords_count': len(sentio_result.get('risk_keywords', [])),
                'text_risk_level': sentio_result.get('risk_level', 'UNKNOWN')
            })
            
            with open(employee_dir / "sentio_result.json", 'w', encoding='utf-8') as f:
                json.dump(sentio_result, f, ensure_ascii=False, indent=2)
        
        # 3. 통합 결과 요약 CSV 저장
        summary_df = pd.DataFrame([results_summary])
        summary_df.to_csv(employee_dir / "analysis_summary.csv", index=False, encoding='utf-8-sig')
        
        # 4. 부서별/직급별 집계에 추가
        if department:
            self._add_to_department_summary(department, results_summary)
        if position:
            self._add_to_position_summary(position, results_summary)
        
        logger.info(f"직원 {employee_id} 결과 저장 완료: {employee_dir}")
        return employee_dir
    
    def _save_structura_visualizations(self, employee_id: str, result: Dict, output_dir: Path):
        """Structura xAI 시각화 저장"""
        if not VISUALIZATION_AVAILABLE:
            logger.warning("시각화 라이브러리가 없어 시각화를 건너뜁니다.")
            return
            
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        try:
            # Feature Importance 시각화
            if 'feature_importance' in result:
                plt.figure(figsize=(12, 8))
                features = list(result['feature_importance'].keys())[:15]
                importances = list(result['feature_importance'].values())[:15]
                
                plt.barh(features, importances)
                plt.title(f'Feature Importance - Employee {employee_id}')
                plt.xlabel('Importance Score')
                plt.tight_layout()
                plt.savefig(viz_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # SHAP 값 시각화
            if 'shap_values' in result:
                plt.figure(figsize=(12, 8))
                features = list(result['shap_values'].keys())[:15]
                shap_vals = list(result['shap_values'].values())[:15]
                
                colors = ['red' if x > 0 else 'blue' for x in shap_vals]
                plt.barh(features, shap_vals, color=colors, alpha=0.7)
                plt.title(f'SHAP Values - Employee {employee_id}')
                plt.xlabel('SHAP Value (Impact on Prediction)')
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                plt.tight_layout()
                plt.savefig(viz_dir / "shap_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.warning(f"Structura 시각화 저장 실패: {e}")
    
    def _save_chronos_visualizations(self, employee_id: str, result: Dict, output_dir: Path):
        """Chronos Attention 시각화 저장"""
        if not VISUALIZATION_AVAILABLE:
            logger.warning("시각화 라이브러리가 없어 시각화를 건너뜁니다.")
            return
            
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        try:
            # Temporal Attention 시각화
            if 'temporal_attention' in result:
                plt.figure(figsize=(12, 6))
                attention_weights = result['temporal_attention']
                time_steps = [f"Week -{len(attention_weights)-i}" for i in range(len(attention_weights))]
                
                plt.plot(time_steps, attention_weights, marker='o', linewidth=2, markersize=8)
                plt.title(f'Temporal Attention Weights - Employee {employee_id}')
                plt.xlabel('Time Steps')
                plt.ylabel('Attention Weight')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(viz_dir / "temporal_attention.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Feature Attention 시각화
            if 'feature_attention' in result:
                plt.figure(figsize=(12, 8))
                features = list(result['feature_attention'].keys())[:15]
                attentions = list(result['feature_attention'].values())[:15]
                
                plt.barh(features, attentions, color='orange', alpha=0.7)
                plt.title(f'Feature Attention Weights - Employee {employee_id}')
                plt.xlabel('Attention Weight')
                plt.tight_layout()
                plt.savefig(viz_dir / "feature_attention.png", dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.warning(f"Chronos 시각화 저장 실패: {e}")
    
    def _add_to_department_summary(self, department: str, employee_summary: Dict):
        """부서별 요약에 직원 결과 추가"""
        dept_dir = self.base_output_dir / "departments" / department
        dept_dir.mkdir(parents=True, exist_ok=True)
        
        summary_file = dept_dir / "department_summary.csv"
        
        # 기존 파일이 있으면 읽어서 추가, 없으면 새로 생성
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            df = pd.concat([df, pd.DataFrame([employee_summary])], ignore_index=True)
        else:
            df = pd.DataFrame([employee_summary])
        
        df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    
    def _add_to_position_summary(self, position: str, employee_summary: Dict):
        """직급별 요약에 직원 결과 추가"""
        pos_dir = self.base_output_dir / "positions" / position
        pos_dir.mkdir(parents=True, exist_ok=True)
        
        summary_file = pos_dir / "position_summary.csv"
        
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            df = pd.concat([df, pd.DataFrame([employee_summary])], ignore_index=True)
        else:
            df = pd.DataFrame([employee_summary])
        
        df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    
    def generate_department_report(self, department: str) -> Dict:
        """부서별 종합 보고서 생성"""
        dept_dir = self.base_output_dir / "departments" / department
        summary_file = dept_dir / "department_summary.csv"
        
        if not summary_file.exists():
            return {"error": f"부서 {department}의 데이터가 없습니다."}
        
        df = pd.read_csv(summary_file)
        
        # 통계 계산
        report = {
            "department": department,
            "total_employees": len(df),
            "analysis_date": datetime.now().isoformat(),
            "risk_distribution": {
                "HIGH": len(df[df['risk_level'] == 'HIGH']),
                "MEDIUM": len(df[df['risk_level'] == 'MEDIUM']),
                "LOW": len(df[df['risk_level'] == 'LOW'])
            },
            "average_scores": {
                "attrition_probability": df['attrition_probability'].mean(),
                "network_risk_score": df['network_risk_score'].mean() if 'network_risk_score' in df.columns else 0,
                "sentiment_score": df['sentiment_score'].mean() if 'sentiment_score' in df.columns else 0
            },
            "high_risk_employees": df[df['risk_level'] == 'HIGH']['employee_id'].tolist()
        }
        
        # 보고서 저장
        report_file = self.base_output_dir / "reports" / f"{department}_report.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report
    
    def get_employee_results(self, employee_id: str) -> Dict:
        """특정 직원의 모든 결과 조회"""
        employee_dir = self.base_output_dir / "employees" / f"employee_{employee_id}"
        
        if not employee_dir.exists():
            return {"error": f"직원 {employee_id}의 결과가 없습니다."}
        
        results = {}
        
        # 각 워커 결과 파일 읽기
        for worker in ['structura', 'cognita', 'chronos', 'sentio']:
            result_file = employee_dir / f"{worker}_result.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    results[worker] = json.load(f)
        
        # 요약 정보 읽기
        summary_file = employee_dir / "analysis_summary.csv"
        if summary_file.exists():
            summary_df = pd.read_csv(summary_file)
            results['summary'] = summary_df.iloc[0].to_dict()
        
        return results
    
    def list_available_visualizations(self, employee_id: str) -> List[str]:
        """특정 직원의 사용 가능한 시각화 파일 목록"""
        employee_dir = self.base_output_dir / "employees" / f"employee_{employee_id}" / "visualizations"
        
        if not employee_dir.exists():
            return []
        
        viz_files = []
        for file in employee_dir.glob("*.png"):
            viz_files.append(str(file))
        
        return viz_files

# 전역 결과 관리자 인스턴스
result_manager = AgenticResultManager()
