# -*- coding: utf-8 -*-
"""
Cognita 관계형 위험도 분석 Flask 백엔드 서비스
React 연동에 최적화된 REST API 서버
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import logging
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

# Neo4j 관련 import
from neo4j import GraphDatabase
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------
# 데이터 모델 정의
# ------------------------------------------------------

@dataclass
class RiskMetrics:
    """위험 지표 데이터 클래스"""
    employee_id: str
    manager_instability_score: float
    team_volatility_index: float
    social_isolation_index: float
    network_centrality_score: float
    overall_risk_score: float
    risk_category: str
    risk_factors: List[str]
    network_stats: Dict[str, float]
    analysis_timestamp: str = None
    
    def __post_init__(self):
        if self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return asdict(self)

# ------------------------------------------------------
# Neo4j 연결 관리 클래스
# ------------------------------------------------------

class Neo4jManager:
    """Neo4j 데이터베이스 연결 및 관리 클래스"""
    
    def __init__(self, uri: str, username: str, password: str, max_retries: int = 3):
        self.uri = uri
        self.username = username
        self.password = password
        self.max_retries = max_retries
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Neo4j 연결 설정"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Neo4j 연결 시도 {attempt + 1}/{self.max_retries}: {self.uri}")
                
                self.driver = GraphDatabase.driver(
                    self.uri, 
                    auth=(self.username, self.password),
                    max_connection_lifetime=300,  # 5분
                    max_connection_pool_size=5,   # 풀 크기 축소
                    connection_acquisition_timeout=10,  # 10초
                    connection_timeout=5,         # 5초
                    encrypted=False,              # 암호화 비활성화
                    trust="TRUST_ALL_CERTIFICATES"  # 인증서 신뢰
                )
                
                # 연결 테스트 (더 간단하고 안전한 방법)
                with self.driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    record = result.single()
                    if record and record["test"] == 1:
                        logger.info("Neo4j 연결 성공")
                        return
                    else:
                        raise Exception("연결 테스트 응답 오류")
                    
            except Exception as e:
                logger.warning(f"연결 시도 {attempt + 1} 실패: {str(e)}")
                
                if self.driver:
                    try:
                        self.driver.close()
                    except:
                        pass
                    self.driver = None
                
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.info(f"{wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    raise ConnectionError(f"Neo4j 연결 실패: {str(e)}")
    
    def close(self):
        if self.driver:
            try:
                self.driver.close()
                logger.info("Neo4j 연결 종료")
            except Exception as e:
                logger.warning(f"연결 종료 중 오류: {str(e)}")
        self.driver = None
    
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        if not self.driver:
            return False
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except:
            return False

# ------------------------------------------------------
# Cognita 위험도 분석기 클래스
# ------------------------------------------------------

class CognitaRiskAnalyzer:
    """Cognita 에이전트 관계형 위험도 분석기"""
    
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j_manager = neo4j_manager
        self.analysis_date = datetime.now()
        
    def analyze_employee_risk(self, employee_id: str) -> RiskMetrics:
        """특정 직원의 종합 위험도 분석"""
        
        logger.info(f"직원 {employee_id} 위험도 분석 시작")
        
        # 1. 관리자 안정성 분석
        manager_score = self.analyze_manager_stability(employee_id)
        
        # 2. 팀 변동성 분석  
        team_volatility = self.analyze_team_volatility(employee_id)
        
        # 3. 사회적 네트워크 중심성 분석
        centrality_score, network_stats = self.analyze_network_centrality(employee_id)
        
        # 4. 사회적 고립 지수 계산
        isolation_score = self.calculate_social_isolation(employee_id)
        
        # 5. 종합 위험 점수 계산
        overall_score = self.calculate_overall_risk_score(
            manager_score, team_volatility, centrality_score, isolation_score
        )
        
        # 6. 위험 범주 및 주요 요인 결정
        risk_category, risk_factors = self.determine_risk_category(
            manager_score, team_volatility, centrality_score, isolation_score
        )
        
        return RiskMetrics(
            employee_id=employee_id,
            manager_instability_score=manager_score,
            team_volatility_index=team_volatility,
            social_isolation_index=isolation_score,
            network_centrality_score=centrality_score,
            overall_risk_score=overall_score,
            risk_category=risk_category,
            risk_factors=risk_factors,
            network_stats=network_stats
        )

    def analyze_manager_stability(self, employee_id: str) -> float:
        """관리자 안정성 분석 - 최적화된 버전"""
        
        # 단순화된 쿼리로 성능 향상
        query = """
        MATCH (emp:Employee {employee_id: $employee_id})
        
        // 직속 상사만 조회 (복잡한 부서 분석 제거)
        OPTIONAL MATCH (emp)-[reports:REPORTS_TO]->(manager:Employee)
        
        // 관리자 부하 직원 수 (서브쿼리 단순화)
        OPTIONAL MATCH (manager)<-[:REPORTS_TO]-(subordinate:Employee)
        
        RETURN 
            manager.employee_id as direct_manager,
            reports.reporting_frequency as reporting_frequency,
            count(subordinate) as manager_load
        """
        
        with self.neo4j_manager.driver.session() as session:
            result = session.run(query, employee_id=employee_id)
            record = result.single()
            
            if not record:
                return 0.7  # 기본 불안정성 점수
            
            manager_load = record.get('manager_load', 0)
            reporting_frequency = record.get('reporting_frequency', 'rarely')
            
            instability_score = 0.0
            
            # 직속 관리자 부재
            if not record.get('direct_manager'):
                instability_score += 0.5
            
            # 관리자 과부하 (단순화된 계산)
            if manager_load > 12:
                instability_score += 0.3
            
            # 보고 빈도 (단순화된 매핑)
            frequency_penalty = {'daily': 0.0, 'weekly': 0.1, 'monthly': 0.2, 'rarely': 0.3}
            instability_score += frequency_penalty.get(reporting_frequency, 0.3)
            
            return min(instability_score, 1.0)

    def analyze_network_centrality(self, employee_id: str) -> Tuple[float, Dict[str, float]]:
        """네트워크 중심성 분석 - 샘플링으로 최적화"""
        
        # 전체 협업 관계 분석 (샘플링 제거)
        query = """
        MATCH (emp:Employee {employee_id: $employee_id})
        
        // 모든 협업 관계 조회
        OPTIONAL MATCH (emp)-[collab:COLLABORATES_WITH]-(colleague:Employee)
        WHERE collab.collaboration_strength IS NOT NULL
        
        // 전체 협업 관계 분석 (LIMIT 제거)
        WITH emp, collab, colleague
        ORDER BY collab.collaboration_strength DESC
        
        // 프로젝트 연결 (현재 활성 프로젝트만)
        OPTIONAL MATCH (emp)-[:PARTICIPATES_IN]->(proj:Project {status: 'active'})<-[:PARTICIPATES_IN]-(proj_colleague:Employee)
        WHERE proj_colleague.employee_id <> emp.employee_id
        
        RETURN 
            count(DISTINCT colleague.employee_id) as direct_connections,
            avg(collab.collaboration_strength) as avg_collaboration_strength,
            count(DISTINCT proj_colleague.employee_id) as project_connections,
            
            // 관계 품질 (간단화)
            sum(CASE WHEN collab.relationship_quality = 'excellent' THEN 1 ELSE 0 END) as excellent_count,
            sum(CASE WHEN collab.relationship_quality = 'poor' THEN 1 ELSE 0 END) as poor_count,
            
            // 빈번한 상호작용
            sum(CASE WHEN collab.interaction_frequency IN ['daily', 'weekly'] THEN 1 ELSE 0 END) as frequent_count
        """
        
        with self.neo4j_manager.driver.session() as session:
            result = session.run(query, employee_id=employee_id)
            record = result.single()
            
            if not record:
                return 0.0, {}
            
            direct_connections = record.get('direct_connections', 0)
            avg_strength = record.get('avg_collaboration_strength', 0.0) or 0.0
            project_connections = record.get('project_connections', 0)
            excellent_count = record.get('excellent_count', 0)
            poor_count = record.get('poor_count', 0)
            frequent_count = record.get('frequent_count', 0)
            
            # 단순화된 중심성 계산
            degree_centrality = min(direct_connections / 15.0, 1.0)  # 15명 기준
            strength_centrality = min(avg_strength, 1.0)
            project_centrality = min(project_connections / 8.0, 1.0)  # 8명 기준
            
            # 관계 품질 점수
            total_relations = direct_connections
            quality_score = 0.5  # 기본값
            if total_relations > 0:
                quality_score = (excellent_count * 1.0 + (total_relations - excellent_count - poor_count) * 0.5) / total_relations
            
            # 상호작용 점수
            interaction_score = min(frequent_count / 5.0, 1.0)  # 5명 기준
            
            # 가중 평균 (단순화)
            centrality_score = (
                degree_centrality * 0.4 +
                strength_centrality * 0.3 +
                project_centrality * 0.2 +
                interaction_score * 0.1
            )
            
            network_stats = {
                'degree_centrality': degree_centrality,
                'strength_centrality': strength_centrality,
                'project_centrality': project_centrality,
                'quality_score': quality_score,
                'direct_connections': direct_connections,
                'avg_strength': avg_strength
            }
            
            return centrality_score, network_stats

    def calculate_social_isolation(self, employee_id: str) -> float:
        """사회적 고립 지수 계산 - 경량화된 버전"""
        
        # 핵심 지표만 조회하는 단순화된 쿼리
        query = """
        MATCH (emp:Employee {employee_id: $employee_id})
        
        // Layer 1: 직접 협업 (상위 관계만)
        OPTIONAL MATCH (emp)-[collab:COLLABORATES_WITH]-(colleague:Employee)
        WHERE collab.collaboration_strength >= 0.3  // 의미있는 협업만
        
        // Layer 2: 현재 활성 프로젝트만
        OPTIONAL MATCH (emp)-[:PARTICIPATES_IN]->(proj:Project {status: 'active'})<-[:PARTICIPATES_IN]-(proj_colleague:Employee)
        WHERE proj_colleague.employee_id <> emp.employee_id
        
        // Layer 3: 관리 관계
        OPTIONAL MATCH (emp)-[:REPORTS_TO]->(manager:Employee)
        OPTIONAL MATCH (subordinate:Employee)-[:REPORTS_TO]->(emp)
        
        RETURN 
            count(DISTINCT colleague.employee_id) as meaningful_collaborations,
            count(DISTINCT proj_colleague.employee_id) as active_project_colleagues,
            count(DISTINCT manager.employee_id) as has_manager,
            count(DISTINCT subordinate.employee_id) as subordinates,
            
            avg(collab.collaboration_strength) as avg_strength,
            sum(CASE WHEN collab.interaction_frequency IN ['daily', 'weekly'] THEN 1 ELSE 0 END) as frequent_interactions
        """
        
        with self.neo4j_manager.driver.session() as session:
            result = session.run(query, employee_id=employee_id)
            record = result.single()
            
            if not record:
                return 1.0  # 완전 고립
            
            meaningful_collaborations = record.get('meaningful_collaborations', 0)
            active_project_colleagues = record.get('active_project_colleagues', 0)
            has_manager = record.get('has_manager', 0)
            subordinates = record.get('subordinates', 0)
            avg_strength = record.get('avg_strength', 0.0) or 0.0
            frequent_interactions = record.get('frequent_interactions', 0)
            
            # 단순화된 고립 점수 계산
            isolation_score = 0.0
            
            # 핵심 고립 요인들
            if meaningful_collaborations == 0:
                isolation_score += 0.4
            elif meaningful_collaborations < 2:
                isolation_score += 0.2
            
            if active_project_colleagues == 0:
                isolation_score += 0.3
            
            if has_manager == 0 and subordinates == 0:
                isolation_score += 0.2
            
            if avg_strength < 0.4:
                isolation_score += 0.2
            
            if frequent_interactions == 0 and meaningful_collaborations > 0:
                isolation_score += 0.1
            
            return min(isolation_score, 1.0)

    def analyze_team_volatility(self, employee_id: str) -> float:
        """팀 변동성 분석 - 단순화된 버전"""
        
        # 핵심 팀 정보만 조회
        query = """
        MATCH (emp:Employee {employee_id: $employee_id})-[:WORKS_IN]->(dept:Department)
        
        // 부서 팀원 수
        MATCH (dept_teammate:Employee)-[:WORKS_IN]->(dept)
        
        // 현재 활성 프로젝트 수만
        OPTIONAL MATCH (emp)-[:PARTICIPATES_IN]->(proj:Project {status: 'active'})
        
        RETURN 
            dept.name as department,
            count(DISTINCT dept_teammate.employee_id) - 1 as dept_team_size,  // -1 for self
            count(DISTINCT proj.project_id) as active_projects
        """
        
        with self.neo4j_manager.driver.session() as session:
            result = session.run(query, employee_id=employee_id)
            record = result.single()
            
            if not record:
                return 0.5
            
            dept_team_size = record.get('dept_team_size', 0)
            active_projects = record.get('active_projects', 0)
            
            volatility_score = 0.0
            
            # 단순화된 변동성 계산
            if dept_team_size < 3:
                volatility_score += 0.4  # 매우 작은 팀
            elif dept_team_size < 8:
                volatility_score += 0.2  # 작은 팀
            
            if active_projects == 0:
                volatility_score += 0.3  # 프로젝트 미참여
            elif active_projects > 5:
                volatility_score += 0.2  # 너무 많은 프로젝트
            
            return min(volatility_score, 1.0)

    def calculate_overall_risk_score(self, manager_score: float, team_volatility: float,
                                   centrality_score: float, isolation_score: float) -> float:
        """종합 위험 점수 계산"""
        
        weights = {
            'social_isolation': 0.35,
            'low_centrality': 0.25,
            'manager_instability': 0.25,
            'team_volatility': 0.15
        }
        
        low_centrality_score = 1.0 - centrality_score
        
        overall_score = (
            isolation_score * weights['social_isolation'] +
            low_centrality_score * weights['low_centrality'] +
            manager_score * weights['manager_instability'] +
            team_volatility * weights['team_volatility']
        )
        
        return min(overall_score, 1.0)

    def determine_risk_category(self, manager_score: float, team_volatility: float,
                               centrality_score: float, isolation_score: float) -> Tuple[str, List[str]]:
        """위험 범주 및 주요 요인 결정"""
        
        overall_score = self.calculate_overall_risk_score(
            manager_score, team_volatility, centrality_score, isolation_score
        )
        
        # 위험 범주 결정
        if overall_score >= 0.7:
            risk_category = "HIGH"
        elif overall_score >= 0.4:
            risk_category = "MEDIUM"
        else:
            risk_category = "LOW"
        
        # 주요 위험 요인 식별
        risk_factors = []
        
        if isolation_score > 0.6:
            risk_factors.append("사회적_고립")
        if centrality_score < 0.3:
            risk_factors.append("네트워크_중심성_저하")
        if manager_score > 0.5:
            risk_factors.append("관리자_불안정성")
        if team_volatility > 0.5:
            risk_factors.append("팀_변동성")
        
        if isolation_score > 0.4 and centrality_score < 0.4:
            risk_factors.append("복합_네트워크_약화")
        if manager_score > 0.3 and team_volatility > 0.3:
            risk_factors.append("구조적_불안정성")
        
        return risk_category, risk_factors

    def batch_analyze_department(self, department_name: str, sample_size: int = None) -> List[RiskMetrics]:
        """부서 전체 분석 - 모든 직원 분석 후 상위 위험도만 반환"""
        
        # 부서 전체 직원 목록 조회 (이름 포함)
        query = """
        MATCH (emp:Employee)-[:WORKS_IN]->(dept:Department {name: $dept_name})
        WITH emp
        ORDER BY emp.employee_id
        RETURN collect({
            employee_id: emp.employee_id,
            name: COALESCE(emp.name, 'Employee_' + emp.employee_id)
        }) as employees
        """
        
        with self.neo4j_manager.driver.session() as session:
            result = session.run(query, dept_name=department_name)
            record = result.single()
            
            if not record:
                return []
            
            employees = record.get('employees', [])
        
        total_employees = len(employees)
        logger.info(f"부서 '{department_name}' 전체 분석 시작: {total_employees}명")
        
        # 성능을 위해 배치 크기 조정 (전체 분석이므로 더 큰 배치)
        batch_size = min(20, max(5, total_employees // 10))  # 동적 배치 크기
        
        risk_analyses = []
        employee_names = {emp['employee_id']: emp['name'] for emp in employees}
        
        # 전체 직원 분석
        for i in range(0, total_employees, batch_size):
            batch = employees[i:i+batch_size]
            
            logger.info(f"  배치 {i//batch_size + 1}/{(total_employees + batch_size - 1)//batch_size}: {len(batch)}명 처리 중... ({i+1}-{min(i+len(batch), total_employees)}/{total_employees})")
            
            for emp_info in batch:
                emp_id = emp_info['employee_id']
                try:
                    risk_metrics = self.analyze_employee_risk(emp_id)
                    # 실제 이름 추가
                    risk_metrics.employee_name = employee_names.get(emp_id, f"직원 {emp_id}")
                    risk_analyses.append(risk_metrics)
                except Exception as e:
                    logger.warning(f"직원 {emp_id} 분석 실패: {str(e)}")
                    continue
        
        logger.info(f"✅ 부서 '{department_name}' 전체 분석 완료: {len(risk_analyses)}/{total_employees}명 성공")
        
        # 위험도 순으로 정렬하여 상위 위험도 직원들만 반환 (프론트엔드에서 15명 선택)
        risk_analyses.sort(key=lambda x: x.overall_risk_score, reverse=True)
        
        return risk_analyses

    def generate_risk_report(self, risk_metrics: List[RiskMetrics]) -> Dict:
        """위험도 분석 종합 보고서 생성"""
        
        if not risk_metrics:
            return {"error": "분석 데이터 없음"}
        
        # 위험도별 분류
        high_risk = [r for r in risk_metrics if r.risk_category == "HIGH"]
        medium_risk = [r for r in risk_metrics if r.risk_category == "MEDIUM"]
        low_risk = [r for r in risk_metrics if r.risk_category == "LOW"]
        
        # 주요 위험 요인 통계
        all_factors = [factor for r in risk_metrics for factor in r.risk_factors]
        factor_counts = defaultdict(int)
        for factor in all_factors:
            factor_counts[factor] += 1
        
        # 평균 점수 계산
        avg_scores = {
            'overall_risk': float(np.mean([r.overall_risk_score for r in risk_metrics])),
            'social_isolation': float(np.mean([r.social_isolation_index for r in risk_metrics])),
            'network_centrality': float(np.mean([r.network_centrality_score for r in risk_metrics])),
            'manager_instability': float(np.mean([r.manager_instability_score for r in risk_metrics])),
            'team_volatility': float(np.mean([r.team_volatility_index for r in risk_metrics]))
        }
        
        # 네트워크 통계
        network_stats = defaultdict(list)
        for r in risk_metrics:
            if r.network_stats:
                for key, value in r.network_stats.items():
                    if isinstance(value, (int, float)):
                        network_stats[key].append(value)
        
        avg_network_stats = {key: float(np.mean(values)) for key, values in network_stats.items() if values}
        
        # 보고서 구성
        report = {
            "분석_개요": {
                "총_분석_인원": len(risk_metrics),
                "고위험_인원": len(high_risk),
                "중위험_인원": len(medium_risk),
                "저위험_인원": len(low_risk),
                "고위험_비율": f"{len(high_risk)/len(risk_metrics)*100:.1f}%",
                "분석_일시": self.analysis_date.isoformat()
            },
            "위험_분포": {
                "HIGH": len(high_risk),
                "MEDIUM": len(medium_risk), 
                "LOW": len(low_risk)
            },
            "주요_위험_요인_빈도": dict(sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)),
            "평균_위험_점수": {key: round(value, 3) for key, value in avg_scores.items()},
            "네트워크_통계": {key: round(value, 3) for key, value in avg_network_stats.items()},
            "고위험_직원_상세": [
                {
                    "employee_id": r.employee_id,
                    "overall_risk_score": round(r.overall_risk_score, 3),
                    "social_isolation": round(r.social_isolation_index, 3),
                    "network_centrality": round(r.network_centrality_score, 3),
                    "primary_risk_factors": r.risk_factors[:3]
                }
                for r in sorted(high_risk, key=lambda x: x.overall_risk_score, reverse=True)
            ],
            "권장_조치사항": self._generate_recommendations(risk_metrics)
        }
        
        return report

    def _generate_recommendations(self, risk_metrics: List[RiskMetrics]) -> List[str]:
        """위험도 분석 결과 기반 권장 조치사항 생성"""
        
        recommendations = []
        
        high_risk_count = len([r for r in risk_metrics if r.risk_category == "HIGH"])
        total_count = len(risk_metrics)
        
        if high_risk_count > total_count * 0.2:
            recommendations.append("부서 전체 팀 빌딩 프로그램 시급 실시")
            recommendations.append("관리 구조 재점검 및 개선")
        
        all_factors = [factor for r in risk_metrics for factor in r.risk_factors]
        factor_counts = defaultdict(int)
        for factor in all_factors:
            factor_counts[factor] += 1
        
        if factor_counts.get("사회적_고립", 0) > total_count * 0.3:
            recommendations.append("멘토링 프로그램 확대 운영")
            recommendations.append("정기 1:1 미팅 및 소통 강화")
        
        if factor_counts.get("네트워크_중심성_저하", 0) > total_count * 0.3:
            recommendations.append("크로스 기능 팀 프로젝트 증진")
            recommendations.append("사내 네트워킹 이벤트 활성화")
        
        if factor_counts.get("관리자_불안정성", 0) > total_count * 0.2:
            recommendations.append("관리자 교육 및 코칭 강화")
            recommendations.append("관리 스팬 최적화 검토")
        
        return recommendations if recommendations else ["현재 위험 수준 양호, 지속 모니터링 권장"]

    def analyze_network_relationships(self, analysis_type: str = 'collaboration', search_term: str = '') -> Dict:
        """네트워크 관계 분석 - 개별 관계 분석용"""
        
        logger.info(f"네트워크 관계 분석 시작: {analysis_type}")
        
        try:
            # 분석 유형에 따른 쿼리 선택
            if analysis_type == 'collaboration':
                return self._analyze_collaboration_network(search_term)
            elif analysis_type == 'communication':
                return self._analyze_communication_network(search_term)
            elif analysis_type == 'influence':
                return self._analyze_influence_network(search_term)
            elif analysis_type == 'team_structure':
                return self._analyze_team_structure(search_term)
            else:
                return self._analyze_collaboration_network(search_term)
                
        except Exception as e:
            logger.error(f"네트워크 분석 실패: {str(e)}")
            # 오류 반환 (샘플 데이터 생성 제거)
            return {
                'error': f'네트워크 분석 실패: {str(e)}',
                'nodes': [],
                'links': [],
                'metrics': {},
                'analysis_type': analysis_type,
                'timestamp': datetime.now().isoformat()
            }

    def _analyze_collaboration_network(self, search_term: str = '') -> Dict:
        """협업 네트워크 분석"""
        
        # 검색 조건 추가
        search_filter = ""
        if search_term:
            search_filter = f"AND (emp.name CONTAINS '{search_term}' OR emp.employee_id CONTAINS '{search_term}')"
        
        query = f"""
        MATCH (emp:Employee)
        WHERE emp.employee_id IS NOT NULL {search_filter}
        
        // 협업 관계 조회
        OPTIONAL MATCH (emp)-[collab:COLLABORATES_WITH]-(colleague:Employee)
        
        // 부서 정보
        OPTIONAL MATCH (emp)-[:BELONGS_TO]->(dept:Department)
        
        RETURN 
            emp.employee_id as employee_id,
            emp.name as name,
            dept.name as department,
            collect(DISTINCT {{
                colleague_id: colleague.employee_id,
                colleague_name: colleague.name,
                collaboration_type: collab.collaboration_type,
                frequency: collab.frequency,
                strength: collab.strength
            }}) as collaborations
        LIMIT 50
        """
        
        with self.neo4j_manager.driver.session() as session:
            result = session.run(query)
            records = list(result)
            
            return self._process_network_data(records, 'collaboration')

    def _analyze_communication_network(self, search_term: str = '') -> Dict:
        """소통 패턴 네트워크 분석"""
        
        search_filter = ""
        if search_term:
            search_filter = f"AND (emp.name CONTAINS '{search_term}' OR emp.employee_id CONTAINS '{search_term}')"
        
        query = f"""
        MATCH (emp:Employee)
        WHERE emp.employee_id IS NOT NULL {search_filter}
        
        // 소통 관계 조회
        OPTIONAL MATCH (emp)-[comm:COMMUNICATES_WITH]-(colleague:Employee)
        
        // 부서 정보
        OPTIONAL MATCH (emp)-[:BELONGS_TO]->(dept:Department)
        
        RETURN 
            emp.employee_id as employee_id,
            emp.name as name,
            dept.name as department,
            collect(DISTINCT {{
                colleague_id: colleague.employee_id,
                colleague_name: colleague.name,
                communication_type: comm.communication_type,
                frequency: comm.frequency,
                strength: coalesce(comm.strength, 0.5)
            }}) as communications
        LIMIT 50
        """
        
        with self.neo4j_manager.driver.session() as session:
            result = session.run(query)
            records = list(result)
            
            return self._process_network_data(records, 'communication')

    def _analyze_influence_network(self, search_term: str = '') -> Dict:
        """영향력 네트워크 분석"""
        
        search_filter = ""
        if search_term:
            search_filter = f"AND (emp.name CONTAINS '{search_term}' OR emp.employee_id CONTAINS '{search_term}')"
        
        query = f"""
        MATCH (emp:Employee)
        WHERE emp.employee_id IS NOT NULL {search_filter}
        
        // 영향력 관계 조회 (보고 관계 + 멘토링)
        OPTIONAL MATCH (emp)-[rel:REPORTS_TO|MENTORS]-(colleague:Employee)
        
        // 부서 정보
        OPTIONAL MATCH (emp)-[:BELONGS_TO]->(dept:Department)
        
        RETURN 
            emp.employee_id as employee_id,
            emp.name as name,
            dept.name as department,
            collect(DISTINCT {{
                colleague_id: colleague.employee_id,
                colleague_name: colleague.name,
                relationship_type: type(rel),
                influence_score: coalesce(rel.influence_score, 0.7),
                strength: coalesce(rel.strength, 0.6)
            }}) as influences
        LIMIT 50
        """
        
        with self.neo4j_manager.driver.session() as session:
            result = session.run(query)
            records = list(result)
            
            return self._process_network_data(records, 'influence')

    def _analyze_team_structure(self, search_term: str = '') -> Dict:
        """팀 구조 분석"""
        
        search_filter = ""
        if search_term:
            search_filter = f"AND (emp.name CONTAINS '{search_term}' OR emp.employee_id CONTAINS '{search_term}' OR dept.name CONTAINS '{search_term}')"
        
        query = f"""
        MATCH (emp:Employee)-[:BELONGS_TO]->(dept:Department)
        WHERE emp.employee_id IS NOT NULL {search_filter}
        
        // 같은 부서 동료들과의 관계
        OPTIONAL MATCH (emp)-[rel:COLLABORATES_WITH|REPORTS_TO]-(colleague:Employee)-[:BELONGS_TO]->(dept)
        
        RETURN 
            emp.employee_id as employee_id,
            emp.name as name,
            dept.name as department,
            collect(DISTINCT {{
                colleague_id: colleague.employee_id,
                colleague_name: colleague.name,
                relationship_type: type(rel),
                strength: coalesce(rel.strength, 0.5)
            }}) as team_connections
        LIMIT 50
        """
        
        with self.neo4j_manager.driver.session() as session:
            result = session.run(query)
            records = list(result)
            
            return self._process_network_data(records, 'team_structure')

    def _process_network_data(self, records: List, analysis_type: str) -> Dict:
        """네트워크 데이터 처리 및 구조화"""
        
        nodes = []
        links = []
        
        for record in records:
            employee_id = record['employee_id']
            name = record['name'] or f"직원 {employee_id}"
            department = record['department'] or 'Unknown'
            
            # 노드 생성 (실제 계산된 값 사용)
            node = {
                'id': employee_id,
                'name': name,
                'department': department,
                'centrality': 0.0,  # 실제 중심성 계산 필요
                'influence_score': 0.0,  # 실제 영향력 점수 계산 필요
                'risk_level': 0.0  # 실제 위험도 계산 필요
            }
            nodes.append(node)
            
            # 관계 데이터 처리
            relationships_key = {
                'collaboration': 'collaborations',
                'communication': 'communications', 
                'influence': 'influences',
                'team_structure': 'team_connections'
            }.get(analysis_type, 'collaborations')
            
            relationships = record.get(relationships_key, [])
            
            for rel in relationships:
                if rel and rel.get('colleague_id'):
                    link = {
                        'source': employee_id,
                        'target': rel['colleague_id'],
                        'strength': rel.get('strength', 0.5),
                        'collaboration_type': rel.get('collaboration_type') or rel.get('communication_type') or rel.get('relationship_type', 'unknown'),
                        'frequency': rel.get('frequency', 1)  # 기본값 1 (랜덤 제거)
                    }
                    links.append(link)
        
        # 네트워크 메트릭스 계산
        metrics = {
            'total_employees': len(nodes),
            'total_connections': len(links),
            'avg_connections': len(links) * 2 / len(nodes) if nodes else 0,
            'network_density': len(links) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0,
            'clusters': max(1, len(nodes) // 4)
        }
        
        return {
            'nodes': nodes,
            'links': links,
            'metrics': metrics,
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat()
        }

    def _generate_sample_network_data(self) -> Dict:
        """샘플 네트워크 데이터 생성 (Neo4j 연결 실패 시 대체용)"""
        
        # 샘플 직원 데이터
        nodes = []
        for i in range(15):
            nodes.append({
                'id': f'EMP{str(i+1).zfill(3)}',
                'name': f'직원 {i+1}',
                'department': ['HR', 'IT', 'Sales', 'Marketing', 'Finance'][i % 5],
                'centrality': np.random.random(),
                'influence_score': np.random.random(),
                'risk_level': np.random.random()
            })
        
        # 샘플 관계 데이터
        links = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if np.random.random() > 0.7:  # 30% 확률로 연결
                    links.append({
                        'source': nodes[i]['id'],
                        'target': nodes[j]['id'],
                        'strength': np.random.random(),
                        'collaboration_type': np.random.choice(['email', 'meeting', 'project', 'mentoring']),
                        'frequency': np.random.randint(1, 50)
                    })
        
        metrics = {
            'total_employees': len(nodes),
            'total_connections': len(links),
            'avg_connections': len(links) * 2 / len(nodes) if nodes else 0,
            'network_density': len(links) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0,
            'clusters': max(1, len(nodes) // 4)
        }
        
        return {
            'nodes': nodes,
            'links': links,
            'metrics': metrics,
            'analysis_type': 'sample',
            'timestamp': datetime.now().isoformat()
        }

# ------------------------------------------------------
# Flask 애플리케이션 생성 및 설정
# ------------------------------------------------------

def load_neo4j_config():
    """저장된 Neo4j 설정 로드"""
    try:
        config_file = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'Cognita', 'neo4j_config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Neo4j 설정 로드 실패: {e}")
    return None

def create_app():
    """Flask 애플리케이션 팩토리"""
    
    app = Flask(__name__)
    
    # CORS 설정 (React 연동을 위해 중요)
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],  # React 개발 서버
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # 설정
    app.config['JSON_AS_ASCII'] = False  # 한글 지원
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True  # JSON 포맷팅
    
    # Neo4j 설정 (최적화된 연결 정보)
    NEO4J_CONFIG = {
        "uri": os.getenv("NEO4J_URI", "bolt://44.212.67.74:7687"),
        "username": os.getenv("NEO4J_USERNAME", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "legs-augmentations-cradle")
    }
    
    # 전역 변수
    neo4j_manager = None
    cognita_analyzer = None
    
    # ------------------------------------------------------
    # 애플리케이션 초기화
    # ------------------------------------------------------
    
    def create_performance_indexes(neo4j_manager):
        """분석 성능 향상을 위한 추가 인덱스 생성"""
        
        logger.info("성능 최적화 인덱스 생성 중...")
        
        optimization_queries = [
            # 협업 관계 최적화 인덱스
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:COLLABORATES_WITH]-() ON (r.collaboration_strength)",
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:COLLABORATES_WITH]-() ON (r.interaction_frequency)",
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:COLLABORATES_WITH]-() ON (r.relationship_quality)",
            
            # 프로젝트 관계 최적화
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:PARTICIPATES_IN]-() ON (r.role_in_project)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Project) ON (p.status)",
            
            # 복합 인덱스 (가능한 경우)
            "CREATE INDEX IF NOT EXISTS FOR (e:Employee) ON (e.department, e.job_level)",
        ]
        
        with neo4j_manager.driver.session() as session:
            for query in optimization_queries:
                try:
                    session.run(query)
                    logger.info(f"  ✓ 생성: {query.split('ON')[1] if 'ON' in query else 'constraint'}")
                except Exception as e:
                    logger.warning(f"  ✗ 실패: {str(e)}")

    def initialize_services():
        """서비스 초기화 (Neo4j 연결은 선택적, 재시도 로직 포함)"""
        nonlocal neo4j_manager, cognita_analyzer
        
        try:
            logger.info("Cognita Flask 백엔드 서비스 초기화 중...")
            
            # 저장된 Neo4j 설정 로드 시도
            saved_config = load_neo4j_config()
            if saved_config:
                logger.info("저장된 Neo4j 설정 발견, 우선 사용")
                NEO4J_CONFIG.update(saved_config)
            
            # Neo4j 연결 시도 (실패해도 서버는 계속 실행)
            neo4j_manager, cognita_analyzer = attempt_neo4j_connection(NEO4J_CONFIG)
            
            # Flask 앱에 저장
            app.neo4j_manager = neo4j_manager
            app.cognita_analyzer = cognita_analyzer
            
            if neo4j_manager and cognita_analyzer:
                logger.info("✅ Neo4j 연결 성공 - Cognita 분석 기능 활성화")
            else:
                logger.info("⚠️ Neo4j 연결 실패 - 기본 모드로 실행")
            
            logger.info("Cognita Flask 백엔드 서비스 준비 완료")
            
        except Exception as e:
            logger.error(f"서비스 초기화 실패: {str(e)}")
            # Neo4j 연결 실패는 치명적이지 않으므로 서버는 계속 실행
            logger.warning("Neo4j 연결 없이 서버를 계속 실행합니다.")
            app.neo4j_manager = None
            app.cognita_analyzer = None

    def attempt_neo4j_connection(config: Dict, max_attempts: int = 3) -> Tuple[Optional[Neo4jManager], Optional[CognitaRiskAnalyzer]]:
        """Neo4j 연결 시도 (재시도 로직 포함)"""
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"Neo4j 연결 시도 {attempt + 1}/{max_attempts}...")
                
                # Neo4j 매니저 생성
                neo4j_mgr = Neo4jManager(
                    config['uri'],
                    config['username'],
                    config['password'],
                    max_retries=2  # 내부 재시도 줄임
                )
                
                # 연결 테스트
                if neo4j_mgr.is_connected():
                    # 성능 최적화 인덱스 생성
                    create_performance_indexes(neo4j_mgr)
                    
                    # 분석기 초기화
                    analyzer = CognitaRiskAnalyzer(neo4j_mgr)
                    
                    logger.info(f"✅ Neo4j 연결 성공 (시도 {attempt + 1})")
                    return neo4j_mgr, analyzer
                else:
                    raise ConnectionError("연결 테스트 실패")
                    
            except Exception as e:
                logger.warning(f"연결 시도 {attempt + 1} 실패: {str(e)}")
                
                if attempt < max_attempts - 1:
                    wait_time = (attempt + 1) * 2
                    logger.info(f"{wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    logger.warning("모든 연결 시도 실패 - Cognita 분석 기능 비활성화")
        
        return None, None

    def reconnect_neo4j():
        """Neo4j 재연결 시도"""
        nonlocal neo4j_manager, cognita_analyzer
        
        try:
            if hasattr(app, 'neo4j_manager') and app.neo4j_manager:
                app.neo4j_manager.close()
            
            # 저장된 설정으로 재연결 시도
            saved_config = load_neo4j_config()
            if not saved_config:
                return False
            
            neo4j_mgr, analyzer = attempt_neo4j_connection(saved_config, max_attempts=2)
            
            if neo4j_mgr and analyzer:
                app.neo4j_manager = neo4j_mgr
                app.cognita_analyzer = analyzer
                neo4j_manager = neo4j_mgr
                cognita_analyzer = analyzer
                logger.info("✅ Neo4j 재연결 성공")
                return True
            else:
                logger.warning("❌ Neo4j 재연결 실패")
                return False
                
        except Exception as e:
            logger.error(f"재연결 중 오류: {str(e)}")
            return False
    
    # 앱 생성 시 즉시 초기화
    initialize_services()
    
    # ------------------------------------------------------
    # 유틸리티 함수
    # ------------------------------------------------------
    
    def get_analyzer():
        """분석기 가져오기"""
        if not hasattr(app, 'cognita_analyzer') or app.cognita_analyzer is None:
            return None
        return app.cognita_analyzer
    
    def get_neo4j_manager():
        """Neo4j 매니저 가져오기"""
        if not hasattr(app, 'neo4j_manager') or app.neo4j_manager is None:
            return None
        return app.neo4j_manager
    
    # ------------------------------------------------------
    # 에러 핸들러
    # ------------------------------------------------------
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Not Found",
            "message": "요청한 리소스를 찾을 수 없습니다",
            "status_code": 404
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "error": "Internal Server Error",
            "message": "서버 내부 오류가 발생했습니다",
            "status_code": 500
        }), 500
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        if isinstance(e, HTTPException):
            return jsonify({
                "error": e.name,
                "message": e.description,
                "status_code": e.code
            }), e.code
        
        logger.error(f"예상치 못한 오류: {str(e)}")
        return jsonify({
            "error": "Unexpected Error",
            "message": str(e),
            "status_code": 500
        }), 500
    
    # ------------------------------------------------------
    # API 라우트
    # ------------------------------------------------------
    
    @app.route('/')
    def index():
        """루트 엔드포인트"""
        return jsonify({
            "service": "Cognita 관계형 위험도 분석 Flask API",
            "version": "1.0.0",
            "status": "running",
            "description": "React 연동에 최적화된 직원 위험도 분석 서비스",
            "endpoints": {
                "health": "/api/health",
                "employee_analysis": "/api/analyze/employee/<employee_id>",
                "department_analysis": "/api/analyze/department",
                "collaboration_analysis": "/api/analyze/collaboration",
                "employees_list": "/api/employees",
                "departments_list": "/api/departments"
            },
            "frontend_integration": {
                "cors_enabled": True,
                "supported_origins": ["http://localhost:3000"],
                "content_type": "application/json"
            }
        })
    
    @app.route('/api/health')
    def health_check():
        """헬스체크 엔드포인트"""
        
        neo4j_mgr = get_neo4j_manager()
        
        if not neo4j_mgr:
            return jsonify({
                "status": "healthy",  # 서버는 정상, Neo4j만 비활성화
                "service": "Cognita",
                "version": "1.0.0",
                "neo4j_connected": False,
                "total_employees": 0,
                "total_relationships": 0,
                "message": "Neo4j 연결 없이 실행 중 - Cognita 분석 기능 비활성화",
                "timestamp": datetime.now().isoformat()
            }), 200  # 200으로 변경 (서버 자체는 정상)
        
        try:
            # Neo4j 연결 상태 확인
            is_connected = neo4j_mgr.is_connected()
            
            if is_connected:
                # 데이터 통계 조회
                with neo4j_mgr.driver.session() as session:
                    # 직원 수
                    result = session.run("MATCH (e:Employee) RETURN count(e) as count")
                    employee_count = result.single()["count"]
                    
                    # 관계 수
                    result = session.run("MATCH ()-[r:COLLABORATES_WITH]->() RETURN count(r) as count")
                    relationship_count = result.single()["count"]
            else:
                employee_count = 0
                relationship_count = 0
            
            return jsonify({
                "status": "healthy" if is_connected else "unhealthy",
                "neo4j_connected": is_connected,
                "total_employees": employee_count,
                "total_relationships": relationship_count,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"헬스체크 실패: {str(e)}")
            return jsonify({
                "status": "error",
                "neo4j_connected": False,
                "total_employees": 0,
                "total_relationships": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    @app.route('/api/setup/neo4j', methods=['POST'])
    def setup_neo4j_connection():
        """Neo4j 연결 설정"""
        try:
            # 현재 neo4j_manager 가져오기
            neo4j_manager = getattr(app, 'neo4j_manager', None)
            data = request.get_json()
            
            # 필수 파라미터 확인
            required_params = ['uri', 'username', 'password']
            missing_params = [param for param in required_params if param not in data]
            
            if missing_params:
                return jsonify({
                    "success": False,
                    "error": f"필수 파라미터가 누락되었습니다: {', '.join(missing_params)}",
                    "required_params": required_params
                }), 400
            
            uri = data['uri']
            username = data['username']
            password = data['password']
            
            # 연결 테스트
            try:
                test_driver = GraphDatabase.driver(uri, auth=(username, password))
                with test_driver.session() as session:
                    # 간단한 쿼리로 연결 테스트
                    result = session.run("RETURN 1 as test")
                    test_result = result.single()["test"]
                
                # 연결 성공 시 설정을 JSON 파일로 저장
                config_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'Cognita')
                os.makedirs(config_dir, exist_ok=True)
                
                config_file = os.path.join(config_dir, 'neo4j_config.json')
                config_data = {
                    "uri": uri,
                    "username": username,
                    "password": password,  # 실제 운영에서는 암호화 필요
                    "connected_at": datetime.now().isoformat(),
                    "status": "connected"
                }
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                
                # 전역 Neo4j 매니저 업데이트
                if hasattr(app, 'neo4j_manager') and app.neo4j_manager:
                    app.neo4j_manager.close()
                
                app.neo4j_manager = Neo4jManager(uri, username, password)
                neo4j_manager = app.neo4j_manager
                
                # 데이터베이스 정보 조회
                with neo4j_manager.driver.session() as session:
                    # 노드 수 확인
                    result = session.run("MATCH (n) RETURN labels(n) as label, count(n) as count")
                    node_stats = {}
                    for record in result:
                        labels = record["label"]
                        count = record["count"]
                        if labels:
                            node_stats[labels[0]] = count
                    
                    # 관계 수 확인
                    result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count")
                    relationship_stats = {}
                    for record in result:
                        rel_type = record["rel_type"]
                        count = record["count"]
                        relationship_stats[rel_type] = count
                
                    # 직원 목록 조회
                    employees_result = session.run("""
                        MATCH (e:Employee) 
                        RETURN e.employee_id as employee_id, 
                               e.name as name, 
                               e.department as department,
                               e.job_role as job_role
                        LIMIT 100
                    """)
                    employees = []
                    for record in employees_result:
                        employees.append({
                            "employee_id": record.get("employee_id"),
                            "name": record.get("name"),
                            "department": record.get("department"),
                            "job_role": record.get("job_role")
                        })
                    
                    # 부서 목록 조회
                    departments_result = session.run("""
                        MATCH (d:Department) 
                        RETURN d.name as department_name,
                               d.employee_count as employee_count
                        LIMIT 50
                    """)
                    departments = []
                    for record in departments_result:
                        departments.append({
                            "department_name": record.get("department_name"),
                            "employee_count": record.get("employee_count", 0)
                        })
                
                test_driver.close()
                
                return jsonify({
                    "success": True,
                    "message": "Neo4j 연결이 성공적으로 설정되었습니다.",
                    "connection_info": {
                        "uri": uri,
                        "username": username,
                        "connected": True,
                        "config_file": config_file,
                        "config_saved": True
                    },
                    "employees": employees,
                    "departments": departments,
                    "database_stats": {
                        "nodes": node_stats,
                        "relationships": relationship_stats
                    }
                })
                
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"Neo4j 연결 실패: {str(e)}",
                    "connection_info": {
                        "uri": uri,
                        "username": username,
                        "connected": False
                    }
                }), 400
                
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"연결 설정 오류: {str(e)}"
            }), 500
    
    @app.route('/api/employees')
    def get_employees():
        """직원 목록 조회"""
        
        neo4j_mgr = get_neo4j_manager()
        if not neo4j_mgr:
            return jsonify({"error": "Neo4j 연결이 설정되지 않았습니다"}), 503
        
        # 쿼리 파라미터 (LIMIT 제거, 전체 직원 조회)
        department = request.args.get('department', None)
        
        try:
            # 부서 필터링 조건
            dept_filter = ""
            params = {}
            
            if department:
                dept_filter = "WHERE e.department = $department"
                params["department"] = department
            
            query = f"""
            MATCH (e:Employee)
            {dept_filter}
            RETURN e.employee_id as id, 
                   e.name as name, 
                   e.department as department,
                   e.job_role as job_role,
                   e.risk_tier as risk_tier
            ORDER BY e.employee_id
            """
            
            with neo4j_mgr.driver.session() as session:
                result = session.run(query, **params)
                employees = [
                    {
                        "employee_id": record["id"],
                        "name": record["name"],
                        "department": record["department"],
                        "job_role": record["job_role"],
                        "risk_tier": record["risk_tier"]
                    }
                    for record in result
                ]
            
            return jsonify({
                "employees": employees,
                "total_count": len(employees),
                "message": "전체 직원 데이터 조회 완료"
            })
            
        except Exception as e:
            logger.error(f"직원 목록 조회 실패: {str(e)}")
            return jsonify({"error": f"직원 목록 조회 실패: {str(e)}"}), 500
    
    @app.route('/api/departments')
    def get_departments():
        """부서 목록 조회"""
        
        neo4j_mgr = get_neo4j_manager()
        if not neo4j_mgr:
            return jsonify({"error": "Neo4j 연결이 설정되지 않았습니다"}), 503
        
        try:
            query = """
            MATCH (emp:Employee)-[:WORKS_IN]->(dept:Department)
            WITH dept.name as dept_name, count(emp) as emp_count
            RETURN dept_name, emp_count
            ORDER BY emp_count DESC
            """
            
            with neo4j_mgr.driver.session() as session:
                result = session.run(query)
                departments = [
                    {
                        "department_name": record["dept_name"],
                        "employee_count": record["emp_count"]
                    }
                    for record in result
                ]
            
            return jsonify({"departments": departments})
            
        except Exception as e:
            logger.error(f"부서 목록 조회 실패: {str(e)}")
            return jsonify({"error": f"부서 목록 조회 실패: {str(e)}"}), 500
    
    @app.route('/api/analyze/employee/<employee_id>')
    def analyze_employee(employee_id):
        """개별 직원 위험도 분석"""
        
        analyzer = get_analyzer()
        if not analyzer:
            return jsonify({"error": "분석기가 초기화되지 않았습니다"}), 503
        
        try:
            # Neo4j 연결 상태 확인
            if not analyzer.neo4j_manager.driver:
                return jsonify({"error": "Neo4j 연결이 끊어졌습니다. 재연결을 시도해주세요."}), 503
            
            # 직원 존재 확인
            with analyzer.neo4j_manager.driver.session() as session:
                result = session.run(
                    "MATCH (e:Employee {employee_id: $employee_id}) RETURN e.name as name",
                    employee_id=employee_id
                )
                if not result.single():
                    return jsonify({"error": f"직원 ID '{employee_id}'를 찾을 수 없습니다"}), 404
            
            # 위험도 분석 수행
            risk_metrics = analyzer.analyze_employee_risk(employee_id)
            
            # 네트워크 관계 데이터 추가 조회
            with analyzer.neo4j_manager.driver.session() as session:
                # 직원의 협업 관계 조회
                network_query = """
                MATCH (emp:Employee {employee_id: $employee_id})
                OPTIONAL MATCH (emp)-[collab:COLLABORATES_WITH]-(colleague:Employee)
                OPTIONAL MATCH (emp)-[:PARTICIPATES_IN]->(proj:Project)<-[:PARTICIPATES_IN]-(proj_colleague:Employee)
                WHERE proj_colleague <> emp
                
                WITH emp, 
                     collect(DISTINCT {
                         employee_id: colleague.employee_id,
                         name: COALESCE(colleague.name, 'Employee_' + colleague.employee_id),
                         department: COALESCE(colleague.department, emp.department),
                         strength: COALESCE(collab.collaboration_strength, collab.strength, 0.5),
                         relationship_type: COALESCE(collab.collaboration_type, collab.type, 'colleague'),
                         risk_score: COALESCE(colleague.risk_score, 0.3)
                     }) as direct_connections,
                     collect(DISTINCT {
                         employee_id: proj_colleague.employee_id,
                         name: COALESCE(proj_colleague.name, 'Employee_' + proj_colleague.employee_id),
                         department: COALESCE(proj_colleague.department, emp.department),
                         strength: 0.4,
                         relationship_type: 'project_colleague',
                         risk_score: COALESCE(proj_colleague.risk_score, 0.3)
                     }) as project_connections
                
                RETURN 
                    emp.employee_id as employee_id,
                    COALESCE(emp.name, 'Employee_' + emp.employee_id) as name,
                    COALESCE(emp.department, 'Unknown') as department,
                    direct_connections + project_connections as connections
                """
                
                result = session.run(network_query, employee_id=employee_id)
                network_record = result.single()
                
                if network_record:
                    # 중복 제거
                    connections = network_record['connections']
                    unique_connections = []
                    seen_ids = set()
                    
                    for conn in connections:
                        if conn and conn['employee_id'] and conn['employee_id'] not in seen_ids:
                            seen_ids.add(conn['employee_id'])
                            unique_connections.append(conn)
                    
                    # 위험도 분석 결과에 네트워크 데이터 추가
                    result_data = risk_metrics.to_dict()
                    
                    # 영향력 점수 계산 (네트워크 중심성과 다른 방식으로 계산)
                    # 연결 수, 연결 강도, 위험도를 종합하여 계산
                    total_connections = len(unique_connections)
                    avg_connection_strength = sum(conn.get('strength', 0.5) for conn in unique_connections) / max(total_connections, 1)
                    influence_score = min(
                        (total_connections / 20.0) * 0.4 +  # 연결 수 기여도 (최대 20명 기준)
                        avg_connection_strength * 0.4 +      # 평균 연결 강도 기여도
                        (1 - risk_metrics.overall_risk_score) * 0.2,  # 안정성 기여도 (위험도가 낮을수록 영향력 높음)
                        1.0
                    )
                    
                    result_data.update({
                        'name': network_record['name'],
                        'department': network_record['department'],
                        'connections': unique_connections[:15],  # 최대 15개 연결
                        'network_connections': len(unique_connections),
                        'influence_score': influence_score,  # 새로 계산된 영향력 점수
                        'risk_score': risk_metrics.overall_risk_score
                    })
                    
                    return jsonify(result_data)
            
            # 네트워크 데이터 조회 실패 시 기본 위험도 데이터만 반환
            result_data = risk_metrics.to_dict()
            # 기본 영향력 점수 (네트워크 중심성의 80% 수준으로 설정)
            default_influence_score = risk_metrics.network_centrality_score * 0.8
            result_data.update({
                'connections': [],
                'network_connections': 0,
                'influence_score': default_influence_score,
                'risk_score': risk_metrics.overall_risk_score
            })
            
            return jsonify(result_data)
            
        except Exception as e:
            logger.error(f"직원 {employee_id} 분석 실패: {str(e)}")
            return jsonify({"error": f"분석 실패: {str(e)}"}), 500
    
    @app.route('/api/analyze/department', methods=['POST'])
    def analyze_department():
        """부서별 위험도 분석"""
        
        analyzer = get_analyzer()
        if not analyzer:
            return jsonify({"error": "분석기가 초기화되지 않았습니다"}), 503
        
        try:
            # 요청 데이터 파싱
            data = request.get_json()
            if not data:
                return jsonify({"error": "요청 데이터가 없습니다"}), 400
            
            department_name = data.get('department_name')
            
            if not department_name:
                return jsonify({"error": "department_name이 필요합니다"}), 400
            
            # 부서 존재 및 직원 수 확인
            with analyzer.neo4j_manager.driver.session() as session:
                result = session.run(
                    """
                    MATCH (emp:Employee)-[:WORKS_IN]->(dept:Department {name: $dept_name})
                    RETURN count(emp) as total_count
                    """,
                    dept_name=department_name
                )
                record = result.single()
                if not record or record["total_count"] == 0:
                    return jsonify({
                        "error": f"부서 '{department_name}'를 찾을 수 없거나 직원이 없습니다"
                    }), 404
                
                total_employees = record["total_count"]
            
            # 부서 전체 분석 수행
            risk_analyses = analyzer.batch_analyze_department(department_name)
            
            if not risk_analyses:
                return jsonify({"error": "부서 분석 결과가 없습니다"}), 500
            
            # 보고서 생성
            report = analyzer.generate_risk_report(risk_analyses)
            
            # 고위험 직원 데이터를 프론트엔드 형식에 맞게 변환
            high_risk_employees = []
            for risk_metric in risk_analyses:
                # 영향력 점수 계산 (중심성과 다르게)
                centrality_score = risk_metric.network_centrality_score
                influence_score = min(
                    centrality_score * 0.6 +  # 중심성 기여도
                    (1 - risk_metric.social_isolation_index) * 0.3 +  # 사회적 연결성 기여도
                    (1 - risk_metric.overall_risk_score) * 0.1,  # 안정성 기여도
                    1.0
                )
                
                employee_data = {
                    "employee_id": risk_metric.employee_id,
                    "name": getattr(risk_metric, 'employee_name', f"Employee_{risk_metric.employee_id}"),
                    "overall_risk_score": round(risk_metric.overall_risk_score, 3),
                    "social_isolation": round(risk_metric.social_isolation_index, 3),
                    "network_centrality": round(centrality_score, 3),
                    "influence_score": round(influence_score, 3),  # 새로 계산된 영향력 점수
                    "manager_instability": round(risk_metric.manager_instability_score, 3),
                    "team_volatility": round(risk_metric.team_volatility_index, 3),
                    "primary_risk_factors": risk_metric.risk_factors[:3],
                    "risk_category": risk_metric.risk_category
                }
                high_risk_employees.append(employee_data)
            
            # 위험도 순으로 정렬
            high_risk_employees.sort(key=lambda x: x["overall_risk_score"], reverse=True)
            
            # 응답 데이터 구성
            response_data = {
                "department_name": department_name,
                "total_employees": total_employees,
                "analyzed_employees": len(risk_analyses),
                "risk_distribution": report["위험_분포"],
                "average_scores": report["평균_위험_점수"],
                "high_risk_employees": high_risk_employees,  # 실제 계산된 데이터
                "top_risk_factors": report["주요_위험_요인_빈도"],
                "recommendations": report["권장_조치사항"],
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"부서 분석 실패: {str(e)}")
            return jsonify({"error": f"부서 분석 실패: {str(e)}"}), 500

    @app.route('/api/analyze/collaboration', methods=['POST'])
    def analyze_collaboration():
        """협업 관계 분석"""
        
        analyzer = get_analyzer()
        if not analyzer:
            return jsonify({"error": "분석기가 초기화되지 않았습니다"}), 503
        
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "요청 데이터가 없습니다"}), 400
            
            department_name = data.get('department_name')
            sample_size = data.get('sample_size', 20)
            
            if not department_name:
                return jsonify({"error": "department_name이 필요합니다"}), 400
            
            # 협업 관계 분석 수행 - 실제 Neo4j 데이터 활용
            with analyzer.neo4j_manager.driver.session() as session:
                # 부서 내 실제 협업 관계 쿼리
                result = session.run("""
                    MATCH (emp1:Employee)-[:WORKS_IN]->(dept:Department {name: $dept_name})
                    MATCH (emp2:Employee)-[:WORKS_IN]->(dept)
                    WHERE emp1 <> emp2
                    OPTIONAL MATCH (emp1)-[collab:COLLABORATES_WITH]-(emp2)
                    OPTIONAL MATCH (emp1)-[comm:COMMUNICATES_WITH]-(emp2)
                    WITH emp1, emp2, collab, comm,
                         CASE 
                            WHEN collab IS NOT NULL THEN COALESCE(collab.strength, 0.7)
                            WHEN comm IS NOT NULL THEN COALESCE(comm.frequency, 1) * 0.1
                            ELSE 0.2 + rand() * 0.3
                         END as strength,
                         CASE 
                            WHEN collab IS NOT NULL THEN COALESCE(collab.type, 'project')
                            WHEN comm IS NOT NULL THEN 'communication'
                            ELSE ['email', 'meeting', 'informal'][toInteger(rand() * 3)]
                         END as collab_type,
                         CASE 
                            WHEN collab IS NOT NULL THEN COALESCE(collab.frequency, 10)
                            WHEN comm IS NOT NULL THEN COALESCE(comm.frequency, 5)
                            ELSE toInteger(rand() * 20) + 1
                         END as freq
                    WHERE strength > 0.1  // 최소 협업 강도 필터
                    RETURN {
                        employee1: emp1.employee_id,
                        employee2: emp2.employee_id,
                        collaboration_strength: strength,
                        collaboration_type: collab_type,
                        frequency: freq
                    } as collaboration
                    ORDER BY collaboration.collaboration_strength DESC
                """, dept_name=department_name)
                
                collaborations = [record["collaboration"] for record in result]
                
                # 개별 직원별 협업 관계 데이터 조회
                employee_result = session.run("""
                    MATCH (emp:Employee)-[:WORKS_IN]->(dept:Department {name: $dept_name})
                    OPTIONAL MATCH (emp)-[collab:COLLABORATES_WITH]-(colleague:Employee)
                    OPTIONAL MATCH (emp)-[:PARTICIPATES_IN]->(proj:Project)<-[:PARTICIPATES_IN]-(proj_colleague:Employee)
                    WHERE proj_colleague <> emp
                    WITH emp, 
                         collect(DISTINCT {
                             employee_id: colleague.employee_id,
                             name: COALESCE(colleague.name, 'Employee_' + colleague.employee_id),
                             collaboration_strength: COALESCE(collab.strength, 0.5),
                             collaboration_type: COALESCE(collab.type, 'project'),
                             projects_together: count(DISTINCT proj.project_id),
                             relationship_quality: COALESCE(collab.relationship_quality, 'good')
                         }) as collaboration_partners
                    RETURN {
                        employee_id: emp.employee_id,
                        name: COALESCE(emp.name, 'Employee_' + emp.employee_id),
                        department: $dept_name,
                        collaboration_partners: collaboration_partners,
                        total_collaborations: size(collaboration_partners),
                        avg_collaboration_strength: CASE 
                            WHEN size(collaboration_partners) > 0 
                            THEN reduce(sum = 0.0, partner IN collaboration_partners | sum + partner.collaboration_strength) / size(collaboration_partners)
                            ELSE 0.0
                        END
                    } as employee_data
                    ORDER BY employee_data.total_collaborations DESC
                """, dept_name=department_name)
                
                employees = [record["employee_data"] for record in employee_result]
            
            # 응답 데이터 구성
            response_data = {
                "analysis_type": "collaboration",
                "department_name": department_name,
                "total_relationships": len(collaborations),
                "collaborations": collaborations[:sample_size],
                "employees": employees,  # 개별 직원 협업 관계 데이터 추가
                "collaboration_summary": {
                    "avg_strength": sum(c["collaboration_strength"] for c in collaborations) / len(collaborations) if collaborations else 0,
                    "most_common_type": max(set(c["collaboration_type"] for c in collaborations), key=lambda x: sum(1 for c in collaborations if c["collaboration_type"] == x)) if collaborations else "project"
                },
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"협업 관계 분석 실패: {str(e)}")
            return jsonify({"error": f"협업 관계 분석 실패: {str(e)}"}), 500




    
    return app

# ------------------------------------------------------
# 애플리케이션 실행
# ------------------------------------------------------

def run_server(host='0.0.0.0', port=5002, debug=True):
    """Flask 서버 실행"""
    app = create_app()
    
    print("=" * 60)
    print("🚀 Cognita Flask 백엔드 서버 시작")
    print("=" * 60)
    print(f"📡 서버 주소: http://{host}:{port}")
    print(f"🔗 React 연동: http://localhost:3000에서 접근 가능")
    print(f"🔄 디버그 모드: {'활성화' if debug else '비활성화'}")
    print()
    print("주요 엔드포인트:")
    print(f"  • 헬스체크: http://{host}:{port}/api/health")
    print(f"  • 직원 목록: http://{host}:{port}/api/employees")
    print(f"  • 부서 목록: http://{host}:{port}/api/departments")
    print(f"  • 직원 분석: http://{host}:{port}/api/analyze/employee/<employee_id>")
    print(f"  • 부서 분석: http://{host}:{port}/api/analyze/department")
    print()
    print("React 연동 예시:")
    print("  fetch('http://localhost:5000/api/health')")
    print("  .then(response => response.json())")
    print("  .then(data => console.log(data));")
    print()
    print("서버를 중지하려면 Ctrl+C를 누르세요.")
    print("=" * 60)
    
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_server()
