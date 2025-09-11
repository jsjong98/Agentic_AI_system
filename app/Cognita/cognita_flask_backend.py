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
                    max_connection_lifetime=30,
                    max_connection_pool_size=10,
                    connection_acquisition_timeout=30,
                    connection_timeout=10
                )
                
                # 연결 테스트
                with self.driver.session() as session:
                    result = session.run("RETURN 'Connected' as status")
                    logger.info("Neo4j 연결 성공")
                    return
                    
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
        
        # 협업 관계 샘플링 (성능 향상)
        query = """
        MATCH (emp:Employee {employee_id: $employee_id})
        
        // 직접 협업 관계만 조회 (2-hop 제거로 성능 향상)
        OPTIONAL MATCH (emp)-[collab:COLLABORATES_WITH]-(colleague:Employee)
        WHERE collab.collaboration_strength IS NOT NULL
        
        // 상위 협업 관계만 샘플링 (강도 기준)
        WITH emp, collab, colleague
        ORDER BY collab.collaboration_strength DESC
        LIMIT 50  // 최대 50개 관계만 분석
        
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

    def batch_analyze_department(self, department_name: str, sample_size: int = 20) -> List[RiskMetrics]:
        """부서 분석 - 샘플링과 배치 최적화"""
        
        # 샘플링된 직원 목록 조회
        query = """
        MATCH (emp:Employee)-[:WORKS_IN]->(dept:Department {name: $dept_name})
        WITH emp
        ORDER BY emp.employee_id
        LIMIT $sample_size
        RETURN collect({
            employee_id: emp.employee_id,
            name: emp.name
        }) as employees
        """
        
        with self.neo4j_manager.driver.session() as session:
            result = session.run(query, dept_name=department_name, sample_size=sample_size)
            record = result.single()
            
            if not record:
                return []
            
            employees = record.get('employees', [])
        
        logger.info(f"부서 '{department_name}' 샘플 분석: {len(employees)}명")
        
        risk_analyses = []
        batch_size = 5  # 작은 배치로 처리
        
        for i in range(0, len(employees), batch_size):
            batch = employees[i:i+batch_size]
            
            logger.info(f"  배치 {i//batch_size + 1}: {len(batch)}명 처리 중...")
            
            for emp_info in batch:
                emp_id = emp_info['employee_id']
                try:
                    risk_metrics = self.analyze_employee_risk(emp_id)
                    risk_analyses.append(risk_metrics)
                except Exception as e:
                    logger.warning(f"직원 {emp_id} 분석 실패: {str(e)}")
                    continue
        
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

# ------------------------------------------------------
# Flask 애플리케이션 생성 및 설정
# ------------------------------------------------------

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
        "uri": os.getenv("NEO4J_URI", "bolt://34.227.31.16:7687"),
        "username": os.getenv("NEO4J_USERNAME", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "cover-site-establishment")
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

    @app.before_first_request
    def initialize_services():
        """첫 요청 전 서비스 초기화"""
        nonlocal neo4j_manager, cognita_analyzer
        
        try:
            logger.info("Cognita Flask 백엔드 서비스 초기화 중...")
            
            # Neo4j 연결
            neo4j_manager = Neo4jManager(
                NEO4J_CONFIG['uri'],
                NEO4J_CONFIG['username'],
                NEO4J_CONFIG['password']
            )
            
            # 성능 최적화 인덱스 생성
            create_performance_indexes(neo4j_manager)
            
            # 분석기 초기화
            cognita_analyzer = CognitaRiskAnalyzer(neo4j_manager)
            
            # Flask 앱에 저장
            app.neo4j_manager = neo4j_manager
            app.cognita_analyzer = cognita_analyzer
            
            logger.info("Cognita Flask 백엔드 서비스 준비 완료")
            
        except Exception as e:
            logger.error(f"서비스 초기화 실패: {str(e)}")
            raise
    
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
                "status": "error",
                "neo4j_connected": False,
                "total_employees": 0,
                "total_relationships": 0,
                "message": "Neo4j 연결이 설정되지 않았습니다",
                "timestamp": datetime.now().isoformat()
            }), 503
        
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
    
    @app.route('/api/employees')
    def get_employees():
        """직원 목록 조회"""
        
        neo4j_mgr = get_neo4j_manager()
        if not neo4j_mgr:
            return jsonify({"error": "Neo4j 연결이 설정되지 않았습니다"}), 503
        
        # 쿼리 파라미터
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        department = request.args.get('department', None)
        
        try:
            # 부서 필터링 조건
            dept_filter = ""
            params = {"limit": limit, "offset": offset}
            
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
            SKIP $offset
            LIMIT $limit
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
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "count": len(employees)
                }
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
            
            # JSON 직렬화 가능한 형태로 변환
            return jsonify(risk_metrics.to_dict())
            
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
            sample_size = data.get('sample_size', 20)
            
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
            
            # 부서 분석 수행
            risk_analyses = analyzer.batch_analyze_department(department_name, sample_size)
            
            if not risk_analyses:
                return jsonify({"error": "부서 분석 결과가 없습니다"}), 500
            
            # 보고서 생성
            report = analyzer.generate_risk_report(risk_analyses)
            
            # 응답 데이터 구성
            response_data = {
                "department_name": department_name,
                "total_employees": total_employees,
                "analyzed_employees": len(risk_analyses),
                "risk_distribution": report["위험_분포"],
                "average_scores": report["평균_위험_점수"],
                "high_risk_employees": report["고위험_직원_상세"],
                "top_risk_factors": report["주요_위험_요인_빈도"],
                "recommendations": report["권장_조치사항"],
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"부서 분석 실패: {str(e)}")
            return jsonify({"error": f"부서 분석 실패: {str(e)}"}), 500
    
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
