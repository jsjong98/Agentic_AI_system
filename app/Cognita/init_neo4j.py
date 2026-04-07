#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j Graph DB initialization script for Railway deployment.
Checks if DB is empty; if so, loads employee_relationships.xml from Railway Volume at /app/data/.
Run once at container startup via start.sh.
"""

import os
import sys
import time
import logging
import xml.etree.ElementTree as ET

from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

NEO4J_URI      = os.environ.get('NEO4J_URI',      'bolt://localhost:7687')
NEO4J_USERNAME = os.environ.get('NEO4J_USERNAME',  'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD',  '')

XML_PATH = os.environ.get('NEO4J_INIT_XML', '/app/data/employee_relationships.xml')


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_driver():
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        max_connection_lifetime=3600,
        max_connection_pool_size=50,
        connection_acquisition_timeout=60
    )
    # verify connectivity
    with driver.session() as session:
        session.run("RETURN 1")
    logger.info("Neo4j 연결 성공")
    return driver


def wait_for_neo4j(retries=10, delay=5):
    for attempt in range(1, retries + 1):
        try:
            driver = get_driver()
            return driver
        except Exception as e:
            logger.warning(f"Neo4j 연결 시도 {attempt}/{retries} 실패: {e}")
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError("Neo4j에 연결할 수 없습니다.")


# ---------------------------------------------------------------------------
# Check if already initialised
# ---------------------------------------------------------------------------

def is_db_populated(driver):
    with driver.session() as session:
        count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
    logger.info(f"현재 노드 수: {count}")
    return count > 0


# ---------------------------------------------------------------------------
# Reset + Constraints
# ---------------------------------------------------------------------------

def reset_database(driver):
    logger.info("데이터베이스 초기화 시작...")
    with driver.session() as session:
        session.run("MATCH ()-[r]->() DELETE r")
        session.run("MATCH (n) DELETE n")

        constraints = list(session.run("SHOW CONSTRAINTS"))
        for row in constraints:
            name = row.get('name')
            if name:
                try:
                    session.run(f"DROP CONSTRAINT {name}")
                except Exception:
                    pass

        indexes = list(session.run("SHOW INDEXES"))
        for row in indexes:
            name = row.get('name')
            if name and not name.startswith('btree_'):
                try:
                    session.run(f"DROP INDEX {name}")
                except Exception:
                    pass

    time.sleep(3)
    logger.info("데이터베이스 초기화 완료")


def create_constraints_and_indexes(driver):
    constraints = [
        "CREATE CONSTRAINT emp_id_unique FOR (e:Employee) REQUIRE e.employee_id IS UNIQUE",
        "CREATE CONSTRAINT dept_name_unique FOR (d:Department) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT proj_id_unique FOR (p:Project) REQUIRE p.project_id IS UNIQUE",
    ]
    indexes = [
        "CREATE INDEX emp_job_level FOR (e:Employee) ON (e.job_level)",
        "CREATE INDEX emp_dept     FOR (e:Employee) ON (e.department)",
    ]
    time.sleep(2)
    with driver.session() as session:
        for q in constraints:
            try:
                session.run(q)
            except Exception as e:
                logger.warning(f"제약조건 생성 건너뜀: {e}")
        for q in indexes:
            try:
                session.run(q)
            except Exception as e:
                logger.warning(f"인덱스 생성 건너뜀: {e}")
    logger.info("제약조건 및 인덱스 생성 완료")


# ---------------------------------------------------------------------------
# XML Parsing
# ---------------------------------------------------------------------------

def parse_xml(xml_path):
    logger.info(f"XML 파싱 중: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = {
        'employees': [],
        'departments': set(),
        'projects': [],
        'hierarchy_relationships': [],
        'collaboration_relationships': [],
        'project_relationships': [],
    }

    # Projects
    projects_section = root.find('ProjectRegistry')
    if projects_section is not None:
        for p in projects_section.findall('Project'):
            data['projects'].append({
                'project_id':   p.get('project_id'),
                'project_name': p.get('project_name', p.get('project_id')),
                'project_type': p.get('project_type'),
                'team_size':    int(p.get('team_size', 0)),
                'start_date':   p.get('start_date'),
                'end_date':     p.get('end_date'),
                'status':       p.get('status'),
                'priority':     p.get('priority', 'medium'),
            })

    # Employees
    employees_section = root.find('Employees')
    if employees_section is not None:
        for emp in employees_section.findall('Employee'):
            eid = emp.get('id')
            data['employees'].append({
                'employee_id':      eid,
                'name':             emp.get('name'),
                'department':       emp.get('department'),
                'job_role':         emp.get('job_role'),
                'job_level':        int(emp.get('job_level', 1)),
                'years_at_company': int(emp.get('years_at_company', 0)),
            })
            data['departments'].add(emp.get('department'))

            # Hierarchy
            hierarchy = emp.find('HierarchyRelationships')
            if hierarchy is not None:
                reports_to = hierarchy.find('ReportsTo')
                if reports_to is not None:
                    data['hierarchy_relationships'].append({
                        'subordinate_id':     eid,
                        'manager_id':         reports_to.get('manager_id'),
                        'reporting_frequency': reports_to.get('reporting_frequency'),
                        'relationship_start': reports_to.get('reporting_since', '2023-01-01'),
                    })

            # Collaboration
            collab = emp.find('CollaborationRelationships')
            if collab is not None:
                for c in collab.findall('Colleague'):
                    data['collaboration_relationships'].append({
                        'employee1_id':         eid,
                        'employee2_id':         c.get('colleague_id'),
                        'collaboration_strength': float(c.get('collaboration_strength', 0)),
                        'interaction_frequency':  float(c.get('interaction_frequency', 0)),
                        'collaboration_type':     c.get('collaboration_type'),
                        'relationship_quality':   c.get('relationship_quality'),
                        'common_projects':        int(c.get('common_projects', 0)),
                    })

            # Projects
            projects = emp.find('ProjectRelationships')
            if projects is not None:
                for pr in projects.findall('ProjectParticipation'):
                    data['project_relationships'].append({
                        'employee_id':     eid,
                        'project_id':      pr.get('project_id'),
                        'role_in_project': pr.get('role_in_project'),
                        'contribution_level': float(pr.get('contribution_level', 0)),
                        'project_start':   pr.get('project_start'),
                        'project_end':     pr.get('project_end'),
                        'project_status':  pr.get('project_status'),
                    })

    data['departments'] = [{'name': d} for d in data['departments']]

    logger.info(
        f"파싱 완료 — 직원: {len(data['employees'])}, 부서: {len(data['departments'])}, "
        f"프로젝트: {len(data['projects'])}, 계층: {len(data['hierarchy_relationships'])}, "
        f"협업: {len(data['collaboration_relationships'])}, 프로젝트참여: {len(data['project_relationships'])}"
    )
    return data


# ---------------------------------------------------------------------------
# Node creation
# ---------------------------------------------------------------------------

def create_employee_nodes(driver, employees):
    batch_size = 500
    total = 0
    query = """
    UNWIND $employees AS employee
    MERGE (e:Employee {employee_id: employee.employee_id})
    ON CREATE SET
        e.name             = employee.name,
        e.department       = employee.department,
        e.job_role         = employee.job_role,
        e.job_level        = employee.job_level,
        e.years_at_company = employee.years_at_company,
        e.created_at       = datetime()
    ON MATCH SET
        e.name             = employee.name,
        e.department       = employee.department,
        e.job_role         = employee.job_role,
        e.job_level        = employee.job_level,
        e.years_at_company = employee.years_at_company
    """
    for i in range(0, len(employees), batch_size):
        batch = employees[i:i + batch_size]
        with driver.session() as session:
            result = session.run(query, employees=batch)
            total += result.consume().counters.nodes_created
    logger.info(f"직원 노드 생성: {total}개")


def create_department_nodes(driver, departments):
    query = """
    UNWIND $departments AS dept
    MERGE (d:Department {name: dept.name})
    ON CREATE SET d.created_at = datetime()
    """
    with driver.session() as session:
        result = session.run(query, departments=departments)
        logger.info(f"부서 노드 생성: {result.consume().counters.nodes_created}개")


def create_project_nodes(driver, projects):
    query = """
    UNWIND $projects AS project
    MERGE (p:Project {project_id: project.project_id})
    ON CREATE SET
        p.project_name = project.project_name,
        p.project_type = project.project_type,
        p.team_size    = project.team_size,
        p.start_date   = CASE WHEN project.start_date IS NOT NULL AND project.start_date <> ''
                              THEN date(project.start_date) ELSE null END,
        p.end_date     = CASE WHEN project.end_date IS NOT NULL AND project.end_date <> ''
                              THEN date(project.end_date) ELSE null END,
        p.status       = project.status,
        p.priority     = project.priority,
        p.created_at   = datetime()
    """
    with driver.session() as session:
        result = session.run(query, projects=projects)
        logger.info(f"프로젝트 노드 생성: {result.consume().counters.nodes_created}개")


# ---------------------------------------------------------------------------
# Relationship creation
# ---------------------------------------------------------------------------

def create_hierarchy_relationships(driver, hierarchy_data):
    query = """
    UNWIND $relationships AS rel
    MATCH (sub:Employee {employee_id: rel.subordinate_id})
    MATCH (mgr:Employee {employee_id: rel.manager_id})
    CREATE (sub)-[:REPORTS_TO {
        reporting_frequency: rel.reporting_frequency,
        relationship_start: CASE
            WHEN rel.relationship_start IS NOT NULL AND rel.relationship_start <> ''
            THEN date(rel.relationship_start) ELSE null END,
        created_at: datetime()
    }]->(mgr)
    """
    with driver.session() as session:
        result = session.run(query, relationships=hierarchy_data)
        logger.info(f"REPORTS_TO 관계 생성: {result.consume().counters.relationships_created}개")


def create_works_in_relationships(driver, employees):
    """Employee → Department WORKS_IN 관계"""
    query = """
    UNWIND $employees AS emp
    MATCH (e:Employee   {employee_id: emp.employee_id})
    MATCH (d:Department {name: emp.department})
    CREATE (e)-[:WORKS_IN {created_at: datetime()}]->(d)
    """
    batch_size = 500
    total = 0
    for i in range(0, len(employees), batch_size):
        batch = employees[i:i + batch_size]
        with driver.session() as session:
            result = session.run(query, employees=batch)
            total += result.consume().counters.relationships_created
    logger.info(f"WORKS_IN 관계 생성: {total}개")


def create_collaboration_relationships(driver, collaboration_data):
    # deduplicate bidirectional pairs
    unique, seen = [], set()
    for c in collaboration_data:
        pair = tuple(sorted([c['employee1_id'], c['employee2_id']]))
        if pair not in seen:
            seen.add(pair)
            unique.append(c)

    query = """
    UNWIND $relationships AS rel
    MATCH (e1:Employee {employee_id: rel.employee1_id})
    MATCH (e2:Employee {employee_id: rel.employee2_id})
    CREATE (e1)-[:COLLABORATES_WITH {
        collaboration_strength: rel.collaboration_strength,
        interaction_frequency:  rel.interaction_frequency,
        collaboration_type:     rel.collaboration_type,
        relationship_quality:   rel.relationship_quality,
        common_projects:        rel.common_projects,
        created_at:             datetime()
    }]->(e2)
    CREATE (e2)-[:COLLABORATES_WITH {
        collaboration_strength: rel.collaboration_strength,
        interaction_frequency:  rel.interaction_frequency,
        collaboration_type:     rel.collaboration_type,
        relationship_quality:   rel.relationship_quality,
        common_projects:        rel.common_projects,
        created_at:             datetime()
    }]->(e1)
    """
    batch_size = 1000
    total = 0
    for i in range(0, len(unique), batch_size):
        batch = unique[i:i + batch_size]
        with driver.session() as session:
            result = session.run(query, relationships=batch)
            total += result.consume().counters.relationships_created
    logger.info(f"COLLABORATES_WITH 관계 생성: {total}개")


def create_project_participation_relationships(driver, project_relationships):
    query = """
    UNWIND $relationships AS rel
    MATCH (e:Employee {employee_id: rel.employee_id})
    MATCH (p:Project  {project_id:  rel.project_id})
    CREATE (e)-[:PARTICIPATES_IN {
        role_in_project:    rel.role_in_project,
        contribution_level: rel.contribution_level,
        project_start: CASE
            WHEN rel.project_start IS NOT NULL AND rel.project_start <> ''
            THEN date(rel.project_start) ELSE null END,
        project_end: CASE
            WHEN rel.project_end IS NOT NULL AND rel.project_end <> ''
            THEN date(rel.project_end) ELSE null END,
        project_status: rel.project_status,
        created_at:     datetime()
    }]->(p)
    """
    batch_size = 1000
    total = 0
    for i in range(0, len(project_relationships), batch_size):
        batch = project_relationships[i:i + batch_size]
        with driver.session() as session:
            result = session.run(query, relationships=batch)
            total += result.consume().counters.relationships_created
    logger.info(f"PARTICIPATES_IN 관계 생성: {total}개")


def create_project_colleague_relationships(driver):
    query = """
    MATCH (e1:Employee)-[:PARTICIPATES_IN]->(p:Project)<-[:PARTICIPATES_IN]-(e2:Employee)
    WHERE e1.employee_id < e2.employee_id
    CREATE (e1)-[:PROJECT_COLLEAGUE {shared_project: p.project_id, created_at: datetime()}]->(e2)
    CREATE (e2)-[:PROJECT_COLLEAGUE {shared_project: p.project_id, created_at: datetime()}]->(e1)
    """
    with driver.session() as session:
        result = session.run(query)
        logger.info(f"PROJECT_COLLEAGUE 관계 생성: {result.consume().counters.relationships_created}개")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=== Neo4j 초기화 스크립트 시작 ===")

    driver = wait_for_neo4j()

    if is_db_populated(driver):
        logger.info("Neo4j에 이미 데이터가 있습니다. 초기화를 건너뜁니다.")
        driver.close()
        return

    if not os.path.exists(XML_PATH):
        logger.warning(
            f"XML 파일이 없습니다: {XML_PATH}\n"
            "Railway Volume에 employee_relationships.xml을 업로드한 뒤 재시작하세요.\n"
            "초기화 없이 Flask 서버를 시작합니다."
        )
        driver.close()
        return

    logger.info(f"XML 파일 발견: {XML_PATH}. Neo4j 초기화를 시작합니다.")

    reset_database(driver)
    create_constraints_and_indexes(driver)

    data = parse_xml(XML_PATH)

    # Nodes
    create_employee_nodes(driver,   data['employees'])
    create_department_nodes(driver, data['departments'])
    create_project_nodes(driver,    data['projects'])

    # Relationships
    create_hierarchy_relationships(driver,            data['hierarchy_relationships'])
    create_works_in_relationships(driver,             data['employees'])
    create_collaboration_relationships(driver,        data['collaboration_relationships'])
    create_project_participation_relationships(driver, data['project_relationships'])
    create_project_colleague_relationships(driver)

    with driver.session() as session:
        n = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        r = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    logger.info(f"=== 초기화 완료 — 노드: {n}개, 관계: {r}개 ===")

    driver.close()


if __name__ == '__main__':
    main()
