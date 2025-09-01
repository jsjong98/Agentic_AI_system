# Neo4j 그래프 데이터베이스 구축 가이드
## `Cognita_graph_development.ipynb`

## 개요
이 노트북은 XML 형태의 직원 관계 데이터를 Neo4j 그래프 데이터베이스로 변환하여 시각적 분석과 고급 쿼리를 가능하게 하는 시스템입니다. 조직 네트워크 분석, 위험 전파 경로 추적, 협업 패턴 분석 등 다양한 인사이트를 도출할 수 있습니다.

## 시스템 아키텍처

### 입력 데이터
- **XML 파일**: `./data/employee_relationships.xml` (289.44 MB)
- **데이터 규모**: 1,470명 직원, 3개 부서, 25개 프로젝트
- **관계 데이터**: 계층관계 295개, 협업관계 1,098,084개, 프로젝트 참여 138개

### 출력 결과
- **Neo4j 그래프 데이터베이스**: 완전한 조직 네트워크 그래프
- **시각화 인터페이스**: Neo4j Browser를 통한 대화형 탐색
- **쿼리 지원**: Cypher 언어를 통한 고급 분석

## 구현 단계별 가이드

### 1단계: 환경 설정 및 Neo4j 연결

#### 필수 라이브러리
```python
# 설치 필요 라이브러리
pip install neo4j pandas xmltodict
```

#### Neo4j 서버 설정
```python
NEO4J_CONFIG = {
    "uri": "bolt://18.208.202.177:7687",    # Neo4j 서버 주소
    "username": "neo4j",                     # 사용자명
    "password": "handwriting-collar-dive"    # 비밀번호 (실제 환경에 맞게 수정)
}
```

#### Neo4jManager 클래스
- **연결 관리**: 자동 연결 및 상태 확인
- **데이터베이스 초기화**: 기존 데이터 삭제
- **제약조건 설정**: 유니크 제약 및 인덱스 생성
- **에러 핸들링**: 연결 실패 시 상세 정보 제공

### 2단계: XML 데이터 파싱 및 구조 변환

#### XML 파싱 프로세스
```python
def parse_xml_to_graph_data(xml_file_path):
    """XML을 그래프 데이터 구조로 변환"""
```

**추출되는 데이터 구조:**
- **직원 정보** (1,470개)
  - 기본 속성: employee_id, name, department, job_role, job_level
  - 분석 속성: years_at_company, risk_tier, persona
- **부서 정보** (3개)
  - Sales, Research & Development, Human Resources
- **프로젝트 정보** (25개)
  - 프로젝트 유형: single_department (60%), cross_department (40%)
  - 상태: active (80%), completed (20%)

#### 관계 데이터 추출
1. **계층 관계**: 상사-부하 관계 (295개)
2. **협업 관계**: 동료 간 협업 강도 및 품질 (1,098,084개)
3. **프로젝트 참여**: 직원-프로젝트 연결 (138개)

### 3단계: 그래프 노드 생성

#### Employee 노드
```cypher
CREATE (e:Employee {
    employee_id: "1",
    name: "Employee_1",
    department: "Sales",
    job_role: "Sales Executive",
    job_level: 2,
    years_at_company: 6,
    risk_tier: "High",
    persona: "Burnout (저성과자/번아웃)",
    created_at: datetime()
})
```

#### Department 노드
```cypher
CREATE (d:Department {
    name: "Sales",
    created_at: datetime()
})
```

#### Project 노드
```cypher
CREATE (p:Project {
    project_id: "PRJ_001",
    project_name: "Project_PRJ_001",
    project_type: "single_department",
    team_size: 8,
    start_date: date("2023-01-15"),
    end_date: date("2023-12-15"),
    status: "active",
    priority: "high",
    created_at: datetime()
})
```

### 4단계: 관계 생성 및 최적화

#### 기본 관계
1. **WORKS_IN**: 직원 → 부서 (1,470개)
```cypher
MATCH (e:Employee), (d:Department)
WHERE e.department = d.name
CREATE (e)-[:WORKS_IN {created_at: datetime()}]->(d)
```

#### 조직 관계
2. **REPORTS_TO**: 부하직원 → 상사 (295개)
```cypher
CREATE (sub)-[:REPORTS_TO {
    reporting_frequency: "weekly",
    relationship_start: date("2023-01-01"),
    created_at: datetime()
}]->(mgr)
```

3. **COLLABORATES_WITH**: 직원 ↔ 동료 (549,042개, 단방향 저장)
```cypher
CREATE (e1)-[:COLLABORATES_WITH {
    collaboration_strength: 0.8,
    interaction_frequency: 0.6,
    collaboration_type: "same_department",
    relationship_quality: "excellent",
    common_projects: 2,
    is_bidirectional: true,
    created_at: datetime()
}]->(e2)
```

4. **PARTICIPATES_IN**: 직원 → 프로젝트 (138개)
```cypher
CREATE (e)-[:PARTICIPATES_IN {
    role_in_project: "lead",
    contribution_level: 0.9,
    project_start: date("2023-01-15"),
    project_end: date("2023-12-15"),
    project_status: "active",
    created_at: datetime()
}]->(p)
```

#### 파생 관계
5. **PROJECT_COLLEAGUE**: 같은 프로젝트 팀원 간 관계
```cypher
MATCH (e1:Employee)-[:PARTICIPATES_IN]->(p:Project)<-[:PARTICIPATES_IN]-(e2:Employee)
WHERE e1.employee_id < e2.employee_id
CREATE (e1)-[:PROJECT_COLLEAGUE {
    shared_project: p.project_id,
    project_name: p.project_name,
    is_bidirectional: true,
    created_at: datetime()
}]->(e2)
```

6. **SAME_TEAM**: 같은 부서/직급 직원 간 관계
```cypher
MATCH (e1:Employee), (e2:Employee)
WHERE e1.department = e2.department 
AND e1.job_level = e2.job_level 
AND e1.employee_id < e2.employee_id
CREATE (e1)-[:SAME_TEAM {
    department: e1.department,
    job_level: e1.job_level,
    is_bidirectional: true,
    created_at: datetime()
}]->(e2)
```

### 5단계: 데이터베이스 최적화

#### 제약조건 및 인덱스
```cypher
-- 유니크 제약조건
CREATE CONSTRAINT FOR (e:Employee) REQUIRE e.employee_id IS UNIQUE;
CREATE CONSTRAINT FOR (d:Department) REQUIRE d.name IS UNIQUE;
CREATE CONSTRAINT FOR (p:Project) REQUIRE p.project_id IS UNIQUE;

-- 성능 최적화 인덱스
CREATE INDEX FOR (e:Employee) ON (e.risk_tier);
CREATE INDEX FOR (e:Employee) ON (e.job_level);
CREATE INDEX FOR (e:Employee) ON (e.persona);
```

#### 중복 제거 최적화
- **협업 관계**: 양방향 관계를 단방향으로 저장하여 중복 제거
- **ID 정렬**: 작은 ID → 큰 ID 방향으로 일관성 유지
- **메모리 효율성**: 549,042개 관계로 50% 저장 공간 절약

## 시각화 및 분석 쿼리

### 조직 구조 분석
```cypher
// 부서별 계층 구조
MATCH (e:Employee)-[:WORKS_IN]->(d:Department {name: 'Sales'})
OPTIONAL MATCH (e)-[r:REPORTS_TO]->(manager)
RETURN e, r, manager, d
LIMIT 20
```

### 위험 전파 분석
```cypher
// 고위험 직원으로부터 2단계 이내 영향 경로
MATCH (high_risk:Employee {risk_tier: 'High'})
MATCH path = (high_risk)-[*1..2]-(connected:Employee)
WHERE connected.risk_tier IN ['High', 'Neutral']
RETURN path
LIMIT 40
```

### 협업 네트워크 분석
```cypher
// 부서간 강한 협업 관계
MATCH (e1:Employee)-[:WORKS_IN]->(d1:Department)
MATCH (e2:Employee)-[:WORKS_IN]->(d2:Department)
MATCH (e1)-[c:COLLABORATES_WITH]-(e2)
WHERE d1 <> d2 AND c.collaboration_strength > 0.6
RETURN e1, e2, c, d1, d2
LIMIT 25
```

### 프로젝트 팀 분석
```cypher
// 활성 프로젝트의 팀 구성 및 협업 관계
MATCH (p:Project {status: 'active'})<-[:PARTICIPATES_IN]-(e:Employee)
OPTIONAL MATCH (e)-[c:COLLABORATES_WITH]-(colleague)-[:PARTICIPATES_IN]->(p)
RETURN p, e, c, colleague
LIMIT 30
```

## 성능 및 통계

### 최종 그래프 통계
- **총 노드 수**: 1,498개 (Employee: 1,470, Department: 3, Project: 25)
- **총 관계 수**: 약 550,000개
- **메모리 사용량**: 약 100-200MB (인덱스 포함)
- **쿼리 성능**: 평균 응답시간 < 100ms

### 관계별 분포
- **WORKS_IN**: 1,470개 (직원-부서)
- **REPORTS_TO**: 295개 (계층관계)
- **COLLABORATES_WITH**: 549,042개 (협업관계, 단방향)
- **PARTICIPATES_IN**: 138개 (프로젝트 참여)
- **PROJECT_COLLEAGUE**: 자동 생성 (프로젝트 내 팀원)
- **SAME_TEAM**: 자동 생성 (같은 팀)

## 활용 사례

### 1. 조직 네트워크 분석
- **중심성 분석**: 영향력 있는 직원 식별
- **클러스터 분석**: 부서 간 협업 패턴 발견
- **경로 분석**: 정보 전파 경로 추적

### 2. 위험 관리
- **위험 전파**: 고위험 직원의 영향 범위 분석
- **조기 경보**: 위험 신호 패턴 탐지
- **개입 전략**: 타겟 직원 및 관계 식별

### 3. 팀 최적화
- **팀 구성**: 협업 강도 기반 최적 팀 구성
- **프로젝트 배정**: 과거 협업 이력 활용
- **갈등 예방**: 관계 품질 모니터링

### 4. 조직 개발
- **페르소나별 전략**: 맞춤형 개입 프로그램
- **경력 개발**: 성장 경로 시각화
- **멘토링**: 최적 멘토-멘티 매칭

## 기술적 고려사항

### 확장성
- **수평 확장**: Neo4j 클러스터링 지원
- **데이터 증가**: 인덱스 최적화로 성능 유지
- **실시간 업데이트**: 증분 업데이트 지원

### 보안
- **접근 제어**: 역할 기반 권한 관리
- **데이터 암호화**: 전송 및 저장 시 암호화
- **감사 로그**: 모든 접근 기록 추적

### 모니터링
- **성능 메트릭**: 쿼리 실행 시간 모니터링
- **리소스 사용량**: 메모리 및 CPU 사용률 추적
- **데이터 품질**: 관계 일관성 검증

## Neo4j Browser 활용 가이드

### 접속 정보
- **URL**: http://localhost:7474
- **인증**: username: neo4j, password: [설정한 비밀번호]

### 시각화 설정
- **노드 크기**: 직급(job_level) 또는 위험도(risk_tier)에 따라 조정
- **노드 색상**: 부서별 또는 위험등급별로 구분
- **관계 두께**: 협업 강도(collaboration_strength)에 따라 조정
- **레이아웃**: Force-directed 또는 Hierarchical 레이아웃 사용

### 대화형 탐색
1. **드릴다운**: 특정 노드 클릭으로 연결된 관계 확장
2. **필터링**: 속성 값에 따른 노드/관계 필터링
3. **경로 탐색**: 두 노드 간 최단 경로 찾기
4. **클러스터링**: 유사한 특성의 노드 그룹화

## 문제 해결 가이드

### 연결 문제
```bash
# Neo4j 서버 상태 확인
systemctl status neo4j

# 포트 확인
netstat -tlnp | grep 7687

# 방화벽 설정
sudo ufw allow 7687
```

### 성능 최적화
```cypher
-- 쿼리 성능 분석
PROFILE MATCH (n:Employee) RETURN count(n);

-- 인덱스 사용 확인
EXPLAIN MATCH (e:Employee {risk_tier: 'High'}) RETURN e;
```

### 데이터 검증
```cypher
-- 고아 노드 확인
MATCH (n) WHERE NOT (n)--() RETURN count(n);

-- 관계 일관성 확인
MATCH ()-[r:COLLABORATES_WITH]->() 
WHERE r.is_bidirectional = true 
RETURN count(r);
```

이 시스템을 통해 복잡한 조직 네트워크를 직관적으로 시각화하고 분석하여 데이터 기반의 인사 의사결정을 지원할 수 있습니다.
