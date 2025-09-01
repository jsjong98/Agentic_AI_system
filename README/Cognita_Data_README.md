# Cognita 데이터 구성 가이드

## 개요
이 문서는 IBM HR 데이터셋을 기반으로 직원 관계 네트워크 데이터를 구성하는 과정을 설명합니다. 총 3단계의 데이터 처리 과정을 통해 최종적으로 Neo4j 그래프 데이터베이스까지 구축합니다.

## 데이터 처리 파이프라인

### 1단계: 직원 페르소나 할당 (`ibmHR_persona_assignment.ipynb`)

#### 목적
IBM HR 데이터셋의 각 직원에게 7가지 페르소나를 할당하여 직원의 특성과 위험도를 분류합니다.

#### 입력 데이터
- **파일**: `./data/IBM_HR.csv`
- **규모**: 1,470명의 직원 데이터
- **주요 컬럼**: JobSatisfaction, PerformanceRating, JobInvolvement, YearsAtCompany, RelationshipSatisfaction 등

#### 처리 과정

1. **데이터 전처리**
   - 필수 컬럼 존재 여부 확인
   - OverTime (Yes/No) → 이진값 (1/0) 변환
   - Min-Max 정규화를 통해 모든 수치형 변수를 0~1 범위로 스케일링

2. **페르소나 점수 계산**
   7가지 페르소나에 대해 가중합 방식으로 적합도 점수 계산:
   
   **문제 직원 (Problem) - 위험도: High**
   - **P01 Burnout (번아웃)**: 초과근무↑, 직무만족↓, 성과↓, 몰입↓
   - **P02 Onboarding Failure (온보딩 실패)**: 근속↓, 관계만족↓, 직무만족↓
   - **P03 Career Stall (성장 정체)**: 승진 후 기간↑, 교육횟수↓, 몰입↓

   **안정 직원 (Stable) - 위험도: Stable**
   - **S01 Anchor (앵커)**: 근속↑, 총경력↑, 이직횟수↓
   - **S02 Rising Star (라이징 스타)**: 성과↑, 최근 승진, 인상률↑

   **중립 직원 (Neutral) - 위험도: Neutral**
   - **N01 Coaster (현상 유지자)**: 몰입↓, 인상률↓, 성과↓
   - **N02 Competent Malcontent (유능한 불만자)**: 성과↑, 관계만족↓

3. **페르소나 할당**
   - **결정적 할당 (Argmax)**: 가장 높은 점수의 페르소나 선택
   - **확률적 할당 (Softmax)**: 점수를 확률로 변환하여 무작위 선택 (재현성을 위한 시드 고정)

#### 출력 데이터
- **파일**: `./data/IBM_HR_personas_assigned.csv`
- **추가 컬럼**: 
  - 각 페르소나별 점수 (Score_P01~N02)
  - 각 페르소나별 확률 (Prob_P01~N02)
  - 결정적/확률적 할당 결과 (argmax_*, softmax_*)

### 2단계: 직원 관계 데이터 생성 (`Cognita_data_assignment.ipynb`)

#### 목적
페르소나가 할당된 직원 데이터를 기반으로 조직 내 다양한 관계를 시뮬레이션하여 XML/JSON 형태의 관계 데이터를 생성합니다.

#### 입력 데이터
- **파일**: `./data/IBM_HR_personas_assigned.csv`
- **규모**: 1,470명의 직원 (페르소나 할당 완료)

#### 관계 생성 로직

1. **계층 관계 (Hierarchy Relationships)**
   - 부서별로 JobLevel과 YearsAtCompany를 기준으로 상사-부하 관계 생성
   - 관리 범위(span of control): 3-8명
   - 생성된 관계: 891개

2. **협업 관계 (Collaboration Relationships)**
   - 같은 부서 내 직원들 간의 협업 강도 계산
   - 직급 차이, 직무 유사성을 고려한 가중치 적용
   - 협업 강도 0.4 이상인 관계만 저장
   - 생성된 관계: 1,098,084개 (양방향)

3. **프로젝트 관계 (Project Relationships)**
   - 25개의 가상 프로젝트 생성 (부서 내/부서 간)
   - 팀 규모: 3-9명
   - 프로젝트 유형: single_department (60%), cross_department (40%)
   - 생성된 관계: 138개

#### 출력 데이터
- **XML 파일**: `./data/employee_relationships.xml` (289.44 MB)
- **JSON 파일**: `./data/employee_relationships.json` (445.85 MB)

#### XML 구조
```xml
<OrganizationNetwork>
  <ProjectRegistry>
    <Project project_id="PRJ_001" project_type="single_department" .../>
  </ProjectRegistry>
  <Employees>
    <Employee id="1" department="Sales" job_role="Sales Executive" ...>
      <HierarchyRelationships>
        <ReportsTo manager_id="..." reporting_frequency="weekly"/>
        <Manages>
          <Subordinate employee_id="..." job_role="..."/>
        </Manages>
      </HierarchyRelationships>
      <CollaborationRelationships>
        <Colleague colleague_id="..." collaboration_strength="0.8"/>
      </CollaborationRelationships>
      <ProjectRelationships>
        <ProjectParticipation project_id="PRJ_001" role_in_project="lead">
          <Teammates>
            <Teammate teammate_id="..." collaboration_intensity="0.9"/>
          </Teammates>
        </ProjectParticipation>
      </ProjectRelationships>
    </Employee>
  </Employees>
</OrganizationNetwork>
```

### 3단계: Neo4j 그래프 데이터베이스 구축 (`Cognita_graph_development.ipynb`)

#### 목적
XML 관계 데이터를 Neo4j 그래프 데이터베이스로 변환하여 시각적 분석과 고급 쿼리를 가능하게 합니다.

#### 입력 데이터
- **파일**: `./data/employee_relationships.xml`
- **Neo4j 서버**: `bolt://18.208.202.177:7687`

#### 그래프 구조

**노드 타입**
- **Employee**: 직원 (1,470개)
  - 속성: employee_id, name, department, job_role, job_level, risk_tier, persona 등
- **Department**: 부서 (3개)
  - 속성: name
- **Project**: 프로젝트 (25개)
  - 속성: project_id, project_name, project_type, team_size, status 등

**관계 타입**
- **WORKS_IN**: 직원 → 부서 (1,470개)
- **REPORTS_TO**: 부하직원 → 상사 (295개)
- **COLLABORATES_WITH**: 직원 ↔ 동료 (549,042개, 단방향으로 저장)
- **PARTICIPATES_IN**: 직원 → 프로젝트 (138개)
- **PROJECT_COLLEAGUE**: 같은 프로젝트 팀원 간 관계
- **SAME_TEAM**: 같은 부서/직급 직원 간 관계

#### 데이터베이스 구축 과정

1. **연결 설정 및 초기화**
   - Neo4j 서버 연결
   - 기존 데이터 삭제
   - 제약조건 및 인덱스 생성

2. **XML 파싱 및 데이터 추출**
   - XML 파일을 파싱하여 노드/관계 데이터 추출
   - 데이터 구조 검증 및 변환

3. **노드 생성**
   - Employee, Department, Project 노드 생성
   - 속성 값 설정 및 데이터 타입 변환

4. **관계 생성**
   - 기본 관계 (WORKS_IN) 생성
   - 조직 관계 (REPORTS_TO, COLLABORATES_WITH, PARTICIPATES_IN) 생성
   - 파생 관계 (PROJECT_COLLEAGUE, SAME_TEAM) 생성

#### 시각화 쿼리 예시

```cypher
// 고위험 직원과 연결된 관계 시각화
MATCH (high_risk:Employee {risk_tier: 'High'})
OPTIONAL MATCH (high_risk)-[r1]-(connected)
RETURN high_risk, r1, connected
LIMIT 50

// 부서간 협업 네트워크
MATCH (e1:Employee)-[:WORKS_IN]->(d1:Department)
MATCH (e2:Employee)-[:WORKS_IN]->(d2:Department)
MATCH (e1)-[c:COLLABORATES_WITH]-(e2)
WHERE d1 <> d2 AND c.collaboration_strength > 0.6
RETURN e1, e2, c, d1, d2
LIMIT 25
```

## 최종 데이터 구성 결과

### 파일 구조
```
data/
├── IBM_HR.csv                          # 원본 데이터 (1,470명)
├── IBM_HR_personas_assigned.csv        # 페르소나 할당 결과
├── employee_relationships.xml          # XML 관계 데이터 (289.44 MB)
└── employee_relationships.json         # JSON 관계 데이터 (445.85 MB)
```

### 데이터 통계
- **직원 수**: 1,470명
- **부서 수**: 3개 (Sales, Research & Development, Human Resources)
- **프로젝트 수**: 25개
- **계층 관계**: 295개
- **협업 관계**: 1,098,084개 (양방향)
- **프로젝트 참여**: 138개

### 페르소나 분포
각 페르소나별 직원 분포와 위험도 등급이 균형있게 할당되어 조직 내 다양한 직원 유형을 반영합니다.

## 활용 방안

1. **조직 네트워크 분석**: 부서 간 협업 패턴, 영향력 있는 직원 식별
2. **위험 전파 분석**: 고위험 직원으로부터의 영향 경로 추적
3. **프로젝트 팀 최적화**: 협업 강도와 페르소나를 고려한 팀 구성
4. **조직 개발**: 페르소나별 맞춤형 개입 전략 수립

## 기술 스택

- **데이터 처리**: Python, Pandas, NumPy
- **관계 데이터**: XML, JSON
- **그래프 데이터베이스**: Neo4j
- **시각화**: Neo4j Browser, Cypher 쿼리

이 데이터 구성 과정을 통해 단순한 HR 데이터를 복잡한 조직 네트워크 데이터로 확장하여 다양한 분석과 인사이트 도출이 가능한 형태로 변환했습니다.
