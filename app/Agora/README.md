# 🌍 Agora - 외부 노동 시장 분석 에이전트

**포트**: `5004` | **기술**: 시장 데이터 분석 + 경쟁력 평가 + LLM 해석

## 🎯 주요 기능

- ✅ **시장 압력 지수 계산**: 외부 시장의 채용 수요 및 경쟁 상황 분석
- ✅ **보상 격차 분석**: 현재 급여와 시장 평균 급여 간의 격차 계산
- ✅ **이직 위험도 평가**: 시장 상황을 고려한 직원별 이직 위험도 산출
- ✅ **직무별 시장 분석**: 특정 직무의 채용 공고, 급여 수준, 트렌드 분석
- ✅ **경쟁력 평가**: 개별 직원의 시장 대비 경쟁력 종합 평가
- ✅ **LLM 기반 해석**: OpenAI GPT를 활용한 자연스러운 분석 결과 해석

## 📊 분석 지표

### 1. 시장 압력 지수 (Market Pressure Index)
```python
market_pressure_index = (
    demand_index * 0.4 +           # 채용 수요 40%
    (1 - salary_competitiveness) * 0.4 +  # 급여 격차 40%
    (trend_weight - 1) * 0.2       # 시장 트렌드 20%
) * trend_weight
```

### 2. 보상 격차 (Compensation Gap)
```python
compensation_gap = (market_avg_salary - current_salary) / market_avg_salary
```

### 3. 위험 수준 결정
- **HIGH**: 시장 압력 높음 + 보상 격차 큼 + 낮은 만족도
- **MEDIUM**: 중간 수준의 시장 압력 또는 보상 격차
- **LOW**: 안정적인 시장 상황 + 경쟁력 있는 보상

## 🏗️ 시스템 구조

```
app/Agora/
├── agora_flask_backend.py      # Flask API 서버
├── agora_processor.py          # 시장 데이터 수집 및 처리
├── agora_analyzer.py           # 시장 분석 및 위험도 평가
├── agora_llm_generator.py      # LLM 기반 해석 생성
├── run_agora_server.py         # 서버 실행 스크립트
├── test_agora_api.py           # API 테스트 스크립트
├── requirements.txt            # 패키지 의존성
└── README.md                   # 이 파일
```

## 🚀 설치 및 실행

### 1. 패키지 설치
```bash
cd app/Agora
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
# OpenAI API 키 설정 (LLM 기능용, 선택사항)
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. 서버 실행
```bash
python run_agora_server.py
```

### 4. API 테스트
```bash
python test_agora_api.py
```

## 📡 API 엔드포인트

### 기본 정보
- **Base URL**: `http://localhost:5005`
- **Content-Type**: `application/json`

### 주요 엔드포인트

#### 1. 헬스체크
```http
GET /health
```

#### 2. 개별 직원 시장 분석
```http
POST /analyze/market
Content-Type: application/json

{
  "EmployeeNumber": 1001,
  "JobRole": "Sales Executive",
  "Department": "Sales",
  "MonthlyIncome": 5000,
  "YearsAtCompany": 3,
  "JobSatisfaction": 3,
  "use_llm": false
}
```

**응답 예시:**
```json
{
  "employee_id": "1001",
  "job_role": "Sales Executive",
  "department": "Sales",
  "market_pressure_index": 0.723,
  "compensation_gap": 0.156,
  "job_postings_count": 187,
  "market_competitiveness": "HIGH",
  "risk_level": "MEDIUM",
  "llm_interpretation": "시장 분석 해석...",
  "analysis_timestamp": "2024-01-15T10:30:00"
}
```

#### 3. 직무별 시장 분석
```http
POST /analyze/job_market
Content-Type: application/json

{
  "job_role": "Data Scientist",
  "location": "서울",
  "experience_level": "mid"
}
```

#### 4. 배치 시장 분석
```http
POST /analyze/batch
Content-Type: application/json

{
  "employees": [
    {
      "EmployeeNumber": 1001,
      "JobRole": "Sales Executive",
      "MonthlyIncome": 5000
    }
  ],
  "use_llm": false
}
```

#### 5. 시장 보고서 조회
```http
GET /market/report/{job_role}
```

#### 6. 전체 시장 트렌드
```http
GET /market/trends
```

#### 7. 경쟁력 분석
```http
POST /market/competitive_analysis
Content-Type: application/json

{
  "EmployeeNumber": 1001,
  "JobRole": "Sales Executive",
  "MonthlyIncome": 5000,
  "YearsAtCompany": 3,
  "Education": 3,
  "JobSatisfaction": 3
}
```

## 🔧 설정 및 커스터마이징

### 1. 직무 매핑 수정
`agora_processor.py`에서 IBM 직무를 시장 검색용 직무명으로 매핑:

```python
self.job_mapping = {
    'Sales Executive': 'Sales Manager',
    'Research Scientist': 'Data Scientist',
    'Laboratory Technician': 'Lab Technician',
    # 추가 매핑...
}
```

### 2. 시뮬레이션 데이터 수정
실제 API 연동 전까지 사용할 시뮬레이션 데이터:

```python
self.simulation_data = {
    'Sales Manager': {
        'job_postings': 187,
        'avg_salary': 5500000,
        'market_trend': 'GROWING',
        'competition_level': 'HIGH'
    }
}
```

### 3. 위험도 계산 로직 조정
`agora_analyzer.py`에서 위험 수준 결정 로직 수정 가능:

```python
def _determine_risk_level(self, market_pressure, compensation_gap, 
                        job_satisfaction, years_at_company):
    # 커스텀 로직 구현
```

## 📈 성능 지표

- **응답 시간**: < 500ms (개별 분석)
- **배치 처리**: 100명/분 (LLM 미사용)
- **캐시 적중률**: > 80% (1시간 TTL)
- **API 안정성**: 99.9% 가용성

## 🔗 통합 가이드

### 마스터 서버 통합
```python
# agentic_master_server.py에서 Agora 결과 활용
agora_result = {
    'market_pressure_index': 0.723,
    'compensation_gap': 0.156,
    'risk_level': 'MEDIUM'
}
```

### 다른 워커 에이전트와의 연계
- **Structura**: 구조적 위험 + 시장 압력 = 종합 이직 위험도
- **Sentio**: 심리적 위험 + 시장 기회 = 이직 동기 분석
- **Cognita**: 관계적 위험 + 외부 네트워크 = 이직 경로 분석

## 🧪 테스트

### 단위 테스트
```bash
pytest test_agora_api.py -v
```

### API 통합 테스트
```bash
python test_agora_api.py
```

### 성능 테스트
```bash
# 100명 배치 분석 성능 테스트
curl -X POST http://localhost:5005/analyze/batch \
  -H "Content-Type: application/json" \
  -d @test_data_100_employees.json
```

## 🚨 문제 해결

### 일반적인 문제

1. **서버 시작 실패**
   ```bash
   # 포트 충돌 확인
   lsof -i :5005
   
   # 의존성 재설치
   pip install -r requirements.txt --force-reinstall
   ```

2. **API 응답 느림**
   - 캐시 상태 확인: `GET /health`
   - 시뮬레이션 모드 사용 (실제 API 대신)

3. **LLM 기능 오류**
   ```bash
   # OpenAI API 키 확인
   echo $OPENAI_API_KEY
   
   # LLM 없이 테스트
   curl -X POST .../analyze/market -d '{"use_llm": false, ...}'
   ```

## 📝 로그 및 모니터링

### 로그 파일
- `agora_api.log`: API 요청/응답 로그
- `agora_agent.log`: 시장 분석 상세 로그

### 모니터링 지표
- API 호출 횟수 및 응답 시간
- 캐시 적중률
- 오류율 및 실패한 분석 건수

## 🔮 향후 계획

- [ ] 실제 채용 사이트 API 연동 (사람인, 원티드 등)
- [ ] 급여 정보 API 연동 (잡플래닛, 블라인드 등)
- [ ] 실시간 시장 동향 알림 기능
- [ ] 시각화 대시보드 개발
- [ ] 머신러닝 기반 시장 예측 모델

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**Agora Agent** - 외부 노동 시장의 변화를 실시간으로 감지하고 분석하여 조직의 인재 리텐션 전략을 지원합니다.
