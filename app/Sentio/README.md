# Sentio - HR Text Analysis Service

Sentio는 HR 텍스트 데이터의 감정 분석, 키워드 추출, 퇴직 위험 신호 탐지를 수행하는 AI 워커 에이전트입니다.

## 🎯 주요 기능

### 1. 텍스트 분석
- **키워드 추출**: 명사 중심의 정확한 키워드 추출
- **감정 분석**: 텍스트의 긍정/부정 감정 점수 계산
- **퇴직 위험 분석**: 5가지 퇴직 원인별 위험 신호 탐지

### 2. 키워드 분석
- **차별적 키워드**: 퇴직자 vs 재직자 특징적 키워드 분석
- **명사 중심 추출**: 조사, 부사 등 노이즈 제거
- **빈도 기반 필터링**: 의미있는 키워드만 선별

### 3. 텍스트 생성
- **페르소나 기반**: 10가지 직원 페르소나별 맞춤 텍스트 생성
- **퇴직 위험 반영**: 각 페르소나의 퇴직 위험 키워드 자동 포함
- **다양한 텍스트 타입**: 자기평가, 동료피드백, 주간설문 등

## 🏗️ 시스템 구조

```
app/Sentio/
├── sentio_flask_backend.py     # Flask API 서버
├── sentio_processor.py         # 텍스트 처리 및 분석
├── sentio_analyzer.py          # 키워드 분석 (명사 중심)
├── sentio_generator.py         # 페르소나 기반 텍스트 생성
├── run_sentio_server.py        # 서버 실행 스크립트
├── test_sentio_api.py          # API 테스트 스크립트
├── requirements.txt            # 패키지 의존성
└── README.md                   # 이 파일
```

## 🚀 설치 및 실행

### 1. 패키지 설치
```bash
cd app/Sentio
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
# OpenAI API 키 설정 (텍스트 생성용)
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. 서버 실행
```bash
# 방법 1: 실행 스크립트 사용
python run_sentio_server.py

# 방법 2: 직접 실행
python sentio_flask_backend.py
```

### 4. API 테스트
```bash
# 별도 터미널에서 테스트 실행
python test_sentio_api.py
```

## 📡 API 엔드포인트

### 기본 정보
- **서버 주소**: `http://localhost:5003`
- **헬스체크**: `GET /health`
- **API 문서**: `GET /`

### 주요 API

#### 1. 텍스트 분석
```http
POST /analyze/text
Content-Type: application/json

{
  "text": "분석할 텍스트",
  "employee_id": "직원ID (선택)",
  "text_type": "텍스트 타입 (선택)"
}
```

**응답 예시:**
```json
{
  "employee_id": "emp_001",
  "text_type": "WEEKLY_SURVEY",
  "original_text": "업무량이 너무 많아서...",
  "keywords": ["업무량", "압박", "스트레스"],
  "sentiment_score": 0.3,
  "attrition_risk_score": 0.8,
  "risk_factors": ["업무량", "압박감", "번아웃"],
  "analysis_timestamp": "2024-01-15T10:30:00"
}
```

#### 2. 키워드 분석
```http
POST /analyze/keywords
Content-Type: application/json

{
  "min_frequency": 5,
  "text_columns": ["SELF_REVIEW_text", "WEEKLY_SURVEY_text"]
}
```

#### 3. 퇴직 위험 분석
```http
POST /analyze/risk
Content-Type: application/json

{
  "texts": [
    {
      "employee_id": "emp_001",
      "text": "분석할 텍스트",
      "text_type": "SELF_REVIEW"
    }
  ]
}
```

#### 4. 텍스트 생성
```http
POST /generate/text
Content-Type: application/json

{
  "employee_data": {
    "EmployeeNumber": 1001,
    "JobRole": "Software Engineer",
    "YearsAtCompany": 3,
    "PerformanceRating": 3,
    "JobSatisfaction": 2,
    "JobInvolvement": 2,
    "softmax_Persona_Code": "P01",
    "softmax_Persona": "번아웃 위험군",
    "Attrition": "Yes"
  },
  "text_type": "SELF_REVIEW"
}
```

#### 5. 페르소나 정보 조회
```http
GET /data/personas
```

## 🎭 페르소나 유형

### 퇴직 위험 높음 (P01-P04)
- **P01**: 번아웃 및 소진 - 과도한 업무부담
- **P02**: 조직 부적응 - 심리적 안전감 부족
- **P03**: 성장 정체 - 발전 기회 부족
- **P04**: 불공정 대우 - 보상/평가 불만

### 퇴직 위험 낮음 (S01-S03)
- **S01**: 안정형 - 장기적 관점, 책임감
- **S02**: 성장형 - 적극적, 미래지향적
- **S03**: 몰입형 - 업무 자체에 집중

### 중립형 (N01-N03)
- **N01**: 무관심형 - 형식적, 무미건조
- **N02**: 비판형 - 객관적이지만 냉소적
- **N03**: 균형형 - 워라밸 중시

## 🔍 퇴직 원인 분석

### 5가지 주요 퇴직 원인
1. **불공정한 보상과 평가**
   - 급여, 복리후생, 평가 공정성
   
2. **성장 정체와 동기부여 상실**
   - 커리어 발전, 업무 의미, 인정/성취
   
3. **과도한 업무부담과 번아웃**
   - 업무량, 스트레스, 워라밸, 역할 모호성
   
4. **건강하지 못한 조직문화와 관계**
   - 리더십, 문화, 심리적 안전감, 팀워크
   
5. **불안정한 고용 및 비효율적 시스템**
   - 고용 안정성, 정책/제도, 근무 조건

## 🛠️ 개발 정보

### 기술 스택
- **Backend**: Flask, Python 3.8+
- **AI/ML**: OpenAI GPT-4, pandas, numpy
- **Text Processing**: 정규표현식, 한국어 NLP
- **API**: RESTful API, JSON

### 데이터 요구사항
- **HR 데이터**: `IBM_HR.csv` (페르소나 정보 제거됨)
- **텍스트 데이터**: `IBM_HR_text.csv` 또는 `sample_hr_texts.csv`

### 환경 변수
```bash
OPENAI_API_KEY=your-openai-api-key  # 텍스트 생성용 (필수)
FLASK_ENV=development               # 개발 모드 (선택)
```

## 🧪 테스트

### API 테스트 실행
```bash
python test_sentio_api.py
```

### 테스트 항목
- ✅ 헬스체크
- ✅ 텍스트 분석
- ✅ 키워드 분석  
- ✅ 퇴직 위험 분석
- ✅ 텍스트 생성
- ✅ 페르소나 정보 조회

## 📊 사용 예시

### Python 클라이언트 예시
```python
import requests

# 텍스트 분석
response = requests.post('http://localhost:5003/analyze/text', json={
    "text": "업무량이 너무 많아서 번아웃이 올 것 같습니다.",
    "employee_id": "emp_001"
})

result = response.json()
print(f"퇴직 위험 점수: {result['attrition_risk_score']}")
print(f"위험 요소: {result['risk_factors']}")
```

### cURL 예시
```bash
curl -X POST http://localhost:5003/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "팀워크가 좋고 성장할 수 있어서 만족합니다."}'
```

## 🔧 트러블슈팅

### 일반적인 문제들

1. **OpenAI API 오류**
   ```bash
   # API 키 확인
   echo $OPENAI_API_KEY
   
   # API 키 설정
   export OPENAI_API_KEY="your-key"
   ```

2. **데이터 파일 없음**
   - `data/` 폴더에 필요한 CSV 파일 확인
   - 상대 경로 설정 확인

3. **포트 충돌**
   ```bash
   # 다른 포트로 실행
   python sentio_flask_backend.py --port 5004
   ```

4. **패키지 의존성 오류**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

## 📈 성능 최적화

### 키워드 추출 개선사항
- **명사 중심 추출**: 조사, 부사 등 노이즈 제거
- **패턴 매칭**: 업무, 감정, 성장 관련 명사 우선 선별
- **불용어 강화**: 500+ 불용어로 정확도 향상

### API 응답 최적화
- **비동기 처리**: 대용량 텍스트 배치 처리
- **캐싱**: 자주 사용되는 분석 결과 캐시
- **스트리밍**: 실시간 텍스트 생성

## 🤝 기여 방법

1. 이슈 리포트
2. 기능 제안
3. 코드 개선
4. 문서화

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
