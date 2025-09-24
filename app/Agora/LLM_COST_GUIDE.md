 # 🎛️ Agora LLM 비용 절약 가이드

## 📊 현재 기본 설정

**✅ 기본적으로 LLM이 꺼져있습니다** (비용 절약)
- 배치 분석: `use_llm: false` (기본값)
- 사후 분석: `use_llm: false` (기본값)
- 개별 분석: `use_llm: false` (기본값)

## 💰 비용 절약 전략

### 1. **일반적인 분석 (권장)**: LLM 끄기 ❌
```json
{
    "JobRole": "Sales Executive",
    "MonthlyIncome": 5500000,
    "use_llm": false
}
```
→ **규칙 기반 해석** 사용 (무료, 빠름)

### 2. **중요한 케이스만**: LLM 켜기 ✅
```json
{
    "JobRole": "Sales Executive", 
    "MonthlyIncome": 5500000,
    "use_llm": true
}
```
→ **자연어 해석** 사용 (유료, 상세함)

### 3. **배치 분석**: 기본적으로 LLM 끄기
```json
{
    "employees": [...1000명...],
    "use_llm": false
}
```
→ 1000명 분석 시 **대폭 비용 절약**

## 🔧 LLM 제어 방법

### 1. 환경변수로 완전 비활성화
```env
# .env 파일에서 API 키 제거
# OPENAI_API_KEY=your-api-key
```

### 2. 개별 분석 시 선택적 사용
```bash
# 일반 분석 (LLM 사용 안함, 기본값)
curl -X POST /api/analyze/individual \
     -d '{"EmployeeNumber": 1001, "JobRole": "Data Scientist"}'

# 레포트 생성 시 (LLM 사용)
curl -X POST /api/analyze/individual \
     -d '{"EmployeeNumber": 1001, "JobRole": "Data Scientist", "generate_report": true}'
```

### 3. 배치 분석 시 선택적 사용
```bash
# 배치 분석 (LLM 사용 안함, 기본값)
curl -X POST /api/analyze/batch \
     -d '{"employees": [...], "use_llm": false}'

# 중요한 케이스만 LLM 사용
curl -X POST /api/analyze/batch \
     -d '{"employees": [...고위험군...], "use_llm": true}'
```

## 📈 규칙 기반 해석 예시 (무료)

LLM 없이도 충분히 유용한 분석을 제공합니다:

```
=== 직원 1001 (Sales Executive) 시장 분석 해석 ===

📊 현재 시장 상황:
Sales Executive 직무의 채용 시장에서 현재 187개의 관련 공고가 활발히 게시되어 있습니다.
시장 트렌드는 'GROWING'이며, 경쟁 수준은 'HIGH'입니다.

🎯 이직 위험도 분석:
종합 위험도는 'HIGH'로 평가됩니다.
- 시장 압력 지수: 0.723/1.0
- 보상 격차: 0.156

⚠️ 높은 이직 위험 상황입니다:
- 외부 시장에서 해당 직무에 대한 수요가 매우 높습니다
- 즉각적인 관심과 조치가 필요한 상황입니다

🔧 권장 조치:
1. 급여 및 복리후생 패키지 재검토
2. 상급자와의 긴급 면담 실시
3. 경력 개발 기회 및 승진 계획 논의
```

## 💡 권장 사용 패턴

1. **일상적인 모니터링**: LLM 끄기 (수치만 확인)
2. **개별 직원 레포트**: LLM 켜기 (`generate_report: true`)
3. **고위험군 발견 시**: LLM 켜기 (상세 분석)
4. **경영진 보고용**: LLM 켜기 (자연어 해석)
5. **대량 배치 처리**: LLM 끄기 (비용 절약)

## 🎯 결론

- **기본값**: LLM 꺼짐 (비용 절약)
- **필요시만**: LLM 켜기 (상세 분석)
- **규칙 기반 해석**도 충분히 실무에 유용함
- **JobSpy 실제 데이터**는 LLM과 무관하게 항상 활용됨