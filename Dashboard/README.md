# Agentic AI System Dashboard

HR 이직 예측을 위한 다중 에이전트 시스템의 통합 대시보드입니다.

## 🚀 주요 기능

### 1. 기본 분석 단계
- **데이터 업로드**: HR 데이터 파일 업로드 및 검증
- **임계값 계산**: 각 Score의 최적 임계값 자동 계산
- **가중치 최적화**: Bayesian, Grid Search 등 다양한 방법으로 가중치 최적화
- **직원 예측**: 개별 직원의 이직 위험도 예측

### 2. 고급 분석 단계 (NEW!)
- **Integration 분석**: 다중 에이전트 결과 통합 및 종합 분석
- **Supervisor 워크플로우**: 슈퍼바이저 에이전트를 통한 지능형 협업 분석
- **XAI (설명 가능한 AI)**: 모델 예측 결과에 대한 상세한 설명 제공

### 3. 시각화 및 리포트
- **결과 시각화**: 인터랙티브 차트와 그래프
- **결과 내보내기**: CSV, JSON 등 다양한 형식으로 결과 다운로드

## 🎯 새로운 기능 하이라이트

### Integration 분석
- 5개 에이전트(Agora, Chronos, Cognita, Sentio, Structura)의 결과를 통합
- 단계별 파일 업로드 시스템
- 자동 통합 분석 및 리포트 생성
- 실시간 진행률 표시

### Supervisor 워크플로우
- **실시간 타이핑 애니메이션**: 생성형 LLM 스타일의 한글자씩 타이핑 효과
- **다중 에이전트 협업**: 워커 에이전트들 간의 지능형 협업
- **워크플로우 제어**: 시작/일시정지/재시작/중지 기능
- **실시간 로그**: 워크플로우 진행 상황 실시간 모니터링

### XAI (설명 가능한 AI)
- **특성 중요도 분석**: 모델 예측에 가장 영향을 미치는 특성 식별
- **SHAP 분석**: SHAP 값을 통한 개별 예측 설명
- **LIME 설명**: 지역적 설명을 통한 모델 해석
- **의사결정 트리**: 시각적 의사결정 과정 표현
- **반사실적 설명**: 다른 결과를 위한 최소 변경사항 제시

## 🛠 기술 스택

### Frontend
- **React 18**: 모던 React 훅 기반 개발
- **Ant Design 5**: 전문적인 UI 컴포넌트 라이브러리
- **Axios**: HTTP 클라이언트
- **Recharts**: 데이터 시각화

### Backend Integration
- **Integration API**: 통합 분석 및 리포트 생성
- **Supervisor API**: 워크플로우 관리 및 에이전트 협업
- **XAI API**: 설명 가능한 AI 분석

## 📋 워크플로우

### 전체 분석 프로세스
1. **데이터 업로드** → 2. **임계값 계산** → 3. **가중치 최적화** → 4. **Integration 분석** → 5. **Supervisor 워크플로우** → 6. **XAI 분석** → 7. **결과 시각화**

### 단계별 체크 시스템
- 각 단계 완료 시 자동으로 다음 단계 활성화
- 실시간 진행률 표시
- 헤더에 모든 단계 상태 표시

## 🎨 UI/UX 특징

### 타이핑 애니메이션
```javascript
// 생성형 LLM 스타일 타이핑 효과
const TypingText = ({ text, speed = 50 }) => {
  // 한글자씩 타이핑되는 애니메이션 구현
  // 커서 깜빡임 효과 포함
}
```

### 반응형 디자인
- 모바일, 태블릿, 데스크톱 모든 화면 크기 지원
- 적응형 레이아웃 및 컴포넌트 크기 조정

### 실시간 상태 표시
- 서버 연결 상태
- 각 단계별 완료 상태
- 실시간 진행률 바

## 🚀 시작하기

### 설치
```bash
cd Dashboard
npm install
```

### 개발 서버 실행
```bash
npm start
```

### 빌드
```bash
npm run build
```

## 📡 API 엔드포인트

### Integration API
- `POST /integration/integrate` - 결과 통합
- `POST /integration/generate_report` - 리포트 생성

### Supervisor API
- `POST /supervisor/start_session` - 세션 시작
- `POST /supervisor/analyze/{sessionId}` - 분석 실행
- `POST /supervisor/collaborate/{sessionId}` - 에이전트 협업
- `POST /supervisor/synthesize/{sessionId}` - 결과 종합

### XAI API
- `POST /xai/feature_importance` - 특성 중요도 분석
- `POST /xai/shap` - SHAP 분석
- `POST /xai/lime` - LIME 설명
- `POST /xai/decision_tree` - 의사결정 트리 생성
- `POST /xai/counterfactuals` - 반사실적 설명 생성

## 🔧 설정

### 환경 변수
```env
REACT_APP_API_URL=http://localhost:5007
```

### 프록시 설정
`package.json`에서 프록시 설정:
```json
{
  "proxy": "http://localhost:5007"
}
```

## 📊 데이터 플로우

1. **파일 업로드** → 각 에이전트별 분석 결과 파일
2. **Integration** → 통합 분석 및 최적화
3. **Supervisor** → 다중 에이전트 협업 분석
4. **XAI** → 모델 해석 및 설명 생성
5. **시각화** → 결과 차트 및 그래프 생성

## 🎯 주요 컴포넌트

### IntegrationAnalysis
- 다중 에이전트 결과 통합
- 단계별 파일 업로드
- 자동 분석 진행

### SupervisorWorkflow
- 워크플로우 관리
- 실시간 타이핑 애니메이션
- 에이전트 메시지 표시

### XAIResults
- 설명 가능한 AI 결과
- 다양한 해석 기법 제공
- 인터랙티브 시각화

## 🔍 트러블슈팅

### 일반적인 문제
1. **서버 연결 실패**: 백엔드 서버가 실행 중인지 확인
2. **파일 업로드 실패**: 파일 형식 및 크기 확인
3. **타이핑 애니메이션 끊김**: 브라우저 성능 확인

### 성능 최적화
- React.memo를 사용한 컴포넌트 최적화
- 지연 로딩을 통한 초기 로딩 시간 단축
- API 호출 최적화

## 📝 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

**개발자**: PwC RA Team  
**버전**: 2.0.0  
**업데이트**: 2024년 Integration/Supervisor/XAI 기능 추가