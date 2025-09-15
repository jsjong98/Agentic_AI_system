# Final_calc Dashboard

Final_calc 백엔드 시스템을 위한 React 기반 프론트엔드 대시보드입니다.

## 🚀 주요 기능

### 📊 대시보드
- 시스템 전체 상태 모니터링
- 진행률 추적 및 최근 활동 표시
- 주요 성능 지표 요약

### 📁 파일 업로드
- 드래그 앤 드롭 파일 업로드
- CSV 파일 미리보기 및 유효성 검사
- 데이터 품질 분석

### 🎯 임계값 계산
- 각 Score별 최적 임계값 자동 계산
- F1-score 기반 성능 최적화
- 상세 성능 지표 및 시각화

### ⚖️ 가중치 최적화
- 3가지 최적화 알고리즘 지원
  - Bayesian Optimization
  - Grid Search
  - Scipy Optimization
- 실시간 최적화 진행률 표시
- 방법별 성능 비교

### 👤 개별 직원 예측
- 실시간 Attrition 위험도 예측
- 직관적인 위험도 게이지
- 개별 Score 분석 및 해석

### 📈 결과 시각화
- 6가지 차트 유형 지원
- 성능 지표 비교 및 분석
- 인터랙티브 차트

### 📤 결과 내보내기
- CSV/JSON 형식 지원
- 내보내기 기록 관리
- 파일 다운로드 기능

## 🛠️ 기술 스택

- **Frontend**: React 18, Ant Design
- **Charts**: Recharts
- **HTTP Client**: Axios
- **File Processing**: Papa Parse, React Dropzone
- **Styling**: Styled Components, CSS3

## 📦 설치 및 실행

### 1. 의존성 설치
```bash
cd Dashboard
npm install
```

### 2. 환경 설정
`.env` 파일 생성 (선택사항):
```env
REACT_APP_API_URL=http://localhost:5007
```

### 3. 개발 서버 실행
```bash
npm start
```

브라우저에서 `http://localhost:3000`으로 접속

### 4. 프로덕션 빌드
```bash
npm run build
```

## 🔧 프로젝트 구조

```
Dashboard/
├── public/
│   └── index.html
├── src/
│   ├── components/           # React 컴포넌트
│   │   ├── Dashboard.js      # 메인 대시보드
│   │   ├── FileUpload.js     # 파일 업로드
│   │   ├── ThresholdCalculation.js  # 임계값 계산
│   │   ├── WeightOptimization.js    # 가중치 최적화
│   │   ├── EmployeePrediction.js    # 직원 예측
│   │   ├── ResultVisualization.js   # 결과 시각화
│   │   └── ExportResults.js         # 결과 내보내기
│   ├── services/
│   │   └── apiService.js     # API 서비스 모듈
│   ├── App.js               # 메인 앱 컴포넌트
│   ├── index.js             # 앱 진입점
│   └── index.css            # 글로벌 스타일
├── package.json
└── README.md
```

## 🎨 UI/UX 특징

### 반응형 디자인
- 모바일, 태블릿, 데스크톱 지원
- Ant Design Grid 시스템 활용

### 직관적인 네비게이션
- 사이드바 메뉴
- 진행 상태 표시
- 단계별 가이드

### 시각적 피드백
- 로딩 상태 표시
- 성공/실패 알림
- 진행률 바

### 데이터 시각화
- 다양한 차트 유형
- 색상 코딩된 위험도
- 인터랙티브 요소

## 🔌 API 연동

### 백엔드 연결
대시보드는 Final_calc Flask 백엔드와 통신합니다:
- 기본 URL: `http://localhost:5007`
- 자동 에러 처리 및 재시도
- 타임아웃: 5분 (최적화 작업용)

### 주요 API 엔드포인트
- `GET /health` - 서버 상태 확인
- `POST /load_data` - 데이터 로드
- `POST /calculate_thresholds` - 임계값 계산
- `POST /optimize_weights` - 가중치 최적화
- `POST /predict_employee` - 개별 예측

## 📱 사용 방법

### 1. 시스템 설정
1. **서버 연결**: Final_calc 백엔드 서버 실행 확인
2. **데이터 업로드**: CSV 파일 업로드 또는 기본 데이터 사용
3. **임계값 계산**: 각 Score별 최적 임계값 계산
4. **가중치 최적화**: 최적 가중치 탐색

### 2. 예측 및 분석
1. **개별 예측**: 직원 Score 입력하여 위험도 예측
2. **결과 시각화**: 다양한 차트로 결과 분석
3. **결과 내보내기**: CSV/JSON 형식으로 결과 저장

### 3. 샘플 데이터
각 기능에는 테스트용 샘플 데이터가 포함되어 있습니다:
- 고위험/중위험/저위험 직원 샘플
- 기본 데이터셋 (`Total_score.csv`)

## 🎯 주요 컴포넌트 설명

### Dashboard.js
- 시스템 전체 상태 모니터링
- 진행률 추적 (서버연결 → 데이터로드 → 임계값계산 → 가중치최적화)
- 최근 활동 타임라인
- 다음 단계 가이드

### FileUpload.js
- 드래그 앤 드롭 파일 업로드
- CSV 파일 미리보기 (첫 10행)
- 데이터 유효성 검사 (필수 컬럼, Score 컬럼, 데이터 품질)
- 기본 데이터 로드 옵션

### ThresholdCalculation.js
- F1-score 기반 최적 임계값 계산
- 성능 지표 테이블 (정밀도, 재현율, 정확도)
- 성능 등급 시스템 (A/B/C/D)
- 성능 비교 차트

### WeightOptimization.js
- 3가지 최적화 방법 지원
- 실시간 최적화 진행률
- 가중치 분포 파이 차트
- 위험도 분포 분석
- 방법별 성능 비교

### EmployeePrediction.js
- 5개 Score 입력 폼
- 위험도 게이지 차트
- 개별 Score 예측 테이블
- 3가지 샘플 데이터 (고/중/저위험)
- 위험도 구간별 해석 가이드

### ResultVisualization.js
- 6가지 차트 유형:
  - 성능 지표 비교 (막대 차트)
  - 임계값 분포 (선 차트)
  - 가중치 분포 (파이 차트)
  - 위험도 분포 (막대 차트)
  - 성능 레이더 (레이더 차트)
  - 정밀도-재현율 산점도
- 차트별 해석 가이드

### ExportResults.js
- CSV/JSON 형식 지원
- 상세 데이터 포함 옵션
- 내보내기 기록 관리
- 파일 다운로드 기능
- 형식별 활용 가이드

## 🔍 문제 해결

### 일반적인 문제
1. **서버 연결 실패**
   - Final_calc 백엔드 서버가 실행 중인지 확인
   - 포트 5007이 사용 가능한지 확인

2. **파일 업로드 실패**
   - CSV 파일 형식 확인
   - 파일 크기 50MB 이하 확인
   - 필수 컬럼 (`attrition`) 존재 확인

3. **차트가 표시되지 않음**
   - 브라우저 콘솔에서 JavaScript 오류 확인
   - 데이터가 올바르게 로드되었는지 확인

### 개발 관련
1. **의존성 설치 오류**
   ```bash
   npm cache clean --force
   rm -rf node_modules package-lock.json
   npm install
   ```

2. **빌드 오류**
   - Node.js 버전 확인 (권장: 16+)
   - 메모리 부족 시: `NODE_OPTIONS=--max_old_space_size=4096 npm run build`

## 🚀 배포

### 개발 환경
```bash
npm start
```

### 프로덕션 환경
```bash
npm run build
# build/ 폴더를 웹 서버에 배포
```

### Docker 배포 (선택사항)
```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## 📈 성능 최적화

### 권장사항
- 큰 데이터셋의 경우 페이지네이션 사용
- 차트 렌더링 시 데이터 샘플링
- 이미지 최적화 및 lazy loading
- 코드 스플리팅 적용

### 모니터링
- React DevTools로 컴포넌트 성능 확인
- Network 탭에서 API 응답 시간 모니터링
- Lighthouse로 전체적인 성능 측정

## 🤝 기여 방법

1. 이슈 리포트: 버그나 개선사항을 GitHub Issues에 등록
2. 기능 요청: 새로운 차트나 분석 기능 제안
3. 코드 기여: Pull Request를 통한 코드 개선
4. 문서 개선: README나 주석 개선

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**Final_calc Dashboard v1.0.0** - HR Attrition 예측을 위한 직관적이고 강력한 웹 인터페이스
