# 🔑 OpenAI API 키 설정 가이드

## 문제 상황
서버 로그에서 다음과 같은 경고가 표시됩니다:
```
⚠️ 경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.
   Supervisor, Integration, Sentio, Agora의 LLM 기능이 제한됩니다.
```

## 해결 방법

### 방법 1: .env 파일 생성 (권장)

1. **app 폴더에 .env 파일 생성**:
```bash
cd app
touch .env
```

2. **.env 파일에 다음 내용 추가**:
```env
# OpenAI API 키 설정
OPENAI_API_KEY=your-actual-api-key-here

# Neo4j 설정 (이미 설정됨)
NEO4J_URI=bolt://44.212.67.74:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=legs-augmentations-cradle

# Flask 설정
FLASK_ENV=development
FLASK_DEBUG=1
```

3. **실제 API 키로 교체**:
   - `your-actual-api-key-here`를 실제 OpenAI API 키로 변경
   - API 키는 `sk-`로 시작하는 문자열입니다

### 방법 2: 환경변수 직접 설정

**Windows (PowerShell)**:
```powershell
$env:OPENAI_API_KEY="your-actual-api-key-here"
```

**Windows (CMD)**:
```cmd
set OPENAI_API_KEY=your-actual-api-key-here
```

**Linux/Mac**:
```bash
export OPENAI_API_KEY="your-actual-api-key-here"
```

### 방법 3: 시스템 환경변수 설정

1. **Windows**: 시스템 속성 → 고급 → 환경 변수
2. **Linux/Mac**: `~/.bashrc` 또는 `~/.zshrc`에 export 명령 추가

## OpenAI API 키 발급 방법

1. **OpenAI 웹사이트 방문**: https://platform.openai.com/
2. **계정 생성/로그인**
3. **API Keys 페이지 이동**: https://platform.openai.com/api-keys
4. **"Create new secret key" 클릭**
5. **키 이름 입력 후 생성**
6. **생성된 키 복사** (한 번만 표시되므로 안전한 곳에 저장)

## 설정 확인

설정 후 서버를 다시 시작하면 경고 메시지가 사라집니다:

```bash
python run_agentic_system.py
```

성공적으로 설정되면 다음과 같은 메시지가 표시됩니다:
```
✅ Agora LLM 생성기 초기화 완료
✅ Supervisor 에이전트 초기화 완료
✅ Integration 에이전트 초기화 완료
```

## 주의사항

- **.env 파일을 Git에 커밋하지 마세요** (API 키 노출 위험)
- **API 키는 안전한 곳에 보관하세요**
- **사용량에 따라 요금이 부과됩니다** (OpenAI 요금제 확인)

## 문제 해결

### API 키가 여전히 인식되지 않는 경우:
1. `.env` 파일이 `app/` 폴더에 있는지 확인
2. 파일 이름이 정확히 `.env`인지 확인 (확장자 없음)
3. API 키에 따옴표나 공백이 없는지 확인
4. 서버를 완전히 재시작

### API 키 유효성 확인:
```python
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# 간단한 테스트
try:
    response = openai.models.list()
    print("✅ API 키가 유효합니다!")
except Exception as e:
    print(f"❌ API 키 오류: {e}")
```
