import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import platform

# 한국어 형태소 분석을 위한 라이브러리
KONLPY_AVAILABLE = False
okt_instance = None
KONLPY_TRIED = False  # KoNLPy 시도 여부 플래그

def try_import_konlpy():
    """KoNLPy 안전 import 함수 - Java 문제 시 완전 우회"""
    global KONLPY_AVAILABLE, okt_instance, KONLPY_TRIED
    
    # 이미 시도했다면 재시도하지 않음
    if KONLPY_TRIED:
        return KONLPY_AVAILABLE
    
    # Java 환경 변수 먼저 확인 (Java 실행 없이)
    import os
    java_home = os.environ.get('JAVA_HOME')
    if not java_home:
        print("ℹ️ JAVA_HOME 환경변수가 설정되지 않았습니다.")
        KONLPY_AVAILABLE = False
        KONLPY_TRIED = True
        return False
    
    # Java 실행 파일 존재 확인
    java_exe = os.path.join(java_home, 'bin', 'java.exe')
    if not os.path.exists(java_exe):
        java_exe = os.path.join(java_home, 'bin', 'java')
        if not os.path.exists(java_exe):
            print("ℹ️ Java 실행 파일을 찾을 수 없습니다.")
            KONLPY_AVAILABLE = False
            KONLPY_TRIED = True
            return False
    
    # 매우 조심스럽게 KoNLPy import 시도
    try:
        # 먼저 KoNLPy 모듈만 import 확인
        import konlpy
        
        # 그 다음 Okt import 시도
        from konlpy.tag import Okt
        
        # 매우 간단한 테스트만 수행
        test_okt = Okt()
        
        # 성공하면 저장
        okt_instance = test_okt
        KONLPY_AVAILABLE = True
        KONLPY_TRIED = True
        print("✅ KoNLPy 형태소 분석기가 성공적으로 로드되었습니다.")
        return True
        
    except ImportError as e:
        print("ℹ️ KoNLPy가 설치되지 않았습니다.")
        KONLPY_AVAILABLE = False
        KONLPY_TRIED = True
    except Exception as e:
        # Java 관련 오류 시 완전히 포기
        print("ℹ️ Java 환경 문제로 KoNLPy를 사용할 수 없습니다.")
        KONLPY_AVAILABLE = False
        KONLPY_TRIED = True
    
    return False

# 초기에는 import 시도하지 않음 (필요할 때만 시도)

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정"""
    system = platform.system()
    
    if system == "Windows":
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif system == "Darwin":  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    else:  # Linux
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

class HRKeywordAnalyzer:
    def __init__(self, csv_file_path):
        """
        IBM HR 데이터 키워드 분석기 초기화
        
        Args:
            csv_file_path (str): CSV 파일 경로
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.resigned_data = None
        self.stayed_data = None
        
        # 한국어 형태소 분석기는 필요할 때 초기화
        self.okt = None
        
        # 대폭 강화된 한국어 불용어 리스트
        self.stopwords = set([
            # 조사
            '은', '는', '이', '가', '을', '를', '에', '에서', '로', '으로', '와', '과', '의', '도',
            '만', '부터', '까지', '에게', '한테', '께', '에게서', '한테서', '께서', '라', '야',
            '아', '어', '여', '이랑', '랑', '하고', '에다', '에다가', '로서', '로써', '처럼',
            
            # 어미 및 완전한 어미 표현들 (가장 문제가 되는 부분)
            '다', '습니다', '입니다', '했습니다', '있습니다', '없습니다', '됩니다', 
            '니다', '였습니다', '이었습니다', '하였습니다', '것입니다', '싶습니다',
            '보입니다', '같습니다', '겠습니다', '드립니다', '부탁드립니다',
            '하겠습니다', '진행하겠', '줄였습니다', '했어요', '해요', '이에요', '예요',
            '했네요', '했죠', '했어', '해서', '하여', '하고', '하며', '하니', '하면',
            '했네', '했지', '했군', '했구나', '하네', '하지', '하군', '하구나',
            '되었습니다', '되겠습니다', '되어야', '되도록', '되었어요', '되네요',
            '필요합니다', '좋겠습니다', '어떻습니다', '그렇습니다', '맞습니다',
            
            # 대명사 및 지시어
            '그', '이', '저', '것', '그것', '이것', '저것', '그런', '이런', '저런',
            '그러한', '이러한', '저러한', '그렇', '이렇', '저렇', '그곳', '이곳', '저곳',
            '그때', '이때', '저때', '그분', '이분', '저분', '그들', '이들', '저들',
            '그녀', '그는', '그의', '그를', '그에게', '그에서', '그로', '그와',
            
            # 부사
            '매우', '정말', '너무', '조금', '약간', '다소', '상당히', '꽤', '아주',
            '더', '가장', '훨씬', '좀', '잘', '안', '못', '별로', '거의', '완전히',
            '전혀', '결코', '절대', '다시', '또', '또다시', '계속', '항상', '늘',
            '자주', '가끔', '때때로', '종종', '보통', '일반적으로', '대체로',
            '대부분', '모두', '전부', '모든', '각각', '서로', '함께', '같이',
            
            # 접속사
            '그리고', '또한', '하지만', '그러나', '따라서', '그래서', '또는', '혹은',
            '그런데', '그러면', '만약', '그럼', '근데', '그치만', '하지만', '그렇지만',
            '그러므로', '그러니까', '그런데도', '그럼에도', '그럼에도불구하고',
            
            # 일반적인 명사 (의미가 약한 것들)
            '때', '경우', '시간', '상황', '문제', '방법', '방식', '부분', '측면',
            '정도', '수준', '범위', '기준', '결과', '효과', '영향', '변화', '차이',
            '내용', '형태', '모습', '모양', '크기', '종류', '유형', '성격', '특성',
            '상태', '조건', '환경', '분위기', '느낌', '생각', '의견', '견해',
            
            # 시간 표현
            '지난', '다음', '이번', '올해', '작년', '내년', '오늘', '어제', '내일',
            '최근', '현재', '앞으로', '향후', '이후', '이전', '전에', '후에',
            '동안', '사이', '중', '때문', '위해', '통해', '대해', '관해',
            
            # 감탄사 및 의성어/의태어
            '아', '어', '오', '우', '에', '이', '음', '으음', '흠', '허',
            '아하', '어머', '오호', '우와', '에이', '이런', '어떻게', '왜',
            '하하', '헤헤', '히히', '호호', '후후', '크크', '킥킥',
            
            # 기타 불용어 (의미가 약하거나 일반적인 표현)
            '등', '및', '또', '역시', '물론', '특히', '즉', '예를', '통해',
            '위한', '위해', '대한', '관한', '에서', '에게', '으로', '로서',
            '같은', '다른', '새로운', '기존', '현재', '과거', '미래', '전체',
            '일부', '부분', '전부', '모든', '각', '여러', '다양한', '많은',
            '적은', '큰', '작은', '높은', '낮은', '좋은', '나쁜', '빠른', '느린',
            
            # 자주 나타나는 무의미한 패턴들
            '있는', '없는', '되는', '하는', '가는', '오는', '보는', '듣는',
            '말하는', '생각하는', '느끼는', '알고', '모르고', '보고', '듣고',
            '말하고', '생각하고', '느끼고', '해는', '되는', '가는', '오는',
            
            # 업무 관련 일반적 표현 (너무 일반적이어서 의미가 없는 것들)
            '진행', '완료', '시작', '마무리', '준비', '계획', '예정', '예상',
            '확인', '검토', '점검', '체크', '처리', '해결', '개선', '보완',
            '수정', '변경', '조정', '업데이트', '갱신', '추가', '삭제', '제거',
            
            # 조사가 붙은 일반적 패턴들 (직접 불용어로 처리)
            '시간을', '시간이', '시간에', '시간의', '시간도', '시간만',
            '기준을', '기준이', '기준에', '기준의', '기준도', '기준만',
            '속도가', '속도를', '속도에', '속도의', '속도도', '속도만',
            '품질을', '품질이', '품질에', '품질의', '품질도', '품질만',
            '업무를', '업무가', '업무에', '업무의', '업무도', '업무만',
            '일정을', '일정이', '일정에', '일정의', '일정도', '일정만',
            '목표를', '목표가', '목표에', '목표의', '목표도', '목표만',
            '결과를', '결과가', '결과에', '결과의', '결과도', '결과만',
            '문제를', '문제가', '문제에', '문제의', '문제도', '문제만',
            '방법을', '방법이', '방법에', '방법의', '방법도', '방법만',
            '상황을', '상황이', '상황에', '상황의', '상황도', '상황만',
            '과제를', '과제가', '과제에', '과제의', '과제도', '과제만',
            
            # 문제가 되는 특정 패턴들 직접 추가
            '필요합니', '그녀는', '그는', '팀의', '의지는', '마무리했습니다',
            '진행하겠습니다', '계획대로', '예정된', '동일한', '안정적인'
        ])
    
    def load_data(self):
        """CSV 데이터를 로드하고 퇴직자/재직자로 분류"""
        try:
            self.data = pd.read_csv(self.csv_file_path, encoding='utf-8')
            print(f"데이터 로드 완료: {len(self.data)}행, {len(self.data.columns)}열")
            print(f"컬럼명: {list(self.data.columns)}")
            
            # 퇴직자/재직자 분류
            self.resigned_data = self.data[self.data['Attrition'] == 'Yes']
            self.stayed_data = self.data[self.data['Attrition'] == 'No']
            
            print(f"퇴직자: {len(self.resigned_data)}명, 재직자: {len(self.stayed_data)}명")
            
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            return False
        return True
    
    def extract_keywords_morpheme(self, text):
        """
        형태소 분석을 사용한 키워드 추출
        
        Args:
            text (str): 분석할 텍스트
        
        Returns:
            list: 추출된 키워드 리스트
        """
        if pd.isna(text) or not text:
            return []
        
        # KoNLPy 사용 시도 (현재 비활성화 - Java 문제 방지)
        if False and try_import_konlpy() and okt_instance:
            try:
                # 형태소 분석: 명사, 형용사, 동사만 추출
                morphs = okt_instance.pos(text, stem=True)  # stem=True로 어간 추출
                
                # 명사(Noun), 형용사(Adjective), 동사(Verb) 추출
                keywords = []
                for word, pos in morphs:
                    if (pos in ['Noun', 'Adjective', 'Verb'] and 
                        len(word) >= 2 and 
                        word not in self.stopwords and
                        not word.isdigit() and
                        not re.match(r'^[a-zA-Z]+$', word)):  # 영어 단어 제외
                        keywords.append(word)
                
                return keywords
                
            except Exception as e:
                print(f"형태소 분석 오류: {e}")
                return self.extract_keywords_basic(text)
        else:
            return self.extract_keywords_basic(text)
    
    def extract_keywords_basic(self, text):
        """
        개선된 기본 키워드 추출 (형태소 분석 없이)
        한국어 패턴을 고려한 고급 텍스트 처리
        """
        if pd.isna(text) or not text:
            return []
        
        # 1단계: 텍스트 전처리 (안전한 정규표현식 사용)
        # 특수문자들을 하나씩 제거
        clean_text = text
        special_chars = '.,!?;:()[]""''「」~`@#$%^&*+=<>{}|\\'
        for char in special_chars:
            clean_text = clean_text.replace(char, ' ')
        
        # 연속된 공백을 하나로 통합
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # 2단계: 한국어 단어 패턴 추출
        # 한글 + 숫자 조합도 허용 (예: 1차, 2년차 등)
        korean_pattern = r'[가-힣]+(?:[0-9]*[가-힣]*)*'
        words = re.findall(korean_pattern, clean_text)
        
        # 3단계: 개선된 키워드 필터링
        keywords = []
        for word in words:
            # 기본 조건 체크
            if len(word) >= 2:
                # 어미 제거 및 정제
                cleaned_word = self.advanced_word_cleaning(word)
                if (cleaned_word and 
                    cleaned_word not in self.stopwords and
                    len(cleaned_word) >= 2 and
                    not self.is_meaningless_word(cleaned_word)):
                    keywords.append(cleaned_word)
        
        return keywords
    
    def advanced_word_cleaning(self, word):
        """고급 한국어 단어 정제"""
        # 1. 일반적인 어미 제거
        cleaned = self.remove_common_endings(word)
        if not cleaned:
            return None
            
        # 2. 반복되는 문자 정제 (예: "좋좋좋" -> "좋")
        cleaned = re.sub(r'(.)\1{2,}', r'\1', cleaned)
        
        # 3. 의미없는 패턴 제거
        if re.match(r'^[ㄱ-ㅎㅏ-ㅣ]+$', cleaned):  # 자음/모음만 있는 경우
            return None
            
        return cleaned
    
    def is_meaningless_word(self, word):
        """대폭 강화된 의미없는 단어 판별"""
        # 패턴 기반 필터링
        meaningless_patterns = [
            r'^[ㄱ-ㅎㅏ-ㅣ]+$',  # 자음/모음만
            r'^.{1}$',  # 한 글자
            r'^(하하|헤헤|히히|호호|후후|크크|킥킥)+$',  # 웃음 표현
            r'^(음음|어어|아아|으으|흠흠|허허)+$',  # 감탄사 반복
            r'^(네네|예예|아아|어어)+$',  # 대답 반복
            r'^(.)\1+$',  # 같은 글자 반복 (예: 좋좋좋)
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, word):
                return True
        
        # 대폭 확장된 의미없는 단어들
        meaningless_words = {
            # 일반적인 부사/감탄사
            '그냥', '뭔가', '좀', '막', '되게', '진짜', '정말', '완전',
            '엄청', '겁나', '개', '쩔', '짱', '대박', '헐', '와', '우와',
            '아이고', '어머', '오호', '어라', '어어', '음음', '흠흠',
            
            # 너무 일반적인 동사/형용사 (어간)
            '하', '되', '있', '없', '가', '오', '보', '듣', '말하', '생각하',
            '느끼', '알', '모르', '좋', '나쁘', '크', '작', '높', '낮',
            
            # 너무 일반적인 명사
            '것', '거', '게', '걸', '건', '곳', '데', '때', '분', '번',
            '개', '명', '원', '년', '월', '일', '시', '분', '초',
            
            # 의미가 약한 연결어/지시어
            '그거', '이거', '저거', '뭐', '누구', '언제', '어디', '어떻게', '왜',
            '그래', '이래', '저래', '어째', '어쩜', '어쩐지',
            
            # 업무에서 너무 일반적인 표현들
            '함', '됨', '임', '음', '기', '성', '력', '화', '적', '인',
            '로', '으로', '에서', '에게', '와', '과', '의', '도',
            
            # 자주 나타나지만 의미가 약한 단어들
            '저희', '우리', '제가', '저를', '저는', '저의', '저에게',
            '그분', '이분', '저분', '분들', '님들', '씨들',
            
            # 시간/순서 관련 일반적 표현
            '먼저', '나중', '다음', '이전', '전에', '후에', '동안', '사이',
            '중간', '처음', '마지막', '끝', '시작', '중',
            
            # 너무 추상적인 표현들
            '방향', '쪽', '편', '면', '측', '관점', '입장', '상황', '경우',
            '상태', '조건', '환경', '분위기', '느낌', '기분'
        }
        
        # 단어가 의미없는 단어 목록에 있는지 확인
        if word in meaningless_words:
            return True
        
        # 너무 짧거나 긴 단어 필터링
        if len(word) < 2 or len(word) > 10:
            return True
        
        # 숫자만 있는 경우
        if word.isdigit():
            return True
        
        # 영어만 있는 경우
        if re.match(r'^[a-zA-Z]+$', word):
            return True
        
        return False
    
    def remove_common_endings(self, word):
        """대폭 강화된 한국어 어미 제거"""
        # 매우 복합적인 어미들 (가장 긴 것부터)
        very_complex_endings = [
            '부탁드립니다', '하겠습니다', '되겠습니다', '싶습니다', '것입니다', 
            '보입니다', '같습니다', '겠습니다', '드립니다', '줄였습니다',
            '진행하겠습니다', '완료하겠습니다', '시작하겠습니다', '필요합니다',
            '좋겠습니다', '어떻습니다', '그렇습니다', '맞습니다', '틀렸습니다',
            '마무리했습니다', '계획했습니다', '준비했습니다', '확인했습니다'
        ]
        
        # 복합 어미 (긴 것부터 처리)
        complex_endings = [
            '했습니다', '있습니다', '없습니다', '됩니다', '입니다', '습니다',
            '이었습니다', '였습니다', '하였습니다', '되었습니다', '였다',
            '했었다', '있었다', '없었다', '되었다', '하였다', '했어요',
            '해요', '이에요', '예요', '했네요', '했죠', '되었어요', '되네요'
        ]
        
        # 기본 어미
        basic_endings = [
            '했다', '한다', '하다', '이다', '있다', '없다', '된다', '되다',
            '했어', '해서', '하여', '하고', '하며', '하니', '하면', '하자',
            '했네', '했지', '했군', '했구나', '하네', '하지', '하군', '하구나',
            '되어', '되고', '되며', '되니', '되면', '되자', '되네', '되지',
            # 더 많은 어미 패턴
            '합니다', '합니', '니다', '다', '네', '지', '군', '구나',
            '어요', '아요', '여요', '에요', '예요', '죠', '요'
        ]
        
        # 조사 (매우 포괄적으로 - 모든 한국어 조사)
        particles = [
            # 주격조사
            '이', '가', '께서', '에서',
            # 목적격조사  
            '을', '를',
            # 관형격조사
            '의',
            # 부사격조사
            '에', '에서', '에게', '한테', '께', '로', '으로', '와', '과', '이랑', '랑', '하고',
            '에다', '에다가', '로서', '로써', '처럼', '같이', '마냥', '대로', '따라',
            '부터', '까지', '마저', '조차', '밖에', '뿐', '만', '도', '나마', '라도',
            # 호격조사
            '아', '야', '여', '이여', '이시여',
            # 접속조사
            '와', '과', '하고', '이랑', '랑',
            # 보조사
            '은', '는', '도', '만', '조차', '마저', '까지', '부터', '나마', '라도', '밖에', '뿐',
            # 종결조사
            '다', '라', '마', '지', '구나', '군', '네', '요',
            # 추가 조사 패턴들
            '에는', '에도', '에만', '로는', '로도', '로만', '으로는', '으로도', '으로만',
            '와는', '와도', '와만', '과는', '과도', '과만', '하고는', '하고도', '하고만'
        ]
        
        # 자주 나타나는 패턴들 (조사가 붙은 형태들)
        common_patterns = [
            # 동사+조사 패턴
            '하는', '되는', '있는', '없는', '가는', '오는', '보는', '듣는',
            '말하는', '생각하는', '느끼는', '알고', '모르고', '보고', '듣고',
            '말하고', '생각하고', '느끼고', '해는', '되는', '가는', '오는',
            
            # 명사+조사 패턴 (가장 문제가 되는 부분)
            '시간을', '시간이', '시간에', '시간의', '시간도', '시간만', '시간부터', '시간까지',
            '기준을', '기준이', '기준에', '기준의', '기준도', '기준만', '기준으로',
            '속도가', '속도를', '속도에', '속도의', '속도도', '속도만',
            '품질을', '품질이', '품질에', '품질의', '품질도', '품질만',
            '업무를', '업무가', '업무에', '업무의', '업무도', '업무만',
            '일정을', '일정이', '일정에', '일정의', '일정도', '일정만',
            '목표를', '목표가', '목표에', '목표의', '목표도', '목표만',
            '결과를', '결과가', '결과에', '결과의', '결과도', '결과만',
            '문제를', '문제가', '문제에', '문제의', '문제도', '문제만',
            '방법을', '방법이', '방법에', '방법의', '방법도', '방법만',
            '상황을', '상황이', '상황에', '상황의', '상황도', '상황만',
            '과제를', '과제가', '과제에', '과제의', '과제도', '과제만',
            '프로젝트를', '프로젝트가', '프로젝트에', '프로젝트의',
            '회의를', '회의가', '회의에', '회의의', '회의도', '회의만',
            
            # 기타 자주 나타나는 조사 결합
            '에서는', '에서도', '에서만', '에게는', '에게도', '에게만',
            '으로는', '으로도', '으로만', '로는', '로도', '로만',
            '와는', '와도', '와만', '과는', '과도', '과만'
        ]
        
        # 모든 어미를 길이 순으로 정렬하여 처리
        all_endings = very_complex_endings + complex_endings + basic_endings + particles + common_patterns
        all_endings = sorted(set(all_endings), key=len, reverse=True)
        
        original_word = word
        
        # 여러 번 반복하여 중첩된 어미 제거 (더 많이 반복)
        for iteration in range(10):  # 최대 10번 반복으로 증가
            changed = False
            for ending in all_endings:
                if word.endswith(ending):
                    root = word[:-len(ending)]
                    if len(root) >= 1:  # 최소 1글자는 남겨야 함
                        word = root
                        changed = True
                        break
            
            if not changed:
                break
        
        # 추가 정제: 남은 조사들 한번 더 체크
        final_particles = ['은', '는', '이', '가', '을', '를', '의', '도', '만', '에', '로', '와', '과']
        for particle in final_particles:
            if word.endswith(particle) and len(word) > len(particle) + 1:
                word = word[:-len(particle)]
        
        # 추가 정제: 반복 문자 제거
        word = re.sub(r'(.)\1{2,}', r'\1', word)
        
        # 결과 검증 (더 엄격하게)
        if (len(word) >= 2 and 
            word != original_word and 
            word not in self.stopwords and
            not self.is_meaningless_word(word)):
            return word
        elif (len(original_word) >= 2 and 
              original_word not in self.stopwords and
              not self.is_meaningless_word(original_word)):
            return original_word
        else:
            return None
    
    def get_keyword_frequency(self, texts):
        """
        텍스트 리스트에서 키워드 빈도 계산
        """
        all_keywords = []
        for text in texts:
            # 형태소 분석 시도, 실패하면 기본 분석 사용
            keywords = self.extract_keywords_morpheme(text)
            all_keywords.extend(keywords)
        
        return Counter(all_keywords)
    
    def get_top_keywords(self, frequency_counter, limit=15):
        """상위 키워드 추출"""
        return frequency_counter.most_common(limit)
    
    def calculate_keyword_differences(self, resigned_freq, stayed_freq, min_count=3):
        """퇴직자와 재직자 간 키워드 출현율 차이 계산"""
        all_keywords = set()
        for word, count in resigned_freq.items():
            if count >= min_count or stayed_freq.get(word, 0) >= min_count:
                all_keywords.add(word)
        
        for word, count in stayed_freq.items():
            if count >= min_count or resigned_freq.get(word, 0) >= min_count:
                all_keywords.add(word)
        
        differences = []
        resigned_total = len(self.resigned_data)
        stayed_total = len(self.stayed_data)
        
        for word in all_keywords:
            resigned_count = resigned_freq.get(word, 0)
            stayed_count = stayed_freq.get(word, 0)
            
            resigned_rate = (resigned_count / resigned_total) * 100
            stayed_rate = (stayed_count / stayed_total) * 100
            
            difference = resigned_rate - stayed_rate
            
            if abs(difference) > 0.3:
                differences.append({
                    'word': word,
                    'resigned_count': resigned_count,
                    'stayed_count': stayed_count,
                    'resigned_rate': round(resigned_rate, 2),
                    'stayed_rate': round(stayed_rate, 2),
                    'difference': round(difference, 2)
                })
        
        return sorted(differences, key=lambda x: abs(x['difference']), reverse=True)
    
    def analyze_feedback_type(self, column_name, feedback_type):
        """특정 피드백 유형에 대한 키워드 분석"""
        print(f"\n{'='*50}")
        print(f"{feedback_type} 키워드 분석")
        print(f"{'='*50}")
        
        # 키워드 빈도 계산
        resigned_freq = self.get_keyword_frequency(self.resigned_data[column_name].fillna(''))
        stayed_freq = self.get_keyword_frequency(self.stayed_data[column_name].fillna(''))
        
        # 상위 키워드
        top_resigned = self.get_top_keywords(resigned_freq)
        top_stayed = self.get_top_keywords(stayed_freq)
        
        print(f"\n퇴직자 상위 키워드:")
        for i, (word, freq) in enumerate(top_resigned, 1):
            print(f"{i:2d}. {word}: {freq}회")
        
        print(f"\n재직자 상위 키워드:")
        for i, (word, freq) in enumerate(top_stayed, 1):
            print(f"{i:2d}. {word}: {freq}회")
        
        # 차이 분석
        differences = self.calculate_keyword_differences(resigned_freq, stayed_freq)
        
        # 퇴직자에게 더 높게 나타나는 키워드
        resigned_high = [d for d in differences if d['difference'] > 0][:10]
        print(f"\n** 퇴직자에게 더 높게 나타나는 키워드 (상위 10개) **")
        for i, item in enumerate(resigned_high, 1):
            print(f"{i:2d}. {item['word']}: 퇴직자 {item['resigned_rate']}% vs 재직자 {item['stayed_rate']}% (차이: +{item['difference']}%p)")
        
        # 재직자에게 더 높게 나타나는 키워드
        stayed_high = [d for d in differences if d['difference'] < 0][:10]
        print(f"\n** 재직자에게 더 높게 나타나는 키워드 (상위 10개) **")
        for i, item in enumerate(stayed_high, 1):
            print(f"{i:2d}. {item['word']}: 퇴직자 {item['resigned_rate']}% vs 재직자 {item['stayed_rate']}% (차이: {item['difference']}%p)")
        
        return {
            'top_resigned': top_resigned,
            'top_stayed': top_stayed,
            'resigned_high': resigned_high,
            'stayed_high': stayed_high,
            'all_differences': differences
        }
    
    def run_complete_analysis(self):
        """전체 키워드 분석 실행"""
        if not self.load_data():
            return None
        
        print("=== IBM HR 데이터 키워드 분석 결과 ===")
        print("(기본 한국어 분석 사용 - Java 없이 안전하게 실행)")
        
        print(f"\n총 데이터: {len(self.data)}명 (퇴직자: {len(self.resigned_data)}명, 재직자: {len(self.stayed_data)}명)\n")
        
        # 각 피드백 유형별 분석
        results = {}
        
        # 1. Self-Performance Review 분석
        results['self_review'] = self.analyze_feedback_type('Self-Performance Review', '1. Self-Performance Review')
        
        # 2. Peer Feedback 분석
        results['peer_feedback'] = self.analyze_feedback_type('Peer Feedback', '2. Peer Feedback')
        
        # 3. Weekly Survey 분석
        results['weekly_survey'] = self.analyze_feedback_type('Weekly Survey', '3. Weekly Survey')
        
        # 종합 분석
        self.comprehensive_analysis(results)
        
        return results
    
    def analyze_unique_keywords(self, results, min_frequency=30):
        """겹치지 않는 고유 키워드 분석"""
        print(f"\n{'='*50}")
        print(f"고유 키워드 분석 (최소 {min_frequency}회 이상, 겹치지 않는 키워드)")
        print(f"{'='*50}")
        
        for category, result in results.items():
            category_name = category.replace('_', ' ').title()
            print(f"\n--- {category_name} ---")
            
            # 퇴직자와 재직자의 키워드 빈도 수집
            resigned_words = {word: freq for word, freq in result['top_resigned'] if freq >= min_frequency}
            stayed_words = {word: freq for word, freq in result['top_stayed'] if freq >= min_frequency}
            
            # 퇴직자에만 나타나는 키워드 (재직자에서는 빈도가 낮거나 없음)
            resigned_unique = []
            for word, freq in resigned_words.items():
                stayed_freq = dict(result['top_stayed']).get(word, 0)
                if stayed_freq < min_frequency:  # 재직자에서는 50회 미만
                    resigned_unique.append((word, freq, stayed_freq))
            
            # 재직자에만 나타나는 키워드 (퇴직자에서는 빈도가 낮거나 없음)
            stayed_unique = []
            for word, freq in stayed_words.items():
                resigned_freq = dict(result['top_resigned']).get(word, 0)
                if resigned_freq < min_frequency:  # 퇴직자에서는 50회 미만
                    stayed_unique.append((word, freq, resigned_freq))
            
            # 결과 출력
            if resigned_unique:
                print(f"\n🔴 퇴직자 고유 키워드 ({min_frequency}회 이상):")
                resigned_unique.sort(key=lambda x: x[1], reverse=True)
                for i, (word, resigned_freq, stayed_freq) in enumerate(resigned_unique[:10], 1):
                    print(f"{i:2d}. {word}: 퇴직자 {resigned_freq}회 vs 재직자 {stayed_freq}회")
            else:
                print(f"\n🔴 퇴직자 고유 키워드 ({min_frequency}회 이상): 없음")
            
            if stayed_unique:
                print(f"\n🔵 재직자 고유 키워드 ({min_frequency}회 이상):")
                stayed_unique.sort(key=lambda x: x[1], reverse=True)
                for i, (word, stayed_freq, resigned_freq) in enumerate(stayed_unique[:10], 1):
                    print(f"{i:2d}. {word}: 재직자 {stayed_freq}회 vs 퇴직자 {resigned_freq}회")
            else:
                print(f"\n🔵 재직자 고유 키워드 ({min_frequency}회 이상): 없음")
    
    def comprehensive_analysis(self, results):
        """종합 분석 및 요약"""
        print(f"\n{'='*50}")
        print("종합 분석 요약")
        print(f"{'='*50}")
        
        # 전체 퇴직자 특징 키워드 수집
        all_resigned_keywords = []
        
        for category, result in results.items():
            for item in result['resigned_high']:
                all_resigned_keywords.append({
                    'word': item['word'],
                    'category': category.replace('_', '-'),
                    'difference': abs(item['difference'])
                })
        
        # 차이가 큰 순으로 정렬
        top_resigned_overall = sorted(all_resigned_keywords, key=lambda x: x['difference'], reverse=True)[:15]
        
        print(f"\n퇴직자 특징 키워드 종합 (상위 15개):")
        for i, item in enumerate(top_resigned_overall, 1):
            print(f"{i:2d}. [{item['category']}] {item['word']}: +{item['difference']:.2f}%p 차이")
        
        # 고유 키워드 분석 추가
        self.analyze_unique_keywords(results, min_frequency=30)
    
    def create_visualizations(self, results):
        """키워드 분석 결과 시각화 (한글 폰트 적용)"""
        setup_korean_font()  # 한글 폰트 설정
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('IBM HR 키워드 분석 결과', fontsize=16, fontweight='bold')
        
        categories = ['self_review', 'peer_feedback', 'weekly_survey']
        titles = ['Self-Performance Review', 'Peer Feedback', 'Weekly Survey']
        
        for i, (category, title) in enumerate(zip(categories, titles)):
            # 퇴직자 키워드 (상위 10개)
            resigned_words = [item[0] for item in results[category]['top_resigned'][:10]]
            resigned_counts = [item[1] for item in results[category]['top_resigned'][:10]]
            
            axes[0, i].barh(resigned_words, resigned_counts, color='salmon', alpha=0.7)
            axes[0, i].set_title(f'{title}\n퇴직자 상위 키워드', fontweight='bold')
            axes[0, i].set_xlabel('빈도')
            axes[0, i].invert_yaxis()
            
            # 재직자 키워드 (상위 10개)
            stayed_words = [item[0] for item in results[category]['top_stayed'][:10]]
            stayed_counts = [item[1] for item in results[category]['top_stayed'][:10]]
            
            axes[1, i].barh(stayed_words, stayed_counts, color='lightblue', alpha=0.7)
            axes[1, i].set_title(f'{title}\n재직자 상위 키워드', fontweight='bold')
            axes[1, i].set_xlabel('빈도')
            axes[1, i].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    def create_wordcloud(self, results):
        """워드클라우드 생성 (한글 폰트 적용)"""
        setup_korean_font()  # 한글 폰트 설정
        
        # 한글 폰트 경로 설정 (시스템별)
        font_path = None
        system = platform.system()
        
        if system == "Windows":
            font_path = "C:/Windows/Fonts/malgun.ttf"
        elif system == "Darwin":  # macOS
            font_path = "/System/Library/Fonts/AppleGothic.ttf"
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('키워드 워드클라우드', fontsize=16, fontweight='bold')
        
        categories = ['self_review', 'peer_feedback', 'weekly_survey']
        titles = ['Self Review', 'Peer Feedback', 'Weekly Survey']
        
        for i, (category, title) in enumerate(zip(categories, titles)):
            # 퇴직자 워드클라우드
            resigned_dict = dict(results[category]['top_resigned'][:20])
            if resigned_dict:
                try:
                    wordcloud_resigned = WordCloud(
                        width=400, height=300, 
                        background_color='white',
                        colormap='Reds',
                        font_path=font_path,
                        max_words=20
                    ).generate_from_frequencies(resigned_dict)
                    
                    axes[0, i].imshow(wordcloud_resigned, interpolation='bilinear')
                    axes[0, i].set_title(f'{title} - 퇴직자', fontweight='bold')
                    axes[0, i].axis('off')
                except Exception as e:
                    axes[0, i].text(0.5, 0.5, f'워드클라우드 생성 오류\n{str(e)}', 
                                   ha='center', va='center', transform=axes[0, i].transAxes)
            
            # 재직자 워드클라우드
            stayed_dict = dict(results[category]['top_stayed'][:20])
            if stayed_dict:
                try:
                    wordcloud_stayed = WordCloud(
                        width=400, height=300, 
                        background_color='white',
                        colormap='Blues',
                        font_path=font_path,
                        max_words=20
                    ).generate_from_frequencies(stayed_dict)
                    
                    axes[1, i].imshow(wordcloud_stayed, interpolation='bilinear')
                    axes[1, i].set_title(f'{title} - 재직자', fontweight='bold')
                    axes[1, i].axis('off')
                except Exception as e:
                    axes[1, i].text(0.5, 0.5, f'워드클라우드 생성 오류\n{str(e)}', 
                                   ha='center', va='center', transform=axes[1, i].transAxes)
        
        plt.tight_layout()
        plt.show()


def main():
    """메인 실행 함수"""
    # CSV 파일 경로 자동 탐지
    possible_paths = [
        'data/IBM_HR_report.csv',
        '../data/IBM_HR_report.csv',
        'IBM_HR_report.csv',
        'data/IBM_HR.csv',
        '../data/IBM_HR.csv',
        'IBM_HR.csv'
    ]
    
    csv_path = None
    for path in possible_paths:
        try:
            if pd.read_csv(path, nrows=1) is not None:
                csv_path = path
                print(f"데이터 파일을 찾았습니다: {path}")
                break
        except:
            continue
    
    if not csv_path:
        print("오류: CSV 파일을 찾을 수 없습니다.")
        print("다음 경로들을 확인했습니다:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\n파일 경로를 직접 지정하려면 다음과 같이 실행하세요:")
        print("analyzer = HRKeywordAnalyzer('your_file_path.csv')")
        return None, None
    
    # 분석기 초기화
    analyzer = HRKeywordAnalyzer(csv_path)
    
    # 전체 분석 실행
    results = analyzer.run_complete_analysis()
    
    if results:
        print("\n✅ 분석이 완료되었습니다!")
        
        # 시각화 생성 (선택사항)
        try:
            print("\n📊 시각화를 생성하는 중...")
            analyzer.create_visualizations(results)
            analyzer.create_wordcloud(results)
            print("✅ 시각화가 완료되었습니다!")
        except ImportError as e:
            print(f"⚠️ 시각화 라이브러리 오류: {e}")
            print("설치 명령어: pip install matplotlib seaborn wordcloud")
        except Exception as e:
            print(f"⚠️ 시각화 생성 오류: {e}")
    else:
        print("❌ 분석 실행 중 오류가 발생했습니다.")
    
    return analyzer, results


if __name__ == "__main__":
    # 프로그램 시작 안내
    print("🔍 IBM HR 데이터 키워드 분석기")
    print("=" * 50)
    
    # 필요한 라이브러리 확인
    missing_libs = []
    try:
        import pandas
    except ImportError:
        missing_libs.append("pandas")
    
    try:
        import matplotlib
    except ImportError:
        missing_libs.append("matplotlib")
    
    try:
        import seaborn
    except ImportError:
        missing_libs.append("seaborn")
    
    try:
        import wordcloud
    except ImportError:
        missing_libs.append("wordcloud")
    
    if missing_libs:
        print("⚠️ 다음 라이브러리가 필요합니다:")
        print(f"pip install {' '.join(missing_libs)}")
        print()
    
    # KoNLPy 관련 안내 (Java 문제 해결)
    print("📝 한국어 분석 모드: 기본 분석 (Java 없이)")
    print("ℹ️ 더 정확한 분석을 원하시면:")
    print("1. Java 8 이상 설치")
    print("2. pip install konlpy")
    print("3. 환경변수 JAVA_HOME 설정")
    print("4. 코드에서 try_import_konlpy() 호출 활성화")
    print()
    
    print("🚀 분석을 시작합니다...")
    print("=" * 50)
    
    # 분석 실행
    analyzer, results = main()
    
    if analyzer and results:
        print("\n" + "=" * 50)
        print("🎉 모든 작업이 완료되었습니다!")
        print("=" * 50)