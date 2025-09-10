#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 IBM HR 데이터 키워드 분석기
- 명사 중심 추출
- 강화된 불용어 처리
- 더 정확한 퇴직 위험 신호 탐지
"""

import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

class ImprovedHRKeywordAnalyzer:
    def __init__(self, csv_file_path):
        """
        개선된 IBM HR 데이터 키워드 분석기 초기화
        
        Args:
            csv_file_path (str): CSV 파일 경로
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.resigned_data = None
        self.stayed_data = None
        
        # 강화된 불용어 리스트 (조사, 어미, 부사 등)
        self.stopwords = self._initialize_comprehensive_stopwords()
        
        # 명사 패턴 정의
        self.noun_patterns = self._initialize_noun_patterns()
        
    def _initialize_comprehensive_stopwords(self):
        """포괄적인 불용어 리스트 초기화"""
        return set([
            # 조사 (완전한 형태)
            '은', '는', '이', '가', '을', '를', '에', '에서', '로', '으로', '와', '과', '의', '도',
            '만', '부터', '까지', '에게', '한테', '께', '에게서', '한테서', '께서', '라', '야',
            '아', '어', '여', '이랑', '랑', '하고', '에다', '에다가', '로서', '로써', '처럼',
            
            # 어미 및 용언 활용형
            '다', '습니다', '입니다', '했습니다', '있습니다', '없습니다', '됩니다', 
            '니다', '였습니다', '이었습니다', '하였습니다', '것입니다', '싶습니다',
            '보입니다', '같습니다', '겠습니다', '드립니다', '부탁드립니다',
            '하겠습니다', '했어요', '해요', '이에요', '예요', '했네요', '했죠',
            
            # 부사 (문제가 되는 단어들)
            '아직', '이미', '벌써', '곧', '바로', '즉시', '항상', '늘', '자주', '가끔',
            '때때로', '종종', '다시', '또', '또다시', '다음', '이번', '저번', '언제나',
            '그냥', '좀', '조금', '많이', '너무', '매우', '정말', '진짜', '완전',
            '엄청', '되게', '꽤', '상당히', '약간', '조금씩', '점점', '갈수록',
            
            # 대명사 및 지시어
            '그', '이', '저', '것', '그것', '이것', '저것', '그런', '이런', '저런',
            '그러한', '이러한', '저러한', '그렇', '이렇', '저렇', '여기', '거기', '저기',
            '이곳', '그곳', '저곳', '어디', '언제', '누구', '무엇', '뭐', '어떤',
            
            # 감탄사 및 의성어/의태어
            '아', '어', '오', '우', '에', '이', '와', '우와', '어머', '아이고',
            '어라', '어어', '음음', '흠흠', '허허', '하하', '헤헤', '히히', '호호',
            
            # 접속사 및 연결어
            '그리고', '그런데', '하지만', '그러나', '따라서', '그래서', '또한', '또는',
            '혹은', '만약', '만일', '비록', '설령', '심지어', '특히', '예를 들어',
            
            # 일반적이고 의미가 약한 명사들
            '것', '거', '게', '걸', '건', '곳', '데', '때', '분', '번', '개', '명',
            '원', '년', '월', '일', '시', '분', '초', '점', '개수', '수', '양', '정도',
            
            # 문제가 되는 형용사 어간들
            '느낌', '기분', '생각', '마음', '의견', '견해', '입장', '관점', '시각',
            
            # 자주 나오지만 의미가 약한 동사 어간들
            '하', '되', '있', '없', '가', '오', '보', '듣', '말하', '생각하', '느끼',
            '알', '모르', '좋', '나쁘', '크', '작', '높', '낮', '많', '적', '빠르', '느리',
            
            # 업무 텍스트에서 자주 나오는 일반적 표현들
            '저희', '우리', '제가', '저를', '저는', '저의', '저에게', '회사', '직원',
            '사람', '분', '님', '씨', '분들', '님들', '씨들', '여러분', '모두',
            
            # 시간 관련 일반적 표현
            '먼저', '나중', '다음', '이전', '전에', '후에', '동안', '사이', '중간',
            '처음', '마지막', '끝', '시작', '중', '현재', '지금', '오늘', '어제', '내일',
            
            # 추상적이고 의미가 약한 표현들
            '방향', '쪽', '편', '면', '측', '관점', '입장', '상황', '경우', '상태',
            '조건', '환경', '분위기', '느낌', '기분', '방법', '방식', '형태', '모습',
            
            # 문제가 되는 특정 단어들 (터미널에서 확인된 것들)
            '느낌이', '아직도', '주어진', '남아있', '이슈가', '파악하', '되어', '하여',
            '이며', '하고', '하니', '하면', '되면', '이면', '라면', '다면', '하기',
            '되기', '이기', '하는', '되는', '이는', '하여서', '되어서', '이어서'
        ])
    
    def _initialize_noun_patterns(self):
        """명사 패턴 정의"""
        return {
            # 업무 관련 명사 패턴
            'work_nouns': [
                r'업무', r'일', r'직무', r'과업', r'임무', r'역할', r'책임', r'담당',
                r'프로젝트', r'과제', r'계획', r'목표', r'성과', r'결과', r'실적'
            ],
            
            # 조직 관련 명사 패턴  
            'org_nouns': [
                r'팀', r'부서', r'조직', r'회사', r'기업', r'직장', r'사업부', r'본부',
                r'팀장', r'부장', r'과장', r'대리', r'사원', r'동료', r'상사', r'부하'
            ],
            
            # 감정/상태 관련 명사 패턴
            'emotion_nouns': [
                r'만족', r'불만', r'스트레스', r'압박', r'부담', r'피로', r'소진', r'번아웃',
                r'열정', r'동기', r'의욕', r'흥미', r'관심', r'집중', r'몰입', r'참여'
            ],
            
            # 성장/발전 관련 명사 패턴
            'growth_nouns': [
                r'성장', r'발전', r'향상', r'개선', r'발달', r'진보', r'진전', r'도약',
                r'교육', r'학습', r'연수', r'교육훈련', r'역량', r'능력', r'기술', r'스킬'
            ],
            
            # 보상/평가 관련 명사 패턴
            'reward_nouns': [
                r'급여', r'연봉', r'월급', r'보상', r'임금', r'수당', r'보너스', r'인센티브',
                r'평가', r'고과', r'심사', r'검토', r'승진', r'승격', r'인사', r'발령'
            ]
        }
    
    def load_data(self):
        """CSV 데이터 로드 및 전처리"""
        try:
            self.data = pd.read_csv(self.csv_file_path, encoding='utf-8-sig')
            print(f"✅ 데이터 로드 완료: {len(self.data)}개 행")
            
            # Attrition 컬럼 확인
            if 'Attrition' not in self.data.columns:
                print("❌ 'Attrition' 컬럼을 찾을 수 없습니다.")
                return False
            
            # 퇴직자/재직자 데이터 분리
            self.resigned_data = self.data[self.data['Attrition'] == 'Yes']
            self.stayed_data = self.data[self.data['Attrition'] == 'No']
            
            print(f"📊 퇴직자: {len(self.resigned_data)}명, 재직자: {len(self.stayed_data)}명")
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로드 오류: {e}")
            return False
    
    def extract_nouns_only(self, text):
        """명사 중심 키워드 추출 (개선된 방식)"""
        if pd.isna(text) or not text:
            return []
        
        # 1단계: 텍스트 정제
        clean_text = self._clean_text(text)
        
        # 2단계: 명사 후보 추출
        noun_candidates = self._extract_noun_candidates(clean_text)
        
        # 3단계: 명사 필터링 및 정제
        filtered_nouns = self._filter_and_clean_nouns(noun_candidates)
        
        return filtered_nouns
    
    def _clean_text(self, text):
        """텍스트 정제"""
        # 특수문자 제거
        clean_text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 연속 공백 정리
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def _extract_noun_candidates(self, text):
        """명사 후보 추출"""
        # 한글 단어 추출 (2글자 이상)
        words = re.findall(r'[가-힣]{2,}', text)
        
        # 명사 패턴 매칭으로 우선순위 부여
        noun_candidates = []
        
        for word in words:
            # 명사 패턴에 매칭되는 단어들 우선 선택
            is_likely_noun = self._is_likely_noun(word)
            if is_likely_noun:
                noun_candidates.append(word)
        
        return noun_candidates
    
    def _is_likely_noun(self, word):
        """명사일 가능성이 높은 단어인지 판단"""
        # 명사 패턴 매칭
        for category, patterns in self.noun_patterns.items():
            for pattern in patterns:
                if re.search(pattern, word):
                    return True
        
        # 명사 어미 패턴 (한국어 명사의 일반적 특징)
        noun_endings = [
            r'.*성$',    # ~성 (특성, 중요성 등)
            r'.*도$',    # ~도 (만족도, 참여도 등) 
            r'.*력$',    # ~력 (능력, 역량 등)
            r'.*감$',    # ~감 (만족감, 성취감 등)
            r'.*율$',    # ~율 (효율, 비율 등)
            r'.*제$',    # ~제 (제도, 복리후생제 등)
            r'.*화$',    # ~화 (개선화, 체계화 등)
            r'.*관$',    # ~관 (가치관, 직업관 등)
            r'.*점$',    # ~점 (관점, 시점 등)
            r'.*면$',    # ~면 (측면, 방면 등)
        ]
        
        for pattern in noun_endings:
            if re.match(pattern, word):
                return True
        
        # 복합명사 패턴
        compound_patterns = [
            r'.*업무.*',   # 업무 관련
            r'.*관리.*',   # 관리 관련  
            r'.*개발.*',   # 개발 관련
            r'.*평가.*',   # 평가 관련
            r'.*교육.*',   # 교육 관련
        ]
        
        for pattern in compound_patterns:
            if re.match(pattern, word):
                return True
        
        return False
    
    def _filter_and_clean_nouns(self, candidates):
        """명사 후보 필터링 및 정제"""
        filtered = []
        
        for word in candidates:
            # 불용어 제거
            if word in self.stopwords:
                continue
            
            # 어미 제거
            cleaned_word = self._remove_endings(word)
            if not cleaned_word or len(cleaned_word) < 2:
                continue
            
            # 의미없는 단어 제거
            if self._is_meaningless_noun(cleaned_word):
                continue
            
            filtered.append(cleaned_word)
        
        return filtered
    
    def _remove_endings(self, word):
        """한국어 어미 제거 (명사 중심)"""
        # 명사 어미 제거 패턴
        noun_ending_patterns = [
            r'(.+)들$',      # 복수형 (~들)
            r'(.+)이$',      # 주격조사 (~이)  
            r'(.+)가$',      # 주격조사 (~가)
            r'(.+)을$',      # 목적격조사 (~을)
            r'(.+)를$',      # 목적격조사 (~를)
            r'(.+)에서$',    # 처격조사 (~에서)
            r'(.+)으로$',    # 도구격조사 (~으로)
            r'(.+)로$',      # 도구격조사 (~로)
            r'(.+)와$',      # 접속조사 (~와)
            r'(.+)과$',      # 접속조사 (~과)
            r'(.+)의$',      # 관형격조사 (~의)
        ]
        
        for pattern in noun_ending_patterns:
            match = re.match(pattern, word)
            if match:
                return match.group(1)
        
        return word
    
    def _is_meaningless_noun(self, word):
        """의미없는 명사 판별"""
        meaningless_nouns = {
            # 너무 일반적인 명사들
            '것', '거', '게', '걸', '건', '곳', '데', '때', '분', '번',
            '개', '명', '원', '년', '월', '일', '시', '분', '초',
            
            # 추상적이고 의미가 약한 명사들
            '방향', '쪽', '편', '면', '측', '부분', '전체', '일부',
            '상황', '경우', '상태', '조건', '환경', '분위기',
            
            # 너무 일반적인 사람 지칭어
            '사람', '분', '님', '씨', '이', '자', '者',
            
            # 의미가 약한 시간 표현
            '시간', '기간', '동안', '사이', '중', '때문',
        }
        
        return word in meaningless_nouns
    
    def analyze_text_columns(self):
        """텍스트 컬럼 분석"""
        if self.data is None:
            print("❌ 데이터를 먼저 로드해주세요.")
            return None
        
        # 텍스트 컬럼 찾기
        text_columns = []
        for col in self.data.columns:
            if col.endswith('_text') or 'text' in col.lower():
                text_columns.append(col)
        
        if not text_columns:
            print("❌ 텍스트 컬럼을 찾을 수 없습니다.")
            return None
        
        print(f"📝 발견된 텍스트 컬럼: {text_columns}")
        
        results = {}
        
        for col in text_columns:
            print(f"\n🔍 '{col}' 컬럼 분석 중...")
            
            # 퇴직자 키워드 추출
            resigned_keywords = []
            for text in self.resigned_data[col].dropna():
                resigned_keywords.extend(self.extract_nouns_only(text))
            
            # 재직자 키워드 추출  
            stayed_keywords = []
            for text in self.stayed_data[col].dropna():
                stayed_keywords.extend(self.extract_nouns_only(text))
            
            # 키워드 빈도 계산
            resigned_counter = Counter(resigned_keywords)
            stayed_counter = Counter(stayed_keywords)
            
            # 결과 저장
            results[col] = {
                'resigned_keywords': resigned_counter,
                'stayed_keywords': stayed_counter,
                'resigned_total': len(resigned_keywords),
                'stayed_total': len(stayed_keywords)
            }
            
            print(f"  퇴직자 키워드: {len(resigned_keywords)}개 (유니크: {len(resigned_counter)}개)")
            print(f"  재직자 키워드: {len(stayed_keywords)}개 (유니크: {len(stayed_counter)}개)")
        
        return results
    
    def find_distinctive_keywords(self, results, min_frequency=5):
        """차별적 키워드 찾기 (개선된 방식)"""
        if not results:
            return None
        
        print(f"\n🎯 차별적 키워드 분석 (최소 빈도: {min_frequency}회)")
        print("=" * 60)
        
        all_distinctive = {}
        
        for col, data in results.items():
            print(f"\n📊 [{col}] 컬럼 분석 결과:")
            
            resigned_counter = data['resigned_keywords']
            stayed_counter = data['stayed_keywords']
            
            # 퇴직자 고유 키워드 (재직자에서는 거의 나타나지 않음)
            resigned_unique = []
            for word, freq in resigned_counter.most_common():
                if freq >= min_frequency:
                    stayed_freq = stayed_counter.get(word, 0)
                    # 퇴직자에서 많이 나오고 재직자에서는 적게 나오는 경우
                    if stayed_freq <= freq * 0.3:  # 재직자 빈도가 퇴직자의 30% 이하
                        ratio = freq / max(stayed_freq, 1)
                        resigned_unique.append((word, freq, stayed_freq, ratio))
            
            # 재직자 고유 키워드
            stayed_unique = []
            for word, freq in stayed_counter.most_common():
                if freq >= min_frequency:
                    resigned_freq = resigned_counter.get(word, 0)
                    if resigned_freq <= freq * 0.3:
                        ratio = freq / max(resigned_freq, 1)
                        stayed_unique.append((word, freq, resigned_freq, ratio))
            
            # 결과 출력
            print(f"\n🔴 퇴직자 특징적 키워드 (상위 10개):")
            for i, (word, r_freq, s_freq, ratio) in enumerate(resigned_unique[:10], 1):
                print(f"{i:2d}. {word}: 퇴직자 {r_freq}회 vs 재직자 {s_freq}회 (비율: {ratio:.1f})")
            
            print(f"\n🔵 재직자 특징적 키워드 (상위 10개):")
            for i, (word, s_freq, r_freq, ratio) in enumerate(stayed_unique[:10], 1):
                print(f"{i:2d}. {word}: 재직자 {s_freq}회 vs 퇴직자 {r_freq}회 (비율: {ratio:.1f})")
            
            all_distinctive[col] = {
                'resigned_unique': resigned_unique,
                'stayed_unique': stayed_unique
            }
        
        return all_distinctive

def main():
    """메인 실행 함수"""
    print("🔍 개선된 IBM HR 데이터 키워드 분석기 (명사 중심)")
    print("=" * 60)
    
    # 데이터 파일 경로 (실제 경로로 수정 필요)
    csv_file_path = "../sample_hr_texts.csv"  # 또는 실제 데이터 파일 경로
    
    # 파일이 없으면 다른 경로 시도
    import os
    if not os.path.exists(csv_file_path):
        alternative_paths = [
            "../data/IBM_HR_text_sample.csv",
            "sample_hr_texts.csv",
            "../sample_hr_texts.csv"
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                csv_file_path = path
                print(f"📁 데이터 파일 발견: {path}")
                break
        else:
            print("❌ 테스트용 데이터 파일을 찾을 수 없습니다.")
            print("💡 다음 중 하나의 파일이 필요합니다:")
            for path in alternative_paths:
                print(f"   - {path}")
            return None, None
    
    # 분석기 초기화
    analyzer = ImprovedHRKeywordAnalyzer(csv_file_path)
    
    # 데이터 로드
    if not analyzer.load_data():
        return None, None
    
    # 텍스트 분석
    results = analyzer.analyze_text_columns()
    if not results:
        return analyzer, None
    
    # 차별적 키워드 찾기
    distinctive_keywords = analyzer.find_distinctive_keywords(results, min_frequency=3)
    
    return analyzer, {
        'analysis_results': results,
        'distinctive_keywords': distinctive_keywords
    }

if __name__ == "__main__":
    analyzer, results = main()
    
    if analyzer and results:
        print("\n" + "=" * 60)
        print("🎉 개선된 키워드 분석 완료!")
        print("📝 명사 중심 추출로 더 정확한 결과를 얻었습니다.")
        print("=" * 60)
