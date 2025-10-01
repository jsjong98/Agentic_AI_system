# -*- coding: utf-8 -*-
"""
Sentio 키워드 분석기
개선된 명사 중심 키워드 추출 및 퇴직자 vs 재직자 차별적 키워드 분석
"""

import pandas as pd
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SentioKeywordAnalyzer:
    """개선된 HR 키워드 분석기 (명사 중심)"""
    
    def __init__(self, csv_file_path: str):
        """
        키워드 분석기 초기화
        
        Args:
            csv_file_path (str): CSV 파일 경로
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.resigned_data = None
        self.stayed_data = None
        
        # 강화된 불용어 리스트
        self.stopwords = self._initialize_comprehensive_stopwords()
        
        # 명사 패턴 정의
        self.noun_patterns = self._initialize_noun_patterns()
        
    def _initialize_comprehensive_stopwords(self) -> set:
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
    
    def _initialize_noun_patterns(self) -> Dict[str, List[str]]:
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
    
    def load_data(self) -> bool:
        """CSV 데이터 로드 및 전처리"""
        try:
            self.data = pd.read_csv(self.csv_file_path, encoding='utf-8-sig')
            logger.info(f"✅ 데이터 로드 완료: {len(self.data)}개 행")
            
            # Attrition 컬럼 확인
            if 'Attrition' not in self.data.columns:
                logger.warning("❌ 'Attrition' 컬럼을 찾을 수 없습니다.")
                return False
            
            # 퇴직자/재직자 데이터 분리
            self.resigned_data = self.data[self.data['Attrition'] == 'Yes']
            self.stayed_data = self.data[self.data['Attrition'] == 'No']
            
            logger.info(f"📊 퇴직자: {len(self.resigned_data)}명, 재직자: {len(self.stayed_data)}명")
            return True
            
        except Exception as e:
            logger.error(f"❌ 데이터 로드 오류: {e}")
            return False
    
    def extract_nouns_only(self, text: str) -> List[str]:
        """명사 중심 키워드 추출 (개선된 방식, persona 텍스트 제외)"""
        if pd.isna(text) or not text:
            return []
        
        # 0단계: persona 텍스트 확인 및 제외
        if self._is_persona_text(text):
            return []
        
        # 1단계: 텍스트 정제
        clean_text = self._clean_text(text)
        
        # 2단계: 명사 후보 추출
        noun_candidates = self._extract_noun_candidates(clean_text)
        
        # 3단계: 명사 필터링 및 정제
        filtered_nouns = self._filter_and_clean_nouns(noun_candidates)
        
        return filtered_nouns
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        # 특수문자 제거
        clean_text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 연속 공백 정리
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def _extract_noun_candidates(self, text: str) -> List[str]:
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
    
    def _is_likely_noun(self, word: str) -> bool:
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
    
    def _filter_and_clean_nouns(self, candidates: List[str]) -> List[str]:
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
    
    def _remove_endings(self, word: str) -> str:
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
    
    def _is_meaningless_noun(self, word: str) -> bool:
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
    
    def _is_persona_text(self, text_content: str) -> bool:
        """persona 관련 텍스트인지 판별"""
        if not text_content or pd.isna(text_content):
            return False
        
        text_lower = str(text_content).lower()
        
        # persona 관련 키워드 패턴 (강화된 버전)
        persona_indicators = [
            'persona', '페르소나', 'p01', 'p02', 'p03', 'p04', 'p05', 'n01', 's02',
            '번아웃 위험군', '성장 추구형', '안정 지향형', '도전 추구형', '균형 추구형', '현상 유지자', '라이징 스타', '저평가된 전문가',
            'burnout risk', 'growth seeker', 'stability oriented', 'challenge seeker', 'balance seeker',
            'softmax_persona', 'persona_code', 'persona_type', 'persona_name',
            # 추가 persona 관련 패턴
            '위험군', '추구형', '지향형', '유지자', '전문가', '스타'
        ]
        
        # persona 텍스트 패턴 확인
        for indicator in persona_indicators:
            if indicator in text_lower:
                return True
        
        # persona 설명 패턴 (일반적인 persona 설명 형태)
        persona_patterns = [
            r'높은\s+업무\s+부담',
            r'스트레스\s+수준이\s+높',
            r'번아웃\s+위험',
            r'성장\s+기회를\s+추구',
            r'안정적인\s+환경을\s+선호',
            r'새로운\s+도전을\s+추구',
            r'일과\s+삶의\s+균형'
        ]
        
        for pattern in persona_patterns:
            if re.search(pattern, text_content):
                return True
        
        return False
    
    def calculate_jdr_scores(self, text: str) -> Dict[str, Any]:
        """JD-R (Job Demands-Resources) 모델 기반 점수 계산"""
        
        # 직무 요구 관련 키워드 (부정적 요인)
        job_demands_keywords = {
            '업무량', '야근', '스트레스', '압박', '마감', '과로', '번아웃', '피로',
            '바쁘다', '힘들다', '어렵다', '복잡하다', '많다', '부족하다', '급하다',
            '걱정', '불안', '긴장', '부담', '책임', '문제', '어려움', '곤란',
            '시간부족', '인력부족', '자원부족', '예산부족', '지원부족'
        }
        
        # 직무 자원 결핍 관련 키워드 (자원 부족)
        job_resources_deficiency_keywords = {
            '지원부족', '도움없음', '혼자', '외롭다', '소통부족', '피드백부족',
            '교육부족', '훈련부족', '정보부족', '자료부족', '장비부족', '시설부족',
            '인정받지못함', '성장기회없음', '발전없음', '승진어려움', '보상부족',
            '불공정', '차별', '무시', '배제', '소외', '관심없음', '방치',
            '자율성부족', '권한없음', '결정권없음', '참여기회없음'
        }
        
        # 텍스트에서 키워드 추출
        detected_keywords = self.extract_nouns_only(text)
        
        # 각 카테고리별 매칭 키워드 찾기
        job_demands_matches = []
        job_resources_deficiency_matches = []
        
        for keyword in detected_keywords:
            if keyword in job_demands_keywords:
                job_demands_matches.append(keyword)
            if keyword in job_resources_deficiency_keywords:
                job_resources_deficiency_matches.append(keyword)
        
        # 점수 계산 (키워드 빈도 기반, 0-1 정규화)
        total_keywords = len(detected_keywords) if detected_keywords else 1
        
        job_demands_score = min(len(job_demands_matches) / total_keywords * 2.0, 1.0)
        job_resources_deficiency_score = min(len(job_resources_deficiency_matches) / total_keywords * 2.0, 1.0)
        
        return {
            'job_demands_score': job_demands_score,
            'job_resources_deficiency_score': job_resources_deficiency_score,
            'detected_keywords': detected_keywords,
            'job_demands_matches': job_demands_matches,
            'job_resources_deficiency_matches': job_resources_deficiency_matches
        }
    
    def analyze_employee_text(self, employee_id, self_review, peer_feedback, weekly_survey):
        """직원 텍스트 종합 분석 (개선된 JD-R 모델 기반)"""
        
        # 텍스트 결합
        texts = [self_review, peer_feedback, weekly_survey]
        combined_text = ' '.join([str(text) for text in texts if pd.notna(text)])
        
        if not combined_text.strip():
            return {
                'employee_id': employee_id,
                'psychological_risk_score': 0.5,
                'jd_r_indicators': {
                    'job_demands_score': 0.0,
                    'job_resources_deficiency_score': 0.5
                },
                'detected_keywords': [],
                'sentiment_score': 0.5,  # 하위 호환성
                'risk_keywords': [],     # 하위 호환성
                'risk_level': 'MEDIUM',
                'analysis_details': "텍스트 데이터가 부족하여 정확한 분석이 어렵습니다. 추가적인 피드백 수집을 권장합니다.",
                'attrition_prediction': 0
            }
        
        # JD-R 지표 계산
        jdr_result = self.calculate_jdr_scores(combined_text)
        
        # 심리적 위험 점수 계산 (개선된 가중치)
        psychological_risk_score = (
            jdr_result['job_demands_score'] * 0.75 +
            jdr_result['job_resources_deficiency_score'] * 0.25
        )
        psychological_risk_score = min(max(psychological_risk_score, 0.0), 1.0)
        
        # 최적 임계값 (기본값, 실제로는 학습을 통해 결정)
        optimal_threshold = getattr(self, 'optimal_threshold', 0.45)
        
        # 이진 예측 (최적 임계값 사용)
        attrition_prediction = 1 if psychological_risk_score > optimal_threshold else 0
        
        # 위험 수준 결정 (개선된 기준)
        if psychological_risk_score >= 0.7:
            risk_level = 'HIGH'
        elif psychological_risk_score >= 0.3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # 하위 호환성을 위한 기존 필드들
        sentiment_score = 1.0 - psychological_risk_score  # 감정 점수는 위험 점수의 역
        risk_keywords = jdr_result['detected_keywords'][:10]  # 상위 10개만
        
        # CSV 저장용 핵심 결과 (LLM 없이, 빠른 처리)
        return {
            'employee_id': employee_id,
            'psychological_risk_score': psychological_risk_score,
            'jd_r_indicators': {
                'job_demands_score': jdr_result['job_demands_score'],
                'job_resources_deficiency_score': jdr_result['job_resources_deficiency_score']
            },
            'detected_keywords': jdr_result['detected_keywords'],
            'job_demands_matches': jdr_result['job_demands_matches'],
            'job_resources_deficiency_matches': jdr_result['job_resources_deficiency_matches'],
            'sentiment_score': sentiment_score,  # 하위 호환성
            'risk_keywords': risk_keywords,     # 하위 호환성
            'risk_level': risk_level,
            'attrition_prediction': attrition_prediction,
            'analysis_timestamp': datetime.now().isoformat(),
            # CSV 저장을 위한 간단한 요약 (LLM 없이)
            'analysis_summary': f"위험도: {risk_level}, 직무요구: {jdr_result['job_demands_score']:.3f}, 자원결핍: {jdr_result['job_resources_deficiency_score']:.3f}"
        }
    
    def generate_csv_batch_analysis(self, text_data_list: List[Dict]) -> pd.DataFrame:
        """대량 텍스트 데이터를 CSV 저장용으로 빠르게 분석 (LLM 없이)"""
        
        results = []
        
        for i, text_data in enumerate(text_data_list):
            employee_id = text_data.get('employee_id', f'emp_{i+1:04d}')
            self_review = text_data.get('self_review', '')
            peer_feedback = text_data.get('peer_feedback', '')
            weekly_survey = text_data.get('weekly_survey', '')
            
            # JD-R 기반 빠른 분석 (LLM 없이)
            analysis_result = self.analyze_employee_text(
                employee_id=employee_id,
                self_review=self_review,
                peer_feedback=peer_feedback,
                weekly_survey=weekly_survey
            )
            
            # CSV용 플랫 구조로 변환
            csv_row = {
                'employee_id': employee_id,
                'psychological_risk_score': analysis_result['psychological_risk_score'],
                'job_demands_score': analysis_result['jd_r_indicators']['job_demands_score'],
                'job_resources_deficiency_score': analysis_result['jd_r_indicators']['job_resources_deficiency_score'],
                'risk_level': analysis_result['risk_level'],
                'attrition_prediction': analysis_result['attrition_prediction'],
                'sentiment_score': analysis_result['sentiment_score'],
                'detected_keywords_count': len(analysis_result['detected_keywords']),
                'job_demands_keywords': ', '.join(analysis_result['job_demands_matches'][:5]),  # 상위 5개
                'job_resources_deficiency_keywords': ', '.join(analysis_result['job_resources_deficiency_matches'][:5]),  # 상위 5개
                'top_risk_keywords': ', '.join(analysis_result['risk_keywords'][:5]),  # 상위 5개
                'analysis_timestamp': analysis_result['analysis_timestamp'],
                'analysis_summary': analysis_result['analysis_summary']
            }
            
            results.append(csv_row)
            
            # 진행 상황 출력 (100개마다)
            if (i + 1) % 100 == 0:
                logger.info(f"CSV 분석 진행: {i + 1}/{len(text_data_list)} 완료")
        
        # DataFrame으로 변환
        df = pd.DataFrame(results)
        logger.info(f"CSV 분석 완료: 총 {len(results)}명 처리")
        
        return df
    
    def save_analysis_to_csv(self, df: pd.DataFrame, output_path: str = "sentio_analysis_results.csv") -> str:
        """분석 결과를 CSV 파일로 저장"""
        
        try:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"CSV 저장 완료: {output_path} ({len(df)}행)")
            return output_path
        except Exception as e:
            logger.error(f"CSV 저장 실패: {e}")
            raise
    
    def generate_individual_comprehensive_report(self, employee_id: str, all_worker_results: Dict, use_llm: bool = False) -> Dict:
        """개별 직원의 모든 워커 에이전트 결과를 종합한 최종 요약 레포트 생성"""
        
        if not all_worker_results:
            return {"error": f"직원 {employee_id}의 분석 결과가 없습니다."}
        
        # 각 워커별 결과 추출
        structura_result = all_worker_results.get('structura', {})
        cognita_result = all_worker_results.get('cognita', {})
        chronos_result = all_worker_results.get('chronos', {})
        sentio_result = all_worker_results.get('sentio', {})
        
        # 종합 위험도 계산 (각 워커의 점수를 가중평균)
        risk_scores = []
        
        # Structura: 퇴직 확률
        if structura_result.get('attrition_probability'):
            risk_scores.append(('structura', structura_result['attrition_probability'], 0.3))
        
        # Cognita: 전체 위험도
        if cognita_result.get('overall_risk_score'):
            risk_scores.append(('cognita', cognita_result['overall_risk_score'], 0.25))
        
        # Chronos: 예측 확률
        if chronos_result.get('probability'):
            risk_scores.append(('chronos', chronos_result['probability'], 0.2))
        
        # Sentio: 심리적 위험 점수
        if sentio_result.get('psychological_risk_score'):
            risk_scores.append(('sentio', sentio_result['psychological_risk_score'], 0.25))
        
        # 가중평균 계산
        if risk_scores:
            weighted_sum = sum(score * weight for _, score, weight in risk_scores)
            total_weight = sum(weight for _, _, weight in risk_scores)
            comprehensive_risk_score = weighted_sum / total_weight
        else:
            comprehensive_risk_score = 0.5
        
        # 종합 위험 수준 결정
        if comprehensive_risk_score >= 0.7:
            overall_risk_level = "HIGH"
            risk_color = "🔴"
        elif comprehensive_risk_score >= 0.4:
            overall_risk_level = "MEDIUM" 
            risk_color = "🟡"
        else:
            overall_risk_level = "LOW"
            risk_color = "🟢"
        
        # 주요 위험 요인 집계
        primary_concerns = []
        
        if structura_result.get('top_risk_factors'):
            primary_concerns.extend([f"구조적: {factor}" for factor in structura_result['top_risk_factors'][:2]])
        
        if cognita_result.get('risk_factors'):
            primary_concerns.extend([f"관계적: {factor}" for factor in cognita_result['risk_factors'][:2]])
        
        if chronos_result.get('risk_indicators'):
            primary_concerns.extend([f"시계열: {factor}" for factor in chronos_result['risk_indicators'][:2]])
        
        if sentio_result.get('job_demands_matches'):
            primary_concerns.extend([f"심리적: {factor}" for factor in sentio_result['job_demands_matches'][:2]])
        
        # 기본 레포트 구조 (LLM 없이)
        comprehensive_report = {
            'employee_id': employee_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'comprehensive_assessment': {
                'overall_risk_score': round(comprehensive_risk_score, 3),
                'overall_risk_level': overall_risk_level,
                'risk_indicator': risk_color,
                'confidence_level': 'HIGH' if len(risk_scores) >= 3 else 'MEDIUM'
            },
            'worker_scores': {
                'structura': {
                    'attrition_probability': structura_result.get('attrition_probability', 0),
                    'prediction': structura_result.get('prediction', 'Unknown'),
                    'confidence': structura_result.get('confidence', 0)
                },
                'cognita': {
                    'overall_risk_score': cognita_result.get('overall_risk_score', 0),
                    'risk_category': cognita_result.get('risk_category', 'Unknown'),
                    'network_centrality': cognita_result.get('network_centrality', 0)
                },
                'chronos': {
                    'prediction': chronos_result.get('prediction', 'Unknown'),
                    'probability': chronos_result.get('probability', 0),
                    'trend': chronos_result.get('trend', 'Stable')
                },
                'sentio': {
                    'psychological_risk_score': sentio_result.get('psychological_risk_score', 0),
                    'risk_level': sentio_result.get('risk_level', 'MEDIUM'),
                    'job_demands_score': sentio_result.get('jd_r_indicators', {}).get('job_demands_score', 0),
                    'resources_deficiency_score': sentio_result.get('jd_r_indicators', {}).get('job_resources_deficiency_score', 0)
                }
            },
            'primary_concerns': primary_concerns[:6],  # 상위 6개
            'llm_interpretation': None  # LLM 해석은 선택적으로 추가
        }
        
        return comprehensive_report
    
    def generate_comprehensive_llm_interpretation(self, comprehensive_report: Dict, use_llm: bool = False) -> str:
        """개별 직원의 종합 레포트에 대한 LLM 해석 생성 (선택적)"""
        
        if not use_llm:
            # 규칙 기반 해석 (LLM 없이)
            employee_id = comprehensive_report['employee_id']
            assessment = comprehensive_report['comprehensive_assessment']
            worker_scores = comprehensive_report['worker_scores']
            concerns = comprehensive_report['primary_concerns']
            
            interpretation = f"""
=== 직원 {employee_id} 종합 분석 결과 ===

{assessment['risk_indicator']} 전체 위험도: {assessment['overall_risk_level']} ({assessment['overall_risk_score']:.3f}/1.0)
📊 신뢰도: {assessment['confidence_level']}

🔍 워커별 상세 분석:
"""
            
            # Structura 분석
            structura = worker_scores['structura']
            if structura['attrition_probability'] > 0:
                interpretation += f"📈 구조적 분석 (Structura): 퇴직 확률 {structura['attrition_probability']:.1%}, 예측 '{structura['prediction']}'\n"
            
            # Cognita 분석
            cognita = worker_scores['cognita']
            if cognita['overall_risk_score'] > 0:
                interpretation += f"🌐 관계적 분석 (Cognita): 위험도 {cognita['overall_risk_score']:.3f}, 카테고리 '{cognita['risk_category']}'\n"
            
            # Chronos 분석
            chronos = worker_scores['chronos']
            if chronos['probability'] > 0:
                interpretation += f"⏰ 시계열 분석 (Chronos): 확률 {chronos['probability']:.1%}, 트렌드 '{chronos['trend']}'\n"
            
            # Sentio 분석
            sentio = worker_scores['sentio']
            if sentio['psychological_risk_score'] > 0:
                interpretation += f"🧠 심리적 분석 (Sentio): 위험도 {sentio['psychological_risk_score']:.3f}, 수준 '{sentio['risk_level']}'\n"
                interpretation += f"   - 직무 요구: {sentio['job_demands_score']:.3f}, 자원 결핍: {sentio['resources_deficiency_score']:.3f}\n"
            
            interpretation += f"\n⚠️ 주요 우려사항:\n"
            for i, concern in enumerate(concerns[:5], 1):
                interpretation += f"{i}. {concern}\n"
            
            interpretation += f"\n💡 권장 조치:\n"
            
            # 위험 수준별 권장사항
            if assessment['overall_risk_level'] == 'HIGH':
                interpretation += "🚨 즉시 개입 필요:\n"
                interpretation += "- 상급자와의 긴급 면담 실시\n"
                interpretation += "- 업무 조정 및 지원 방안 검토\n"
                interpretation += "- 정기적 모니터링 체계 구축\n"
            elif assessment['overall_risk_level'] == 'MEDIUM':
                interpretation += "⚠️ 예방적 관리 필요:\n"
                interpretation += "- 정기적 상담 및 피드백 제공\n"
                interpretation += "- 업무 환경 개선 검토\n"
                interpretation += "- 스트레스 관리 프로그램 참여 권장\n"
            else:
                interpretation += "✅ 현재 상태 유지:\n"
                interpretation += "- 정기적 모니터링 지속\n"
                interpretation += "- 긍정적 요소 강화\n"
                interpretation += "- 성장 기회 제공 검토\n"
            
            return interpretation.strip()
        
        else:
            # LLM 기반 해석 (선택적 사용)
            # TODO: OpenAI API 호출로 더 상세한 해석 생성
            return "LLM 기반 개별 직원 상세 해석 (구현 예정)"
    
    def analyze_text_columns(self, text_columns: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """텍스트 컬럼 분석"""
        if self.data is None:
            logger.error("❌ 데이터를 먼저 로드해주세요.")
            return None
        
        # 텍스트 컬럼 찾기 (persona 관련 컬럼 제외)
        if text_columns is None:
            text_columns = []
            # persona 관련 컬럼 제외 목록
            persona_columns_to_exclude = [
                'Persona_Code', 'Persona_Name', 'persona_code', 'persona_name', 
                'persona_type', 'Persona_Type', 'softmax_persona'
            ]
            
            for col in self.data.columns:
                # persona 관련 컬럼은 제외
                if col in persona_columns_to_exclude:
                    continue
                    
                # 텍스트 컬럼만 포함
                if col.endswith('_text') or 'text' in col.lower():
                    text_columns.append(col)
        
        if not text_columns:
            logger.error("❌ 텍스트 컬럼을 찾을 수 없습니다.")
            return None
        
        logger.info(f"📝 발견된 텍스트 컬럼: {text_columns}")
        
        results = {}
        
        for col in text_columns:
            logger.info(f"🔍 '{col}' 컬럼 분석 중...")
            
            # 퇴직자 키워드 추출 (persona 텍스트 제외)
            resigned_keywords = []
            for text in self.resigned_data[col].dropna():
                if not self._is_persona_text(text):
                    resigned_keywords.extend(self.extract_nouns_only(text))
            
            # 재직자 키워드 추출 (persona 텍스트 제외)
            stayed_keywords = []
            for text in self.stayed_data[col].dropna():
                if not self._is_persona_text(text):
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
            
            logger.info(f"  퇴직자 키워드: {len(resigned_keywords)}개 (유니크: {len(resigned_counter)}개)")
            logger.info(f"  재직자 키워드: {len(stayed_keywords)}개 (유니크: {len(stayed_counter)}개)")
        
        return results
    
    def find_distinctive_keywords(self, results: Dict[str, Any], min_frequency: int = 5) -> Optional[Dict[str, Any]]:
        """차별적 키워드 찾기 (개선된 방식)"""
        if not results:
            return None
        
        logger.info(f"🎯 차별적 키워드 분석 (최소 빈도: {min_frequency}회)")
        
        all_distinctive = {}
        
        for col, data in results.items():
            logger.info(f"📊 [{col}] 컬럼 분석 결과:")
            
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
            
            # 결과 저장
            all_distinctive[col] = {
                'resigned_unique': resigned_unique,
                'stayed_unique': stayed_unique
            }
            
            logger.info(f"🔴 퇴직자 특징적 키워드: {len(resigned_unique)}개")
            logger.info(f"🔵 재직자 특징적 키워드: {len(stayed_unique)}개")
        
        return all_distinctive
