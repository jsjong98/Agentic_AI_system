# -*- coding: utf-8 -*-
"""
Agora Market Data Processor
외부 시장 데이터 수집 및 처리 모듈
"""

import requests
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

class AgoraMarketProcessor:
    """외부 시장 데이터 수집 및 처리 클래스"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'api_rate_limit': 1.0,  # API 호출 간 대기 시간 (초)
            'max_retries': 3,       # 재시도 횟수
            'cache_ttl': 3600,      # 캐시 유효 시간 (1시간)
            'timeout': 30           # 요청 타임아웃 (초)
        }
        
        # 캐시 저장소
        self.cache = {}
        self.cache_file = Path("data/market_cache.json")
        
        # 직무 매핑 (IBM JobRole → 시장 검색용)
        self.job_mapping = {
            'Sales Executive': 'Sales Manager',
            'Research Scientist': 'Data Scientist', 
            'Laboratory Technician': 'Lab Technician',
            'Manufacturing Director': 'Operations Manager',
            'Healthcare Representative': 'Sales Representative',
            'Manager': 'Manager',
            'Sales Representative': 'Sales Representative',
            'Research Director': 'Research Manager',
            'Human Resources': 'HR Specialist'
        }
        
        # 시뮬레이션 데이터 (실제 API 대신 사용)
        self.simulation_data = self._initialize_simulation_data()
        
        # 캐시 로드
        self._load_cache()
        
        logger.info("Agora Market Processor 초기화 완료")
    
    def _initialize_simulation_data(self) -> Dict:
        """시뮬레이션용 시장 데이터 초기화"""
        return {
            'Sales Manager': {
                'job_postings': 187,
                'avg_salary': 5500000,
                'salary_range': {'min': 3500000, 'max': 8000000},
                'market_trend': 'GROWING',
                'competition_level': 'HIGH',
                'key_skills': ['영업관리', '고객관리', '팀리더십', '성과관리']
            },
            'Data Scientist': {
                'job_postings': 234,
                'avg_salary': 6200000,
                'salary_range': {'min': 4000000, 'max': 9500000},
                'market_trend': 'HOT',
                'competition_level': 'VERY_HIGH',
                'key_skills': ['Python', 'Machine Learning', 'SQL', 'Statistics']
            },
            'Lab Technician': {
                'job_postings': 89,
                'avg_salary': 3200000,
                'salary_range': {'min': 2500000, 'max': 4500000},
                'market_trend': 'STABLE',
                'competition_level': 'MEDIUM',
                'key_skills': ['실험기술', '품질관리', '데이터분석', '안전관리']
            },
            'Operations Manager': {
                'job_postings': 156,
                'avg_salary': 5800000,
                'salary_range': {'min': 4200000, 'max': 8500000},
                'market_trend': 'GROWING',
                'competition_level': 'HIGH',
                'key_skills': ['운영관리', '프로세스개선', '팀관리', '비용관리']
            },
            'Sales Representative': {
                'job_postings': 312,
                'avg_salary': 3800000,
                'salary_range': {'min': 2800000, 'max': 5500000},
                'market_trend': 'STABLE',
                'competition_level': 'MEDIUM',
                'key_skills': ['영업', '고객응대', '제품지식', '협상']
            },
            'Research Manager': {
                'job_postings': 67,
                'avg_salary': 7200000,
                'salary_range': {'min': 5500000, 'max': 10000000},
                'market_trend': 'GROWING',
                'competition_level': 'HIGH',
                'key_skills': ['연구기획', '프로젝트관리', '팀리더십', '기술분석']
            },
            'HR Specialist': {
                'job_postings': 143,
                'avg_salary': 4200000,
                'salary_range': {'min': 3200000, 'max': 6000000},
                'market_trend': 'STABLE',
                'competition_level': 'MEDIUM',
                'key_skills': ['인사관리', '채용', '교육기획', '노무관리']
            },
            'Manager': {
                'job_postings': 298,
                'avg_salary': 5200000,
                'salary_range': {'min': 3800000, 'max': 7500000},
                'market_trend': 'STABLE',
                'competition_level': 'MEDIUM',
                'key_skills': ['팀관리', '전략기획', '의사결정', '커뮤니케이션']
            }
        }
    
    def _load_cache(self):
        """캐시 파일 로드"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.debug(f"캐시 로드 완료: {len(self.cache)}개 항목")
        except Exception as e:
            logger.warning(f"캐시 로드 실패: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """캐시 파일 저장"""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.debug("캐시 저장 완료")
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {e}")
    
    def _get_cache_key(self, job_role: str, location: str = "서울") -> str:
        """캐시 키 생성"""
        return f"{job_role}_{location}_{datetime.now().strftime('%Y%m%d')}"
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """캐시 유효성 검사"""
        if 'timestamp' not in cache_entry:
            return False
        
        cache_time = datetime.fromisoformat(cache_entry['timestamp'])
        return (datetime.now() - cache_time).seconds < self.config['cache_ttl']
    
    def normalize_job_title(self, job_role: str) -> str:
        """IBM 직무를 시장 검색용 직무명으로 변환"""
        return self.job_mapping.get(job_role, job_role)
    
    def collect_job_postings(self, job_role: str, location: str = "서울") -> Dict:
        """채용 공고 데이터 수집 (시뮬레이션)"""
        
        # 캐시 확인
        cache_key = self._get_cache_key(job_role, location)
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            logger.debug(f"캐시에서 데이터 반환: {job_role}")
            return self.cache[cache_key]['data']
        
        # 시장 직무명으로 변환
        market_job_title = self.normalize_job_title(job_role)
        
        # 시뮬레이션 데이터 반환
        if market_job_title in self.simulation_data:
            data = self.simulation_data[market_job_title].copy()
            
            # 약간의 랜덤성 추가 (실제 시장 변동 시뮬레이션)
            import random
            data['job_postings'] = int(data['job_postings'] * random.uniform(0.8, 1.2))
            data['avg_salary'] = int(data['avg_salary'] * random.uniform(0.95, 1.05))
            
            # 캐시에 저장
            cache_entry = {
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            self.cache[cache_key] = cache_entry
            self._save_cache()
            
            logger.info(f"시장 데이터 수집 완료: {job_role} ({data['job_postings']}개 공고)")
            return data
        
        else:
            # 기본값 반환
            default_data = {
                'job_postings': 50,
                'avg_salary': 4000000,
                'salary_range': {'min': 3000000, 'max': 6000000},
                'market_trend': 'STABLE',
                'competition_level': 'MEDIUM',
                'key_skills': ['기본업무', '커뮤니케이션', '문제해결']
            }
            
            logger.warning(f"시뮬레이션 데이터 없음, 기본값 반환: {job_role}")
            return default_data
    
    def analyze_job_market(self, job_role: str, location: str = "서울", 
                          experience_level: str = "mid") -> Dict:
        """직무별 시장 분석"""
        
        try:
            # 채용 공고 데이터 수집
            market_data = self.collect_job_postings(job_role, location)
            
            # 경력 수준별 조정
            experience_multiplier = {
                'junior': 0.8,
                'mid': 1.0,
                'senior': 1.3,
                'lead': 1.6
            }
            
            multiplier = experience_multiplier.get(experience_level, 1.0)
            
            # 분석 결과 구성
            analysis = {
                'job_role': job_role,
                'location': location,
                'experience_level': experience_level,
                'total_postings': market_data['job_postings'],
                'average_salary': int(market_data['avg_salary'] * multiplier),
                'salary_range': {
                    'min': int(market_data['salary_range']['min'] * multiplier),
                    'max': int(market_data['salary_range']['max'] * multiplier)
                },
                'market_trend': market_data['market_trend'],
                'competition_level': market_data['competition_level'],
                'key_skills': market_data['key_skills'],
                'market_activity_score': min(market_data['job_postings'] / 100.0, 1.0),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"직무별 시장 분석 실패: {e}")
            raise
    
    def calculate_market_pressure_index(self, job_role: str, current_salary: float) -> float:
        """시장 압력 지수 계산"""
        
        try:
            market_data = self.collect_job_postings(job_role)
            
            # 채용 공고 수 기반 수요 지수 (0-1)
            demand_index = min(market_data['job_postings'] / 200.0, 1.0)
            
            # 급여 경쟁력 지수 (0-1)
            market_avg_salary = market_data['avg_salary']
            if market_avg_salary > 0:
                salary_competitiveness = min(current_salary / market_avg_salary, 1.5) / 1.5
            else:
                salary_competitiveness = 0.5
            
            # 시장 트렌드 가중치
            trend_weights = {
                'HOT': 1.2,
                'GROWING': 1.1,
                'STABLE': 1.0,
                'DECLINING': 0.8,
                'COLD': 0.6
            }
            trend_weight = trend_weights.get(market_data['market_trend'], 1.0)
            
            # 최종 시장 압력 지수 계산
            market_pressure_index = (
                demand_index * 0.4 +           # 수요 40%
                (1 - salary_competitiveness) * 0.4 +  # 급여 격차 40%
                (trend_weight - 1) * 0.2       # 트렌드 20%
            ) * trend_weight
            
            # 0-1 범위로 정규화
            market_pressure_index = max(0.0, min(1.0, market_pressure_index))
            
            logger.debug(f"시장 압력 지수 계산: {job_role} = {market_pressure_index:.3f}")
            return market_pressure_index
            
        except Exception as e:
            logger.error(f"시장 압력 지수 계산 실패: {e}")
            return 0.5  # 기본값
    
    def calculate_compensation_gap(self, job_role: str, current_salary: float) -> float:
        """보상 격차 계산"""
        
        try:
            market_data = self.collect_job_postings(job_role)
            market_avg_salary = market_data['avg_salary']
            
            if market_avg_salary <= 0:
                return 0.0
            
            # 보상 격차 = (시장 평균 - 현재 급여) / 시장 평균
            compensation_gap = (market_avg_salary - current_salary) / market_avg_salary
            
            # -1 ~ 1 범위로 제한 (음수는 시장 대비 높은 급여)
            compensation_gap = max(-1.0, min(1.0, compensation_gap))
            
            logger.debug(f"보상 격차 계산: {job_role} = {compensation_gap:.3f}")
            return compensation_gap
            
        except Exception as e:
            logger.error(f"보상 격차 계산 실패: {e}")
            return 0.0  # 기본값
    
    def generate_market_report(self, job_role: str) -> Dict:
        """직무별 종합 시장 보고서 생성"""
        
        try:
            market_data = self.collect_job_postings(job_role)
            
            # 경쟁 수준 점수화
            competition_scores = {
                'VERY_HIGH': 0.9,
                'HIGH': 0.7,
                'MEDIUM': 0.5,
                'LOW': 0.3,
                'VERY_LOW': 0.1
            }
            
            competition_score = competition_scores.get(market_data['competition_level'], 0.5)
            
            # 시장 활성도 계산
            market_activity = min(market_data['job_postings'] / 150.0, 1.0)
            
            # 종합 보고서 구성
            report = {
                'job_role': job_role,
                'market_summary': {
                    'total_postings': market_data['job_postings'],
                    'average_salary': market_data['avg_salary'],
                    'salary_range': market_data['salary_range'],
                    'market_trend': market_data['market_trend'],
                    'competition_level': market_data['competition_level']
                },
                'market_metrics': {
                    'market_activity_score': round(market_activity, 3),
                    'competition_score': round(competition_score, 3),
                    'demand_level': 'HIGH' if market_activity > 0.7 else 'MEDIUM' if market_activity > 0.4 else 'LOW'
                },
                'key_insights': {
                    'top_skills': market_data['key_skills'],
                    'market_outlook': self._generate_market_outlook(market_data),
                    'recommendations': self._generate_recommendations(market_data)
                },
                'report_timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"시장 보고서 생성 실패: {e}")
            raise
    
    def _generate_market_outlook(self, market_data: Dict) -> str:
        """시장 전망 생성"""
        trend = market_data['market_trend']
        postings = market_data['job_postings']
        
        if trend == 'HOT':
            return f"매우 활발한 채용 시장 ({postings}개 공고). 인재 확보 경쟁이 치열합니다."
        elif trend == 'GROWING':
            return f"성장하는 시장 ({postings}개 공고). 좋은 기회들이 지속적으로 나타나고 있습니다."
        elif trend == 'STABLE':
            return f"안정적인 시장 ({postings}개 공고). 꾸준한 채용 수요가 유지되고 있습니다."
        elif trend == 'DECLINING':
            return f"둔화되는 시장 ({postings}개 공고). 채용 기회가 제한적입니다."
        else:
            return f"침체된 시장 ({postings}개 공고). 채용 활동이 매우 저조합니다."
    
    def _generate_recommendations(self, market_data: Dict) -> List[str]:
        """추천사항 생성"""
        recommendations = []
        
        competition = market_data['competition_level']
        trend = market_data['market_trend']
        
        if competition in ['VERY_HIGH', 'HIGH']:
            recommendations.append("높은 경쟁률로 인해 차별화된 역량 개발이 필요합니다.")
            recommendations.append("네트워킹과 개인 브랜딩에 투자하세요.")
        
        if trend in ['HOT', 'GROWING']:
            recommendations.append("시장이 활발하므로 적극적인 기회 탐색을 권장합니다.")
            recommendations.append("급여 협상에서 유리한 위치에 있습니다.")
        elif trend in ['DECLINING', 'COLD']:
            recommendations.append("현재 포지션 유지를 고려하세요.")
            recommendations.append("추가 스킬 개발로 경쟁력을 강화하세요.")
        
        # 핵심 스킬 기반 추천
        key_skills = market_data.get('key_skills', [])
        if key_skills:
            recommendations.append(f"핵심 스킬 강화 필요: {', '.join(key_skills[:3])}")
        
        return recommendations
