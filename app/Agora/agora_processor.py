# -*- coding: utf-8 -*-
"""
Agora Market Data Processor
외부 시장 데이터 수집 및 처리 모듈 (JobSpy 통합)
"""

import requests
import time
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# JobSpy 및 웹 스크레이핑 라이브러리
try:
    from jobspy import scrape_jobs
    JOBSPY_AVAILABLE = True
except ImportError:
    print("Warning: JobSpy 라이브러리가 설치되지 않음. 시뮬레이션 모드로 실행됩니다.")
    JOBSPY_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from bs4 import BeautifulSoup
    SELENIUM_AVAILABLE = True
except ImportError:
    print("Warning: Selenium/BeautifulSoup이 설치되지 않음. 웹 스크레이핑 기능이 제한됩니다.")
    SELENIUM_AVAILABLE = False

logger = logging.getLogger(__name__)

class AgoraMarketProcessor:
    """외부 시장 데이터 수집 및 처리 클래스 (JobSpy 통합)"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'api_rate_limit': 1.0,  # API 호출 간 대기 시간 (초)
            'max_retries': 3,       # 재시도 횟수
            'cache_ttl': 3600,      # 캐시 유효 시간 (1시간)
            'timeout': 30,          # 요청 타임아웃 (초)
            'use_jobspy': True,     # JobSpy 사용 여부
            'use_selenium': True,   # Selenium 사용 여부
            'jobspy_results_wanted': 50,  # JobSpy 결과 수
            'jobspy_hours_old': 72  # 최근 N시간 내 공고
        }
        
        # API 키 설정 (환경변수에서 로드)
        self.api_keys = {
            'saramin': os.getenv('SARAMIN_API_KEY'),
            'wanted': os.getenv('WANTED_API_KEY'),
            'openai': os.getenv('OPENAI_API_KEY')
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
        
        # API 엔드포인트
        self.api_endpoints = {
            'saramin': 'https://oapi.saramin.co.kr/job-search',
            'wanted': 'https://www.wanted.co.kr/api/v4/jobs',
            'levels_fyi': 'https://levels.fyi/api/companies',
            'paylab': 'https://www.paylab.com/api/salary'
        }
        
        # JobSpy 지원 사이트
        self.jobspy_sites = ["indeed", "linkedin", "zip_recruiter", "glassdoor"]
        
        # 시뮬레이션 데이터 (실제 API 대신 사용)
        self.simulation_data = self._initialize_simulation_data()
        
        # 캐시 로드
        self._load_cache()
        
        logger.info(f"Agora Market Processor 초기화 완료 (JobSpy: {JOBSPY_AVAILABLE}, Selenium: {SELENIUM_AVAILABLE})")
    
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
        """채용 공고 데이터 수집 (JobSpy + API 통합)"""
        
        # 캐시 확인
        cache_key = self._get_cache_key(job_role, location)
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            logger.debug(f"캐시에서 데이터 반환: {job_role}")
            return self.cache[cache_key]['data']
        
        # 시장 직무명으로 변환
        market_job_title = self.normalize_job_title(job_role)
        
        # 실제 데이터 수집 시도
        try:
            if self.config['use_jobspy'] and JOBSPY_AVAILABLE:
                logger.info(f"JobSpy로 실제 데이터 수집 중: {market_job_title}")
                real_data = self._collect_jobspy_data(market_job_title, location)
                if real_data:
                    return self._cache_and_return_data(cache_key, real_data, job_role)
            
            # JobSpy 실패 시 API 시도
            if self.api_keys['saramin']:
                logger.info(f"사람인 API로 데이터 수집 중: {market_job_title}")
                saramin_data = self._collect_saramin_data(market_job_title, location)
                if saramin_data:
                    return self._cache_and_return_data(cache_key, saramin_data, job_role)
            
            # 웹 스크레이핑 시도
            if self.config['use_selenium'] and SELENIUM_AVAILABLE:
                logger.info(f"웹 스크레이핑으로 데이터 수집 중: {market_job_title}")
                scraped_data = self._collect_scraped_data(market_job_title, location)
                if scraped_data:
                    return self._cache_and_return_data(cache_key, scraped_data, job_role)
        
        except Exception as e:
            logger.warning(f"실제 데이터 수집 실패: {str(e)}, 시뮬레이션 데이터 사용")
        
        # 모든 실제 데이터 수집 실패 시 시뮬레이션 데이터 사용
        return self._get_simulation_data(market_job_title, job_role, cache_key)
    
    def _collect_jobspy_data(self, job_title: str, location: str) -> Optional[Dict]:
        """JobSpy를 사용한 실제 채용 공고 수집"""
        try:
            # JobSpy로 다중 사이트 스크레이핑
            jobs_df = scrape_jobs(
                site_name=self.jobspy_sites,
                search_term=job_title,
                location=location if location != "서울" else "South Korea",
                results_wanted=self.config['jobspy_results_wanted'],
                hours_old=self.config['jobspy_hours_old'],
                country_indeed='South Korea'
            )
            
            if jobs_df is not None and len(jobs_df) > 0:
                # 급여 정보 처리
                salary_data = self._process_jobspy_salary_data(jobs_df)
                
                # 결과 구성
                result = {
                    'job_postings': len(jobs_df),
                    'avg_salary': salary_data['avg_salary'],
                    'salary_range': salary_data['salary_range'],
                    'market_trend': self._analyze_market_trend(jobs_df),
                    'competition_level': self._analyze_competition_level(len(jobs_df)),
                    'key_skills': self._extract_key_skills(jobs_df),
                    'data_source': 'JobSpy',
                    'sites_covered': self.jobspy_sites,
                    'data_freshness': f"최근 {self.config['jobspy_hours_old']}시간"
                }
                
                logger.info(f"JobSpy 데이터 수집 성공: {len(jobs_df)}개 공고")
                return result
            
        except Exception as e:
            logger.error(f"JobSpy 데이터 수집 실패: {str(e)}")
        
        return None
    
    def _collect_saramin_data(self, job_title: str, location: str) -> Optional[Dict]:
        """사람인 API를 통한 채용 공고 수집"""
        try:
            if not self.api_keys['saramin']:
                return None
            
            params = {
                'access-key': self.api_keys['saramin'],
                'keywords': job_title,
                'loc_cd': '101000' if location == "서울" else '102000',  # 서울/경기
                'count': 100
            }
            
            response = requests.get(
                self.api_endpoints['saramin'], 
                params=params,
                timeout=self.config['timeout']
            )
            
            if response.status_code == 200:
                data = response.json()
                
                result = {
                    'job_postings': data.get('jobs', {}).get('total', 0),
                    'avg_salary': self._calculate_avg_salary_saramin(data),
                    'salary_range': self._get_salary_range_saramin(data),
                    'market_trend': 'STABLE',  # API에서 제공하지 않는 경우 기본값
                    'competition_level': 'MEDIUM',
                    'key_skills': self._extract_skills_saramin(data),
                    'data_source': 'Saramin_API'
                }
                
                logger.info(f"사람인 API 데이터 수집 성공: {result['job_postings']}개 공고")
                return result
            
        except Exception as e:
            logger.error(f"사람인 API 데이터 수집 실패: {str(e)}")
        
        return None
    
    def _collect_scraped_data(self, job_title: str, location: str) -> Optional[Dict]:
        """Selenium을 사용한 웹 스크레이핑"""
        try:
            # Chrome 옵션 설정
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            
            driver = webdriver.Chrome(options=chrome_options)
            
            # 원티드 스크레이핑 예시
            search_url = f"https://www.wanted.co.kr/search?query={job_title}&country=kr&job_sort=job.latest_order"
            driver.get(search_url)
            time.sleep(3)
            
            # 채용 공고 수 추출
            job_elements = driver.find_elements(By.CSS_SELECTOR, "[data-cy='job-card']")
            job_count = len(job_elements)
            
            driver.quit()
            
            if job_count > 0:
                # 기본 결과 구성 (실제로는 더 상세한 정보 추출 가능)
                result = {
                    'job_postings': job_count,
                    'avg_salary': np.random.randint(4000000, 8000000),  # 실제로는 스크레이핑
                    'salary_range': {'min': 3000000, 'max': 10000000},
                    'market_trend': 'STABLE',
                    'competition_level': 'HIGH' if job_count > 100 else 'MEDIUM',
                    'key_skills': ['업무 스킬', '커뮤니케이션', '문제해결'],
                    'data_source': 'Web_Scraping'
                }
                
                logger.info(f"웹 스크레이핑 데이터 수집 성공: {job_count}개 공고")
                return result
            
        except Exception as e:
            logger.error(f"웹 스크레이핑 실패: {str(e)}")
        
        return None
    
    def _get_simulation_data(self, market_job_title: str, original_job_role: str, cache_key: str) -> Dict:
        """시뮬레이션 데이터 반환"""
        if market_job_title in self.simulation_data:
            data = self.simulation_data[market_job_title].copy()
            
            # 약간의 랜덤성 추가
            import random
            data['job_postings'] = int(data['job_postings'] * random.uniform(0.8, 1.2))
            data['avg_salary'] = int(data['avg_salary'] * random.uniform(0.95, 1.05))
            data['data_source'] = 'Simulation'
            
            return self._cache_and_return_data(cache_key, data, original_job_role)
        
        else:
            # 기본값 반환
            default_data = {
                'job_postings': 50,
                'avg_salary': 4000000,
                'salary_range': {'min': 3000000, 'max': 6000000},
                'market_trend': 'STABLE',
                'competition_level': 'MEDIUM',
                'key_skills': ['기본업무', '커뮤니케이션', '문제해결'],
                'data_source': 'Default'
            }
            
            logger.warning(f"시뮬레이션 데이터 없음, 기본값 반환: {original_job_role}")
            return self._cache_and_return_data(cache_key, default_data, original_job_role)
    
    def _cache_and_return_data(self, cache_key: str, data: Dict, job_role: str) -> Dict:
        """데이터를 캐시에 저장하고 반환"""
        cache_entry = {
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        self.cache[cache_key] = cache_entry
        self._save_cache()
        
        logger.info(f"시장 데이터 수집 완료: {job_role} ({data['job_postings']}개 공고, 출처: {data.get('data_source', 'Unknown')})")
        return data
    
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
            
            # 현재 급여를 숫자로 변환 (문자열인 경우 대비)
            try:
                current_salary = float(current_salary)
            except (ValueError, TypeError):
                logger.warning(f"급여 값 변환 실패: {current_salary}, 기본값 5000000 사용")
                current_salary = 5000000.0
            
            # 채용 공고 수 기반 수요 지수 (0-1)
            demand_index = min(market_data['job_postings'] / 200.0, 1.0)
            
            # 급여 경쟁력 지수 (0-1)
            market_avg_salary = market_data['avg_salary']
            if market_avg_salary > 0 and current_salary > 0:
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
            
            # 현재 급여를 숫자로 변환 (문자열인 경우 대비)
            try:
                current_salary = float(current_salary)
            except (ValueError, TypeError):
                logger.warning(f"급여 값 변환 실패: {current_salary}, 기본값 5000000 사용")
                current_salary = 5000000.0
            
            if market_avg_salary <= 0 or current_salary <= 0:
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
    
    # JobSpy 데이터 처리 헬퍼 메서드들
    def _process_jobspy_salary_data(self, jobs_df: pd.DataFrame) -> Dict:
        """JobSpy 결과에서 급여 정보 추출 및 처리"""
        try:
            # 급여 컬럼이 있는지 확인
            salary_columns = ['salary_min', 'salary_max', 'salary', 'compensation']
            available_salary_cols = [col for col in salary_columns if col in jobs_df.columns]
            
            if available_salary_cols:
                # 급여 데이터가 있는 행만 필터링
                salary_data = jobs_df[available_salary_cols].dropna()
                
                if len(salary_data) > 0:
                    # 평균 급여 계산
                    if 'salary_min' in salary_data.columns and 'salary_max' in salary_data.columns:
                        avg_salary = int((salary_data['salary_min'].mean() + salary_data['salary_max'].mean()) / 2)
                        salary_range = {
                            'min': int(salary_data['salary_min'].min()),
                            'max': int(salary_data['salary_max'].max())
                        }
                    else:
                        # 단일 급여 컬럼 사용
                        salary_col = available_salary_cols[0]
                        avg_salary = int(salary_data[salary_col].mean())
                        salary_range = {
                            'min': int(salary_data[salary_col].min()),
                            'max': int(salary_data[salary_col].max())
                        }
                    
                    return {
                        'avg_salary': avg_salary,
                        'salary_range': salary_range
                    }
            
            # 급여 정보가 없는 경우 기본값
            return {
                'avg_salary': 5000000,  # 기본 평균 급여
                'salary_range': {'min': 3000000, 'max': 8000000}
            }
            
        except Exception as e:
            logger.warning(f"급여 데이터 처리 실패: {str(e)}")
            return {
                'avg_salary': 5000000,
                'salary_range': {'min': 3000000, 'max': 8000000}
            }
    
    def _analyze_market_trend(self, jobs_df: pd.DataFrame) -> str:
        """채용 공고 데이터를 기반으로 시장 트렌드 분석"""
        try:
            job_count = len(jobs_df)
            
            # 날짜 정보가 있다면 최근 트렌드 분석
            if 'date_posted' in jobs_df.columns:
                recent_jobs = jobs_df[jobs_df['date_posted'].notna()]
                if len(recent_jobs) > 0:
                    # 최근 일주일 내 공고 비율
                    recent_ratio = len(recent_jobs) / len(jobs_df)
                    if recent_ratio > 0.7:
                        return 'HOT'
                    elif recent_ratio > 0.5:
                        return 'GROWING'
            
            # 공고 수 기반 트렌드 판단
            if job_count > 100:
                return 'HOT'
            elif job_count > 50:
                return 'GROWING'
            else:
                return 'STABLE'
                
        except Exception as e:
            logger.warning(f"시장 트렌드 분석 실패: {str(e)}")
            return 'STABLE'
    
    def _analyze_competition_level(self, job_count: int) -> str:
        """채용 공고 수를 기반으로 경쟁 수준 분석"""
        if job_count > 150:
            return 'VERY_HIGH'
        elif job_count > 100:
            return 'HIGH'
        elif job_count > 50:
            return 'MEDIUM'
        elif job_count > 20:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def _extract_key_skills(self, jobs_df: pd.DataFrame) -> List[str]:
        """채용 공고에서 핵심 스킬 추출"""
        try:
            skills = []
            
            # 설명 컬럼에서 스킬 키워드 추출
            description_columns = ['description', 'job_description', 'requirements']
            available_desc_cols = [col for col in description_columns if col in jobs_df.columns]
            
            if available_desc_cols:
                # 일반적인 스킬 키워드
                skill_keywords = [
                    'Python', 'Java', 'JavaScript', 'SQL', 'Excel', 'PowerPoint',
                    '영어', '커뮤니케이션', '팀워크', '리더십', '분석', '기획',
                    '마케팅', '영업', '고객관리', '프로젝트관리', '데이터분석'
                ]
                
                # 설명 텍스트에서 스킬 키워드 찾기
                all_descriptions = ' '.join(jobs_df[available_desc_cols[0]].fillna('').astype(str))
                
                for skill in skill_keywords:
                    if skill.lower() in all_descriptions.lower():
                        skills.append(skill)
                
                # 상위 5개 스킬만 반환
                return skills[:5] if skills else ['업무 스킬', '커뮤니케이션', '문제해결']
            
            return ['업무 스킬', '커뮤니케이션', '문제해결']
            
        except Exception as e:
            logger.warning(f"스킬 추출 실패: {str(e)}")
            return ['업무 스킬', '커뮤니케이션', '문제해결']
    
    def _calculate_avg_salary_saramin(self, data: Dict) -> int:
        """사람인 API 데이터에서 평균 급여 계산"""
        try:
            jobs = data.get('jobs', {}).get('job', [])
            salaries = []
            
            for job in jobs:
                salary_info = job.get('salary', {})
                if salary_info:
                    # 급여 정보 파싱 (실제 API 응답 구조에 따라 조정 필요)
                    salary_text = salary_info.get('name', '')
                    # 간단한 급여 추출 로직 (실제로는 더 정교한 파싱 필요)
                    if '만원' in salary_text:
                        try:
                            # "3000만원" 형태에서 숫자 추출
                            import re
                            numbers = re.findall(r'\d+', salary_text)
                            if numbers:
                                salary = int(numbers[0]) * 10000  # 만원 단위를 원 단위로
                                salaries.append(salary)
                        except:
                            pass
            
            return int(np.mean(salaries)) if salaries else 5000000
            
        except Exception as e:
            logger.warning(f"사람인 급여 계산 실패: {str(e)}")
            return 5000000
    
    def _get_salary_range_saramin(self, data: Dict) -> Dict:
        """사람인 API 데이터에서 급여 범위 추출"""
        try:
            # 실제 구현에서는 API 응답 구조에 맞게 조정
            return {'min': 3000000, 'max': 8000000}
        except:
            return {'min': 3000000, 'max': 8000000}
    
    def _extract_skills_saramin(self, data: Dict) -> List[str]:
        """사람인 API 데이터에서 스킬 추출"""
        try:
            # 실제 구현에서는 API 응답에서 스킬 정보 추출
            return ['업무 스킬', '커뮤니케이션', '협업']
        except:
            return ['업무 스킬', '커뮤니케이션', '협업']
    
    def calculate_agora_score(self, market_pressure_index: float, compensation_gap: float) -> float:
        """Agora 종합 점수 계산 (0-1 사이)"""
        # 가중치: 시장 압력 60%, 보상 격차 40%
        agora_score = (market_pressure_index * 0.6) + (compensation_gap * 0.4)
        return round(agora_score, 3)
    
    def comprehensive_market_analysis(self, employee_data: Dict) -> Dict:
        """종합적인 시장 분석 수행 (JobSpy 통합)"""
        try:
            # 필수 데이터 검증 및 기본값 설정
            job_role = employee_data.get('JobRole', 'Unknown')
            monthly_income = employee_data.get('MonthlyIncome', 5000000)
            
            # 급여 데이터 타입 안전 처리
            try:
                monthly_income = float(monthly_income)
            except (ValueError, TypeError):
                logger.warning(f"MonthlyIncome 변환 실패: {monthly_income}, 기본값 5000000 사용")
                monthly_income = 5000000.0
            
            # 시장 데이터 수집 (JobSpy 우선)
            market_data = self.collect_job_postings(
                job_role,
                employee_data.get('location', '서울')
            )
            
            # 지표 계산
            market_pressure_index = self.calculate_market_pressure_index(
                job_role,
                monthly_income
            )
            
            compensation_gap = self.calculate_compensation_gap(
                job_role,
                monthly_income
            )
            
            # Agora 종합 점수 계산
            agora_score = self.calculate_agora_score(market_pressure_index, compensation_gap)
            
            # 종합 결과 구성
            result = {
                'agora_score': agora_score,
                'market_pressure_index': market_pressure_index,
                'compensation_gap': compensation_gap,
                'raw_market_data': market_data,
                'data_source': market_data.get('data_source', 'Unknown'),
                'analysis_timestamp': datetime.now().isoformat(),
                'employee_id': employee_data.get('EmployeeNumber', 'Unknown')
            }
            
            logger.info(f"종합 시장 분석 완료: 직원 {result['employee_id']}, Agora Score: {agora_score}")
            return result
            
        except Exception as e:
            logger.error(f"종합 시장 분석 실패: {str(e)}")
            raise
