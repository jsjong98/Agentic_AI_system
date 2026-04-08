import React, { useState, useEffect } from 'react';
import {
  Card,
  Button,
  Table,
  message,
  Typography,
  Row,
  Col,
  Statistic,
  Tag,
  Modal,
  Spin,
  Alert,
  Space,
  Select
} from 'antd';
import {
  FileTextOutlined,
  UserOutlined,
  BarChartOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  DownloadOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

const SUPERVISOR_URL = process.env.REACT_APP_SUPERVISOR_URL || 'http://localhost:5006';
const INTEGRATION_URL = process.env.REACT_APP_INTEGRATION_URL || 'http://localhost:5007';

// ── 보고서 생성 헬퍼 ──────────────────────────────────────────────────────────

const AGENT_DESCS = {
  Structura: '정형 데이터 (보상·직무 구조·재직 기간)',
  Cognita:   '관계망 분석 (협업 네트워크·소속감)',
  Chronos:   '시계열 행동 (로그인 패턴·이메일·출퇴근)',
  Sentio:    '자연어 감성 (면담·자기평가 텍스트)',
  Agora:     '외부 시장 (보상 벤치마크·이직 플랫폼)',
};

const AGENT_PERSONA = {
  Structura: { code: 'P04', name: '저평가된 전문가',      desc: '보상 체계 및 직무 구조 불만족이 주요 이탈 요인입니다.' },
  Chronos:   { code: 'P01', name: '번아웃에 직면한 직원', desc: '과도한 업무 부담과 행동 패턴 이상이 주요 이탈 요인입니다.' },
  Cognita:   { code: 'P02', name: '온보딩에 실패한 직원', desc: '관계망 단절 및 조직 내 소외감이 주요 이탈 요인입니다.' },
  Sentio:    { code: 'P01', name: '번아웃에 직면한 직원', desc: '심리적 소진 및 감성적 부정 상태가 주요 이탈 요인입니다.' },
  Agora:     { code: 'P03', name: '성장이 정체된 직원',   desc: '외부 시장 매력도에 의한 이탈 위험이 높습니다.' },
};

const AGENT_INTERPRETATION = {
  Structura: (lv) => `직무 만족도·보상·직급 등 정형 HR 데이터 분석에서 이탈 가능성이 ${lv}으로 측정되었습니다. 동료 대비 보상 격차 또는 직무 설계 문제가 의심됩니다.`,
  Cognita:   (lv) => `조직 내 협업 네트워크·커뮤니케이션 분석에서 관계망 단절 위험이 ${lv}으로 측정되었습니다. 팀 소속감 약화 및 관계 복원 개입이 필요합니다.`,
  Chronos:   (lv) => `로그인 패턴·이메일 빈도·업무 시간 등 행동 데이터에서 이직 준비 징후가 ${lv}으로 탐지되었습니다. 즉각적인 행동 모니터링이 필요합니다.`,
  Sentio:    (lv) => `자기평가·코칭 면담 텍스트에서 부정 감성 지수가 ${lv}으로 측정되었습니다. 심리적 소진 또는 강한 불만족 상태일 가능성이 높습니다.`,
  Agora:     (lv) => `외부 채용 플랫폼 접속·시장 보상 비교 행동이 ${lv}으로 탐지되었습니다. 경쟁사 또는 외부 기회를 적극적으로 탐색 중일 가능성이 높습니다.`,
};

const AGENT_INTERVENTIONS = {
  Structura: [
    '총체적 보상 체계 재검토 및 시장 경쟁력 확보 (동료 대비 벤치마킹)',
    '투명한 승진 기준 및 경력 경로 명확화',
    '직무 범위·업무량 분배 재검토 및 초과근무 구조 개선',
  ],
  Cognita: [
    '관리자 주도 1:1 커뮤니케이션 체계 정기화',
    '프로젝트 페어링을 통한 협업 기회 확대',
    '멘토링 프로그램 연계 및 팀 네트워크 복원 지원',
  ],
  Chronos: [
    '업무 자율성 부여 및 유의미한 업무 재할당',
    '업무량 및 기대 수준 재조정, 초과근무 억제',
    '시의적절한 인정·격려 제공 및 주간 체크인 시작',
  ],
  Sentio: [
    'JD-R(Job Demands-Resources) 모델 기반 관리적 개입',
    'Job Crafting 기법 도입 — 직원 주도 역할 재설계',
    '웰니스 프로그램 지원 및 EAP(Employee Assistance Program) 연계',
  ],
  Agora: [
    "내부 '탤런트 마켓플레이스' 관점의 관리 — 사내 이동 기회 제공",
    '시장 벤치마크 기반 전략적 보상 조정',
    '경쟁력 있는 EVP(Employee Value Proposition) 강화',
  ],
};

const getScoreLevel = (score) =>
  score >= 0.8 ? '매우 높음' : score >= 0.6 ? '높음' : score >= 0.4 ? '중간' : '낮음';

const getScoreBadge = (score) =>
  score >= 0.8 ? '🔴 위험' : score >= 0.6 ? '🟡 주의' : score >= 0.4 ? '🟠 관찰' : '🟢 양호';

const getActionPlan = (riskLevel, topAgent) => {
  const agentNote = `${topAgent} 기반 개입 전략을 최우선 적용`;
  if (riskLevel === 'high') return [
    { phase: '⚡ 즉시 (72시간 이내)', items: [`관리자 + HR BP 공동 면담 일정 즉시 수립`, `개인 보상 패키지 재검토 의뢰`, agentNote] },
    { phase: '📅 단기 (1~2주)',        items: ['주간 1:1 체크인 모니터링 시작', '경력 개발 로드맵 수립 면담 예약'] },
    { phase: '📆 중기 (1~3개월)',      items: ['Retention Interview 실시', '프로젝트 재배치·역할 조정 검토'] },
  ];
  if (riskLevel === 'medium') return [
    { phase: '📋 단기 (1주 이내)',     items: ['Retention Interview 예약', agentNote] },
    { phase: '📅 단기 (2~4주)',        items: ['격주 체크인 모니터링 시작', '성장 기회 및 프로젝트 배치 논의'] },
    { phase: '📆 중기 (2~3개월)',      items: ['역량 강화 프로그램 연계', '팀 네트워크 활성화 지원'] },
  ];
  return [
    { phase: '📋 월간 관리',           items: ['월간 1:1 코칭 면담 정기화', '성장 기회 우선 배정 검토'] },
    { phase: '📆 분기별',              items: ['분기 Pulse Survey 참여 독려', '동료 네트워크 활성화 지원'] },
  ];
};

const buildReport = (employee) => {
  const scores = {
    Structura: employee.structura_score || 0,
    Cognita:   employee.cognita_score   || 0,
    Chronos:   employee.chronos_score   || 0,
    Sentio:    employee.sentio_score    || 0,
    Agora:     employee.agora_score     || 0,
  };
  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const [topAgent, topScore] = sorted[0];
  const [secondAgent, secondScore] = sorted[1];
  const persona = AGENT_PERSONA[topAgent];
  const riskLabelMap = { HIGH: '고위험', MED: '잠재적 위험', MEDIUM: '잠재적 위험', LOW: '안정', high: '고위험', medium: '잠재적 위험', low: '안정' };
  const riskIconMap  = { HIGH: '🔴', MED: '🟡', MEDIUM: '🟡', LOW: '🟢', high: '🔴', medium: '🟡', low: '🟢' };
  const today = new Date().toLocaleDateString('ko-KR', { year: 'numeric', month: 'long', day: 'numeric' });

  return {
    meta: { today, employee, riskLabel: riskLabelMap[employee.risk_level] || employee.risk_level, riskIcon: riskIconMap[employee.risk_level] || '⚪', persona },
    sorted, topAgent, topScore, secondAgent, secondScore,
    agentScores: scores,
    actionPlan: getActionPlan(
      String(employee.risk_level).toLowerCase().replace('medium', 'medium').replace('med', 'medium'),
      topAgent
    ),
  };
};

// ─────────────────────────────────────────────────────────────────────────────

const ReportGeneration = () => {
  const [batchResults, setBatchResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedEmployee, setSelectedEmployee] = useState(null);
  const [reportModalVisible, setReportModalVisible] = useState(false);
  const [generatedReport, setGeneratedReport] = useState('');
  const [reportGenerating, setReportGenerating] = useState(false);
  const [riskFilter, setRiskFilter] = useState('all');
  const [departmentFilter, setDepartmentFilter] = useState('all');

  // 컴포넌트 로드 시 배치 분석 결과 로드
  useEffect(() => {
    loadBatchResults();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // IndexedDB에서 데이터 로드
  const loadFromIndexedDB = async (dbName = 'BatchAnalysisDB', storeName = 'results') => {
    return new Promise((resolve) => {
      const request = indexedDB.open(dbName, 1);
      
      request.onsuccess = function(event) {
        const db = event.target.result;
        
        if (!db.objectStoreNames.contains(storeName)) {
          console.log('IndexedDB: Object Store가 존재하지 않음');
          resolve(null);
          return;
        }
        
        try {
          const transaction = db.transaction([storeName], 'readonly');
          const store = transaction.objectStore(storeName);
          const getAllRequest = store.getAll();
          
          getAllRequest.onsuccess = function() {
            const records = getAllRequest.result;
            if (records && records.length > 0) {
              const latestRecord = records.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0];
              console.log(`✅ IndexedDB에서 데이터 로드: ${latestRecord.total_employees}명`);
              resolve(latestRecord.full_data);
            } else {
              resolve(null);
            }
          };
          
          getAllRequest.onerror = function() {
            console.error('IndexedDB 조회 실패:', getAllRequest.error);
            resolve(null);
          };
        } catch (error) {
          console.error('IndexedDB 트랜잭션 오류:', error);
          resolve(null);
        }
      };
      
      request.onerror = function() {
        console.error('IndexedDB 열기 실패:', request.error);
        resolve(null);
      };
    });
  };

  // 청크 데이터 복원
  const loadFromChunks = async () => {
    try {
      const metadata = localStorage.getItem('batchAnalysisMetadata');
      if (!metadata) {
        console.log('청크 메타데이터 없음');
        return null;
      }

      const meta = JSON.parse(metadata);
      console.log(`🔍 청크 데이터 복원 시도: ${meta.total_chunks}개 청크`);

      const allResults = [];
      for (let i = 0; i < meta.total_chunks; i++) {
        const chunkKey = `batchAnalysisResults_chunk_${i}`;
        const chunkData = localStorage.getItem(chunkKey);
        
        if (chunkData) {
          const chunk = JSON.parse(chunkData);
          allResults.push(...chunk.results);
        } else {
          console.warn(`청크 ${i} 누락`);
        }
      }

      if (allResults.length > 0) {
        console.log(`✅ 청크에서 데이터 복원: ${allResults.length}명`);
        return {
          success: true,
          results: allResults,
          total_employees: allResults.length,
          completed_employees: allResults.length
        };
      }

      return null;
    } catch (error) {
      console.error('청크 복원 실패:', error);
      return null;
    }
  };

  // 서버에서 최근 저장된 파일 로드
  const loadFromServer = async () => {
    try {
      console.log('🌐 서버에서 저장된 파일 조회 중...');
      const response = await fetch(`${INTEGRATION_URL}/api/batch-analysis/list-saved-files`);
      
      if (!response.ok) {
        console.log('서버에서 파일 목록 조회 실패');
        return null;
      }

      const data = await response.json();
      if (!data.success || !data.files || data.files.length === 0) {
        console.log('서버에 저장된 파일 없음');
        return null;
      }

      // 가장 최근 파일 로드
      const latestFile = data.files[0];
      console.log(`📥 최근 파일 로드 시도: ${latestFile.filename}`);

      const fileResponse = await fetch(`${INTEGRATION_URL}/api/batch-analysis/load-file/${latestFile.filename}`);
      if (!fileResponse.ok) {
        console.log('파일 로드 실패');
        return null;
      }

      const fileData = await fileResponse.json();
      
      if (fileData.success && fileData.data) {
        console.log('🔍 서버 데이터 구조 확인:', {
          hasData: !!fileData.data,
          dataKeys: Object.keys(fileData.data),
          hasResults: !!fileData.data.results,
          isArray: Array.isArray(fileData.data),
          totalEmployees: fileData.data.total_employees,
          resultsLength: fileData.data.results?.length,
          firstItemKeys: fileData.data.results?.[0] ? Object.keys(fileData.data.results[0]).slice(0, 5) : 'N/A'
        });
        
        const employeeCount = fileData.data.total_employees || fileData.data.results?.length || 0;
        console.log(`✅ 서버에서 데이터 로드: ${employeeCount}명`);
        
        // 데이터 정규화 - 여러 구조 지원
        let normalizedData = null;
        
        // Case 1: results 배열이 있는 경우
        if (fileData.data.results && Array.isArray(fileData.data.results)) {
          normalizedData = {
            success: true,
            results: fileData.data.results,
            total_employees: fileData.data.total_employees || fileData.data.results.length,
            completed_employees: fileData.data.completed_employees || fileData.data.results.length,
            timestamp: fileData.data.timestamp || new Date().toISOString(),
            source: 'server_file'
          };
          console.log('✅ Case 1: results 배열 구조로 정규화');
        }
        // Case 2: 최상위가 배열인 경우
        else if (Array.isArray(fileData.data)) {
          normalizedData = {
            success: true,
            results: fileData.data,
            total_employees: fileData.data.length,
            completed_employees: fileData.data.length,
            timestamp: new Date().toISOString(),
            source: 'server_file'
          };
          console.log('✅ Case 2: 배열 구조로 정규화');
        }
        // Case 3: individual_results 배열이 있는 경우
        else if (fileData.data.individual_results && Array.isArray(fileData.data.individual_results)) {
          normalizedData = {
            success: true,
            results: fileData.data.individual_results,
            total_employees: fileData.data.individual_results.length,
            completed_employees: fileData.data.individual_results.length,
            timestamp: fileData.data.timestamp || new Date().toISOString(),
            source: 'server_file'
          };
          console.log('✅ Case 3: individual_results 구조로 정규화');
        }
        // Case 4: 다른 구조들 확인
        else {
          console.warn('⚠️ 알 수 없는 데이터 구조. 전체 데이터:', fileData.data);
          
          // 가능한 배열 키 찾기
          const possibleArrayKeys = Object.keys(fileData.data).filter(key => 
            Array.isArray(fileData.data[key]) && fileData.data[key].length > 0
          );
          
          if (possibleArrayKeys.length > 0) {
            const arrayKey = possibleArrayKeys[0];
            console.log(`🔄 발견된 배열 키 사용: ${arrayKey}`);
            normalizedData = {
              success: true,
              results: fileData.data[arrayKey],
              total_employees: fileData.data[arrayKey].length,
              completed_employees: fileData.data[arrayKey].length,
              timestamp: fileData.data.timestamp || new Date().toISOString(),
              source: 'server_file'
            };
          }
        }
        
        if (normalizedData && normalizedData.results && normalizedData.results.length > 0) {
          console.log('✅ 데이터 정규화 성공:', {
            resultsCount: normalizedData.results.length,
            firstEmployee: normalizedData.results[0]?.employee_id || normalizedData.results[0]?.employee_number
          });
          return normalizedData;
        } else {
          console.error('❌ 데이터 정규화 실패 - 유효한 results 배열을 찾을 수 없음');
        }
      }

      return null;
    } catch (error) {
      console.error('서버에서 로드 실패:', error);
      return null;
    }
  };

  // results 폴더에서 직접 직원 목록 로드
  const loadFromResultsFolder = async () => {
    try {
      console.log('📂 results 폴더에서 직원 목록 조회 중...');
      const response = await fetch(`${INTEGRATION_URL}/api/results/list-all-employees`);
      
      if (!response.ok) {
        console.log('results 폴더 조회 실패');
        return null;
      }

      const data = await response.json();
      
      if (data.success && data.results && Array.isArray(data.results)) {
        console.log(`✅ results 폴더에서 ${data.results.length}명의 직원 정보 로드`);
        
        // 데이터 구조 확인
        if (data.results.length > 0) {
          console.log('👤 첫 번째 직원 샘플:', {
            employee_id: data.results[0].employee_id,
            department: data.results[0].department,
            job_role: data.results[0].job_role,
            risk_score: data.results[0].risk_score,
            risk_level: data.results[0].risk_level,
            structura_score: data.results[0].structura_score,
            chronos_score: data.results[0].chronos_score,
            cognita_score: data.results[0].cognita_score,
            sentio_score: data.results[0].sentio_score,
            agora_score: data.results[0].agora_score
          });
        }
        
        return {
          success: true,
          results: data.results,
          total_employees: data.total_employees,
          completed_employees: data.completed_employees,
          timestamp: data.timestamp,
          source: 'results_folder'
        };
      }

      return null;
    } catch (error) {
      console.error('results 폴더 조회 실패:', error);
      return null;
    }
  };

  // 배치 분석 결과 로드 (개선된 버전)
  const loadBatchResults = async () => {
    try {
      setLoading(true);
      
      // 1. results 폴더에서 직접 로드 (최우선) - 항상 최신 데이터!
      console.log('🔄 Step 1: results 폴더에서 comprehensive_report.json 기반 로드...');
      const resultsData = await loadFromResultsFolder();
      if (resultsData && resultsData.results && resultsData.results.length > 0) {
        console.log(`✅ API에서 로드한 위험도 분포:`, {
          high: resultsData.results.filter(r => r.risk_level === 'HIGH').length,
          medium: resultsData.results.filter(r => r.risk_level === 'MEDIUM').length,
          low: resultsData.results.filter(r => r.risk_level === 'LOW').length
        });
        setBatchResults(resultsData);
        message.success(`최신 데이터 로드: ${resultsData.total_employees}명 (comprehensive_report.json 기준)`);
        return;
      }
      
      // 2. localStorage에서 배치 분석 결과 확인
      console.log('🔄 Step 2: localStorage 확인...');
      const savedResults = localStorage.getItem('batchAnalysisResults');
      console.log('🔍 localStorage 확인:', !!savedResults);
      
      if (savedResults) {
        try {
          const results = JSON.parse(savedResults);
          console.log('📊 저장된 데이터 구조:', {
            keys: Object.keys(results),
            storageMethod: results.storage_method,
            dataLocation: results.data_location
          });
          
          // Case 1: 참조 데이터 (IndexedDB 또는 청크 방식)
          if (results.storage_method) {
            console.log(`🔄 참조 데이터 감지: ${results.storage_method}`);
            
            let actualData = null;
            
            // IndexedDB에서 로드
            if (results.storage_method === 'indexeddb') {
              actualData = await loadFromIndexedDB();
            }
            
            // 청크에서 로드
            if (!actualData && results.data_location === 'LocalStorage_Chunks') {
              actualData = await loadFromChunks();
            }
            
            if (actualData) {
              setBatchResults(actualData);
              console.log('✅ 참조 데이터에서 실제 데이터 로드 성공');
              message.success(`배치 분석 결과 로드 완료 (${actualData.total_employees}명)`);
              return;
            }
          }
          
          // Case 2: 직접 저장된 전체 데이터
          else if (results.results && Array.isArray(results.results)) {
            setBatchResults(results);
            console.log('✅ 직접 저장된 데이터 로드:', results.results.length, '명');
            message.success(`배치 분석 결과 로드 완료 (${results.results.length}명)`);
            return;
          }
          
          // Case 3: 배열 형태
          else if (Array.isArray(results)) {
            const normalizedResults = {
              success: true,
              results: results,
              total_employees: results.length,
              completed_employees: results.length
            };
          setBatchResults(normalizedResults);
            console.log('✅ 배열 데이터 로드:', results.length, '명');
            message.success(`배치 분석 결과 로드 완료 (${results.length}명)`);
            return;
          }
          
        } catch (parseError) {
          console.error('JSON 파싱 실패:', parseError);
        }
      }
      
      // 3. IndexedDB에서 직접 시도
      console.log('🔄 Step 3: IndexedDB 직접 확인...');
      const indexedDBData = await loadFromIndexedDB();
      if (indexedDBData) {
        setBatchResults(indexedDBData);
        message.success(`IndexedDB에서 데이터 로드 완료 (${indexedDBData.total_employees}명)`);
        return;
      }
      
      // 4. 청크에서 직접 시도
      console.log('🔄 Step 4: 청크 데이터 직접 확인...');
      const chunkData = await loadFromChunks();
      if (chunkData) {
        setBatchResults(chunkData);
        message.success(`청크에서 데이터 로드 완료 (${chunkData.total_employees}명)`);
        return;
      }
      
      // 5. 서버에서 최근 파일 로드
      console.log('🔄 Step 5: 서버에서 저장된 파일 확인...');
      const serverData = await loadFromServer();
      if (serverData) {
        setBatchResults(serverData);
        message.success(`서버에서 데이터 로드 완료 (${serverData.total_employees}명)`);
        return;
      }
      
      // 6. 모든 시도 실패
      console.log('❌ 모든 소스에서 데이터를 찾을 수 없음');
      message.info('배치 분석 결과가 없습니다. 먼저 배치 분석을 실행해주세요.');
      
    } catch (error) {
      console.error('배치 분석 결과 로드 실패:', error);
      message.error('배치 분석 결과를 불러올 수 없습니다.');
    } finally {
      setLoading(false);
    }
  };

  // 위험도별 직원 분류
  const getEmployeesByRisk = () => {
    if (!batchResults || !batchResults.results) {
      console.log('❌ 배치 결과가 없거나 results 속성이 없음:', batchResults);
      return { high: [], medium: [], low: [] };
    }

    console.log('📊 직원 분류 시작:', {
      totalResults: batchResults.results.length,
      source: batchResults.source,
      firstEmployee: batchResults.results[0]
    });

    const employees = batchResults.results.map((emp, index) => {
      // results 폴더에서 직접 로드한 경우 (이미 정규화된 데이터)
      if (batchResults.source === 'results_folder') {
        // 위험도 레벨 변환
        let riskLevel = 'low';
        const riskLevelMap = {
          'HIGH': 'high',
          'MEDIUM': 'medium',
          'LOW': 'low',
          'UNKNOWN': 'low'
        };
        riskLevel = riskLevelMap[emp.risk_level] || 'low';
        
        return {
          key: emp.employee_id || emp.employee_number || index,
          employee_id: emp.employee_id || emp.employee_number,
          employee_number: emp.employee_number || emp.employee_id,
          name: emp.name || `직원 ${emp.employee_id}`,
          department: emp.department || '미분류',
          job_role: emp.job_role || emp.department,
          position: emp.position,
          risk_score: emp.risk_score || 0,
          risk_level: riskLevel,
          structura_score: emp.structura_score || 0,
          chronos_score: emp.chronos_score || 0,
          cognita_score: emp.cognita_score || 0,
          sentio_score: emp.sentio_score || 0,
          agora_score: emp.agora_score || 0,
          has_comprehensive_report: emp.has_comprehensive_report,
          folder_path: emp.folder_path
        };
      }
      
      // 배치 분석 결과에서 로드한 경우 (기존 로직)
      // 여러 경로에서 위험도 점수 추출 시도
      let riskScore = 0;
      
      // 1. 직접 저장된 risk_score 사용 (배치 분석 결과)
      if (emp.risk_score && emp.risk_score > 0) {
        riskScore = emp.risk_score;
      }
      // 2. combined_analysis 경로
      else if (emp.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score) {
        riskScore = emp.analysis_result.combined_analysis.integrated_assessment.overall_risk_score;
      }
      // 3. 개별 에이전트 점수들로 계산
      else {
        const structuraScore = emp.analysis_result?.structura_result?.prediction?.attrition_probability || 
                              emp.structura_result?.prediction?.attrition_probability || 0;
        const chronosScore = emp.analysis_result?.chronos_result?.prediction?.risk_score || 
                            emp.chronos_result?.prediction?.risk_score || 0;
        const cognitaScore = emp.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score || 
                            emp.cognita_result?.risk_analysis?.overall_risk_score || 0;
        const sentioScore = emp.analysis_result?.sentio_result?.sentiment_analysis?.risk_score || 
                           emp.sentio_result?.sentiment_analysis?.risk_score || 0;
        const agoraScore = emp.analysis_result?.agora_result?.market_analysis?.risk_score || 
                          emp.agora_result?.market_analysis?.risk_score || 0;
        
        const scores = [structuraScore, chronosScore, cognitaScore, sentioScore, agoraScore];
        const validScores = scores.filter(score => score > 0);
        
        if (validScores.length > 0) {
          riskScore = validScores.reduce((sum, score) => sum + score, 0) / validScores.length;
        }
      }
      
      // 부서 정보 추출
      const department = emp.analysis_result?.employee_data?.Department || 
                        emp.employee_data?.Department ||
                        emp.department || 
                        emp.Department || 
                        '미분류';
      
      // 직무 정보 추출
      const jobRole = emp.analysis_result?.employee_data?.JobRole || 
                     emp.employee_data?.JobRole ||
                     emp.job_role ||
                     emp.JobRole || 
                     department; // 기본값으로 부서명 사용
      
      // 직급 정보 추출
      const position = emp.analysis_result?.employee_data?.JobLevel || 
                      emp.employee_data?.JobLevel ||
                      emp.position ||
                      emp.Position ||
                      emp.JobLevel ||
                      null;
      
      // 직원 이름 추출
      const name = emp.analysis_result?.employee_data?.Name || 
                  emp.employee_data?.Name ||
                  emp.name ||
                  emp.Name ||
                  `직원 ${emp.employee_number || emp.employee_id || index + 1}`;
      
      // 사후 분석 최적화 설정 적용
      const finalSettings = JSON.parse(localStorage.getItem('finalRiskSettings') || '{}');
      const highThreshold = finalSettings.risk_thresholds?.high_risk_threshold || 0.7;
      const lowThreshold = finalSettings.risk_thresholds?.low_risk_threshold || 0.3;
      
      let riskLevel = 'low';
      if (riskScore >= highThreshold) riskLevel = 'high';
      else if (riskScore >= lowThreshold) riskLevel = 'medium';

      const employeeData = {
        key: emp.employee_number || emp.employee_id || index,
        employee_id: emp.employee_number || emp.employee_id || index,
        name: name,
        department: department,
        job_role: jobRole,
        position: position,
        risk_score: riskScore,
        risk_level: riskLevel,
        structura_score: emp.analysis_result?.structura_result?.prediction?.attrition_probability || 
                        emp.structura_result?.prediction?.attrition_probability || 0,
        chronos_score: emp.analysis_result?.chronos_result?.prediction?.risk_score || 
                      emp.chronos_result?.prediction?.risk_score || 0,
        cognita_score: emp.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score || 
                      emp.cognita_result?.risk_analysis?.overall_risk_score || 0,
        sentio_score: emp.analysis_result?.sentio_result?.sentiment_analysis?.risk_score || 
                     emp.sentio_result?.sentiment_analysis?.risk_score || 0,
        agora_score: emp.analysis_result?.agora_result?.market_analysis?.risk_score || 
                    emp.agora_result?.market_analysis?.risk_score || 0
      };
      
      if (index < 3) { // 처음 3명만 로그 출력
        console.log(`👤 직원 ${employeeData.employee_id} 데이터:`, employeeData);
      }
      
      return employeeData;
    });

    const result = {
      high: employees.filter(emp => emp.risk_level === 'high'),
      medium: employees.filter(emp => emp.risk_level === 'medium'),
      low: employees.filter(emp => emp.risk_level === 'low')
    };
    
    console.log('📊 위험도별 분류 결과:', {
      high: result.high.length,
      medium: result.medium.length,
      low: result.low.length,
      total: employees.length
    });

    return result;
  };

  // 필터링된 직원 목록
  const getFilteredEmployees = () => {
    const employeesByRisk = getEmployeesByRisk();
    let allEmployees = [...employeesByRisk.high, ...employeesByRisk.medium, ...employeesByRisk.low];

    // 위험도 필터
    if (riskFilter !== 'all') {
      allEmployees = employeesByRisk[riskFilter] || [];
    }

    // 부서 필터
    if (departmentFilter !== 'all') {
      allEmployees = allEmployees.filter(emp => emp.department === departmentFilter);
    }

    return allEmployees;
  };

  // 부서 목록 추출
  const getDepartments = () => {
    if (!batchResults || !batchResults.results) return [];
    const departments = [...new Set(batchResults.results.map(emp => 
      emp.analysis_result?.employee_data?.Department || 
      emp.employee_data?.Department ||
      emp.department || 
      emp.Department || 
      '미분류'
    ))];
    return departments.filter(dept => dept && dept !== '미분류').concat(['미분류']);
  };

  // 개별 직원 맞춤형 보고서 생성 (5개 Agent 데이터 기반)
  const generateEmployeeReport = async (employee) => {
    setSelectedEmployee(employee);
    setReportModalVisible(true);
    setReportGenerating(true);

    // 클라이언트 측 즉시 생성 (Agent 점수 기반)
    setTimeout(() => {
      try {
        const report = buildReport(employee);
        setGeneratedReport(report);
        message.success('맞춤형 보고서가 생성되었습니다.');
      } catch (err) {
        console.error('보고서 생성 오류:', err);
        message.error('보고서 생성에 실패했습니다.');
        setGeneratedReport(null);
      } finally {
        setReportGenerating(false);
      }
    }, 600); // 생성 중 UX 피드백용 딜레이
  };

  // 테이블 컬럼 정의
  const columns = [
    {
      title: '직원 ID',
      dataIndex: 'employee_id',
      key: 'employee_id',
      width: 100,
      fixed: 'left',
    },
    {
      title: '이름',
      dataIndex: 'name',
      key: 'name',
      width: 120,
    },
    {
      title: '부서',
      dataIndex: 'department',
      key: 'department',
      width: 140,
    },
    {
      title: '직무',
      dataIndex: 'job_role',
      key: 'job_role',
      width: 140,
    },
    {
      title: '직급',
      dataIndex: 'position',
      key: 'position',
      width: 80,
      render: (position) => position || '-',
    },
    {
      title: '위험도',
      dataIndex: 'risk_level',
      key: 'risk_level',
      width: 100,
      render: (level) => {
        const config = {
          high: { color: 'red', text: '고위험군' },
          medium: { color: 'orange', text: '주의군' },
          low: { color: 'green', text: '안전군' }
        };
        return <Tag color={config[level]?.color}>{config[level]?.text}</Tag>;
      },
      filters: [
        { text: '고위험군', value: 'high' },
        { text: '주의군', value: 'medium' },
        { text: '안전군', value: 'low' },
      ],
      onFilter: (value, record) => record.risk_level === value,
    },
    {
      title: '위험 점수',
      dataIndex: 'risk_score',
      key: 'risk_score',
      width: 100,
      render: (score) => (score * 100).toFixed(1) + '%',
      sorter: (a, b) => a.risk_score - b.risk_score,
      defaultSortOrder: 'descend',
    },
    {
      title: 'Structura',
      dataIndex: 'structura_score',
      key: 'structura_score',
      width: 90,
      render: (score) => (score * 100).toFixed(1) + '%',
    },
    {
      title: 'Cognita',
      dataIndex: 'cognita_score',
      key: 'cognita_score',
      width: 90,
      render: (score) => (score * 100).toFixed(1) + '%',
    },
    {
      title: 'Chronos',
      dataIndex: 'chronos_score',
      key: 'chronos_score',
      width: 90,
      render: (score) => (score * 100).toFixed(1) + '%',
    },
    {
      title: 'Sentio',
      dataIndex: 'sentio_score',
      key: 'sentio_score',
      width: 90,
      render: (score) => (score * 100).toFixed(1) + '%',
    },
    {
      title: 'Agora',
      dataIndex: 'agora_score',
      key: 'agora_score',
      width: 90,
      render: (score) => (score * 100).toFixed(1) + '%',
    },
    {
      title: '액션',
      key: 'action',
      width: 120,
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <Button
            type="primary"
            icon={<FileTextOutlined />}
            size="small"
            onClick={() => generateEmployeeReport(record)}
          >
            보고서
          </Button>
        </Space>
      ),
    },
  ];

  const employeesByRisk = getEmployeesByRisk();
  const filteredEmployees = getFilteredEmployees();

  if (loading) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>
          <Text>배치 분석 결과를 불러오는 중...</Text>
        </div>
      </div>
    );
  }

  if (!batchResults) {
    return (
      <div style={{ padding: '24px' }}>
        <Alert
          message="배치 분석 결과 없음"
          description="보고서를 생성하려면 먼저 배치 분석을 실행해주세요."
          type="info"
          showIcon
          action={
            <Button size="small" onClick={loadBatchResults}>
              다시 시도
            </Button>
          }
        />
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <FileTextOutlined /> 보고서 출력
      </Title>
      
      <Paragraph>
        배치 분석 결과를 기반으로 개별 직원의 상세 보고서를 생성합니다.
        각 직원의 위험도 분석 결과와 XAI 설명을 포함한 종합 보고서를 제공합니다.
      </Paragraph>

      {/* 통계 요약 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Statistic
            title="총 직원 수"
            value={batchResults.total_employees || 0}
            prefix={<UserOutlined />}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="고위험군"
            value={employeesByRisk.high.length}
            valueStyle={{ color: '#cf1322' }}
            prefix={<ExclamationCircleOutlined />}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="주의군"
            value={employeesByRisk.medium.length}
            valueStyle={{ color: '#fa8c16' }}
            prefix={<ExclamationCircleOutlined />}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="안전군"
            value={employeesByRisk.low.length}
            valueStyle={{ color: '#52c41a' }}
            prefix={<CheckCircleOutlined />}
          />
        </Col>
      </Row>

      {/* 필터 */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row gutter={16}>
          <Col span={8}>
            <Text strong>위험도 필터:</Text>
            <Select
              style={{ width: '100%', marginTop: 8 }}
              value={riskFilter}
              onChange={setRiskFilter}
            >
              <Option value="all">전체</Option>
              <Option value="high">고위험군</Option>
              <Option value="medium">주의군</Option>
              <Option value="low">안전군</Option>
            </Select>
          </Col>
          <Col span={8}>
            <Text strong>부서 필터:</Text>
            <Select
              style={{ width: '100%', marginTop: 8 }}
              value={departmentFilter}
              onChange={setDepartmentFilter}
            >
              <Option value="all">전체 부서</Option>
              {getDepartments().map(dept => (
                <Option key={dept} value={dept}>{dept}</Option>
              ))}
            </Select>
          </Col>
          <Col span={8}>
            <Text strong>필터링된 결과:</Text>
            <div style={{ marginTop: 8 }}>
              <Text>{filteredEmployees.length}명</Text>
            </div>
          </Col>
        </Row>
      </Card>

      {/* 직원 목록 테이블 */}
      <Card title="직원 목록" extra={<BarChartOutlined />}>
        <Table
          columns={columns}
          dataSource={filteredEmployees}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `${range[0]}-${range[1]} / 총 ${total}명`,
          }}
          scroll={{ x: 1500 }}
          size="small"
        />
      </Card>

      {/* 보고서 모달 */}
      <Modal
        title={
          <span>
            <FileTextOutlined style={{ marginRight: 8, color: '#d93954' }} />
            맞춤형 퇴사위험 분석 보고서 — 직원 {selectedEmployee?.employee_id}
          </span>
        }
        open={reportModalVisible}
        onCancel={() => { setReportModalVisible(false); setGeneratedReport(null); }}
        width={860}
        footer={[
          <Button key="close" onClick={() => { setReportModalVisible(false); setGeneratedReport(null); }}>
            닫기
          </Button>,
          <Button
            key="download"
            type="primary"
            icon={<DownloadOutlined />}
            disabled={!generatedReport}
            onClick={() => {
              if (!generatedReport) return;
              const { meta, sorted, topAgent, secondAgent, actionPlan } = generatedReport;
              const lines = [
                '════════════════════════════════════════════════════════',
                '   RETAIN SENTINEL 360 — 맞춤형 퇴사위험 분석 보고서',
                '════════════════════════════════════════════════════════',
                `생성일: ${meta.today}`,
                `직원 ID: ${meta.employee.employee_id}  |  부서: ${meta.employee.department}`,
                `직무: ${meta.employee.job_role || '-'}  |  직급: Level ${meta.employee.position || '-'}`,
                '',
                '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
                '1. 종합 위험 진단',
                '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
                `위험 등급: ${meta.riskIcon} ${meta.riskLabel.toUpperCase()}`,
                `종합 위험 점수: ${(meta.employee.risk_score * 100).toFixed(1)}%`,
                `Persona: [${meta.persona.code}] ${meta.persona.name}`,
                `→ ${meta.persona.desc}`,
                '',
                '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
                '2. 5대 Agent 위험 분석',
                '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
                ...sorted.map(([a, s], i) => `  ${i+1}. ${a}  ${AGENT_DESCS[a]}\n     점수: ${(s*100).toFixed(1)}%  ${getScoreBadge(s)}`),
                '',
                '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
                '3. 주요 위험 요인 해석',
                '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
                `[주요] ${topAgent}: ${AGENT_INTERPRETATION[topAgent](getScoreLevel(sorted[0][1]))}`,
                `[보조] ${secondAgent}: ${AGENT_INTERPRETATION[secondAgent](getScoreLevel(sorted[1][1]))}`,
                '',
                '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
                `4. 맞춤형 개입 전략 (${topAgent} 중심)`,
                '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
                ...AGENT_INTERVENTIONS[topAgent].map(t => `  • ${t}`),
                `\n[보완 — ${secondAgent}]`,
                `  • ${AGENT_INTERVENTIONS[secondAgent][0]}`,
                '',
                '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
                '5. 우선순위 액션 플랜',
                '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━',
                ...actionPlan.flatMap(p => [p.phase, ...p.items.map(i => `  • ${i}`)]),
                '',
                '════════════════════════════════════════════════════════',
                '본 보고서는 Retain Sentinel 360 Agentic AI 시스템이',
                '5개 전문 Worker Agent 분석 결과를 종합하여 생성했습니다.',
                'PwC Consulting HR Tech © 2025',
                '════════════════════════════════════════════════════════',
              ];
              const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `report_${meta.employee.employee_id}_${new Date().toISOString().slice(0,10)}.txt`;
              a.click();
              URL.revokeObjectURL(url);
            }}
          >
            다운로드
          </Button>,
        ]}
      >
        {reportGenerating ? (
          <div style={{ textAlign: 'center', padding: '60px' }}>
            <Spin size="large" />
            <div style={{ marginTop: 16 }}>
              <Text>5개 Agent 분석 데이터로 맞춤형 보고서 생성 중...</Text>
            </div>
          </div>
        ) : generatedReport ? (() => {
          const { meta, sorted, topAgent, secondAgent, actionPlan } = generatedReport;
          const riskColor = { '고위험': '#d93954', '잠재적 위험': '#e8721a', '안정': '#2ea44f' }[meta.riskLabel] || '#888';
          const sec = (title) => (
            <div style={{ fontWeight: 700, fontSize: 13, color: '#d93954', borderBottom: '1px solid #f0f0f0', paddingBottom: 6, marginBottom: 10, marginTop: 18 }}>
              {title}
            </div>
          );
          return (
            <div style={{ fontSize: 13, lineHeight: 1.7, maxHeight: '70vh', overflowY: 'auto' }}>
              {/* 헤더 */}
              <div style={{ background: 'linear-gradient(135deg,#2d2d2d,#1a1a2e)', color: '#fff', borderRadius: 10, padding: '16px 20px', marginBottom: 16 }}>
                <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 4 }}>RETAIN SENTINEL 360 — 맞춤형 퇴사위험 분석 보고서</div>
                <div style={{ fontSize: 11, color: '#aaa' }}>생성일: {meta.today} | 직원 ID: {meta.employee.employee_id}</div>
                <div style={{ display: 'flex', gap: 12, marginTop: 8, flexWrap: 'wrap' }}>
                  {[
                    { label: '부서', val: meta.employee.department },
                    { label: '직무', val: meta.employee.job_role || '-' },
                    { label: '직급', val: `Level ${meta.employee.position || '-'}` },
                  ].map(({ label, val }) => (
                    <span key={label} style={{ fontSize: 11, background: 'rgba(255,255,255,.1)', padding: '2px 10px', borderRadius: 12 }}>
                      {label}: <strong>{val}</strong>
                    </span>
                  ))}
                </div>
              </div>

              {/* 1. 종합 위험 진단 */}
              {sec('1. 종합 위험 진단')}
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 8 }}>
                <div style={{ flex: 1, minWidth: 140, background: '#fef2f2', border: `2px solid ${riskColor}`, borderRadius: 8, padding: '10px 14px', textAlign: 'center' }}>
                  <div style={{ fontSize: 10, color: '#888', marginBottom: 4 }}>위험 등급</div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: riskColor }}>{meta.riskIcon} {meta.riskLabel}</div>
                </div>
                <div style={{ flex: 1, minWidth: 140, background: '#f5f5f5', borderRadius: 8, padding: '10px 14px', textAlign: 'center' }}>
                  <div style={{ fontSize: 10, color: '#888', marginBottom: 4 }}>종합 위험 점수</div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: riskColor }}>{(meta.employee.risk_score * 100).toFixed(1)}%</div>
                </div>
                <div style={{ flex: 2, minWidth: 200, background: '#fde8ec', borderRadius: 8, padding: '10px 14px' }}>
                  <div style={{ fontSize: 10, color: '#888', marginBottom: 4 }}>Persona 분류</div>
                  <div style={{ fontWeight: 700, color: '#d93954' }}>[{meta.persona.code}] {meta.persona.name}</div>
                  <div style={{ fontSize: 11, color: '#555', marginTop: 4 }}>{meta.persona.desc}</div>
                </div>
              </div>

              {/* 2. 5대 Agent 분석 */}
              {sec('2. 5대 Agent 위험 분석')}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {sorted.map(([agent, score], i) => {
                  const badge = getScoreBadge(score);
                  const agentColors = { Structura: '#d93954', Cognita: '#2563eb', Chronos: '#e8721a', Sentio: '#7c3aed', Agora: '#2ea44f' };
                  const c = agentColors[agent];
                  return (
                    <div key={agent} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px', background: i === 0 ? '#fef8f8' : '#fafafa', borderRadius: 8, border: i === 0 ? `1px solid ${c}40` : '1px solid #f0f0f0' }}>
                      <span style={{ width: 18, fontSize: 11, color: '#999', fontWeight: 700 }}>{i + 1}</span>
                      <span style={{ width: 70, fontWeight: 700, fontSize: 12, color: c }}>{agent}</span>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 10, color: '#888' }}>{AGENT_DESCS[agent]}</div>
                        <div style={{ height: 6, background: '#eee', borderRadius: 3, marginTop: 4, overflow: 'hidden' }}>
                          <div style={{ width: `${score * 100}%`, height: '100%', background: c, borderRadius: 3 }} />
                        </div>
                      </div>
                      <span style={{ fontSize: 12, fontWeight: 700, color: c, width: 42, textAlign: 'right' }}>{(score * 100).toFixed(0)}%</span>
                      <span style={{ fontSize: 11, width: 60 }}>{badge}</span>
                    </div>
                  );
                })}
              </div>

              {/* 3. 해석 */}
              {sec('3. 주요 위험 요인 해석')}
              <div style={{ background: '#fde8ec', borderLeft: '4px solid #d93954', borderRadius: 6, padding: '10px 14px', marginBottom: 8 }}>
                <div style={{ fontWeight: 700, fontSize: 12, marginBottom: 4 }}>[주요] {topAgent}</div>
                <div style={{ fontSize: 12, color: '#444' }}>{AGENT_INTERPRETATION[topAgent](getScoreLevel(sorted[0][1]))}</div>
              </div>
              <div style={{ background: '#f5f5f5', borderLeft: '4px solid #aaa', borderRadius: 6, padding: '10px 14px' }}>
                <div style={{ fontWeight: 700, fontSize: 12, marginBottom: 4 }}>[보조] {secondAgent}</div>
                <div style={{ fontSize: 12, color: '#444' }}>{AGENT_INTERPRETATION[secondAgent](getScoreLevel(sorted[1][1]))}</div>
              </div>

              {/* 4. 개입 전략 */}
              {sec(`4. 맞춤형 개입 전략 (${topAgent} 중심)`)}
              <div style={{ marginBottom: 10 }}>
                {AGENT_INTERVENTIONS[topAgent].map((t, i) => (
                  <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 5, fontSize: 12 }}>
                    <span style={{ color: '#d93954', fontWeight: 700, flexShrink: 0 }}>✅</span>
                    <span>{t}</span>
                  </div>
                ))}
              </div>
              <div style={{ background: '#f9f9f9', borderRadius: 6, padding: '8px 12px', marginTop: 4 }}>
                <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>보완 전략 — {secondAgent}</div>
                <div style={{ display: 'flex', gap: 8, fontSize: 12 }}>
                  <span style={{ color: '#aaa', flexShrink: 0 }}>•</span>
                  <span>{AGENT_INTERVENTIONS[secondAgent][0]}</span>
                </div>
              </div>

              {/* 5. 액션 플랜 */}
              {sec('5. 우선순위 액션 플랜')}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {actionPlan.map((p, i) => (
                  <div key={i} style={{ background: '#fafafa', borderRadius: 8, padding: '10px 14px', border: '1px solid #f0f0f0' }}>
                    <div style={{ fontWeight: 700, fontSize: 12, marginBottom: 6 }}>{p.phase}</div>
                    {p.items.map((item, j) => (
                      <div key={j} style={{ display: 'flex', gap: 8, fontSize: 12, marginBottom: 3 }}>
                        <span style={{ color: '#d93954' }}>•</span><span>{item}</span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>

              <div style={{ marginTop: 20, padding: '10px 16px', background: '#f5f5f5', borderRadius: 8, fontSize: 11, color: '#999', textAlign: 'center' }}>
                본 보고서는 Retain Sentinel 360 Agentic AI 시스템이 5개 전문 Worker Agent의 분석 결과를 종합하여 생성했습니다.
                PwC Consulting HR Tech © 2025
              </div>
            </div>
          );
        })() : (
          <div style={{ textAlign: 'center', color: '#888', padding: 40 }}>보고서 데이터가 없습니다.</div>
        )}
      </Modal>
    </div>
  );
};

export default ReportGeneration;
