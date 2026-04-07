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
  Select,
  Input,
  Divider
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
const { TextArea } = Input;
const { Option } = Select;

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
      const response = await fetch('http://localhost:5007/api/batch-analysis/list-saved-files');
      
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

      const fileResponse = await fetch(`http://localhost:5007/api/batch-analysis/load-file/${latestFile.filename}`);
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
      const response = await fetch('http://localhost:5007/api/results/list-all-employees');
      
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

  // 개별 직원 보고서 생성
  const generateEmployeeReport = async (employee) => {
    try {
      setReportGenerating(true);
      setSelectedEmployee(employee);
      setReportModalVisible(true);

      console.log('📝 직원 보고서 생성 시작:', {
        employee_id: employee.employee_id,
        department: employee.department,
        job_role: employee.job_role,
        position: employee.position
      });

      // Integration 서버에 보고서 생성 요청
      const response = await fetch('http://localhost:5007/api/generate-employee-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          employee_id: employee.employee_id,
          department: employee.department,
          job_role: employee.job_role,
          position: employee.position,
          risk_level: employee.risk_level,
          risk_score: employee.risk_score,
          agent_scores: {
            structura: employee.structura_score,
            chronos: employee.chronos_score,
            cognita: employee.cognita_score,
            sentio: employee.sentio_score,
            agora: employee.agora_score
          }
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setGeneratedReport(result.report);
        console.log('✅ 직원 보고서 생성 완료', {
          has_comprehensive_report: result.has_comprehensive_report,
          visualization_files: result.visualization_files?.length || 0
        });
        
        if (result.has_comprehensive_report) {
          message.success('저장된 종합 보고서를 불러왔습니다.');
        } else {
          message.success('LLM 기반 보고서가 생성되었습니다.');
        }
      } else {
        throw new Error(result.error || '보고서 생성 실패');
      }

    } catch (error) {
      console.error('보고서 생성 실패:', error);
      message.error(`보고서 생성에 실패했습니다: ${error.message}`);
      setGeneratedReport('보고서 생성에 실패했습니다. 다시 시도해주세요.');
    } finally {
      setReportGenerating(false);
    }
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
        title={`직원 보고서 - ${selectedEmployee?.name || selectedEmployee?.employee_id}`}
        open={reportModalVisible}
        onCancel={() => setReportModalVisible(false)}
        width={800}
        footer={[
          <Button key="close" onClick={() => setReportModalVisible(false)}>
            닫기
          </Button>,
          <Button
            key="download"
            type="primary"
            icon={<DownloadOutlined />}
            onClick={() => {
              // 보고서 다운로드 기능 (추후 구현)
              message.info('다운로드 기능은 추후 구현 예정입니다.');
            }}
          >
            다운로드
          </Button>,
        ]}
      >
        {reportGenerating ? (
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <Spin size="large" />
            <div style={{ marginTop: 16 }}>
              <Text>LLM으로 보고서를 생성하는 중...</Text>
            </div>
          </div>
        ) : (
          <div>
            {selectedEmployee && (
              <div style={{ marginBottom: 16, padding: 16, backgroundColor: '#f5f5f5', borderRadius: 6 }}>
                <Row gutter={16}>
                  <Col span={8}>
                    <Text strong>직원 ID:</Text> {selectedEmployee.employee_id}<br />
                    <Text strong>이름:</Text> {selectedEmployee.name}<br />
                    <Text strong>부서:</Text> {selectedEmployee.department}<br />
                    <Text strong>직무:</Text> {selectedEmployee.job_role || '-'}<br />
                    <Text strong>직급:</Text> {selectedEmployee.position || '-'}
                  </Col>
                  <Col span={8}>
                    <Text strong>위험도:</Text> <Tag color={
                      selectedEmployee.risk_level === 'high' ? 'red' : 
                      selectedEmployee.risk_level === 'medium' ? 'orange' : 'green'
                    }>
                      {selectedEmployee.risk_level === 'high' ? '고위험군' : 
                       selectedEmployee.risk_level === 'medium' ? '주의군' : '안전군'}
                    </Tag><br />
                    <Text strong>위험 점수:</Text> {(selectedEmployee.risk_score * 100).toFixed(1)}%<br />
                    <Text strong>Structura:</Text> {(selectedEmployee.structura_score * 100).toFixed(1)}%<br />
                    <Text strong>Chronos:</Text> {(selectedEmployee.chronos_score * 100).toFixed(1)}%
                  </Col>
                  <Col span={8}>
                    <Text strong>Cognita:</Text> {(selectedEmployee.cognita_score * 100).toFixed(1)}%<br />
                    <Text strong>Sentio:</Text> {(selectedEmployee.sentio_score * 100).toFixed(1)}%<br />
                    <Text strong>Agora:</Text> {(selectedEmployee.agora_score * 100).toFixed(1)}%
                  </Col>
                </Row>
              </div>
            )}
            
            <Divider>생성된 보고서</Divider>
            
            <TextArea
              value={generatedReport}
              readOnly
              rows={15}
              style={{ fontSize: 'var(--font-base)', lineHeight: '1.6' }}
              placeholder="보고서가 생성되면 여기에 표시됩니다..."
            />
          </div>
        )}
      </Modal>
    </div>
  );
};

export default ReportGeneration;
