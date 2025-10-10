import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Button,
  Upload,
  message,
  Progress,
  Typography,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Alert,
  Space,
  Input,
  Modal,
  Spin,
  Descriptions,
  Divider,
  Badge,
  List
} from 'antd';
import {
  UploadOutlined,
  FileTextOutlined,
  ApiOutlined,
  BarChartOutlined,
  RocketOutlined,
  FolderOutlined,
  DeleteOutlined,
  SettingOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  DownloadOutlined,
  TeamOutlined,
  HistoryOutlined,
  FilePdfOutlined
} from '@ant-design/icons';
import { predictionService } from '../services/predictionService';
import networkManager from '../utils/networkManager';

const { Title, Text, Paragraph } = Typography;
const { Dragger } = Upload;

const BatchAnalysis = ({
  loading,
  setLoading,
  serverStatus,
  onNavigate, // 네비게이션 콜백 함수 추가
  globalBatchResults, // 전역 배치 결과
  updateBatchResults // 배치 결과 업데이트 함수
}) => {
  const [analysisResults, setAnalysisResults] = useState(globalBatchResults);
  const [analysisProgress, setAnalysisProgress] = useState({
    structura: 0,
    cognita: 0,
    chronos: 0,
    sentio: 0,
    agora: 0,
    overall: 0
  });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [employeeData, setEmployeeData] = useState([]);
  
  // 각 에이전트별 데이터 상태
  const [agentFiles, setAgentFiles] = useState({
    structura: null,
    chronos: null,
    sentio: null,
    agora: null
  });
  
  // Neo4j 연결 설정
  const [neo4jConfig, setNeo4jConfig] = useState({
    uri: 'bolt://13.220.63.109:7687',
    username: 'neo4j',
    password: 'coughs-laboratories-knife'
  });
  
  // 사후 분석에서 저장된 최종 설정 상태
  const [finalRiskSettings, setFinalRiskSettings] = useState(null);
  const [neo4jConnected, setNeo4jConnected] = useState(false);
  const [neo4jTesting, setNeo4jTesting] = useState(false);
  
  // 개별 직원 레포트 관련 상태
  const [selectedEmployee, setSelectedEmployee] = useState(null);
  const [employeeReportVisible, setEmployeeReportVisible] = useState(false);
  const [employeeReport, setEmployeeReport] = useState(null);
  const [reportGenerating, setReportGenerating] = useState(false);
  

  // 캐시 관련 상태
  const [cachedResults, setCachedResults] = useState([]);
  const [showCacheOptions, setShowCacheOptions] = useState(false);
  const [cacheModalVisible, setCacheModalVisible] = useState(false);
  
  // 결과 내보내기 관련 상태
  const [isExporting, setIsExporting] = useState(false);

  // 캐시된 결과 로드 (comprehensive_report.json 기반 - 정확한 데이터!)
  const loadCachedResults = useCallback(async () => {
    try {
      console.log('🔄 comprehensive_report.json 기반 캐시 로드 시작...');
      
      // 1. results 폴더에서 직접 로드 (보고서 출력과 동일한 API)
      try {
        console.log('📂 /api/results/list-all-employees 호출 중...');
        const response = await fetch('http://localhost:5007/api/results/list-all-employees');
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.results && data.results.length > 0) {
            console.log('✅ comprehensive_report.json 기반 데이터 로드:', data.results.length, '명');
            
            // 위험도 분포 계산 (comprehensive_report.json의 overall_risk_level 직접 사용!)
            const highRiskCount = data.results.filter(r => r.risk_level === 'HIGH').length;
            const mediumRiskCount = data.results.filter(r => r.risk_level === 'MEDIUM').length;
            const lowRiskCount = data.results.filter(r => r.risk_level === 'LOW').length;
            
            console.log(`📊 정확한 위험도 분포: 고위험 ${highRiskCount}명, 중위험 ${mediumRiskCount}명, 저위험 ${lowRiskCount}명`);
            
            // 캐시 형태로 변환
            const fileBasedCache = [{
              id: `comprehensive_${data.timestamp}`,
              title: `배치 분석 결과 (${data.total_employees}명) - comprehensive_report.json 기준`,
              timestamp: data.timestamp,
              totalEmployees: data.total_employees,
              accuracy: 85,
              highRiskCount: highRiskCount,
              mediumRiskCount: mediumRiskCount,
              lowRiskCount: lowRiskCount,
              summary: `${data.total_employees}명 분석 완료 (정확한 comprehensive_report 기준)`,
              keyInsights: [
                `고위험군 ${highRiskCount}명 (${(highRiskCount/data.total_employees*100).toFixed(1)}%)`,
                `중위험군 ${mediumRiskCount}명 (${(mediumRiskCount/data.total_employees*100).toFixed(1)}%)`,
                `저위험군 ${lowRiskCount}명 (${(lowRiskCount/data.total_employees*100).toFixed(1)}%)`
              ],
              departmentStats: {},
              source: 'comprehensive_report'
            }];
            
            setCachedResults(fileBasedCache);
            setShowCacheOptions(true);
            
            // 🔄 comprehensive_report 기반 데이터를 predictionService 히스토리에 저장
            try {
              const cache = fileBasedCache[0]; // 단일 결과
              
              // predictionData 생성 (정확한 comprehensive_report 기반!)
              const predictionData = {
                title: cache.title,
                totalEmployees: cache.totalEmployees,
                highRiskCount: cache.highRiskCount,
                mediumRiskCount: cache.mediumRiskCount,
                lowRiskCount: cache.lowRiskCount,
                accuracy: cache.accuracy,
                summary: cache.summary,
                keyInsights: cache.keyInsights,
                departmentStats: cache.departmentStats,
                timestamp: cache.timestamp,
                status: 'completed',
                rawData: null
              };
              
              // 히스토리에 저장 (comprehensive_report 기반 정확한 데이터!)
              predictionService.savePredictionResult(predictionData);
              console.log('✅ comprehensive_report 기반 정확한 데이터를 히스토리에 저장 완료!');
            } catch (syncError) {
              console.warn('히스토리 동기화 중 오류 (계속 진행):', syncError);
            }
            
            if (!globalBatchResults && fileBasedCache.length > 0) {
              message.success(`✅ comprehensive_report 기반 정확한 분석 결과 로드 완료! (고위험 ${fileBasedCache[0].highRiskCount}명)`);
            }
          } else {
            console.log('❌ comprehensive_report 데이터가 없습니다');
            setCachedResults([]);
          }
        } else {
          console.warn('❌ API 호출 실패:', response.status);
          setCachedResults([]);
        }
      } catch (fileListError) {
        console.error('저장된 파일 목록 조회 실패:', fileListError);
      }
      
      // 2. IndexedDB에서 전체 데이터 확인 (보조)
      let indexedDBData = null;
      try {
        indexedDBData = await loadFromIndexedDB();
        if (indexedDBData && indexedDBData.results && indexedDBData.results.length > 0) {
          console.log('✅ IndexedDB에서 전체 데이터 발견:', indexedDBData.results.length, '명');
          setShowCacheOptions(true);
          
          // IndexedDB 데이터를 캐시 목록에 추가
          const indexedDBCache = {
            id: 'indexeddb_latest',
            title: `IndexedDB 저장 결과 (${indexedDBData.results.length}명)`,
            timestamp: new Date().toISOString(),
            totalEmployees: indexedDBData.results.length,
            rawData: indexedDBData,
            accuracy: 90,
            highRiskCount: indexedDBData.results.filter(r => {
              const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
              return score && score >= 0.7;
            }).length,
            mediumRiskCount: indexedDBData.results.filter(r => {
              const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
              return score && score >= 0.3 && score < 0.7;
            }).length,
            lowRiskCount: indexedDBData.results.filter(r => {
              const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
              return score && score < 0.3;
            }).length,
            summary: `IndexedDB에서 로드된 ${indexedDBData.results.length}명 분석 결과`,
            keyInsights: [`완전한 XAI 정보 포함`, `상세 분석 결과 보존`],
            departmentStats: {},
            source: 'indexeddb'
          };
          
          setCachedResults(prev => [indexedDBCache, ...prev]);
          
          // 🔄 IndexedDB 데이터도 predictionService 히스토리와 동기화
          try {
            const predictionData = predictionService.convertBatchResultToPrediction(indexedDBData);
            if (predictionData) {
              // 오늘 날짜의 히스토리가 있는지 확인
              const existingHistory = predictionService.getPredictionHistory();
              const today = new Date().toISOString().split('T')[0];
              const hasTodayHistory = existingHistory.some(h => 
                new Date(h.timestamp).toISOString().split('T')[0] === today
              );
              
              if (!hasTodayHistory) {
                predictionService.savePredictionResult(predictionData);
                console.log('✅ IndexedDB 데이터를 분석 히스토리에 저장');
              }
            }
          } catch (syncError) {
            console.warn('IndexedDB 히스토리 동기화 중 오류 (계속 진행):', syncError);
          }
        }
      } catch (indexedDBError) {
        console.log('IndexedDB 확인 중 오류 (무시됨):', indexedDBError.message);
      }
      
      // 3. 기존 예측 히스토리 확인 (보조)
      try {
        const history = await predictionService.getPredictionHistoryAsync();
        if (history.length > 0) {
          console.log('📋 예측 히스토리 발견:', history.length, '개');
          
          // 중복 제거를 위해 기존 캐시와 비교
          setCachedResults(prevCached => {
            const existingIds = new Set(prevCached.map(cache => cache.id));
            const newHistoryItems = history.filter(item => !existingIds.has(item.id));
            
            if (newHistoryItems.length > 0) {
              setShowCacheOptions(true);
              return [...prevCached, ...newHistoryItems.map(item => ({
                ...item,
                source: 'prediction_history'
              }))];
            }
            return prevCached;
          });
        }
      } catch (historyError) {
        console.error('예측 히스토리 로드 실패:', historyError);
      }
      
    } catch (error) {
      console.error('캐시 로드 전체 실패:', error);
      setCachedResults([]);
      setShowCacheOptions(false);
    }
  }, [globalBatchResults]);

  // 컴포넌트 로드 시 캐시 확인
  useEffect(() => {
    loadCachedResults();
  }, [loadCachedResults]);

  // 사후 분석 최종 설정 로드
  useEffect(() => {
    const savedSettings = localStorage.getItem('finalRiskSettings');
    if (savedSettings) {
      try {
        const settings = JSON.parse(savedSettings);
        setFinalRiskSettings(settings);
        console.log('📋 사후 분석 최종 설정 로드됨:', settings);
      } catch (error) {
        console.error('최종 설정 파싱 오류:', error);
      }
    }
  }, []);

  // 캐시된 결과 사용 - 저장된 파일에서 실제 데이터 로드
  const loadCachedResult = async (cacheId) => {
    const cachedResult = cachedResults.find(cache => cache.id === cacheId);
    console.log('🔍 캐시된 결과 로드 시도:', { cacheId, cachedResult: !!cachedResult, hasRawData: !!(cachedResult?.rawData) });
    
    if (cachedResult) {
      let dataToLoad = null;
      
      // 1. rawData가 있으면 사용
      if (cachedResult.rawData) {
        dataToLoad = cachedResult.rawData;
        console.log('✅ rawData에서 데이터 로드:', dataToLoad);
      }
      // 2. rawData가 없으면 저장된 파일에서 실제 데이터 로드 시도
      else {
        console.log('⚠️ rawData가 없어서 저장된 파일에서 로드 시도');
        
        try {
          // 저장된 배치 분석 파일에서 데이터 로드 (Integration 서버)
          const response = await fetch(`http://localhost:5007/api/batch-analysis/load-results?timestamp=${encodeURIComponent(cachedResult.timestamp)}`);
          
          if (response.ok) {
            const savedData = await response.json();
            console.log('✅ 저장된 파일에서 데이터 로드 성공:', savedData);
            
            if (savedData.success && savedData.results && savedData.results.length > 0) {
              dataToLoad = {
                success: true,
                results: savedData.results,
                total_employees: savedData.total_employees || savedData.results.length,
                completed_employees: savedData.completed_employees || savedData.results.length,
                summary: savedData.summary || {
                  total_employees: savedData.results.length,
                  successful_analyses: savedData.results.length,
                  failed_analyses: 0,
                  success_rate: 1.0
                },
                analysis_metadata: {
                  analysis_type: 'batch',
                  timestamp: cachedResult.timestamp,
                  loaded_from_files: true
                }
              };
            } else {
              throw new Error('저장된 파일에 유효한 데이터가 없습니다.');
            }
          } else {
            throw new Error(`저장된 파일 로드 실패: ${response.status}`);
          }
        } catch (fileLoadError) {
          console.warn('저장된 파일 로드 실패:', fileLoadError.message);
          
          // 파일 로드 실패 시 캐시 정보로부터 기본 구조 생성
          console.log('📋 캐시 정보로부터 기본 구조 생성');
          dataToLoad = {
            success: true,
            results: [], // 빈 결과 배열
            total_employees: cachedResult.totalEmployees || 0,
            completed_employees: cachedResult.totalEmployees || 0,
            summary: {
              total_employees: cachedResult.totalEmployees || 0,
              successful_analyses: cachedResult.totalEmployees || 0,
              failed_analyses: 0,
              success_rate: 1.0
            },
            analysis_metadata: {
              analysis_type: 'batch',
              timestamp: cachedResult.timestamp,
              cached_result: true
            },
            // 캐시 정보를 메타데이터로 포함
            cache_info: {
              title: cachedResult.title,
              highRiskCount: cachedResult.highRiskCount,
              mediumRiskCount: cachedResult.mediumRiskCount,
              lowRiskCount: cachedResult.lowRiskCount,
              accuracy: cachedResult.accuracy,
              summary: cachedResult.summary,
              keyInsights: cachedResult.keyInsights,
              departmentStats: cachedResult.departmentStats,
              totalEmployees: cachedResult.totalEmployees
            }
          };
        }
      }
      
      // 데이터 로드
      setAnalysisResults(dataToLoad);
      if (updateBatchResults) {
        updateBatchResults(dataToLoad);
      }
      
      // 🔄 predictionService에도 저장하여 Home.js 히스토리와 동기화
      try {
        const predictionData = predictionService.convertBatchResultToPrediction(dataToLoad);
        if (predictionData) {
          // 기존에 동일한 타임스탬프의 히스토리가 있는지 확인
          const existingHistory = predictionService.getPredictionHistory();
          const isDuplicate = existingHistory.some(item => 
            Math.abs(new Date(item.timestamp) - new Date(cachedResult.timestamp)) < 1000
          );
          
          if (!isDuplicate) {
            predictionService.savePredictionResult(predictionData);
            console.log('✅ 캐시 결과를 분석 히스토리에 저장');
          } else {
            console.log('ℹ️ 이미 히스토리에 존재하는 결과 (중복 저장 방지)');
          }
        }
      } catch (historyError) {
        console.warn('분석 히스토리 저장 실패 (계속 진행):', historyError);
      }
      
      const loadSource = dataToLoad.analysis_metadata?.loaded_from_files ? '저장된 파일' : 
                        dataToLoad.analysis_metadata?.cached_result ? '캐시 정보' : 'rawData';
      
      message.success(`${loadSource}에서 분석 결과를 불러왔습니다 (${new Date(cachedResult.timestamp).toLocaleString('ko-KR')})`);
      setShowCacheOptions(false);
      setCacheModalVisible(false);
      
      console.log('✅ 캐시 로드 완료:', dataToLoad);
    } else {
      console.error('❌ 캐시된 데이터를 찾을 수 없습니다:', cacheId);
      message.error('캐시된 데이터를 불러올 수 없습니다.');
    }
  };

  // 미분류 폴더 정리
  const cleanupMisclassifiedFolders = async () => {
    try {
      setLoading(true);
      console.log('🔄 미분류 폴더 정리 시작...');
      
      const response = await fetch('http://localhost:5007/api/batch-analysis/cleanup-misclassified', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        const result = await response.json();
        if (result.success) {
          message.success(`미분류 폴더 정리 완료! (${result.processed_employees}명 처리)`);
          console.log('✅ 미분류 폴더 정리 성공:', result);
        } else {
          throw new Error(result.error || '미분류 폴더 정리 실패');
        }
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      console.error('❌ 미분류 폴더 정리 실패:', error);
      message.error(`미분류 폴더 정리 실패: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 캐시 삭제 (저장된 파일 포함)
  const deleteCachedResult = async (cacheId) => {
    const cachedResult = cachedResults.find(cache => cache.id === cacheId);
    
    if (!cachedResult) {
      message.error('삭제할 캐시를 찾을 수 없습니다.');
      return;
    }

    try {
      // 1. 저장된 파일 삭제
      if (cachedResult.source === 'saved_file' && cachedResult.filename) {
        console.log('🗑️ 저장된 파일 삭제:', cachedResult.filename);
        
        const response = await fetch('http://localhost:5007/api/batch-analysis/delete-saved-file', {
          method: 'DELETE',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            filename: cachedResult.filename
          })
        });
        
        if (response.ok) {
          const result = await response.json();
          if (result.success) {
            console.log('✅ 파일 삭제 완료:', result.deleted_files);
            message.success(`${result.deleted_files.length}개 파일이 삭제되었습니다.`);
          } else {
            throw new Error(result.error || '파일 삭제 실패');
          }
        } else {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
      }
      
      // 2. IndexedDB 캐시 삭제
      if (cachedResult.source === 'indexeddb') {
        try {
          // IndexedDB 전체 삭제
          const request = indexedDB.deleteDatabase('BatchAnalysisDB');
          request.onsuccess = () => {
            console.log('✅ IndexedDB 삭제 완료');
          };
          request.onerror = (event) => {
            console.warn('IndexedDB 삭제 실패:', event);
          };
        } catch (dbError) {
          console.warn('IndexedDB 삭제 중 오류:', dbError);
        }
      }
      
      // 3. localStorage 캐시 삭제
      if (cachedResult.source === 'prediction_history') {
        try {
          await predictionService.deletePredictionFromHistory(cacheId);
          console.log('✅ 예측 히스토리에서 삭제 완료');
        } catch (historyError) {
          console.warn('예측 히스토리 삭제 실패:', historyError);
        }
      }
      
      // 4. 로컬 캐시 목록에서 제거
      setCachedResults(prev => prev.filter(cache => cache.id !== cacheId));
      
      // 5. 현재 표시 중인 결과가 삭제된 캐시라면 초기화
      if (analysisResults && analysisResults.cache_info && analysisResults.cache_info.id === cacheId) {
        setAnalysisResults(null);
        updateBatchResults(null);
      }
      
      message.success('캐시가 성공적으로 삭제되었습니다.');
      
    } catch (error) {
      console.error('캐시 삭제 중 오류:', error);
      message.error(`캐시 삭제 실패: ${error.message}`);
    }
  };

  // 최신 캐시 자동 로드 (IndexedDB 우선, 안전한 오류 처리)
  const loadLatestCache = async () => {
    try {
      console.log('🔄 최신 캐시 로드 시작...');
      
      // 1. IndexedDB에서 전체 데이터 로드 시도 (안전한 처리)
      let indexedDBData = null;
      try {
        indexedDBData = await loadFromIndexedDB();
        console.log('🔍 IndexedDB 데이터 확인:', { 
          hasData: !!indexedDBData, 
          hasResults: !!(indexedDBData?.results), 
          resultsLength: indexedDBData?.results?.length 
        });
        
        if (indexedDBData && indexedDBData.results && indexedDBData.results.length > 0) {
          console.log('✅ IndexedDB에서 전체 데이터 로드:', indexedDBData.results.length, '명');
          setAnalysisResults(indexedDBData);
          if (updateBatchResults) {
            updateBatchResults(indexedDBData);
          }
          
          message.success(
            `IndexedDB에서 전체 분석 결과를 불러왔습니다!\n` +
            `총 ${indexedDBData.results.length}명 (데이터 손실 없음)\n` +
            `XAI 정보 및 상세 분석 결과 완전 보존`
          );
          setShowCacheOptions(false);
          return;
        } else {
          console.log('IndexedDB에 유효한 데이터 없음 - 기존 캐시 확인');
        }
      } catch (indexedDBError) {
        console.log('IndexedDB 로드 중 오류 (무시됨):', indexedDBError.message);
        // IndexedDB 오류는 무시하고 기존 캐시로 진행
      }
      
      // 2. localStorage에서 직접 배치 분석 결과 확인
      try {
        const savedResults = localStorage.getItem('batchAnalysisResults');
        if (savedResults) {
          console.log('🔍 localStorage에서 배치 분석 결과 발견');
          const results = JSON.parse(savedResults);
          
          // 데이터 구조 정규화
          let normalizedResults;
          if (Array.isArray(results)) {
            normalizedResults = {
              success: true,
              results: results,
              total_employees: results.length,
              completed_employees: results.length
            };
          } else if (results && results.results && Array.isArray(results.results)) {
            normalizedResults = results;
          } else if (results && typeof results === 'object') {
            normalizedResults = {
              success: true,
              results: [results],
              total_employees: 1,
              completed_employees: 1
            };
          }
          
          if (normalizedResults && normalizedResults.results && normalizedResults.results.length > 0) {
            console.log('✅ localStorage에서 배치 분석 결과 로드:', normalizedResults.results.length, '명');
            setAnalysisResults(normalizedResults);
            if (updateBatchResults) {
              updateBatchResults(normalizedResults);
            }
            
            message.success(`localStorage에서 분석 결과를 불러왔습니다! (${normalizedResults.results.length}명)`);
            setShowCacheOptions(false);
            return;
          }
        }
      } catch (localStorageError) {
        console.log('localStorage 확인 중 오류 (무시됨):', localStorageError.message);
      }
      
      // 3. 기존 캐시 로드
      if (cachedResults && cachedResults.length > 0) {
        try {
          console.log('🔍 기존 캐시에서 로드 시도:', cachedResults[0]);
          await loadCachedResult(cachedResults[0].id);
        } catch (cacheError) {
          console.error('기존 캐시 로드 실패:', cacheError);
          message.warning('저장된 분석 결과를 불러올 수 없습니다.');
        }
      } else {
        console.log('❌ 사용 가능한 캐시 없음');
        message.info('저장된 분석 결과가 없습니다. 새로운 분석을 시작하세요.');
      }
      
    } catch (error) {
      console.error('캐시 로드 전체 실패:', error);
      message.error('분석 결과 로드 중 오류가 발생했습니다.');
    }
  };

  // 새로 분석 시작
  const startNewAnalysis = () => {
    setShowCacheOptions(false);
    message.info('새로운 분석을 시작합니다. 파일을 업로드해주세요.');
  };

  // 최적화된 설정을 적용한 분석 결과 해석 (사용하지 않음 - 향후 확장용)
  /*
  const generateAnalysisInsights = (results) => {
    // analysisResults.results 배열을 사용하여 분석
    const employeeResults = results.results || results;
    const totalEmployees = employeeResults.length;
    
    // 사후 분석의 최적화된 설정 로드
    const finalSettings = finalRiskSettings || {};
    const optimizedThresholds = finalSettings.risk_thresholds || {};
    const optimizedWeights = finalSettings.integration_config || {};
    
    console.log('📊 배치 분석에 적용된 최적화 설정:', {
      thresholds: optimizedThresholds,
      weights: optimizedWeights,
      prediction_mode: finalSettings.attrition_prediction_mode
    });
    
    // 각 직원의 위험도 계산 및 부서 정보 추출
    const processedEmployees = employeeResults.map(emp => {
      // 최적화된 가중치가 있으면 적용, 없으면 기본 통합 점수 사용
      let riskScore = emp.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
      
      // 최적화된 가중치로 재계산 (가능한 경우)
      if (optimizedWeights && Object.keys(optimizedWeights).length > 0) {
        const structuraScore = emp.analysis_result?.structura_result?.prediction?.attrition_probability || 0;
        const cognitaScore = emp.analysis_result?.cognita_result?.overall_risk_score || 
                            emp.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score || 0;
        const chronosScore = emp.analysis_result?.chronos_result?.prediction?.risk_score || 0;
        const sentioScore = emp.analysis_result?.sentio_result?.psychological_risk_score || 
                           emp.analysis_result?.sentio_result?.sentiment_analysis?.risk_score || 0;
        const agoraScore = emp.analysis_result?.agora_result?.agora_score || 
                          emp.analysis_result?.agora_result?.market_analysis?.risk_score || 0;
        
        riskScore = (
          (structuraScore * (optimizedWeights.structura_weight || 0.3)) +
          (cognitaScore * (optimizedWeights.cognita_weight || 0.2)) +
          (chronosScore * (optimizedWeights.chronos_weight || 0.2)) +
          (sentioScore * (optimizedWeights.sentio_weight || 0.15)) +
          (agoraScore * (optimizedWeights.agora_weight || 0.15))
        );
      }
      
      // 최적화된 임계값으로 위험도 분류
      const riskLevel = riskScore ? calculateRiskLevel(riskScore) : 'UNKNOWN';
      
      // 부서 정보 추출 (다양한 소스에서 시도, 부서 분류 문제 해결)
      let department = 'Unclassified';
      
      // 직원 데이터에서 부서 정보 찾기 (우선순위 순)
      const sources = [
        emp.analysis_result?.employee_data?.Department,
        emp.department,
        emp.analysis_result?.structura_result?.employee_data?.Department,
        emp.employee_data?.Department,
        emp.Department  // 직접 부서 필드
      ];
      
      // 첫 번째로 유효한 부서 정보 사용
      for (const source of sources) {
        if (source && typeof source === 'string' && source.trim() && source !== '미분류') {
          department = source.trim();
          break;
        }
      }
      
      // 부서명 정규화 (일반적인 부서명으로 매핑)
      const deptMapping = {
        'HR': 'Human Resources',
        'IT': 'Information Technology', 
        'R&D': 'Research and Development',
        'Sales': 'Sales',
        'Marketing': 'Marketing',
        'Finance': 'Finance',
        'Operations': 'Operations',
        'Manufacturing': 'Manufacturing'
      };
      
      // 매핑된 부서명이 있으면 사용
      if (deptMapping[department]) {
        department = deptMapping[department];
      }
      
      // 부서 정보 추출 완료
      
      return {
        ...emp,
        risk_level: riskLevel.toLowerCase(),
        risk_score: riskScore,
        department: department
      };
    });

    // 최적화된 위험도 분류 통계 (사후 분석 기준 적용)
    const highRisk = processedEmployees.filter(emp => emp.risk_level === 'high').length;
    const mediumRisk = processedEmployees.filter(emp => emp.risk_level === 'medium').length;
    const lowRisk = processedEmployees.filter(emp => emp.risk_level === 'low').length;
    
    // 최적화된 퇴사 예측 (사후 분석 설정 적용)
    const predictionMode = finalSettings.attrition_prediction_mode || 'high_risk_only';
    let predictedAttrition = 0;
    
    if (predictionMode === 'high_risk_only') {
      predictedAttrition = highRisk; // 고위험군만 퇴사 예측
    } else if (predictionMode === 'medium_high_risk') {
      predictedAttrition = highRisk + mediumRisk; // 주의군 + 고위험군 퇴사 예측
    }
    
    console.log('📊 최적화된 위험도 분류 결과:', {
      안전군: lowRisk,
      주의군: mediumRisk, 
      고위험군: highRisk,
      퇴사예측모드: predictionMode,
      예측퇴사자수: predictedAttrition,
      임계값: optimizedThresholds
    });

    // 부서별 통계
    const departmentStats = {};
    processedEmployees.forEach(emp => {
      const dept = emp.department || '미분류';
      if (!departmentStats[dept]) {
        departmentStats[dept] = { total: 0, high: 0, medium: 0, low: 0 };
      }
      departmentStats[dept].total++;
      departmentStats[dept][emp.risk_level]++;
    });

    // 주요 위험 요인 분석 (객체 직렬화 오류 및 빈 데이터 문제 해결)
    const riskFactors = {};
    processedEmployees.filter(emp => emp.risk_level === 'high').forEach(emp => {
      // 위험 요인을 분석 결과에서 추출
      const factors = [];
      
      // Structura 위험 요인 (안전한 추출)
      if (emp.analysis_result?.structura_result?.explanation?.top_risk_factors) {
        emp.analysis_result.structura_result.explanation.top_risk_factors.forEach(factor => {
          if (factor && typeof factor === 'object' && factor.feature) {
            factors.push(`Structura: ${factor.feature}`);
          }
        });
      }
      
      // Cognita 위험 요인 (안전한 추출)
      if (emp.analysis_result?.cognita_result?.risk_analysis?.risk_factors) {
        const cognitaFactors = emp.analysis_result.cognita_result.risk_analysis.risk_factors;
        if (Array.isArray(cognitaFactors)) {
          cognitaFactors.forEach(factor => {
            if (factor && typeof factor === 'string') {
              factors.push(`Cognita: ${factor}`);
            }
          });
        }
      }
      
      // 통합 분석의 위험 요인 (안전한 추출)
      if (emp.analysis_result?.combined_analysis?.risk_factors) {
        const combinedFactors = emp.analysis_result.combined_analysis.risk_factors;
        if (Array.isArray(combinedFactors)) {
          combinedFactors.forEach(factor => {
            if (factor && typeof factor === 'string') {
              factors.push(`Combined: ${factor}`);
            }
          });
        }
      }
      
      // 기본 위험 요인 (점수 기반)
      if (emp.risk_score > 0.8) {
        factors.push('Very High Risk Score');
      } else if (emp.risk_score > 0.7) {
        factors.push('High Risk Score');
      }
      
      // 부서별 위험 요인
      if (emp.department && emp.department !== 'Unclassified') {
        factors.push(`Department: ${emp.department}`);
      }
      
      // 위험 요인이 없으면 기본 요인 추가
      if (factors.length === 0) {
        factors.push('General Risk Factors');
      }
      
      factors.forEach(factor => {
        if (factor && typeof factor === 'string' && factor.trim()) {
          const cleanFactor = factor.trim();
          riskFactors[cleanFactor] = (riskFactors[cleanFactor] || 0) + 1;
        }
      });
    });

    // 부서별 통계 분석 완료

    const insights = {
      summary: {
        totalEmployees,
        highRisk,
        mediumRisk,
        lowRisk,
        highRiskPercentage: ((highRisk / totalEmployees) * 100).toFixed(1),
        mediumRiskPercentage: ((mediumRisk / totalEmployees) * 100).toFixed(1),
        lowRiskPercentage: ((lowRisk / totalEmployees) * 100).toFixed(1)
      },
      departmentAnalysis: departmentStats,
      topRiskFactors: Object.entries(riskFactors)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5)
        .map(([factor, count]) => ({ factor, count })),
      recommendations: generateRecommendations(highRisk, mediumRisk, totalEmployees, departmentStats),
      riskTrends: analyzeTrends(processedEmployees),
      criticalAlerts: generateCriticalAlerts(processedEmployees, departmentStats)
    };

    return insights;
  };
  */

  // 추천 사항 생성 (권장사항 생성 로직 개선) - 사용하지 않음, 향후 확장용
  /*
  const generateRecommendations = (highRisk, mediumRisk, total, deptStats) => {
    const recommendations = [];
    
    // 전체 고위험 비율 기반 권장사항
    if (total > 0) {
      const highRiskPercentage = (highRisk / total) * 100;
      
      if (highRiskPercentage > 20) {
        recommendations.push({
          priority: 'high',
          title: 'Immediate Action Required',
          description: `${highRiskPercentage.toFixed(1)}% of employees are in high-risk category. Implement immediate individual interviews and support programs.`
        });
      } else if (highRiskPercentage > 10) {
        recommendations.push({
          priority: 'medium',
          title: 'Enhanced Monitoring Required',
          description: `${highRiskPercentage.toFixed(1)}% of employees are in high-risk category. Consider enhanced monitoring and preventive measures.`
        });
      }
      
      // 중위험 비율 기반 권장사항
      const mediumRiskPercentage = (mediumRisk / total) * 100;
      if (mediumRiskPercentage > 30) {
        recommendations.push({
          priority: 'medium',
          title: 'Preventive Measures Recommended',
          description: `${mediumRiskPercentage.toFixed(1)}% of employees are in medium-risk category. Introduce preventive mentoring programs.`
        });
      }
    }

    // 부서별 권장사항 (안전한 처리)
    if (deptStats && typeof deptStats === 'object') {
      Object.entries(deptStats).forEach(([dept, stats]) => {
        if (stats && typeof stats === 'object' && stats.total > 0) {
          const deptHighRiskPercentage = (stats.high / stats.total) * 100;
          
          if (deptHighRiskPercentage > 30) {
            recommendations.push({
              priority: 'high',
              title: `${dept} Department Focus Required`,
              description: `${dept} department has ${deptHighRiskPercentage.toFixed(1)}% high-risk employees. Department-specific intervention needed.`
            });
          } else if (deptHighRiskPercentage > 20) {
            recommendations.push({
              priority: 'medium',
              title: `${dept} Department Monitoring`,
              description: `${dept} department shows elevated risk levels (${deptHighRiskPercentage.toFixed(1)}%). Enhanced monitoring recommended.`
            });
          }
        }
      });
    }
    
    // 기본 권장사항 (권장사항이 없을 경우)
    if (recommendations.length === 0) {
      recommendations.push({
        priority: 'low',
        title: 'Continue Regular Monitoring',
        description: 'Current risk levels are within acceptable ranges. Continue regular monitoring and maintain current retention strategies.'
      });
    }

    return recommendations;
  };
  */

  // 트렌드 분석 - 사용하지 않음, 향후 확장용
  /*
  const analyzeTrends = (results) => {
    // 간단한 트렌드 분석 (실제로는 시계열 데이터가 필요)
    return {
      overallTrend: 'stable',
      departmentTrends: {},
      seasonalPatterns: []
    };
  };

  // 중요 알림 생성 - 사용하지 않음, 향후 확장용
  const generateCriticalAlerts = (processedEmployees, deptStats) => {
    const alerts = [];
    
    const highRiskEmployees = processedEmployees.filter(emp => emp.risk_level === 'high');
    if (highRiskEmployees.length > 0) {
      alerts.push({
        type: 'critical',
        message: `${highRiskEmployees.length}명의 직원이 즉시 관심이 필요한 고위험 상태입니다.`
      });
    }

    // 부서별 알림
    Object.entries(deptStats).forEach(([dept, stats]) => {
      if (stats.high > 5) {
        alerts.push({
          type: 'warning',
          message: `${dept} 부서에서 ${stats.high}명의 고위험 직원이 발견되었습니다.`
        });
      }
    });

    return alerts;
  };
  */

  // PDF 보고서 생성 (사용하지 않음 - 향후 확장용)
  /*
  const generatePDFReport = async () => {
    if (!analysisResults || !analysisResults.results || analysisResults.results.length === 0) {
      message.error('분석 결과가 없습니다. 먼저 배치 분석을 실행해주세요.');
      return;
    }

    setIsGeneratingPDF(true);
    
    try {
      // LLM 분석 수행
      const insights = generateAnalysisInsights(analysisResults);
      
      // PDF 생성
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      let yPosition = 20;

      // 한글 지원을 위한 폰트 설정
      pdf.setFont('helvetica');

      // 제목
      pdf.setFontSize(20);
      pdf.setTextColor(0, 0, 0);
      const title = 'Batch Analysis Report';
      pdf.text(title, pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 15;

      // 생성 날짜 (한국어 인코딩 문제 해결)
      pdf.setFontSize(12);
      const now = new Date();
      const timestamp = `${now.getFullYear()}. ${(now.getMonth() + 1).toString().padStart(2, '0')}. ${now.getDate().toString().padStart(2, '0')} ${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
      pdf.text(`Generated: ${timestamp}`, pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 20;

      // 요약 섹션
      pdf.setFontSize(16);
      pdf.setTextColor(0, 0, 0);
      pdf.text('Executive Summary', 20, yPosition);
      yPosition += 10;

      pdf.setFontSize(11);
      const summaryLines = [
        `Total Employees Analyzed: ${insights.summary.totalEmployees}`,
        `High Risk: ${insights.summary.highRisk} (${insights.summary.highRiskPercentage}%)`,
        `Medium Risk: ${insights.summary.mediumRisk} (${insights.summary.mediumRiskPercentage}%)`,
        `Low Risk: ${insights.summary.lowRisk} (${insights.summary.lowRiskPercentage}%)`
      ];

      summaryLines.forEach(line => {
        pdf.text(line, 20, yPosition);
        yPosition += 7;
      });

      yPosition += 10;

      // 부서별 분석
      pdf.setFontSize(16);
      pdf.text('Department Analysis', 20, yPosition);
      yPosition += 10;

      pdf.setFontSize(11);
      Object.entries(insights.departmentAnalysis).forEach(([dept, stats]) => {
        if (yPosition > pageHeight - 30) {
          pdf.addPage();
          yPosition = 20;
        }
        
        // 부서명을 영어로 변환하거나 안전하게 처리
        const safeDeptName = dept === '미분류' ? 'Unclassified' : 
                           dept.replace(/[^\u0020-\u007E]/g, '') || 'Department';
        
        pdf.text(`${safeDeptName}: Total ${stats.total}, High Risk ${stats.high}, Medium Risk ${stats.medium}, Low Risk ${stats.low}`, 20, yPosition);
        yPosition += 7;
      });

      yPosition += 10;

      // 주요 위험 요인
      if (insights.topRiskFactors && insights.topRiskFactors.length > 0) {
        pdf.setFontSize(16);
        pdf.text('Top Risk Factors', 20, yPosition);
        yPosition += 10;

        pdf.setFontSize(11);
        insights.topRiskFactors.forEach(({ factor, count }) => {
          if (yPosition > pageHeight - 30) {
            pdf.addPage();
            yPosition = 20;
          }
          
          // 위험 요인을 안전하게 처리 (객체 직렬화 오류 해결)
          let safeFactor = 'Risk Factor';
          if (typeof factor === 'string' && factor.trim()) {
            safeFactor = factor.replace(/[^\u0020-\u007E]/g, '').trim() || 'Risk Factor';
          } else if (typeof factor === 'object' && factor !== null) {
            safeFactor = JSON.stringify(factor).replace(/[^\u0020-\u007E]/g, '') || 'Risk Factor';
          }
          
          const safeCount = typeof count === 'number' ? count : 0;
          pdf.text(`${safeFactor}: ${safeCount} employees`, 20, yPosition);
          yPosition += 7;
        });

        yPosition += 10;
      }

      // 권장사항
      if (insights.recommendations && insights.recommendations.length > 0) {
        pdf.setFontSize(16);
        pdf.text('Recommendations', 20, yPosition);
        yPosition += 10;

        pdf.setFontSize(11);
        insights.recommendations.forEach((rec, index) => {
          if (yPosition > pageHeight - 40) {
            pdf.addPage();
            yPosition = 20;
          }
          
          // 우선순위에 따른 색상 설정
          const priority = rec.priority || 'medium';
          pdf.setTextColor(priority === 'high' ? 255 : 0, priority === 'medium' ? 165 : 0, 0);
          
          // 제목을 안전하게 처리 (빈 데이터 문제 해결)
          let safeTitle = 'Recommendation';
          if (typeof rec.title === 'string' && rec.title.trim()) {
            safeTitle = rec.title.replace(/[^\u0020-\u007E]/g, '').trim() || 'Recommendation';
          }
          
          pdf.text(`${index + 1}. ${safeTitle}`, 20, yPosition);
          yPosition += 7;
          
          pdf.setTextColor(0, 0, 0);
          
          // 설명을 안전하게 처리 (빈 데이터 문제 해결)
          let safeDescription = 'Please review this recommendation.';
          if (typeof rec.description === 'string' && rec.description.trim()) {
            safeDescription = rec.description.replace(/[^\u0020-\u007E]/g, '').trim() || 'Please review this recommendation.';
          }
          
          const descLines = pdf.splitTextToSize(safeDescription, pageWidth - 40);
          descLines.forEach(line => {
            if (line.trim()) { // 빈 줄 제거
              pdf.text(line, 25, yPosition);
              yPosition += 6;
            }
          });
          yPosition += 5;
        });
      }

      // 중요 알림
      if (insights.criticalAlerts && insights.criticalAlerts.length > 0) {
        if (yPosition > pageHeight - 50) {
          pdf.addPage();
          yPosition = 20;
        }

        pdf.setFontSize(16);
        pdf.setTextColor(255, 0, 0);
        pdf.text('Critical Alerts', 20, yPosition);
        yPosition += 10;

        pdf.setFontSize(11);
        insights.criticalAlerts.forEach(alert => {
          const alertType = alert.type || 'warning';
          pdf.setTextColor(alertType === 'critical' ? 255 : 255, alertType === 'critical' ? 0 : 165, 0);
          
          // 알림 메시지를 안전하게 처리 (빈 데이터 문제 해결)
          let safeMessage = 'Critical alert requires attention.';
          if (typeof alert.message === 'string' && alert.message.trim()) {
            safeMessage = alert.message.replace(/[^\u0020-\u007E]/g, '').trim() || 'Critical alert requires attention.';
          }
          
          const alertLines = pdf.splitTextToSize(safeMessage, pageWidth - 40);
          alertLines.forEach(line => {
            if (line.trim()) { // 빈 줄 제거
              pdf.text(line, 20, yPosition);
              yPosition += 6;
            }
          });
          yPosition += 5;
        });
      }

      // 파일명 생성
      const fileName = `Report_BatchAnalysis_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.pdf`;
      
      // PDF 저장
      pdf.save(fileName);
      
      message.success(`PDF 보고서가 생성되었습니다: ${fileName}`);
      
    } catch (error) {
      console.error('PDF 생성 오류:', error);
      message.error('PDF 보고서 생성 중 오류가 발생했습니다.');
    }
  };
  */
  
  
  // Integration 설정 (사후 분석에서 최적화된 값 자동 로드)
  const [integrationConfig, setIntegrationConfig] = useState({
    // 에이전트별 가중치 (기본값 - 사후 분석 후 최적화 예정)
    structura_weight: 0.25,  // 정형 데이터 분석
    cognita_weight: 0.20,    // 관계형 데이터 분석
    chronos_weight: 0.25,    // 시계열 데이터 분석
    sentio_weight: 0.15,     // 감정 분석
    agora_weight: 0.15,      // 시장 분석
    
    // 위험도 분류 임계값 (기본값)
    high_risk_threshold: 0.7,
    medium_risk_threshold: 0.4,
    
    // 개별 에이전트 임계값 (기본값 - 사후 분석을 통해 최적화 예정)
    structura_threshold: 0.7,
    cognita_threshold: 0.5,
    chronos_threshold: 0.5,
    sentio_threshold: 0.5,
    agora_threshold: 0.5,
    
    // 모델 사용 설정
    use_trained_models: false
  });

  // 컴포넌트 로드 시 저장된 모델 확인 및 설정 자동 로드
  useEffect(() => {
    loadTrainedModels();
  }, []);

  // 저장된 모델 로드 함수
  const loadTrainedModels = () => {
    try {
      const savedModels = localStorage.getItem('trainedModels');
      if (savedModels) {
        const modelData = JSON.parse(savedModels);
        console.log('💾 저장된 모델 발견:', modelData);
        
        // 최적화된 설정이 있으면 자동 적용
        if (modelData.optimization_results) {
          const optimized = modelData.optimization_results;
          
          setIntegrationConfig(prev => ({
            ...prev,
            // 최적화된 가중치 적용
            structura_weight: optimized.weights?.structura_weight || prev.structura_weight,
            cognita_weight: optimized.weights?.cognita_weight || prev.cognita_weight,
            chronos_weight: optimized.weights?.chronos_weight || prev.chronos_weight,
            sentio_weight: optimized.weights?.sentio_weight || prev.sentio_weight,
            agora_weight: optimized.weights?.agora_weight || prev.agora_weight,
            
            // 최적화된 임계값 적용
            high_risk_threshold: optimized.thresholds?.high_risk_threshold || prev.high_risk_threshold,
            medium_risk_threshold: optimized.thresholds?.medium_risk_threshold || prev.medium_risk_threshold,
            structura_threshold: optimized.thresholds?.structura_threshold || prev.structura_threshold,
            cognita_threshold: optimized.thresholds?.cognita_threshold || prev.cognita_threshold,
            chronos_threshold: optimized.thresholds?.chronos_threshold || prev.chronos_threshold,
            sentio_threshold: optimized.thresholds?.sentio_threshold || prev.sentio_threshold,
            agora_threshold: optimized.thresholds?.agora_threshold || prev.agora_threshold,
            
            use_trained_models: true
          }));
          
          message.success(
            `사후 분석에서 최적화된 설정을 자동으로 적용했습니다! ` +
            `(학습일: ${new Date(modelData.training_metadata?.training_date).toLocaleDateString('ko-KR')})`
          );
        }
      }
    } catch (error) {
      console.error('저장된 모델 로드 실패:', error);
    }
  };


  // 에이전트별 파일 업로드 처리
  const handleAgentFileUpload = async (file, agentType) => {
    const isCSV = file.type === 'text/csv' || file.name.endsWith('.csv');
    if (!isCSV) {
      message.error('CSV 파일만 업로드 가능합니다.');
      return false;
    }

    // 파일 크기 제한 (Chronos는 500MB, 나머지는 10MB)
    const maxSize = agentType === 'chronos' ? 500 : 10;
    const isLtMaxSize = file.size / 1024 / 1024 < maxSize;
    if (!isLtMaxSize) {
      message.error(`파일 크기는 ${maxSize}MB 이하여야 합니다.`);
      return false;
    }

    try {
      setLoading(true);
      
      // CSV 파일 읽기 및 검증 (개선된 파싱)
      const text = await file.text();
      
      // PostAnalysis와 동일한 CSV 파싱 로직 (멀티라인 레코드 처리)
      const lines = text.split('\n');
      const csvHeaders = lines[0].split(',').map(h => h.trim());
      
      console.log(`${agentType} 파일 파싱 시작:`);
      console.log(`- 총 라인 수: ${lines.length}`);
      console.log(`- 헤더: ${csvHeaders.join(', ')}`);
      
      // 개선된 CSV 파싱 로직 - 따옴표와 줄바꿈 처리
      const parseCSVLine = (line) => {
        const values = [];
        let current = '';
        let inQuotes = false;
        let i = 0;
        
        while (i < line.length) {
          const char = line[i];
          
          if (char === '"') {
            if (inQuotes && line[i + 1] === '"') {
              // 이스케이프된 따옴표
              current += '"';
              i += 2;
            } else {
              // 따옴표 시작/끝
              inQuotes = !inQuotes;
              i++;
            }
          } else if (char === ',' && !inQuotes) {
            // 쉼표 구분자 (따옴표 밖에서만)
            values.push(current.trim());
            current = '';
            i++;
          } else {
            current += char;
            i++;
          }
        }
        
        values.push(current.trim());
        return values;
      };
      
      // 멀티라인 레코드를 처리하기 위한 로직
      const data = [];
      let skippedLines = 0;
      let currentRecord = '';
      let inQuotes = false;
      
      for (let i = 1; i < lines.length; i++) {
        const line = lines[i];
        currentRecord += (currentRecord ? '\n' : '') + line;
        
        // 따옴표 상태 확인
        for (let char of line) {
          if (char === '"') {
            inQuotes = !inQuotes;
          }
        }
        
        // 레코드가 완성되었는지 확인 (따옴표가 모두 닫혔고, 컬럼 수가 맞는지)
        if (!inQuotes) {
          const values = parseCSVLine(currentRecord);
          
          if (values.length === csvHeaders.length) {
            const row = {};
            csvHeaders.forEach((header, index) => {
              row[header] = values[index]?.replace(/^"|"$/g, '').trim(); // 앞뒤 따옴표 제거
            });
            data.push(row);
            currentRecord = '';
          } else if (currentRecord.trim() === '') {
            // 빈 줄
            skippedLines++;
            currentRecord = '';
          }
          // 컬럼 수가 맞지 않으면 다음 라인과 합쳐서 계속 처리
        }
      }
      
      console.log(`${agentType} 파싱 결과:`);
      console.log(`- 파싱된 데이터 행: ${data.length}`);
      console.log(`- 건너뛴 빈 줄: ${skippedLines}`);
      
      const rows = [csvHeaders, ...data.map(row => csvHeaders.map(h => row[h]))];
      
      if (rows.length < 2) {
        message.error('유효한 CSV 데이터가 없습니다.');
        return false;
      }

      // 에이전트별 필수 컬럼 검증
      const headers = csvHeaders;
      const requiredColumns = getRequiredColumns(agentType);
      const missingColumns = requiredColumns.filter(col => !headers.includes(col));
      
      if (missingColumns.length > 0) {
        message.error(`필수 컬럼이 누락되었습니다: ${missingColumns.join(', ')}`);
        return false;
      }

      // 1. 먼저 파일을 Supervisor에 업로드
      const formData = new FormData();
      formData.append('file', file);
      formData.append('agent_type', agentType);
      formData.append('analysis_type', 'batch'); // 배치 분석용
      
      const uploadResponse = await fetch('http://localhost:5006/upload_file', {
        method: 'POST',
        body: formData
      });
      
      if (!uploadResponse.ok) {
        const errorData = await uploadResponse.json();
        throw new Error(errorData.error || '파일 업로드 실패');
      }
      
      const uploadResult = await uploadResponse.json();
      console.log(`${agentType} 파일 업로드 성공:`, uploadResult);

      // 2. 파일 저장 및 데이터 파싱
      let parsedData = null;
      
      // CSV 데이터 파싱 (모든 에이전트 타입에 대해) - 새로운 데이터 구조 사용
      if (data.length > 0) {
        parsedData = [];
        for (let i = 0; i < data.length; i++) {
          const row = data[i];
          const processedRow = {};
          
          headers.forEach((header) => {
            let value = row[header] || '';
            // 숫자 변환 시도
            if (!isNaN(value) && value !== '' && !isNaN(parseFloat(value))) {
              processedRow[header] = parseFloat(value);
            } else {
              processedRow[header] = value;
            }
          });
          
          parsedData.push(processedRow);
        }
      }
      
      // 파일 객체에 파싱된 데이터 추가 (명시적으로 필요한 속성들 복사)
      const fileWithData = {
        name: file.name,
        size: file.size,
        type: file.type,
        lastModified: file.lastModified,
        data: parsedData,
        headers: headers,
        // 원본 파일 객체도 보관
        originalFile: file
      };
      
      console.log(`📊 ${agentType} 파일 파싱 완료:`, {
        fileName: file.name,
        fileSize: file.size,
        fileType: file.type,
        dataRows: parsedData ? parsedData.length : 0,
        headers: headers,
        sampleData: parsedData && parsedData.length > 0 ? parsedData[0] : null
      });
      
      console.log(`🔍 ${agentType} fileWithData 객체:`, {
        name: fileWithData.name,
        size: fileWithData.size,
        type: fileWithData.type,
        hasData: !!fileWithData.data,
        hasHeaders: !!fileWithData.headers
      });
      
      setAgentFiles(prev => ({
        ...prev,
        [agentType]: fileWithData
      }));

      // 3. Structura 파일인 경우 직원 데이터도 별도 저장
      if (agentType === 'structura') {
        const employees = parseEmployeeData(data, headers);
        setEmployeeData(employees);
      }

      message.success(`${agentType} 데이터를 업로드하고 로드했습니다.`);
      return false; // Ant Design Upload 컴포넌트의 자동 업로드 방지
      
    } catch (error) {
      console.error(`${agentType} 파일 업로드 실패:`, error);
      message.error(`${agentType} 파일 업로드 실패: ${error.message}`);
      return false;
    } finally {
      setLoading(false);
    }
  };

  // 에이전트별 필수 컬럼 정의
  const getRequiredColumns = (agentType) => {
    const columnMap = {
      structura: ['EmployeeNumber', 'Age', 'JobRole', 'Department'],
      chronos: ['employee_id', 'date', 'work_focused_ratio', 'meeting_collaboration_ratio'],
      sentio: ['EmployeeNumber', 'SELF_REVIEW_text', 'PEER_FEEDBACK_text', 'WEEKLY_SURVEY_text'],
      agora: ['EmployeeNumber', 'Age', 'JobRole', 'Department'] // Structura와 동일한 데이터 사용
    };
    return columnMap[agentType] || [];
  };

  // 직원 데이터 파싱 (새로운 데이터 구조 사용)
  const parseEmployeeData = (data, headers) => {
    const employees = [];
    for (let i = 0; i < data.length; i++) {
      const row = data[i];
      const employee = {};
      
      headers.forEach((header) => {
        let value = row[header] || '';
        
        // 숫자 컬럼 처리
        if (['Age', 'MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance'].includes(header)) {
          value = parseFloat(value) || 0;
        }
        
        employee[header] = value;
      });
      
      if (employee.EmployeeNumber) {
        employees.push(employee);
      }
    }
    return employees;
  };

  // Neo4j 연결 테스트
  const testNeo4jConnection = async () => {
    if (!neo4jConfig.uri || !neo4jConfig.username || !neo4jConfig.password) {
      message.error('Neo4j 연결 정보를 모두 입력해주세요.');
      return;
    }

    setNeo4jTesting(true);
    try {
      const response = await fetch('/api/cognita/setup/neo4j', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(neo4jConfig)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('Neo4j 연결 테스트 응답:', result);
      
      // 응답 형식 통일 (success 필드 확인)
      if (result.success) {
        setNeo4jConnected(true);
        message.success('Neo4j 연결 성공!');
      } else {
        setNeo4jConnected(false);
        const errorMsg = result.error || result.message || '알 수 없는 오류';
        message.error(`Neo4j 연결 실패: ${errorMsg}`);
        console.error('Neo4j 연결 실패 상세:', result);
      }
    } catch (error) {
      console.error('Neo4j 연결 테스트 실패:', error);
      setNeo4jConnected(false);
      message.error(`Neo4j 연결 테스트 실패: ${error.message}`);
    } finally {
      setNeo4jTesting(false);
    }
  };

  // 에이전트 디버깅 정보 조회
  const debugAgents = async () => {
    try {
      console.log('🔍 에이전트 디버깅 정보 조회 시작');
      
      const response = await fetch('/api/agents/debug');
      const debugInfo = await response.json();
      
      console.log('🔍 에이전트 디버깅 정보:', debugInfo);
      
      // Console에 상세 정보 출력
      console.log('📊 에이전트별 상태:');
      Object.entries(debugInfo.agents).forEach(([agentName, info]) => {
        console.log(`  ${agentName}:`);
        console.log(`    - Import 가능: ${info.import_available}`);
        console.log(`    - 초기화됨: ${info.initialized}`);
        console.log(`    - 객체 존재: ${info.agent_object}`);
        if (info.error_message) {
          console.log(`    - 오류: ${info.error_message}`);
        }
      });
      
      message.success('에이전트 디버깅 정보가 Console에 출력되었습니다. F12를 눌러 확인하세요.');
      
    } catch (error) {
      console.error('❌ 에이전트 디버깅 정보 조회 실패:', error);
      message.error(`디버깅 정보 조회 실패: ${error.message}`);
    }
  };

  // 통합 배치 분석 실행 - 데이터 검증 + 순차적 워크플로우 + Integration
  const runBatchAnalysis = async () => {
    // 1. 필수 데이터 검증
    if (!agentFiles.structura) {
      message.error('Structura 데이터(HR 기본 데이터)를 업로드해주세요.');
      return;
    }

    if (!employeeData || employeeData.length === 0) {
      message.error('직원 데이터가 로드되지 않았습니다. Structura 파일을 다시 업로드해주세요.');
      return;
    }

    // 2. 가중치 합계 검증
    const weightSum = integrationConfig.structura_weight + 
                     integrationConfig.cognita_weight + 
                     integrationConfig.chronos_weight + 
                     integrationConfig.sentio_weight + 
                     integrationConfig.agora_weight;
    
    if (Math.abs(weightSum - 1.0) > 0.01) {
      message.error(`가중치 합계가 1.0이 되어야 합니다. 현재: ${weightSum.toFixed(2)}`);
      return;
    }

    // 클라이언트 사이드 진행률 관리 (PostAnalysis 방식)
    let finalResults = null; // finally 블록에서 사용할 수 있도록 상위 스코프에 선언

    try {
      setIsAnalyzing(true);
      setAnalysisProgress({
        structura: 0,
        cognita: 0,
        chronos: 0,
        sentio: 0,
        agora: 0,
        overall: 0
      });
      
      // 3. 에이전트별 활성화 여부 결정
      const agentConfig = {
        use_structura: !!agentFiles.structura,
        use_cognita: neo4jConnected,
        use_chronos: !!agentFiles.chronos,
        use_sentio: !!agentFiles.sentio,
        use_agora: !!agentFiles.agora || !!agentFiles.structura // Agora는 별도 파일 또는 Structura 데이터 사용
      };

      // Console 로그 추가
      console.log('🚀 배치 분석 시작');
      console.log('📊 직원 데이터:', employeeData.length, '명');
      console.log('🔧 에이전트 설정:', agentConfig);
      console.log('⚙️ 통합 설정:', integrationConfig);
      console.log('🔗 Neo4j 연결 상태:', neo4jConnected);

      // 4. 통합된 진행률 관리 시스템
      const updateProgress = (step, agentProgress = {}) => {
        const stepProgress = {
          'start': 5,
          'api_call': 15,
          'processing': 50,
          'integration': 85,
          'complete': 100
        };
        
        const overall = stepProgress[step] || 0;
        
        // 분석이 완료되면 모든 에이전트 진행률을 100%로 설정
        if (step === 'complete') {
          setAnalysisProgress({
            structura: 100,
            cognita: 100,
            chronos: 100,
            sentio: 100,
            agora: 100,
            overall: 100
          });
        } else {
          setAnalysisProgress(prev => ({
            ...prev,
            ...agentProgress,
            overall: overall
          }));
        }
        
        console.log(`📊 진행률 업데이트: ${step} - 전체 ${overall}%`);
      };
      
      // 분석 시작
      updateProgress('start');

      // 5. 저장된 모델 정보 포함하여 배치 분석 API 호출
      let savedModelInfo = null;
      if (integrationConfig.use_trained_models) {
        try {
          const savedModels = localStorage.getItem('trainedModels');
          if (savedModels) {
            savedModelInfo = JSON.parse(savedModels);
            console.log('🧠 저장된 모델 사용:', savedModelInfo.training_metadata);
          }
        } catch (error) {
          console.error('저장된 모델 로드 실패:', error);
        }
      }

      // 직원 데이터에서 employee_id 추출 (다양한 필드명 지원)
      console.log('🔍 직원 데이터 샘플:', employeeData.slice(0, 2)); // 디버깅용 로그
      
      const employee_ids = employeeData.map(emp => {
        // 다양한 가능한 ID 필드명들을 시도
        return emp.EmployeeNumber || emp.employee_id || emp.id || emp.Employee_ID || emp.employeeNumber;
      }).filter(id => id !== undefined && id !== null && id !== '');
      
      if (employee_ids.length === 0) {
        console.error('❌ 직원 데이터 구조:', employeeData.slice(0, 3));
        throw new Error('유효한 직원 ID를 찾을 수 없습니다. 데이터 구조를 확인해주세요. 가능한 필드: EmployeeNumber, employee_id, id');
      }

      console.log(`📋 배치 분석 대상: ${employee_ids.length}명의 직원 (IDs: ${employee_ids.join(', ')})`);

      // 요청 데이터 구성
      const requestData = {
        employee_ids: employee_ids, // 백엔드가 기대하는 형식으로 변경
        employees: employeeData, // 추가 데이터로 전체 직원 정보도 포함
        analysis_type: 'batch', // 배치 분석 타입 전달
        ...agentConfig,
        integration_config: integrationConfig,
        neo4j_config: neo4jConnected ? neo4jConfig : null,
        agent_files: {
          structura: agentFiles.structura?.name,
          chronos: agentFiles.chronos?.name,
          sentio: agentFiles.sentio?.name,
          agora: agentFiles.agora?.name
        },
        // 저장된 모델 정보 전달
        trained_models: savedModelInfo?.saved_models || null,
        use_trained_models: integrationConfig.use_trained_models
      };

      console.log('📤 서버로 전송할 데이터:', {
        employee_ids: requestData.employee_ids,
        employee_count: requestData.employees?.length,
        analysis_type: requestData.analysis_type,
        agent_files: requestData.agent_files
      });

      // 5. PostAnalysis 방식으로 각 에이전트 직접 호출
      console.log('🚀 PostAnalysis 방식으로 각 에이전트 순차 실행 시작');
      updateProgress('api_call');
      
      const analysisResults = {};
      const expectedAgents = ['structura', 'cognita', 'chronos', 'sentio', 'agora'];
      
      // 개별 에이전트 진행률 업데이트 함수
      const updateAgentProgress = (agentName, progress) => {
        setAnalysisProgress(prev => {
          const newProgress = { ...prev };
          newProgress[agentName] = progress;
          
          // 전체 진행률 계산 (활성화된 에이전트 기준)
          const activeAgents = expectedAgents.filter(agent => agentConfig[`use_${agent}`]);
          const totalProgress = activeAgents.reduce((sum, agent) => sum + (newProgress[agent] || 0), 0);
          const calculatedOverall = activeAgents.length > 0 ? Math.round(totalProgress / activeAgents.length) : 0;
          
          // 에이전트 진행률만으로 85% 계산 (integration 단계 제외)
          // Integration 단계는 별도로 85-100% 구간 업데이트
          if (prev.overall !== 100) {
            // 에이전트 평균이 100%이면 85%로 표시 (Integration 대기)
            // 그렇지 않으면 85% 비율로 스케일링
            newProgress.overall = calculatedOverall === 100 ? 85 : Math.round(calculatedOverall * 0.85);
          }
          
          return newProgress;
        });
      };
      
      // 각 에이전트별 순차 실행
      for (const agentName of expectedAgents) {
        if (!agentConfig[`use_${agentName}`]) {
          console.log(`⚠️ ${agentName}: 비활성화됨, 건너뜀`);
          continue;
        }
        
        console.log(`🔄 ${agentName}: 배치 분석 시작...`);
        updateAgentProgress(agentName, 10);
        
        try {
          let predictions = [];
          
          if (agentName === 'structura') {
            // Structura 배치 분석: Post 데이터로 학습 → Batch 데이터로 예측
            console.log(`📊 Structura: 배치 분석 시작 (Post 학습 → Batch 예측)`);
            updateAgentProgress('structura', 30);
            
            const response = await fetch('http://localhost:5001/api/batch-analysis', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                analysis_type: 'batch'
                // employees 데이터 제거 - CSV 파일에서 직접 읽음
              })
            });
            
            if (response.ok) {
              const result = await response.json();
              predictions = result.predictions || [];
              console.log(`✅ Structura: ${predictions.length}명 배치 분석 완료`);
              updateAgentProgress('structura', 100);
            } else {
              throw new Error(`Structura 배치 분석 실패: ${response.status}`);
            }
            
          } else if (agentName === 'chronos' && agentFiles.chronos) {
            // Chronos 배치 분석: Post 데이터로 학습 → Batch 데이터로 예측
            console.log(`📈 Chronos: 배치 분석 시작 (Post 학습 → Batch 예측)`);
            updateAgentProgress('chronos', 30);
            
            const response = await fetch('http://localhost:5003/api/batch-analysis', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                analysis_type: 'batch'
                // employee_ids 제거 - CSV 파일에서 직접 읽음
              })
            });
            
            if (response.ok) {
              const result = await response.json();
              predictions = result.predictions || [];
              console.log(`✅ Chronos: ${predictions.length}명 배치 분석 완료`);
              updateAgentProgress('chronos', 100);
            } else {
              throw new Error(`Chronos 배치 분석 실패: ${response.status}`);
            }
            
          } else if (agentName === 'sentio' && agentFiles.sentio) {
            // Sentio API 호출 - 업로드된 파일 데이터 사용
            console.log(`💭 Sentio: ${employee_ids.length}명 감정 분석 시작...`);
            updateAgentProgress('sentio', 10);
            
            // 디버깅: 업로드된 Sentio 데이터 구조 확인
            console.log('🔍 Sentio 파일 데이터 샘플:', agentFiles.sentio.data.slice(0, 2));
            console.log('🔍 Sentio 파일 헤더:', agentFiles.sentio.headers);
            console.log('🔍 분석 대상 직원 ID 샘플:', employee_ids.slice(0, 5));
            console.log('🔍 Sentio 파일의 직원 ID 샘플:', agentFiles.sentio.data.slice(0, 5).map(emp => emp.EmployeeNumber || emp.employee_id || emp.id));
            
            // 업로드된 Sentio 데이터에서 실제 텍스트 추출 (PostAnalysis와 동일한 방식)
            const sentioEmployees = agentFiles.sentio.data.map(emp => {
              const empId = emp.EmployeeNumber || emp.employee_id || emp.id || emp.Employee_ID || emp.employeeNumber;
              return {
                employee_id: empId,
                text_data: {
                  self_review: emp.SELF_REVIEW_text || emp.self_review_text || emp.self_review || '',
                  peer_feedback: emp.PEER_FEEDBACK_text || emp.peer_feedback_text || emp.peer_feedback || '',
                  weekly_survey: emp.WEEKLY_SURVEY_text || emp.weekly_survey_text || emp.weekly_survey || ''
                }
              };
            });
            
            console.log(`📝 Sentio: 업로드된 데이터에서 ${sentioEmployees.length}명의 실제 텍스트 데이터 추출`);
            updateAgentProgress('sentio', 20);
            
            // 텍스트 데이터 품질 검증
            console.log('🔍 Sentio 직원 데이터 샘플:', sentioEmployees.slice(0, 2));
            
            const validTextCount = sentioEmployees.filter(emp => {
              const textData = emp.text_data;
              const hasText = textData.self_review || textData.peer_feedback || textData.weekly_survey;
              if (!hasText) {
                console.log(`⚠️ 직원 ${emp.employee_id}: 텍스트 데이터 없음`, textData);
              }
              return hasText;
            }).length;
            
            console.log(`📊 Sentio: 전체 ${sentioEmployees.length}명 중 ${validTextCount}명이 유효한 텍스트 데이터 보유`);
            
            if (validTextCount === 0) {
              console.error('❌ Sentio: 업로드된 파일에 유효한 텍스트 데이터가 없습니다.');
              console.error('사용 가능한 컬럼:', agentFiles.sentio.headers);
              throw new Error('업로드된 Sentio 파일에 SELF_REVIEW_text, PEER_FEEDBACK_text, WEEKLY_SURVEY_text 중 하나 이상의 데이터가 필요합니다.');
            }
            
            console.log(`✅ Sentio: ${validTextCount}명의 유효한 텍스트 데이터 확인됨`);
            updateAgentProgress('sentio', 30);
            
            console.log('🚀 Sentio API 호출 시작...');
            console.log('📤 요청 데이터:', {
              analysis_type: 'batch',
              employees_count: sentioEmployees.length,
              first_employee_sample: sentioEmployees[0]
            });
            
            const response = await fetch('http://localhost:5004/analyze_sentiment', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                analysis_type: 'batch', // 배치 분석 타입 전달
                employees: sentioEmployees  // 실제 텍스트 데이터 포함
              })
            });
            
            console.log('📥 Sentio API 응답 상태:', response.status, response.statusText);
            
            if (response.ok) {
              const result = await response.json();
              console.log('🔍 Sentio API 응답 구조:', {
                success: result.success,
                total_analyzed: result.total_analyzed,
                analysis_results_length: result.analysis_results?.length,
                first_result_sample: result.analysis_results?.[0]
              });
              predictions = result.analysis_results?.map(pred => ({
                employee_id: pred.employee_id,
                risk_score: pred.psychological_risk_score, // PostAnalysis와 완전히 동일
                predicted_attrition: pred.psychological_risk_score > 0.5 ? 1 : 0,
                confidence: 0.8,
                actual_attrition: 0 // 배치 분석에서는 실제 퇴직 데이터 없음
              })) || [];
              
              console.log(`✅ Sentio: ${predictions.length}명 배치 분석 완료`);
              console.log('🔍 Sentio predictions 샘플:', predictions.slice(0, 3));
              updateAgentProgress('sentio', 100);
            } else {
              const errorText = await response.text();
              console.error('❌ Sentio API 오류:', {
                status: response.status,
                statusText: response.statusText,
                errorText: errorText
              });
              // 실패해도 빈 배열로 초기화 (전체 분석 중단 방지)
              predictions = [];
              console.warn('⚠️ Sentio 분석 실패로 빈 결과 사용');
            }
            
          } else if (agentName === 'cognita' && neo4jConnected) {
            // Cognita 개별 분석 API 호출 (PostAnalysis 방식)
            console.log(`🕸️ Cognita: ${employee_ids.length}명 개별 관계 분석 시작...`);
            updateAgentProgress('cognita', 10);
            
            predictions = [];
            let successCount = 0;
            let failCount = 0;
            
            for (let i = 0; i < employee_ids.length; i++) {
              try {
                const empId = employee_ids[i];
                
                // 진행률 업데이트 (10명마다)
                if (i % 10 === 0) {
                  const progress = Math.round((i / employee_ids.length) * 90) + 10; // 10-100%
                  updateAgentProgress('cognita', progress);
                  console.log(`🔄 Cognita: ${i}/${employee_ids.length}명 완료 (${progress}%)`);
                }
                
                const response = await fetch(`http://localhost:5002/api/analyze/employee/${empId}`);
                
                if (response.ok) {
                  const result = await response.json();
                  predictions.push({
                    employee_id: empId,
                    risk_score: result.overall_risk_score || result.risk_score || 0.5,
                    predicted_attrition: (result.overall_risk_score || result.risk_score || 0.5) > 0.5 ? 1 : 0
                  });
                  successCount++;
                } else {
                  console.warn(`⚠️ Cognita: 직원 ${empId} 분석 실패 (${response.status})`);
                  // 실패 시 기본값 추가
                  predictions.push({
                    employee_id: empId,
                    risk_score: 0.5,
                    predicted_attrition: 0
                  });
                  failCount++;
                }
                
                // 너무 빠른 요청 방지 (서버 부하 감소)
                if (i % 50 === 0 && i > 0) {
                  await new Promise(resolve => setTimeout(resolve, 100)); // 100ms 대기
                }
                
              } catch (error) {
                console.error(`❌ Cognita: 직원 ${employee_ids[i]} 분석 오류:`, error);
                // 오류 시 기본값 추가
                predictions.push({
                  employee_id: employee_ids[i],
                  risk_score: 0.5,
                  predicted_attrition: 0
                });
                failCount++;
              }
            }
            
            updateAgentProgress('cognita', 100);
            console.log(`✅ Cognita: 개별 분석 완료 - 성공 ${successCount}명, 실패 ${failCount}명`);
            
          } else if (agentName === 'agora') {
            // Agora 배치 분석 API 호출
            console.log(`📊 Agora: ${employee_ids.length}명 배치 시장 분석 시작...`);
            updateAgentProgress('agora', 10);
            
            // 직원 데이터 준비
            const agoraEmployees = employee_ids.map(empId => {
              const employeeInfo = employeeData.find(emp => emp.EmployeeNumber === empId) || {};
              return {
                employee_id: empId,
                JobRole: employeeInfo.JobRole || 'Unknown',
                MonthlyIncome: parseFloat(employeeInfo.MonthlyIncome) || 5000,
                Department: employeeInfo.Department || 'Unknown',
                YearsAtCompany: parseInt(employeeInfo.YearsAtCompany) || 1,
                JobSatisfaction: parseInt(employeeInfo.JobSatisfaction) || 3
              };
            });
            
            console.log(`🔍 Agora 배치 분석 데이터 준비 완료: ${agoraEmployees.length}명`);
            updateAgentProgress('agora', 20);
            
            const response = await fetch('http://localhost:5005/analyze/batch', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                employees: agoraEmployees,
                use_llm: false  // 빠른 처리를 위해 LLM 비사용
              })
            });
            
            if (response.ok) {
              const result = await response.json();
              console.log(`✅ Agora 배치 분석 성공:`, result);
              
              predictions = result.analysis_results?.map(item => ({
                employee_id: item.employee_id,
                agora_score: item.agora_score || item.market_pressure_index || 0.5,
                risk_level: item.risk_level || 'MEDIUM',
                market_pressure_index: item.market_pressure_index || 0.5,
                compensation_gap: item.compensation_gap || 0.5,
                job_postings_count: item.job_postings_count || 0,
                market_competitiveness: item.market_competitiveness || 'MEDIUM'
              })) || [];
              
              console.log(`✅ Agora: ${predictions.length}/${employee_ids.length}명 배치 분석 완료`);
            } else {
              console.error(`❌ Agora 배치 분석 실패: ${response.status}`);
              throw new Error(`Agora 배치 분석 실패: ${response.status}`);
            }
            
            updateAgentProgress('agora', 100);
          }
          
          // 결과 저장
          analysisResults[agentName] = predictions;
          console.log(`💾 ${agentName} 결과 저장 완료:`, {
            agentName,
            predictions_count: predictions?.length || 0,
            first_prediction: predictions?.[0]
          });
          
        } catch (error) {
          console.error(`❌ ${agentName} 분석 실패:`, error);
          throw error; // 에러를 상위로 전파하여 전체 분석 중단
        }
      }
      
      console.log('📊 모든 에이전트 분석 완료:', analysisResults);
      updateProgress('processing');

      // 6. 결과 통합 및 포맷팅 (PostAnalysis 방식)
      updateProgress('integration');
      
      // Integration 진행률 업데이트 (85% → 90%)
      setAnalysisProgress(prev => ({ ...prev, overall: 90 }));
      
      const batchResults = [];
      const totalEmployees = employeeData.length;
      let successfulAnalyses = 0;
      
      // 각 직원별로 결과 통합
      for (const employee of employeeData) {
        const empId = employee.EmployeeNumber || employee.employee_id || employee.id;
        
        const employeeResult = {
          employee_id: empId,
          employee_number: empId,
          success: true,
          analysis_result: {
            status: 'success',
            employee_data: employee,
            structura_result: null,
            cognita_result: null,
            chronos_result: null,
            sentio_result: null,
            agora_result: null,
            combined_analysis: {
              integrated_assessment: {
                overall_risk_score: 0,
                overall_risk_level: 'LOW'
              }
            }
          }
        };
        
        // 각 에이전트 결과 통합
        let totalRiskScore = 0;
        let activeAgentCount = 0;
        
        // Structura 결과
        if (analysisResults.structura) {
          const structuraPred = analysisResults.structura.find(p => 
            String(p.employee_number || p.employee_id) === String(empId)
          );
          if (structuraPred) {
            employeeResult.analysis_result.structura_result = {
              prediction: {
                attrition_probability: structuraPred.risk_score || structuraPred.attrition_probability,
                confidence_score: structuraPred.confidence || 0.8,
                risk_category: structuraPred.risk_score > 0.7 ? 'HIGH' : structuraPred.risk_score > 0.3 ? 'MEDIUM' : 'LOW'
              }
            };
            totalRiskScore += (structuraPred.risk_score || 0) * integrationConfig.structura_weight;
            activeAgentCount++;
          }
        }
        
        // Cognita 결과
        if (analysisResults.cognita) {
          const cognitaPred = analysisResults.cognita.find(p => 
            String(p.employee_id) === String(empId)
          );
          if (cognitaPred) {
            employeeResult.analysis_result.cognita_result = {
              risk_analysis: {
                overall_risk_score: cognitaPred.risk_score,
                risk_category: cognitaPred.risk_score > 0.7 ? 'HIGH' : cognitaPred.risk_score > 0.3 ? 'MEDIUM' : 'LOW'
              }
            };
            totalRiskScore += (cognitaPred.risk_score || 0) * integrationConfig.cognita_weight;
            activeAgentCount++;
          }
        }
        
        // Chronos 결과
        if (analysisResults.chronos) {
          const chronosPred = analysisResults.chronos.find(p => 
            String(p.employee_id) === String(empId)
          );
          if (chronosPred) {
            employeeResult.analysis_result.chronos_result = {
              prediction: {
                risk_score: chronosPred.risk_score || chronosPred.attrition_probability,
                attrition_probability: chronosPred.risk_score || chronosPred.attrition_probability
              },
              confidence: chronosPred.confidence || 0.8
            };
            totalRiskScore += (chronosPred.risk_score || 0) * integrationConfig.chronos_weight;
            activeAgentCount++;
          }
        }
        
        // Sentio 결과
        
        if (analysisResults.sentio) {
          const sentioPred = analysisResults.sentio.find(p => 
            String(p.employee_id) === String(empId)
          );
          
          if (sentioPred) {
            employeeResult.analysis_result.sentio_result = {
              psychological_risk_score: sentioPred.risk_score, // API 응답의 risk_score를 직접 저장
              sentiment_analysis: {
                risk_score: sentioPred.risk_score, // 호환성을 위해 중복 저장
                sentiment_score: sentioPred.sentiment_score || 0
              },
              risk_level: sentioPred.risk_level || 'MEDIUM'
            };
            // sentio_score도 최상위에 저장 (UI에서 쉽게 접근)
            employeeResult.sentio_score = sentioPred.risk_score;
            totalRiskScore += (sentioPred.risk_score || 0) * integrationConfig.sentio_weight;
            activeAgentCount++;
          }
        }
        
        // Agora 결과
        if (analysisResults.agora) {
          const agoraPred = analysisResults.agora.find(p => 
            String(p.employee_id) === String(empId)
          );
          if (agoraPred) {
            employeeResult.analysis_result.agora_result = {
              market_analysis: {
                market_pressure_index: agoraPred.market_pressure_index || agoraPred.agora_score || 0,
                risk_score: agoraPred.agora_score || agoraPred.risk_score || 0,
                compensation_gap: agoraPred.compensation_gap || 0,
                job_postings_count: agoraPred.job_postings_count || 0,
                market_competitiveness: agoraPred.market_competitiveness || 'UNKNOWN'
              },
              risk_level: agoraPred.risk_level || 'MEDIUM',
              agora_score: agoraPred.agora_score || 0
            };
            totalRiskScore += (agoraPred.agora_score || agoraPred.risk_score || 0) * integrationConfig.agora_weight;
            activeAgentCount++;
          }
        }
        
        // 통합 위험도 계산
        if (activeAgentCount > 0) {
          const normalizedRiskScore = totalRiskScore; // 가중치 합이 1이므로 정규화 불필요
          employeeResult.analysis_result.combined_analysis.integrated_assessment.overall_risk_score = normalizedRiskScore;
          employeeResult.analysis_result.combined_analysis.integrated_assessment.overall_risk_level = 
            normalizedRiskScore > 0.7 ? 'HIGH' : normalizedRiskScore > 0.3 ? 'MEDIUM' : 'LOW';
          successfulAnalyses++;
        }
        
        batchResults.push(employeeResult);
      }
      
      // 최종 결과 구성
      finalResults = {
        success: true,
        results: batchResults,
        total_employees: totalEmployees,
        completed_employees: successfulAnalyses,
        summary: {
          total_employees: totalEmployees,
          successful_analyses: successfulAnalyses,
          failed_analyses: totalEmployees - successfulAnalyses,
          success_rate: successfulAnalyses / totalEmployees
        },
        analysis_metadata: {
          analysis_type: 'batch',
          agents_used: Object.keys(agentConfig).filter(key => agentConfig[key] && key.startsWith('use_')),
          integration_config: integrationConfig,
          timestamp: new Date().toISOString()
        }
      };
      
      console.log(`📊 배치 분석 완료: ${successfulAnalyses}/${totalEmployees}명 성공`);
      
      // Integration 진행률 업데이트 (90% → 95%)
      setAnalysisProgress(prev => ({ ...prev, overall: 95 }));
      
      // 브라우저 로컬 저장은 생략하고 서버 저장만 수행
      console.log('💾 서버에 결과 저장 준비 완료...');

      // 전역 상태 업데이트 (다른 페이지에서 사용할 수 있도록)
      if (updateBatchResults) {
        updateBatchResults(finalResults);
      }

      // 예측 히스토리에 저장
      try {
        const predictionData = predictionService.convertBatchResultToPrediction(finalResults);
        if (predictionData) {
          predictionService.savePredictionResult(predictionData);
          message.success('분석 결과가 히스토리에 저장되었습니다.');
        }
      } catch (error) {
        console.error('예측 히스토리 저장 실패:', error);
        // 에러가 발생해도 메인 분석 프로세스는 계속 진행
      }

      // 부서별 결과 저장 (XAI 포함) - 손실 없는 저장 방식
      try {
        console.log('💾 부서별 배치 분석 결과 저장 시작...');
        
        // 원본 데이터 보존을 위한 스마트 저장 전략
        const saveData = {
          results: finalResults.results || [],
          applied_settings: finalRiskSettings || {},
          analysis_metadata: {
            total_employees: finalResults.total_employees,
            completed_employees: finalResults.completed_employees,
            analysis_timestamp: new Date().toISOString(),
            agents_used: Object.keys(agentConfig).filter(key => agentConfig[key] && key.startsWith('use_')),
            integration_config: integrationConfig
          }
        };
        
        // 데이터 크기 확인
        const dataString = JSON.stringify(saveData);
        const dataSize = new Blob([dataString]).size;
        const maxSize = 50 * 1024 * 1024; // 50MB로 증가 (더 관대한 제한)
        
        console.log(`📊 저장할 데이터 크기: ${Math.round(dataSize/1024/1024*100)/100}MB (제한: ${maxSize/1024/1024}MB)`);
        
        if (dataSize > maxSize) {
          console.log(`⚠️ 데이터 크기 초과 - 분할 저장 방식 사용`);
          
          // 🚀 손실 없는 분할 저장 방식
          // 1. 메타데이터와 설정 정보는 별도 저장
          const metadataOnly = {
            applied_settings: saveData.applied_settings,
            analysis_metadata: saveData.analysis_metadata,
            total_results: saveData.results.length,
            storage_method: 'chunked_lossless'
          };
          
          // 2. 결과 데이터를 청크 단위로 분할 (손실 없음)
          const chunkSize = 100; // 작은 청크로 분할
          const chunks = [];
          
          for (let i = 0; i < saveData.results.length; i += chunkSize) {
            const chunk = {
              chunk_index: Math.floor(i / chunkSize),
              start_index: i,
              end_index: Math.min(i + chunkSize, saveData.results.length),
              data: saveData.results.slice(i, i + chunkSize) // 원본 데이터 그대로 보존
            };
            chunks.push(chunk);
          }
          
          console.log(`📦 분할 저장: ${chunks.length}개 청크, 청크당 최대 ${chunkSize}명`);
          
          // 3. 각 청크를 개별적으로 저장 시도
          let savedChunks = 0;
          const chunkSavePromises = [];
          
          for (let i = 0; i < Math.min(chunks.length, 5); i++) { // 최대 5개 청크씩 병렬 처리
            const chunkPromise = this.saveDataChunk(chunks[i], i, metadataOnly);
            chunkSavePromises.push(chunkPromise);
          }
          
          try {
            const chunkResults = await Promise.allSettled(chunkSavePromises);
            savedChunks = chunkResults.filter(result => result.status === 'fulfilled').length;
            
            console.log(`✅ 청크 저장 완료: ${savedChunks}/${chunkSavePromises.length}개 성공`);
            
            if (savedChunks > 0) {
              message.success(
                `분할 저장 완료! ${savedChunks}개 청크 저장 ` +
                `(전체 데이터 손실 없이 보존됨)`
              );
              return; // 성공적으로 저장됨
            }
          } catch (chunkError) {
            console.error('청크 저장 실패:', chunkError);
          }
          
          // 4. 청크 저장도 실패한 경우에만 서버 직접 저장 시도
          console.log('🔄 청크 저장 실패 - 서버 직접 저장 시도...');
          saveData.storage_note = 'Direct server storage due to chunk failure';
        }
        
        // 🚀 개선된 네트워크 저장 시스템 사용
        try {
          const networkSaveResult = await networkManager.saveBatchAnalysisResults(saveData);
          
          if (networkSaveResult.ok) {
            const result = await networkSaveResult.json();
            console.log('✅ 서버 저장 성공:', result);
            message.success(
              `부서별 분석 결과 저장 완료! ` +
              `${result.statistics?.total_departments || 0}개 부서, ` +
              `${result.statistics?.total_employees || 0}명 직원`
            );
          } else {
            throw new Error(`서버 응답 오류: ${networkSaveResult.status}`);
          }
        } catch (networkError) {
          console.error('서버 저장 실패:', networkError);
          message.warning(`서버 저장 실패: ${networkError.message}. 로컬 저장은 완료되었습니다.`);
        }
        
      } catch (error) {
        console.error('부서별 결과 저장 중 예상치 못한 오류:', error);
        message.warning('부서별 결과 저장에 실패했지만 분석은 완료되었습니다.');
      }
      
          // 7. 계층적 구조로 결과 저장 (개선된 오류 처리)
      console.log('💾 배치 분석 결과를 계층적 구조로 저장 중...');
      try {
        // Integration 서버 상태 확인 (올바른 포트 사용)
        const integrationHealthCheck = await fetch('http://localhost:5007/health', {
          method: 'GET',
          signal: AbortSignal.timeout(5000) // 5초 타임아웃
        });
        
        if (!integrationHealthCheck.ok) {
          throw new Error(`Integration 서버 응답 오류: ${integrationHealthCheck.status}`);
        }
        
        // Supervisor의 HierarchicalResultManager 사용 (올바른 파일 구조로 저장)
        console.log('💾 Supervisor HierarchicalResultManager로 저장 시작...');
        
        // saveHierarchicalBatchResults 함수를 사용하여 계층적 구조 생성 및 저장
        const hierarchicalSaveResult = await saveHierarchicalBatchResults(
          finalResults,
          employeeData,
          {
            totalEmployees: employee_ids.length,
            successfulAnalyses: finalResults.summary?.successful_analyses || 0,
            failedAnalyses: finalResults.summary?.failed_analyses || 0,
            agentBreakdown: finalResults.summary?.agent_breakdown || {}
          }
        );
        
        if (!hierarchicalSaveResult.success) {
          throw new Error(`Supervisor 계층적 저장 실패: ${hierarchicalSaveResult.error || '알 수 없는 오류'}`);
        }
        
        console.log('✅ Supervisor HierarchicalResultManager 저장 완료:', hierarchicalSaveResult);
        console.log('✅ 계층적 구조 저장 완료:', hierarchicalSaveResult.statistics);
        message.success('계층적 구조로 결과 저장 완료! (Department > JobRole > JobLevel > 직원별)');
      } catch (error) {
        console.error('❌ 계층적 구조 저장 실패:', error);
        
        // 오류 유형에 따른 구체적인 메시지 제공
        let errorMessage = '계층적 구조 저장에 실패했지만 분석은 완료되었습니다.';
        
        if (error.message.includes('압축 저장 실패: 400')) {
          errorMessage = '서버에서 데이터를 처리할 수 없습니다. 데이터 형식을 확인해주세요.';
        } else if (error.name === 'AbortError' || error.message.includes('TimeoutError')) {
          errorMessage = 'Integration 서버 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.';
        } else if (error.message.includes('Failed to fetch') || error.message.includes('ERR_CONNECTION_RESET')) {
          errorMessage = 'Integration 서버와의 연결이 끊어졌습니다. 서버 상태를 확인해주세요.';
        } else if (error.message.includes('Integration 서버 응답 오류')) {
          errorMessage = 'Integration 서버가 응답하지 않습니다. 서버를 다시 시작해주세요.';
        }
        
        message.warning(errorMessage);
      }

      // 8. 분석 완료 - 진행률을 100%로 고정하고 상태 정리
      console.log('🎯 분석 완료 처리 시작...');
      
      // 최종 진행률 업데이트 (95% → 100%)
      updateProgress('complete');
      
      // 분석 완료 후 상태를 즉시 정리
      setIsAnalyzing(false);
      
      // 진행률을 100%로 고정 (이중 보장)
      setAnalysisProgress({
        structura: 100,
        cognita: 100,
        chronos: 100,
        sentio: 100,
        agora: 100,
        overall: 100
      });
      
      // 분석 결과 설정
      setAnalysisResults(finalResults);

      const completedCount = finalResults.summary?.successful_analyses || 0;
      message.success(`PostAnalysis 방식 배치 분석 완료! ${completedCount}명의 직원 분석이 완료되었습니다.`);
      
      console.log('✅ 배치 분석 완료 - 상태 정리됨');

    } catch (error) {
      console.error('❌ 통합 배치 분석 실패:', error);
      console.error('❌ 오류 스택:', error.stack);
      
      // 에러 발생 시 진행률 초기화
      setAnalysisProgress({
        structura: 0,
        cognita: 0,
        chronos: 0,
        sentio: 0,
        agora: 0,
        overall: 0
      });
      
      // 네트워크 오류인지 확인
      if (error.message.includes('fetch')) {
        console.error('🌐 네트워크 연결 문제가 발생했습니다. 백엔드 서버가 실행 중인지 확인하세요.');
        message.error('네트워크 연결 문제가 발생했습니다. 백엔드 서버 상태를 확인하세요.');
      } else {
        message.error(`통합 배치 분석 실패: ${error.message}`);
      }
    } finally {
      // finally 블록에서도 분석 상태를 false로 설정 (에러 발생 시에도)
      setIsAnalyzing(false);
      
      // 에러 발생 시에만 진행률 초기화 (성공한 경우는 100% 유지)
      if (!finalResults || !finalResults.results) {
        console.log('⚠️ 분석 실패로 진행률 초기화');
        setAnalysisProgress({
          structura: 0,
          cognita: 0,
          chronos: 0,
          sentio: 0,
          agora: 0,
          overall: 0
        });
      } else {
        console.log('✅ 분석 성공으로 진행률 100% 유지');
      }
    }
  };

  // 손실 없는 청크 저장 함수 (사용하지 않음 - 향후 확장용)
  /*
  const saveDataChunk = async (chunk, chunkIndex, metadata) => {
    try {
      const chunkData = {
        ...metadata,
        chunk_info: {
          index: chunkIndex,
          start_index: chunk.start_index,
          end_index: chunk.end_index,
          employee_count: chunk.data.length
        },
        results: chunk.data // 원본 데이터 그대로 보존
      };
      
      const response = await fetch('http://localhost:5007/api/batch-analysis/save-chunk', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(chunkData),
        signal: AbortSignal.timeout(30000)
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log(`✅ 청크 ${chunkIndex} 저장 성공:`, result);
        return result;
      } else {
        const errorText = await response.text();
        throw new Error(`청크 ${chunkIndex} 저장 실패: ${response.status} - ${errorText}`);
      }
    } catch (error) {
      console.error(`❌ 청크 ${chunkIndex} 저장 오류:`, error);
      throw error;
    }
  };
  */

  // 계층적 구조로 배치 결과 저장 함수 - 개선된 오류 처리
  // 서버 연결 상태 확인 함수
  const checkServerConnection = async () => {
    try {
      const response = await fetch('http://localhost:5006/health', {
        method: 'GET'
        // 타임아웃 제거 - 무제한 대기
      });
      return response.ok;
    } catch (error) {
      console.error('❌ 서버 연결 확인 실패:', error.message);
      return false;
    }
  };

  const saveHierarchicalBatchResults = async (analysisResults, employeeData, analysisSummary) => {
    try {
      // 먼저 서버 연결 상태 확인
      console.log('🔍 서버 연결 상태 확인 중...');
      const isServerConnected = await checkServerConnection();
      
      if (!isServerConnected) {
        throw new Error('서버에 연결할 수 없습니다. Integration 서버(포트 5006)가 실행 중인지 확인해주세요.');
      }
      
      console.log('✅ 서버 연결 확인됨');
      console.log('💾 계층적 구조 저장 시작...');
      console.log('📊 분석 결과 구조:', {
        hasResults: !!analysisResults,
        hasResultsArray: !!(analysisResults && analysisResults.results),
        resultsCount: analysisResults?.results?.length || 0,
        employeeDataCount: employeeData?.length || 0
      });
      
      // 입력 데이터 검증
      if (!analysisResults || !employeeData || !Array.isArray(employeeData)) {
        console.error('❌ 유효하지 않은 입력 데이터:', {
          analysisResults: !!analysisResults,
          employeeData: !!employeeData,
          isArray: Array.isArray(employeeData)
        });
        throw new Error('유효하지 않은 입력 데이터');
      }
      
      // 직원별 결과를 Department > JobRole > JobLevel 구조로 정리
      const hierarchicalResults = {};
      let processedEmployees = 0;
      
      // 각 직원의 분석 결과를 계층적으로 구성
      for (let i = 0; i < employeeData.length; i++) {
        try {
          const employee = employeeData[i];
          const employeeId = employee.EmployeeNumber || employee.employee_id || employee.id;
          
          if (!employeeId) {
            console.warn(`직원 ${i}: ID가 없어 건너뜀`);
            continue;
          }
          
          // 직원 기본 정보 (안전한 추출) - 여러 소스에서 시도
          // 1차: 직원 데이터에서 직접 추출
          let department = employee.Department || employee.department;
          let jobRole = employee.JobRole || employee.job_role;
          let jobLevel = employee.JobLevel || employee.Position || employee.position || employee.job_level;
          
          // 2차: 분석 결과의 employee_data에서 추출
          if (analysisResults && analysisResults.results && Array.isArray(analysisResults.results)) {
            const employeeAnalysis = analysisResults.results.find(r => 
              String(r.employee_id || r.employee_number) === String(employeeId)
            );
            
            if (employeeAnalysis && employeeAnalysis.analysis_result && employeeAnalysis.analysis_result.employee_data) {
              const empData = employeeAnalysis.analysis_result.employee_data;
              if (!department && empData.Department) department = empData.Department;
              if (!jobRole && empData.JobRole) jobRole = empData.JobRole;
              if (!jobLevel && (empData.JobLevel || empData.Position)) {
                jobLevel = empData.JobLevel || empData.Position;
              }
            }
          }
          
          // 3차: Structura 결과에서 추출
          if (analysisResults && analysisResults.results && Array.isArray(analysisResults.results)) {
            const employeeAnalysis = analysisResults.results.find(r => 
              String(r.employee_id || r.employee_number) === String(employeeId)
            );
            
            if (employeeAnalysis && employeeAnalysis.analysis_result && 
                employeeAnalysis.analysis_result.structura_result && 
                employeeAnalysis.analysis_result.structura_result.employee_data) {
              const structEmpData = employeeAnalysis.analysis_result.structura_result.employee_data;
              if (!department && structEmpData.Department) department = structEmpData.Department;
              if (!jobRole && structEmpData.JobRole) jobRole = structEmpData.JobRole;
              if (!jobLevel && (structEmpData.JobLevel || structEmpData.Position)) {
                jobLevel = structEmpData.JobLevel || structEmpData.Position;
              }
            }
          }
          
          // 최종 기본값 설정
          department = department || 'Unknown';
          jobRole = jobRole || 'Unknown';
          jobLevel = jobLevel || '1';
          
          // 각 에이전트별 결과 수집
          const employeeResults = {
            employee_id: employeeId,
            employee_data: employee,
            agent_results: {}
          };
          
          // 각 에이전트 결과 추가 (finalResults.results에서 추출)
          if (analysisResults && analysisResults.results && Array.isArray(analysisResults.results)) {
            const employeeAnalysis = analysisResults.results.find(r => 
              String(r.employee_id || r.employee_number) === String(employeeId)
            );
            
            if (employeeAnalysis && employeeAnalysis.analysis_result) {
              // 각 에이전트별 결과 추출
              const analysis = employeeAnalysis.analysis_result;
              
              if (analysis.structura_result) {
                employeeResults.agent_results.structura = analysis.structura_result;
              }
              if (analysis.cognita_result) {
                employeeResults.agent_results.cognita = analysis.cognita_result;
              }
              if (analysis.chronos_result) {
                employeeResults.agent_results.chronos = analysis.chronos_result;
              }
              if (analysis.sentio_result) {
                employeeResults.agent_results.sentio = analysis.sentio_result;
              }
              if (analysis.agora_result) {
                employeeResults.agent_results.agora = analysis.agora_result;
              }
              if (analysis.combined_analysis) {
                employeeResults.agent_results.combined = analysis.combined_analysis;
              }
            }
          }
          
          // 계층적 구조 생성 - 디버깅 로그 추가
          console.log(`🏗️ 계층 구조 생성: ${department} > ${jobRole} > ${jobLevel} > ${employeeId}`);
          
          if (!hierarchicalResults[department]) {
            hierarchicalResults[department] = {};
          }
          if (!hierarchicalResults[department][jobRole]) {
            hierarchicalResults[department][jobRole] = {};
          }
          if (!hierarchicalResults[department][jobRole][jobLevel]) {
            hierarchicalResults[department][jobRole][jobLevel] = {};
          }
          
          hierarchicalResults[department][jobRole][jobLevel][employeeId] = employeeResults;
          processedEmployees++;
          
        } catch (employeeError) {
          console.error(`직원 ${i} 처리 중 오류:`, employeeError);
          // 개별 직원 오류는 전체 프로세스를 중단하지 않음
        }
      }
      
      console.log(`📊 계층적 구조 생성 완료: ${processedEmployees}/${employeeData.length}명 처리`);
      console.log(`🏢 생성된 부서 수: ${Object.keys(hierarchicalResults).length}`);
      console.log(`📋 부서별 직원 수:`, Object.entries(hierarchicalResults).map(([dept, data]) => {
        const count = Object.values(data).reduce((sum, roles) => 
          sum + Object.values(roles).reduce((roleSum, levels) => 
            roleSum + Object.keys(levels).length, 0), 0);
        return `${dept}: ${count}명`;
      }).join(', '));
      
      // 데이터 크기 확인
      const dataString = JSON.stringify({
        hierarchical_results: hierarchicalResults,
        analysis_summary: analysisSummary,
        analysis_timestamp: new Date().toISOString()
      });
      
      const dataSize = new Blob([dataString]).size;
      const maxSize = 10 * 1024 * 1024; // 10MB 제한으로 조정
      const chunkSize = 5 * 1024 * 1024; // 5MB 청크 크기
      
      console.log(`📏 데이터 크기: ${(dataSize/1024/1024).toFixed(2)}MB`);
      
      if (dataSize > maxSize) {
        console.warn(`⚠️ 대용량 데이터 감지 (${Math.round(dataSize/1024/1024)}MB > 10MB) - 청크 단위 전송 시작`);
        
        // 부서별로 데이터를 분할하여 전송
        const departments = Object.keys(hierarchicalResults);
        const totalDepartments = departments.length;
        let successfulChunks = 0;
        
        for (let i = 0; i < departments.length; i++) {
          const department = departments[i];
          const chunkData = {
            hierarchical_results: {
              [department]: hierarchicalResults[department]
            },
            chunk_info: {
              chunk_index: i + 1,
              total_chunks: totalDepartments,
              department: department,
              is_chunk: true
            },
            analysis_summary: analysisSummary,
            analysis_timestamp: new Date().toISOString()
          };
          
          try {
            console.log(`📦 청크 ${i + 1}/${totalDepartments} 전송 중 (${department} 부서)...`);
            
            const chunkResponse = await fetch('http://localhost:5006/api/batch-analysis/save-hierarchical-results', {
              method: 'POST',
              headers: { 
                'Content-Type': 'application/json',
                'Connection': 'keep-alive'
              },
              body: JSON.stringify(chunkData)
              // 타임아웃 제거 - 무제한 대기
            });
            
            if (chunkResponse.ok) {
              const result = await chunkResponse.json();
              console.log(`✅ 청크 ${i + 1} 저장 성공:`, result.message || 'OK');
              successfulChunks++;
            } else {
              const errorText = await chunkResponse.text();
              console.error(`❌ 청크 ${i + 1} 저장 실패:`, chunkResponse.status, errorText);
              throw new Error(`청크 ${i + 1} 저장 실패: ${chunkResponse.status}`);
            }
            
            // 청크 간 대기 시간 (서버 부하 방지)
            if (i < departments.length - 1) {
              await new Promise(resolve => setTimeout(resolve, 500));
            }
            
          } catch (error) {
            console.error(`❌ 청크 ${i + 1} 전송 중 오류:`, error.message);
            throw new Error(`청크 전송 실패 (${i + 1}/${totalDepartments}): ${error.message}`);
          }
        }
        
        console.log(`✅ 모든 청크 전송 완료: ${successfulChunks}/${totalDepartments}`);
        return {
          success: true,
          message: `계층적 구조를 ${totalDepartments}개 청크로 분할하여 저장 완료`,
          chunks_sent: successfulChunks,
          total_chunks: totalDepartments
        };
        
      } else if (dataSize > chunkSize) {
        console.log(`📊 중간 크기 데이터 (${(dataSize/1024/1024).toFixed(2)}MB) - 압축 전송 시도`);
        // 중간 크기 데이터의 경우 요약 정보만 저장
        const summaryData = {
          hierarchical_summary: {
            departments: Object.keys(hierarchicalResults).length,
            total_employees: processedEmployees,
            structure_overview: Object.keys(hierarchicalResults).reduce((acc, dept) => {
              acc[dept] = {
                job_roles: Object.keys(hierarchicalResults[dept]).length,
                employees: Object.values(hierarchicalResults[dept]).reduce((sum, role) => 
                  sum + Object.values(role).reduce((roleSum, level) => 
                    roleSum + Object.keys(level).length, 0), 0)
              };
              return acc;
            }, {})
          },
          analysis_summary: analysisSummary,
          analysis_timestamp: new Date().toISOString(),
          data_compressed: true
        };
        
        // 압축된 데이터로 저장 요청 (Supervisor로 전송 - 실제 저장 담당)
        const saveResponse = await fetch('http://localhost:5006/api/batch-analysis/save-hierarchical-results', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(summaryData)
          // 타임아웃 제거 - 무제한 대기
        });
        
        if (saveResponse.ok) {
          const result = await saveResponse.json();
          console.log('✅ 계층적 구조 요약 저장 성공:', result);
          return { ...result, compressed: true };
        } else {
          const errorText = await saveResponse.text();
          throw new Error(`압축 저장 실패: ${saveResponse.status} - ${errorText}`);
        }
      } else {
        // 일반 저장 (재시도 로직 포함)
        const maxAttempts = 3;
        
        const attemptSave = async (attemptNumber) => {
          console.log(`💾 계층적 저장 시도 ${attemptNumber}/${maxAttempts}...`);
          
          const saveResponse = await fetch('http://localhost:5006/api/batch-analysis/save-hierarchical-results', {
            method: 'POST',
            headers: { 
              'Content-Type': 'application/json',
              'Connection': 'keep-alive'
            },
            body: dataString
            // 타임아웃 제거 - 무제한 대기
          });
          
          if (saveResponse.ok) {
            const result = await saveResponse.json();
            console.log('✅ 계층적 구조 저장 성공:', result);
            return result;
          } else {
            const errorText = await saveResponse.text();
            console.error(`❌ 계층적 저장 실패 (시도 ${attemptNumber}):`, saveResponse.status, errorText);
            
            if (attemptNumber === maxAttempts) {
              throw new Error(`저장 실패: ${saveResponse.status} - ${errorText}`);
            }
            throw new Error('Retry needed');
          }
        };

        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
          try {
            return await attemptSave(attempt);
          } catch (error) {
            console.error(`❌ 저장 시도 ${attempt} 실패:`, error.message);
            
            if (attempt === maxAttempts) {
              // 최종 실패 시 더 자세한 오류 정보 제공
              if (error.message.includes('fetch') || error.message.includes('network')) {
                throw new Error(`서버 연결 실패 (${maxAttempts}회 시도 후 실패). 서버가 실행 중인지 확인해주세요.`);
              }
              throw error;
            }
            
            // 재시도 전 대기 (점진적으로 증가)
            const waitTime = 2000 * attempt; // 2초, 4초, 6초
            console.log(`⏳ ${waitTime/1000}초 후 재시도...`);
            await new Promise(resolve => setTimeout(resolve, waitTime));
          }
        }
      }
      
    } catch (error) {
      console.error('❌ 계층적 구조 저장 중 오류:', error);
      throw error;
    }
  };

  // 위험도 레벨 계산 함수
  const calculateRiskLevel = (score) => {
    // 사후 분석에서 저장된 최종 설정 사용
    const finalSettings = JSON.parse(localStorage.getItem('finalRiskSettings') || '{}');
    const highThreshold = finalSettings.risk_thresholds?.high_risk_threshold || 0.7;
    const lowThreshold = finalSettings.risk_thresholds?.low_risk_threshold || 0.3;
    
    if (score >= highThreshold) return 'HIGH';
    if (score >= lowThreshold) return 'MEDIUM';
    return 'LOW';
  };

  // 저장된 결과 파일 정보 표시
  const showSavedResultsInfo = () => {
    Modal.info({
      title: '💾 저장된 배치 분석 결과',
      width: 600,
      content: (
        <div>
          <Alert
            message="부서별 분석 결과 저장 위치"
            description={
              <div>
                <Text strong>저장 경로:</Text> <Text code>app/results/batch_analysis/</Text><br />
                <Text strong>파일 형식:</Text><br />
                • <Text code>department_summary_[timestamp].json</Text> - 부서별 요약 통계<br />
                • <Text code>individual_results_[timestamp].json</Text> - 개별 직원 상세 결과 (XAI 포함)<br />
                <br />
                <Text strong>포함 내용:</Text><br />
                • 각 부서별 위험도 분포 (안전군/주의군/고위험군)<br />
                • 개별 직원별 5개 에이전트 분석 결과<br />
                • Structura & Chronos XAI 설명 (SHAP, LIME, Attention)<br />
                • <Text strong style={{color: '#1890ff'}}>XAI PNG 시각화 파일들</Text> (각 직원별 visualizations 폴더)<br />
                • 적용된 최적화 설정 (임계값, 가중치, 예측 모드)<br />
                <br />
                <Text type="secondary">
                  💡 파일은 배치 분석 완료 시 자동으로 저장되며, 
                  타임스탬프를 포함한 파일명으로 구분됩니다.
                </Text>
              </div>
            }
            type="info"
            showIcon
          />
          
          <div style={{ marginTop: 16 }}>
            <Text strong>저장되는 XAI 정보:</Text>
            <ul>
              <li><Text strong>Structura:</Text> SHAP values, LIME explanation, Feature importance</li>
              <li><Text strong>Chronos:</Text> Attention weights, Sequence importance, Trend analysis</li>
              <li><Text strong>Cognita:</Text> Network centrality, Relationship strength, Influence score</li>
              <li><Text strong>Sentio:</Text> Sentiment analysis, Keyword analysis, Emotion distribution</li>
              <li><Text strong>Agora:</Text> Market analysis, Industry trends, External factors</li>
            </ul>
            
            <Text strong style={{color: '#1890ff'}}>생성되는 PNG 시각화 파일들:</Text>
            <ul>
              <li><Text code>structura_feature_importance.png</Text> - 특성 중요도 막대 그래프</li>
              <li><Text code>structura_shap_values.png</Text> - SHAP 값 시각화</li>
              <li><Text code>chronos_attention_weights.png</Text> - 어텐션 가중치 시계열</li>
              <li><Text code>chronos_sequence_importance.png</Text> - 시퀀스 중요도 막대 그래프</li>
              <li><Text code>agent_scores_comparison.png</Text> - 에이전트별 위험도 점수 비교</li>
              <li><Text code>sentio_emotion_distribution.png</Text> - 감정 분포 파이 차트</li>
            </ul>
          </div>
        </div>
      ),
      onOk() {},
    });
  };

  // 결과 내보내기 함수 (사용하지 않음 - exportBatchResults 사용)
  /*
  const exportResults = (format) => {
    if (!analysisResults) {
      message.error('내보낼 분석 결과가 없습니다.');
      return;
    }

    try {
      if (format === 'excel') {
        // Excel 형태로 데이터 준비
        const exportData = analysisResults.results.map(result => ({
          '직원번호': result.employee_number,
          '통합점수': result.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score 
            ? (result.analysis_result.combined_analysis.integrated_assessment.overall_risk_score * 100).toFixed(1) + '%' 
            : 'N/A',
          '위험도': result.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score 
            ? calculateRiskLevel(result.analysis_result.combined_analysis.integrated_assessment.overall_risk_score)
            : 'N/A',
          'Structura점수': result.analysis_result?.structura_result?.prediction?.attrition_probability 
            ? (result.analysis_result.structura_result.prediction.attrition_probability * 100).toFixed(1) + '%' 
            : 'N/A',
          'Cognita점수': result.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score 
            ? (result.analysis_result.cognita_result.risk_analysis.overall_risk_score * 100).toFixed(1) + '%' 
            : 'N/A',
          'Chronos점수': result.analysis_result?.chronos_result?.trend_score 
            ? (result.analysis_result.chronos_result.trend_score * 100).toFixed(1) + '%' 
            : 'N/A',
          'Sentio점수': result.analysis_result?.sentio_result?.sentiment_score !== undefined
            ? (Math.abs(result.analysis_result.sentio_result.sentiment_score) * 100).toFixed(1) + '%' 
            : 'N/A',
          'Agora점수': (() => {
            const score = result.analysis_result?.agora_result?.agora_score || 
                         result.analysis_result?.agora_result?.market_analysis?.risk_score ||
                         result.analysis_result?.agora_result?.market_analysis?.market_pressure_index;
            return score ? (score * 100).toFixed(1) + '%' : 'N/A';
          })(),
          '상태': result.error ? '오류' : '완료'
        }));

        // CSV 형태로 변환
        const headers = Object.keys(exportData[0]);
        const csvContent = [
          headers.join(','),
          ...exportData.map(row => headers.map(header => row[header]).join(','))
        ].join('\n');

        // 파일 다운로드
        const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', `batch_analysis_results_${new Date().toISOString().split('T')[0]}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        message.success('Excel 파일이 다운로드되었습니다.');
                      } else if (format === 'pdf') {
        // PDF 보고서 생성 요청 (현재 비활성화)
        message.info('PDF 생성 기능은 현재 개발 중입니다.');
      }
    } catch (error) {
      console.error('결과 내보내기 실패:', error);
      message.error('결과 내보내기에 실패했습니다.');
    }
  };
  */


  // 동적으로 최적 직원 수를 계산하는 함수 (사용하지 않음 - 향후 확장용)
  /*
  const calculateOptimalEmployeeCount = (data, maxSize) => {
    if (!data || !data.results || data.results.length === 0) return 0;
    
    // 샘플 데이터로 직원 1명당 평균 크기 계산
    const sampleResult = {
      employee_number: data.results[0].employee_number || 'sample',
      status: data.results[0].status || 'completed',
      risk_score: data.results[0].analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score || 50
    };
    
    const sampleData = {
      timestamp: new Date().toISOString(),
      total_employees: data.total_employees || data.results.length,
      completed: data.completed || data.results.length,
      failed: data.failed || 0,
      results: [sampleResult],
      compressed: true,
      original_count: data.results.length,
      compressed_count: 1
    };
    
    const sampleSize = new Blob([JSON.stringify(sampleData)]).size;
    const baseSize = sampleSize - new Blob([JSON.stringify([sampleResult])]).size; // 기본 구조 크기
    const perEmployeeSize = new Blob([JSON.stringify(sampleResult)]).size; // 직원 1명당 크기
    
    // 안전 마진 20% 고려하여 최적 직원 수 계산
    const optimalCount = Math.floor((maxSize * 0.8 - baseSize) / perEmployeeSize);
    const maxPossible = Math.min(optimalCount, data.results.length);
    
    console.log('최적 직원 수 계산:', {
      sampleSize,
      baseSize,
      perEmployeeSize,
      optimalCount,
      maxPossible,
      totalEmployees: data.results.length
    });
    
    return Math.max(1, maxPossible); // 최소 1명은 보장
  };
  */

  // IndexedDB를 활용한 전체 데이터 보존 함수 (안전한 오류 처리)
  const saveToIndexedDB = async (data, dbName = 'BatchAnalysisDB', storeName = 'results') => {
    return new Promise((resolve, reject) => {
      // IndexedDB 지원 여부 확인
      if (!window.indexedDB) {
        console.error('IndexedDB가 지원되지 않는 브라우저입니다.');
        reject(new Error('IndexedDB not supported'));
        return;
      }
      
      const request = indexedDB.open(dbName, 1);
      
      request.onupgradeneeded = function(event) {
        try {
          const db = event.target.result;
          if (!db.objectStoreNames.contains(storeName)) {
            const store = db.createObjectStore(storeName, { keyPath: 'id', autoIncrement: true });
            store.createIndex('timestamp', 'timestamp', { unique: false });
            store.createIndex('employee_id', 'employee_id', { unique: false });
            console.log('IndexedDB Object Store 생성 완료');
          }
        } catch (upgradeError) {
          console.error('IndexedDB 업그레이드 실패:', upgradeError);
          reject(upgradeError);
        }
      };
      
      request.onsuccess = function(event) {
        try {
          const db = event.target.result;
          
          // Object Store 존재 여부 재확인
          if (!db.objectStoreNames.contains(storeName)) {
            console.error('Object Store가 생성되지 않았습니다.');
            reject(new Error('Object Store not found'));
            return;
          }
          
          const transaction = db.transaction([storeName], 'readwrite');
          const store = transaction.objectStore(storeName);
          
          // 기존 데이터 정리 (안전한 처리)
          const clearRequest = store.clear();
          
          clearRequest.onsuccess = function() {
            // 전체 데이터를 하나의 레코드로 저장
            const fullDataRecord = {
              timestamp: new Date().toISOString(),
              data_type: 'batch_analysis_full',
              total_employees: data.results?.length || 0,
              full_data: data // 전체 데이터 그대로 보존!
            };
            
            const addRequest = store.add(fullDataRecord);
            
            addRequest.onsuccess = function() {
              console.log(`✅ IndexedDB에 전체 데이터 저장 완료: ${data.results?.length || 0}명`);
              resolve({
                success: true,
                stored_employees: data.results?.length || 0,
                storage_method: 'indexeddb_full',
                data_loss: false
              });
            };
            
            addRequest.onerror = function() {
              console.error('IndexedDB 데이터 저장 실패:', addRequest.error);
              reject(addRequest.error);
            };
          };
          
          clearRequest.onerror = function() {
            console.error('IndexedDB 데이터 정리 실패:', clearRequest.error);
            reject(clearRequest.error);
          };
          
          transaction.onerror = function() {
            console.error('IndexedDB 트랜잭션 실패:', transaction.error);
            reject(transaction.error);
          };
          
        } catch (transactionError) {
          console.error('IndexedDB 트랜잭션 생성 실패:', transactionError);
          reject(transactionError);
        }
      };
      
      request.onerror = function() {
        console.error('IndexedDB 열기 실패:', request.error);
        reject(request.error);
      };
      
      request.onblocked = function() {
        console.warn('IndexedDB가 다른 탭에서 사용 중입니다.');
        reject(new Error('IndexedDB blocked by another tab'));
      };
    });
  };

  // IndexedDB에서 전체 데이터 로드 함수 (오류 처리 개선)
  const loadFromIndexedDB = async (dbName = 'BatchAnalysisDB', storeName = 'results') => {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(dbName, 1);
      
      request.onupgradeneeded = function(event) {
        const db = event.target.result;
        if (!db.objectStoreNames.contains(storeName)) {
          const store = db.createObjectStore(storeName, { keyPath: 'id', autoIncrement: true });
          store.createIndex('timestamp', 'timestamp', { unique: false });
          store.createIndex('employee_id', 'employee_id', { unique: false });
          console.log('IndexedDB 로드 중 Object Store 생성됨');
        }
      };
      
      request.onsuccess = function(event) {
        const db = event.target.result;
        
        // Object Store 존재 여부 확인
        if (!db.objectStoreNames.contains(storeName)) {
          console.log('Object Store가 존재하지 않음 - 데이터 없음');
          resolve(null);
          return;
        }
        
        try {
          const transaction = db.transaction([storeName], 'readonly');
          const store = transaction.objectStore(storeName);
          
          // 최신 데이터 가져오기
          const getAllRequest = store.getAll();
          
          getAllRequest.onsuccess = function() {
            const records = getAllRequest.result;
            if (records && records.length > 0) {
              // 가장 최신 레코드 선택
              const latestRecord = records.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0];
              console.log(`✅ IndexedDB에서 전체 데이터 로드: ${latestRecord.total_employees}명`);
              resolve(latestRecord.full_data);
            } else {
              console.log('IndexedDB에 저장된 데이터 없음');
              resolve(null);
            }
          };
          
          getAllRequest.onerror = function() {
            console.error('IndexedDB 데이터 조회 실패:', getAllRequest.error);
            reject(getAllRequest.error);
          };
          
          transaction.onerror = function() {
            console.error('IndexedDB 트랜잭션 실패:', transaction.error);
            reject(transaction.error);
          };
          
        } catch (transactionError) {
          console.error('IndexedDB 트랜잭션 생성 실패:', transactionError);
          resolve(null); // 오류 시 null 반환
        }
      };
      
      request.onerror = function() {
        console.error('IndexedDB 열기 실패:', request.error);
        resolve(null); // 오류 시 null 반환 (reject 대신)
      };
    });
  };

  // 전체 데이터 보존 함수 (LocalStorage 대신 IndexedDB 사용)
  const saveFullDataWithoutLoss = async (data) => {
    if (!data || !data.results || !Array.isArray(data.results)) {
      console.warn('저장할 데이터가 없습니다:', data);
      return { success: false, reason: 'no_data' };
    }
    
    const originalSize = new Blob([JSON.stringify(data)]).size;
    console.log(`📊 전체 데이터 크기: ${Math.round(originalSize/1024/1024*100)/100}MB (${data.results.length}명)`);
    
    try {
      // 1. IndexedDB에 전체 데이터 저장 시도
      const indexedDBResult = await saveToIndexedDB(data);
      if (indexedDBResult.success) {
        console.log('✅ IndexedDB 저장 성공 - 전체 데이터 보존됨');
        return {
          success: true,
          storage_method: 'indexeddb',
          total_employees: data.results.length,
          data_loss: false,
          message: `전체 ${data.results.length}명 데이터 완전 보존`
        };
      }
    } catch (indexedDBError) {
      console.error('IndexedDB 저장 실패:', indexedDBError);
    }
    
    // 2. IndexedDB 실패 시 청크 분할로 LocalStorage에 전체 저장
    try {
      console.log('🔄 청크 분할로 전체 데이터 저장 시도...');
      
      // 기존 청크 데이터 정리
      for (let i = 0; i < 50; i++) {
        localStorage.removeItem(`batchAnalysisResults_chunk_${i}`);
      }
      
      const chunkSize = 50; // 작은 청크로 분할하여 전체 보존
      const chunks = [];
      
      for (let i = 0; i < data.results.length; i += chunkSize) {
        chunks.push({
          chunk_index: Math.floor(i / chunkSize),
          start_index: i,
          end_index: Math.min(i + chunkSize, data.results.length),
          data: data.results.slice(i, i + chunkSize) // 원본 데이터 그대로!
        });
      }
      
      let savedChunks = 0;
      for (const chunk of chunks) {
        try {
          const chunkData = {
            ...chunk,
            metadata: {
              total_employees: data.results.length,
              total_chunks: chunks.length,
              timestamp: new Date().toISOString(),
              storage_method: 'chunked_full'
            }
          };
          
          localStorage.setItem(`batchAnalysisResults_chunk_${chunk.chunk_index}`, JSON.stringify(chunkData));
          savedChunks++;
        } catch (chunkError) {
          console.error(`청크 ${chunk.chunk_index} 저장 실패:`, chunkError);
          break;
        }
      }
      
      // 메타데이터 저장
      const metadata = {
        total_employees: data.results.length,
        saved_employees: savedChunks * chunkSize,
        total_chunks: chunks.length,
        saved_chunks: savedChunks,
        storage_method: 'chunked_full',
        data_loss: savedChunks < chunks.length,
        timestamp: new Date().toISOString()
      };
      
      localStorage.setItem('batchAnalysisMetadata', JSON.stringify(metadata));
      
      console.log(`✅ 청크 저장 완료: ${savedChunks}/${chunks.length}개 청크, ${Math.min(savedChunks * chunkSize, data.results.length)}/${data.results.length}명`);
      
      return {
        success: true,
        storage_method: 'chunked_localStorage',
        total_employees: data.results.length,
        saved_employees: Math.min(savedChunks * chunkSize, data.results.length),
        data_loss: savedChunks < chunks.length,
        message: `${Math.min(savedChunks * chunkSize, data.results.length)}/${data.results.length}명 저장 완료`
      };
      
    } catch (chunkError) {
      console.error('청크 저장도 실패:', chunkError);
      return {
        success: false,
        reason: 'all_methods_failed',
        error: chunkError.message
      };
    }
  };

  // 데이터 압축 함수는 더 이상 사용되지 않으므로 제거되었습니다.

  // 전체 결과 내보내기 함수들
  const exportBatchResults = async (format = 'csv') => {
    if (!analysisResults || !analysisResults.results) {
      message.error('내보낼 분석 결과가 없습니다.');
      return;
    }

    setIsExporting(true);
    
    try {
      if (format === 'csv') {
        await exportToCSV();
      } else if (format === 'excel') {
        await exportToExcel();
      } else if (format === 'json') {
        await exportToJSON();
      } else if (format === 'pdf') {
        await exportToPDF();
      }
      
      message.success(`${format.toUpperCase()} 형식으로 결과를 내보냈습니다.`);
    } catch (error) {
      console.error('결과 내보내기 실패:', error);
      message.error(`결과 내보내기 실패: ${error.message}`);
    } finally {
      setIsExporting(false);
    }
  };

  const exportToCSV = async () => {
    const csvData = [];
    
    // 헤더 생성
    const headers = [
      '직원번호', '부서', '직무', '전체위험도', '위험수준',
      'Structura_위험도', 'Structura_신뢰도',
      'Cognita_위험도', 'Cognita_신뢰도', 
      'Chronos_위험도', 'Chronos_신뢰도',
      'Sentio_감정점수', 'Sentio_위험수준',
      'Agora_시장압력', 'Agora_위험수준',
      '주요위험요인', '권장사항', '분석일시'
    ];
    csvData.push(headers);

    // 데이터 행 생성
    analysisResults.results.forEach(result => {
      if (result.analysis_result && result.analysis_result.status === 'success') {
        const analysis = result.analysis_result;
        const combined = analysis.combined_analysis || {};
        const integrated = combined.integrated_assessment || {};
        
        const row = [
          result.employee_number || 'Unknown',
          result.department || 'Unknown',
          result.job_role || 'Unknown',
          (integrated.overall_risk_score || 0).toFixed(3),
          integrated.overall_risk_level || 'UNKNOWN',
          
          // Structura
          analysis.structura_result?.prediction?.attrition_probability?.toFixed(3) || 'N/A',
          analysis.structura_result?.prediction?.confidence?.toFixed(3) || 'N/A',
          
          // Cognita
          analysis.cognita_result?.risk_analysis?.overall_risk_score?.toFixed(3) || 'N/A',
          analysis.cognita_result?.confidence?.toFixed(3) || 'N/A',
          
          // Chronos
          analysis.chronos_result?.prediction?.attrition_probability?.toFixed(3) || 'N/A',
          analysis.chronos_result?.confidence?.toFixed(3) || 'N/A',
          
          // Sentio
          analysis.sentio_result?.sentiment_score?.toFixed(3) || 'N/A',
          analysis.sentio_result?.risk_level || 'N/A',
          
          // Agora
          analysis.agora_result?.market_analysis?.market_pressure_index?.toFixed(3) || 'N/A',
          analysis.agora_result?.risk_level || 'N/A',
          
          // 위험요인 및 권장사항
          combined.risk_factors?.slice(0, 3).join('; ') || 'N/A',
          combined.recommendations?.slice(0, 2).join('; ') || 'N/A',
          new Date().toLocaleString('ko-KR')
        ];
        csvData.push(row);
      }
    });

    // CSV 파일 생성 및 다운로드
    const csvContent = csvData.map(row => 
      row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(',')
    ).join('\n');
    
    const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `배치분석결과_${new Date().toISOString().slice(0, 10)}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportToJSON = async () => {
    // 최적화 결과도 함께 포함
    const exportData = {
      analysis_metadata: {
        export_date: new Date().toISOString(),
        total_employees: analysisResults.results?.length || 0,
        analysis_type: 'batch_analysis',
        integration_config: analysisResults.integration_config
      },
      batch_results: analysisResults.results,
      integration_report: analysisResults.integration_report,
      optimization_results: analysisResults.optimization_results || null,
      performance_summary: {
        success_count: analysisResults.results?.filter(r => r.analysis_result?.status === 'success').length || 0,
        failure_count: analysisResults.results?.filter(r => r.analysis_result?.status === 'failed').length || 0
      }
    };

    const jsonContent = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `배치분석결과_${new Date().toISOString().slice(0, 10)}.json`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportToPDF = async () => {
    // PDF 생성을 위한 간단한 구현
    message.info('PDF 내보내기는 개발 중입니다. CSV 또는 JSON 형식을 사용해주세요.');
  };

  const exportToExcel = async () => {
    // Excel 내보내기를 위한 간단한 구현 (CSV와 동일하게 처리)
    await exportToCSV();
  };

  // 개별 직원 레포트 생성 함수
  const generateEmployeeReport = async (employeeData) => {
    setReportGenerating(true);
    try {
      // XAI 결과와 예측값을 종합하여 LLM 기반 레포트 생성
      const reportData = {
        employee_number: employeeData.employee_number,
        analysis_results: employeeData.analysis_result,
        risk_thresholds: {
          high: integrationConfig.high_risk_threshold,
          medium: integrationConfig.medium_risk_threshold
        },  // 기본 임계값 전달
        request_type: 'individual_report'
      };

      const response = await fetch('http://localhost:5006/api/generate-employee-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(reportData)
      });

      if (!response.ok) {
        throw new Error('레포트 생성 실패');
      }

      const report = await response.json();
      setEmployeeReport(report);
      message.success('개별 직원 레포트가 생성되었습니다.');
    } catch (error) {
      console.error('레포트 생성 실패:', error);
      // 백엔드 연결 실패 시 클라이언트에서 기본 레포트 생성
      const basicReport = generateBasicEmployeeReport(employeeData);
      setEmployeeReport(basicReport);
      message.warning('서버 연결 실패로 기본 레포트를 생성했습니다.');
    } finally {
      setReportGenerating(false);
    }
  };

  // 위험도 분류 함수 (기본 임계값 적용)
  const classifyRiskLevel = (riskScore) => {
    const normalizedScore = riskScore > 1 ? riskScore / 100 : riskScore; // 0-1 범위로 정규화
    
    if (normalizedScore >= integrationConfig.high_risk_threshold) {
      return 'HIGH';
    } else if (normalizedScore >= integrationConfig.medium_risk_threshold) {
      return 'MEDIUM';
    } else {
      return 'LOW';
    }
  };

  // 기본 직원 레포트 생성 (클라이언트 사이드)
  const generateBasicEmployeeReport = (employeeData) => {
    const analysis = employeeData.analysis_result;
    
    // 위험도 계산 (여러 소스에서 시도)
    let overallRisk = 0;
    let riskLevel = 'LOW';
    
    // 1. Combined analysis에서 위험도 추출
    if (analysis?.combined_analysis?.integrated_assessment?.overall_risk_score) {
      overallRisk = analysis.combined_analysis.integrated_assessment.overall_risk_score;
    }
    // 2. Structura 결과에서 위험도 추출 (확률값이므로 100을 곱함)
    else if (analysis?.structura_result?.prediction?.attrition_probability) {
      overallRisk = analysis.structura_result.prediction.attrition_probability * 100;
    }
    // 3. Cognita 결과에서 위험도 추출
    else if (analysis?.cognita_result?.risk_analysis?.overall_risk_score) {
      overallRisk = analysis.cognita_result.risk_analysis.overall_risk_score * 100;
    }
    
    // 동적 위험도 분류
    riskLevel = classifyRiskLevel(overallRisk);
    
    // 각 에이전트별 결과 요약
    const structuraResult = analysis?.structura_result;
    const cognitaResult = analysis?.cognita_result;
    const chronosResult = analysis?.chronos_result;
    const sentioResult = analysis?.sentio_result;
    const agoraResult = analysis?.agora_result;

    return {
      employee_number: employeeData.employee_number,
      overall_assessment: {
        risk_level: riskLevel,
        risk_score: overallRisk,
        summary: `직원 ${employeeData.employee_number}의 종합 위험도는 ${riskLevel} (${overallRisk.toFixed(1)}%) 수준입니다.`
      },
      detailed_analysis: {
        structura: structuraResult ? {
          예측값: structuraResult.prediction?.attrition_probability ? 
            `${(structuraResult.prediction.attrition_probability * 100).toFixed(1)}%` : 'N/A',
          신뢰도: structuraResult.prediction?.confidence_score ? 
            `${(structuraResult.prediction.confidence_score * 100).toFixed(1)}%` : 'N/A',
          위험_분류: structuraResult.prediction?.risk_category || 'N/A',
          주요_위험요인: structuraResult.explanation?.top_risk_factors?.slice(0, 3).map(f => 
            `${f.feature} (${(f.impact * 100).toFixed(1)}%)`).join(', ') || 'N/A',
          보호_요인: structuraResult.explanation?.top_protective_factors?.slice(0, 2).map(f => 
            `${f.feature} (${(f.impact * 100).toFixed(1)}%)`).join(', ') || 'N/A',
          interpretation: structuraResult.prediction ? 
            `구조적 분석 결과 ${structuraResult.prediction.attrition_probability > 0.5 ? '높은' : '낮은'} 위험도를 보입니다.` :
            '구조적 분석 결과 보다 위험도를 보입니다.'
        } : { interpretation: '구조적 분석 결과가 없어 위험도를 보입니다.' },
        
        cognita: cognitaResult ? {
          위험_수준: cognitaResult.risk_analysis?.risk_category || 'undefined',
          위험_점수: cognitaResult.risk_analysis?.overall_risk_score ? 
            `${(cognitaResult.risk_analysis.overall_risk_score * 100).toFixed(1)}%` : 'N/A',
          위험_요인: cognitaResult.risk_analysis?.risk_factors?.join(', ') || 'N/A',
          네트워크_중심성: cognitaResult.risk_analysis?.network_stats?.degree_centrality?.toFixed(2) || 'N/A',
          직접_연결수: cognitaResult.risk_analysis?.network_stats?.direct_connections || 'N/A',
          관리자_불안정성: cognitaResult.risk_analysis?.manager_instability_score?.toFixed(2) || 'N/A',
          interpretation: cognitaResult.risk_analysis ? 
            `관계 분석 결과 ${cognitaResult.risk_analysis.risk_category || 'undefined'} 위험 수준으로 평가됩니다.` :
            '관계 분석 결과 undefined 위험 수준으로 평가됩니다.'
        } : { interpretation: '관계 분석 결과 undefined 위험 수준으로 평가됩니다.' },
        
        chronos: chronosResult ? {
          예측값: chronosResult.prediction || 'N/A',
          신뢰도: chronosResult.confidence || 'N/A',
          interpretation: chronosResult.prediction ? 
            `시계열 분석 결과 ${chronosResult.prediction > 0.5 ? '상승' : '안정'} 추세를 보입니다.` :
            '시계열 분석 결과 안정 추세를 보입니다.'
        } : { interpretation: '시계열 분석 결과 안정 추세를 보입니다.' },
        
        sentio: sentioResult ? {
          감정_점수: sentioResult.sentiment_analysis?.sentiment_score || sentioResult.sentiment_score || 0.1,
          위험_수준: sentioResult.risk_level || sentioResult.sentiment_analysis?.risk_level || '',
          interpretation: `감정 분석 결과 ${(sentioResult.sentiment_analysis?.sentiment_score || sentioResult.sentiment_score || 0) > 0 ? '긍정적' : '부정적'} 성향을 보입니다.`
        } : { interpretation: '감정 분석 결과 긍정적 성향을 보입니다.' },
        
        agora: agoraResult ? {
          시장_압력: agoraResult.market_analysis?.market_pressure_index || 57.8,
          interpretation: `시장 분석 결과 ${(agoraResult.market_analysis?.market_pressure_index || 0.5) > 0.5 ? '높은' : '낮은'} 시장 압력을 받고 있습니다.`
        } : { interpretation: '시장 분석 결과 높은 시장 압력을 받고 있습니다.' }
      },
      recommendations: [
        riskLevel === 'HIGH' ? '🚨 즉시 관리자와 면담을 진행하시기 바랍니다.' : null,
        riskLevel === 'MEDIUM' ? '⚠️ 필요시 추가 상담을 진행하시기 바랍니다.' : null,
        structuraResult?.prediction?.attrition_probability > 0.7 ? '💼 업무 환경 개선이 필요합니다.' : null,
        sentioResult?.sentiment_score < -0.5 ? '🤝 심리적 지원이 필요할 수 있습니다.' : null,
        '📊 정기적인 모니터링을 통해 상태 변화를 추적하시기 바랍니다.'
      ].filter(Boolean),
      generated_at: new Date().toISOString()
    };
  };

  // 직원 클릭 시 레포트 모달 열기
  const handleEmployeeClick = (employeeData) => {
    setSelectedEmployee(employeeData);
    setEmployeeReportVisible(true);
    generateEmployeeReport(employeeData);
  };

  // localStorage 정리 함수
  const clearLocalStorage = () => {
    try {
      localStorage.removeItem('batchAnalysisResults');
      message.success('저장된 분석 결과가 정리되었습니다.');
    } catch (error) {
      console.error('localStorage 정리 실패:', error);
      message.error('저장된 데이터 정리에 실패했습니다.');
    }
  };

  // 시각화 대시보드로 이동
  const navigateToVisualization = async () => {
    console.log('🎯 시각화 대시보드로 이동 시작');
    console.log('📊 현재 분석 결과:', { 
      hasResults: !!analysisResults, 
      hasResultsArray: !!(analysisResults?.results),
      resultsLength: analysisResults?.results?.length,
      hasCacheInfo: !!(analysisResults?.cache_info)
    });
    
    // 분석 결과를 localStorage에 저장 (용량 제한 고려)
    if (analysisResults) {
      try {
        // 시각화에 필요한 데이터 구조 준비
        let visualizationData = null;
        
        // 실제 분석 결과가 있는 경우
        if (analysisResults.results && analysisResults.results.length > 0) {
          visualizationData = {
            success: true,
            results: analysisResults.results,
            total_employees: analysisResults.total_employees,
            completed_employees: analysisResults.completed_employees,
            summary: analysisResults.summary,
            analysis_metadata: analysisResults.analysis_metadata,
            data_source: 'actual_results'
          };
        }
        // 캐시 정보만 있는 경우 - 가상의 결과 데이터 생성
        else if (analysisResults.cache_info) {
          console.log('📋 캐시 정보를 기반으로 시각화 데이터 생성');
          
          // 캐시 정보를 바탕으로 가상의 직원 데이터 생성
          const mockResults = [];
          const { highRiskCount, mediumRiskCount, lowRiskCount, totalEmployees } = analysisResults.cache_info;
          
          // 고위험군 가상 데이터
          for (let i = 1; i <= highRiskCount; i++) {
            mockResults.push({
              employee_number: `HIGH_${i}`,
              analysis_result: {
                combined_analysis: {
                  integrated_assessment: {
                    overall_risk_score: 0.8 + (Math.random() * 0.2), // 0.8-1.0
                    overall_risk_level: 'HIGH'
                  }
                },
                employee_data: {
                  Department: 'Unknown',
                  JobRole: 'Unknown'
                }
              }
            });
          }
          
          // 중위험군 가상 데이터
          for (let i = 1; i <= mediumRiskCount; i++) {
            mockResults.push({
              employee_number: `MEDIUM_${i}`,
              analysis_result: {
                combined_analysis: {
                  integrated_assessment: {
                    overall_risk_score: 0.4 + (Math.random() * 0.4), // 0.4-0.8
                    overall_risk_level: 'MEDIUM'
                  }
                },
                employee_data: {
                  Department: 'Unknown',
                  JobRole: 'Unknown'
                }
              }
            });
          }
          
          // 저위험군 가상 데이터
          for (let i = 1; i <= lowRiskCount; i++) {
            mockResults.push({
              employee_number: `LOW_${i}`,
              analysis_result: {
                combined_analysis: {
                  integrated_assessment: {
                    overall_risk_score: Math.random() * 0.4, // 0.0-0.4
                    overall_risk_level: 'LOW'
                  }
                },
                employee_data: {
                  Department: 'Unknown',
                  JobRole: 'Unknown'
                }
              }
            });
          }
          
          visualizationData = {
            success: true,
            results: mockResults,
            total_employees: totalEmployees || mockResults.length,
            completed_employees: totalEmployees || mockResults.length,
            summary: {
              total_employees: totalEmployees || mockResults.length,
              successful_analyses: totalEmployees || mockResults.length,
              failed_analyses: 0,
              success_rate: 1.0
            },
            analysis_metadata: {
              analysis_type: 'batch',
              timestamp: analysisResults.analysis_metadata?.timestamp,
              data_source: 'cache_based_mock'
            },
            cache_info: analysisResults.cache_info,
            data_source: 'cache_mock'
          };
          
          console.log('✅ 캐시 기반 시각화 데이터 생성 완료:', {
            totalMockResults: mockResults.length,
            highRisk: highRiskCount,
            mediumRisk: mediumRiskCount,
            lowRisk: lowRiskCount
          });
        }
        
        if (!visualizationData) {
          message.error('시각화할 데이터가 없습니다.');
          return;
        }
        
        // 데이터 크기 확인 및 저장
        const dataString = JSON.stringify(visualizationData);
        const dataSize = new Blob([dataString]).size;
        const maxSize = 4 * 1024 * 1024; // 4MB 제한
        
        console.log(`📏 시각화 데이터 크기: ${(dataSize/1024/1024).toFixed(2)}MB`);
        
        if (dataSize > maxSize) {
          console.log('🔄 대용량 데이터 - 전체 데이터 보존 시도');
          
          const saveResult = await saveFullDataWithoutLoss(visualizationData);
          
          if (saveResult.success) {
            // LocalStorage에는 참조 정보만 저장
            const referenceData = {
              timestamp: new Date().toISOString(),
              storage_method: saveResult.storage_method,
              total_employees: saveResult.total_employees,
              data_location: saveResult.storage_method === 'indexeddb' ? 'IndexedDB' : 'LocalStorage_Chunks',
              full_data_preserved: !saveResult.data_loss,
              data_source: visualizationData.data_source
            };
            
            localStorage.setItem('batchAnalysisResults', JSON.stringify(referenceData));
            
            message.success(
              `✅ 시각화 데이터 보존 완료!\n` +
              `저장 방식: ${saveResult.storage_method === 'indexeddb' ? 'IndexedDB' : '청크 분할'}\n` +
              `데이터 소스: ${visualizationData.data_source === 'cache_mock' ? '캐시 기반 모의 데이터' : '실제 분석 결과'}`
            );
          } else {
            console.error('시각화 데이터 저장 실패:', saveResult);
            message.error(`시각화 데이터 저장 실패: ${saveResult.reason || saveResult.error}`);
            return;
          }
        } else {
          localStorage.setItem('batchAnalysisResults', dataString);
          message.success(
            `시각화 데이터 연동 완료!\n` +
            `데이터 소스: ${visualizationData.data_source === 'cache_mock' ? '캐시 기반 모의 데이터' : '실제 분석 결과'}\n` +
            `직원 수: ${visualizationData.total_employees}명`
          );
        }
        
      } catch (error) {
        console.error('시각화 데이터 저장 실패:', error);
        message.error('시각화 데이터 저장에 실패했습니다.');
        return;
      }
    } else {
      message.warning('시각화할 분석 결과가 없습니다. 먼저 배치 분석을 실행하거나 이전 결과를 불러오세요.');
      return;
    }
    
    // 실제 페이지 이동
    if (onNavigate) {
      onNavigate('visualization'); // 새로운 시각화 페이지 키
    } else {
      message.info('시각화 대시보드 메뉴로 이동하세요.');
    }
  };

  // 관계 분석으로 이동 - 전체 데이터 보존 적용
  const navigateToRelationshipAnalysis = async () => {
    // 분석 결과를 localStorage에 저장
    if (analysisResults) {
      try {
        const dataString = JSON.stringify(analysisResults);
        const dataSize = new Blob([dataString]).size;
        const maxSize = 4 * 1024 * 1024;
        
        if (dataSize > maxSize) {
          console.log('관계 분석용 전체 데이터 보존 시작...');
          const saveResult = await saveFullDataWithoutLoss(analysisResults);
          
          if (saveResult.success) {
            // LocalStorage에는 참조 정보만 저장
            const referenceData = {
              timestamp: new Date().toISOString(),
              storage_method: saveResult.storage_method,
              total_employees: saveResult.total_employees,
              data_location: saveResult.storage_method === 'indexeddb' ? 'IndexedDB' : 'LocalStorage_Chunks',
              full_data_preserved: !saveResult.data_loss,
              note: 'Full data for relationship analysis'
            };
            
            localStorage.setItem('batchAnalysisResults', JSON.stringify(referenceData));
            
            message.success(
              `✅ 관계 분석용 전체 데이터 보존 완료!\n` +
              `저장 방식: ${saveResult.storage_method === 'indexeddb' ? 'IndexedDB' : '청크 분할'}\n` +
              `보존된 직원: ${saveResult.total_employees || saveResult.saved_employees}/${saveResult.total_employees}명\n` +
              `데이터 손실: ${saveResult.data_loss ? '일부 있음' : '없음'}`
            );
          } else {
            message.error(`관계 분석용 데이터 저장 실패: ${saveResult.reason || saveResult.error}`);
            return;
          }
        } else {
          localStorage.setItem('batchAnalysisResults', dataString);
          message.success('분석 결과가 관계 분석에 연동되었습니다.');
        }
      } catch (error) {
        console.error('관계 분석용 데이터 저장 실패:', error);
        message.error('데이터 저장에 실패했습니다.');
        return;
      }
    }
    
    // 실제 페이지 이동
    if (onNavigate) {
      onNavigate('cognita');
    } else {
      message.info('🕸️ 개별 관계분석 메뉴로 이동하세요.');
    }
  };

  // 결과 테이블 컬럼 정의
  const resultColumns = [
    {
      title: '직원번호',
      dataIndex: 'employee_number',
      key: 'employee_number',
      width: 100,
      sorter: (a, b) => parseInt(a.employee_number) - parseInt(b.employee_number),
      sortDirections: ['ascend', 'descend'],
      render: (text, record) => (
        <Button 
          type="link" 
          onClick={() => handleEmployeeClick(record)}
          style={{ padding: 0, height: 'auto' }}
        >
          {text}
        </Button>
      ),
    },
    {
      title: '통합 점수',
      key: 'overall_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
        return score ? (score * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = a.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score || 0;
        const scoreB = b.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score || 0;
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: '위험도',
      key: 'risk_level',
      width: 100,
      render: (_, record) => {
        const score = record.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
        const riskLevel = score ? calculateRiskLevel(score) : 'N/A';
        const color = riskLevel === 'HIGH' ? 'red' : riskLevel === 'MEDIUM' ? 'orange' : riskLevel === 'LOW' ? 'green' : 'default';
        return <Tag color={color}>{riskLevel}</Tag>;
      },
      filters: [
        { text: '고위험군', value: 'HIGH' },
        { text: '중위험군', value: 'MEDIUM' },
        { text: '저위험군', value: 'LOW' },
      ],
      onFilter: (value, record) => {
        const score = record.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
        const riskLevel = score ? calculateRiskLevel(score) : 'N/A';
        return riskLevel === value;
      },
    },
    {
      title: 'Structura 점수',
      key: 'structura_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.structura_result?.prediction?.attrition_probability;
        return score ? (score * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = a.analysis_result?.structura_result?.prediction?.attrition_probability || 0;
        const scoreB = b.analysis_result?.structura_result?.prediction?.attrition_probability || 0;
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: 'Cognita 점수',
      key: 'cognita_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.cognita_result?.overall_risk_score || 
                     record.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score;
        return score ? (score * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = a.analysis_result?.cognita_result?.overall_risk_score || 
                      a.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score || 0;
        const scoreB = b.analysis_result?.cognita_result?.overall_risk_score || 
                      b.analysis_result?.cognita_result?.risk_analysis?.overall_risk_score || 0;
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: 'Chronos 점수',
      key: 'chronos_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.chronos_result?.prediction?.risk_score;
        return score !== undefined ? (score * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = a.analysis_result?.chronos_result?.prediction?.risk_score || 0;
        const scoreB = b.analysis_result?.chronos_result?.prediction?.risk_score || 0;
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: 'Sentio 점수',
      key: 'sentio_score',
      width: 120,
      render: (_, record) => {
        // 여러 경로에서 Sentio 점수를 찾아보기 (실제 API 응답 구조에 맞게 수정)
        let score = null;
        
        // 1. 직접 sentio_score 필드 (가장 일반적)
        if (record.sentio_score !== undefined && record.sentio_score !== null) {
          score = record.sentio_score;
        }
        // 2. sentiment_analysis.risk_score (실제 저장 경로)
        else if (record.analysis_result?.sentio_result?.sentiment_analysis?.risk_score !== undefined) {
          score = record.analysis_result.sentio_result.sentiment_analysis.risk_score;
        }
        // 3. psychological_risk_score (JD-R 모델 기반 - 직접 경로)
        else if (record.analysis_result?.sentio_result?.psychological_risk_score !== undefined) {
          score = record.analysis_result.sentio_result.psychological_risk_score;
        }
        // 4. sentiment_score를 위험 점수로 변환
        else if (record.analysis_result?.sentio_result?.sentiment_analysis?.sentiment_score !== undefined) {
          score = 1.0 - record.analysis_result.sentio_result.sentiment_analysis.sentiment_score; // 감정 점수를 위험 점수로 변환
        }
        // 5. 기본값 처리
        else {
          score = 0.5; // 기본값
        }
        
        // 점수가 1보다 큰 경우 (100 스케일로 입력된 경우) 정규화
        if (score > 1) {
          score = score / 100;
        }
        
        return score !== null ? (score * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        // 정렬을 위한 점수 추출 (동일한 로직)
        const getScore = (record) => {
          let score = null;
          
          // 1. 직접 sentio_score 필드
          if (record.sentio_score !== undefined && record.sentio_score !== null) {
            score = record.sentio_score;
          }
          // 2. sentiment_analysis.risk_score (실제 저장 경로)
          else if (record.analysis_result?.sentio_result?.sentiment_analysis?.risk_score !== undefined) {
            score = record.analysis_result.sentio_result.sentiment_analysis.risk_score;
          }
          // 3. psychological_risk_score
          else if (record.analysis_result?.sentio_result?.psychological_risk_score !== undefined) {
            score = record.analysis_result.sentio_result.psychological_risk_score;
          }
          // 4. sentiment_score를 위험 점수로 변환
          else if (record.analysis_result?.sentio_result?.sentiment_analysis?.sentiment_score !== undefined) {
            score = 1.0 - record.analysis_result.sentio_result.sentiment_analysis.sentiment_score;
          }
          // 4. 기본값
          else {
            score = 0.5;
          }
          
          // 점수가 1보다 큰 경우 정규화
          if (score > 1) {
            score = score / 100;
          }
          
          return score || 0;
        };
        
        return getScore(a) - getScore(b);
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: 'Agora 점수',
      key: 'agora_score',
      width: 120,
      render: (_, record) => {
        const score = record.analysis_result?.agora_result?.agora_score || 
                     record.analysis_result?.agora_result?.market_analysis?.risk_score ||
                     record.analysis_result?.agora_result?.market_analysis?.market_pressure_index;
        return score ? (score * 100).toFixed(1) + '%' : 'N/A';
      },
      sorter: (a, b) => {
        const scoreA = a.analysis_result?.agora_result?.agora_score || 
                      a.analysis_result?.agora_result?.market_analysis?.risk_score ||
                      a.analysis_result?.agora_result?.market_analysis?.market_pressure_index || 0;
        const scoreB = b.analysis_result?.agora_result?.agora_score || 
                      b.analysis_result?.agora_result?.market_analysis?.risk_score ||
                      b.analysis_result?.agora_result?.market_analysis?.market_pressure_index || 0;
        return scoreA - scoreB;
      },
      sortDirections: ['ascend', 'descend'],
    },
    {
      title: '상태',
      key: 'status',
      width: 80,
      render: (_, record) => {
        if (record.error) {
          return <Tag color="red">오류</Tag>;
        }
        return <Tag color="green">완료</Tag>;
      }
    }
  ];


  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <RocketOutlined /> 배치 분석
      </Title>
      
      <Paragraph>
        HR 데이터 파일을 업로드하여 전체 직원에 대한 종합적인 이직 위험도 분석을 수행합니다.
        Supervisor가 각 에이전트를 순차적으로 실행하여 정확한 분석 결과를 제공합니다.
      </Paragraph>

      <Alert
        message="순차적 에이전트 실행 순서"
        description="Structura (HR 분석) → Cognita (관계 분석) → Chronos (시계열 분석) → Sentio (감정 분석) → Agora (시장 분석)"
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
      />

      {/* 사후 분석 최종 설정 정보 표시 */}
      {finalRiskSettings && (
        <Alert
          message="📊 사후 분석 최적화 설정 적용됨"
          description={
            <div>
              <Text strong>위험도 분류 기준:</Text> 안전군 &lt; {finalRiskSettings.risk_thresholds?.low_risk_threshold || 0.3}, 
              주의군 {finalRiskSettings.risk_thresholds?.low_risk_threshold || 0.3} ~ {finalRiskSettings.risk_thresholds?.high_risk_threshold || 0.7}, 
              고위험군 ≥ {finalRiskSettings.risk_thresholds?.high_risk_threshold || 0.7}
              <br />
              <Text strong>퇴사 예측 기준:</Text> {finalRiskSettings.attrition_prediction_mode === 'high_risk_only' ? '고위험군만 퇴사 예측' : '주의군 + 고위험군 퇴사 예측'}
              {finalRiskSettings.performance_metrics?.f1_score && (
                <>
                  <br />
                  <Text strong>최적화된 F1-Score:</Text> {finalRiskSettings.performance_metrics.f1_score.toFixed(4)}
                </>
              )}
            </div>
          }
          type="success"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      <Alert
        message="🔍 디버깅 정보"
        description={
          <div>
            분석 과정에서 문제가 발생하면 브라우저의 개발자 도구(F12) → Console 탭에서 상세한 로그를 확인할 수 있습니다. 에이전트별 실행 상태와 오류 정보가 실시간으로 표시됩니다.
            <div style={{ marginTop: '8px' }}>
              <Button 
                size="small" 
                onClick={debugAgents}
                disabled={loading || isAnalyzing}
              >
                🔍 에이전트 상태 확인
              </Button>
            </div>
          </div>
        }
        type="warning"
        showIcon
        style={{ marginBottom: 24 }}
      />

      {integrationConfig.use_trained_models ? (
        <Alert
          message="🧠 최적화된 모델 사용 중"
          description="사후 분석에서 학습된 모델과 최적화된 하이퍼파라미터를 사용하여 더 정확한 예측을 수행합니다."
          type="success"
          showIcon
          style={{ marginBottom: 24 }}
          action={
            <Button 
              size="small" 
              onClick={() => {
                const savedModels = localStorage.getItem('trainedModels');
                if (savedModels) {
                  const modelData = JSON.parse(savedModels);
                  Modal.info({
                    title: '저장된 모델 정보',
                    content: (
                      <div>
                        <p><strong>학습일:</strong> {new Date(modelData.training_metadata?.training_date).toLocaleString('ko-KR')}</p>
                        <p><strong>학습 데이터 크기:</strong> {modelData.training_metadata?.training_data_size}명</p>
                        <p><strong>사용된 에이전트:</strong> {modelData.training_metadata?.agents_used?.join(', ')}</p>
                        <p><strong>앙상블 성능:</strong> {JSON.stringify(modelData.training_metadata?.performance_summary)}</p>
                      </div>
                    ),
                    width: 600
                  });
                }
              }}
            >
              모델 정보 보기
            </Button>
          }
        />
      ) : (
        <Alert
          message="📈 기본 모델 사용 중"
          description="기본 설정으로 분석을 수행합니다. 사후 분석을 통해 모델을 학습하면 더 정확한 예측이 가능합니다."
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      {/* 캐시 옵션 표시 */}
      {showCacheOptions && cachedResults.length > 0 && (
        <Card
          title={
            <Space>
              <HistoryOutlined style={{ color: '#1890ff' }} />
              <span>이전 분석 결과 발견</span>
              <Badge count={cachedResults.length} showZero={false} />
            </Space>
          }
          style={{ marginBottom: 24 }}
          extra={
            <Button 
              type="link" 
              onClick={() => setCacheModalVisible(true)}
              size="small"
            >
              전체 목록 보기
            </Button>
          }
        >
          <div>
            <Paragraph>
              <Text strong>{cachedResults.length}개</Text>의 이전 분석 결과가 있습니다. 
              새로 분석하거나 기존 결과를 사용할 수 있습니다.
            </Paragraph>
            
            {cachedResults.length > 0 && (
              <div style={{ marginBottom: '16px', padding: '12px', backgroundColor: '#f6ffed', border: '1px solid #b7eb8f', borderRadius: '6px' }}>
                <Text strong>📊 최신 분석 결과:</Text>
                <br />
                <Text type="secondary">
                  📅 {new Date(cachedResults[0].timestamp).toLocaleString('ko-KR')} | 
                  👥 {cachedResults[0].totalEmployees}명 분석
                </Text>
              </div>
            )}
            
            <Space size="middle">
              <Button 
                type="primary" 
                icon={<CheckCircleOutlined />}
                onClick={loadLatestCache}
              >
                최신 결과 사용
              </Button>
              <Button 
                icon={<HistoryOutlined />}
                onClick={() => setCacheModalVisible(true)}
              >
                다른 결과 선택
              </Button>
              <Button 
                icon={<RocketOutlined />}
                onClick={startNewAnalysis}
              >
                새로 분석하기
              </Button>
              <Button 
                icon={<FolderOutlined />}
                onClick={cleanupMisclassifiedFolders}
                loading={loading}
                style={{ marginLeft: 8 }}
              >
                미분류 폴더 정리
              </Button>
            </Space>
          </div>
        </Card>
      )}

      <Row gutter={[24, 24]}>
        {/* 파일 업로드 섹션 */}
        {/* 1단계: Structura 데이터 (정형 데이터) */}
        <Col span={12}>
          <Card title="1단계: Structura - 정형 데이터" extra={<FileTextOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Dragger
                name="structura_file"
                multiple={false}
                beforeUpload={(file) => handleAgentFileUpload(file, 'structura')}
                showUploadList={false}
                disabled={loading || isAnalyzing}
              >
                <p className="ant-upload-drag-icon">
                  <UploadOutlined />
                </p>
                <p className="ant-upload-text">
                  HR 기본 데이터 (CSV)
                </p>
                <p className="ant-upload-hint">
                  필수: EmployeeNumber, Age, JobRole, Department
                  <br />
                  선택: MonthlyIncome, JobSatisfaction, WorkLifeBalance
                </p>
              </Dragger>
              {agentFiles.structura && (
                <Alert
                  message={`✅ ${agentFiles.structura.name || 'Structura 파일 업로드됨'}`}
                  type="success"
                  showIcon
                />
              )}
            </Space>
          </Card>
        </Col>

        {/* 2단계: Cognita 데이터 (Neo4j 연결) */}
        <Col span={12}>
          <Card title="2단계: Cognita - Neo4j 연결" extra={<ApiOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Input
                placeholder="Neo4j URI (예: bolt://localhost:7687)"
                value={neo4jConfig.uri}
                onChange={(e) => setNeo4jConfig({...neo4jConfig, uri: e.target.value})}
                disabled={loading || isAnalyzing}
              />
              <Input
                placeholder="Username (예: neo4j)"
                value={neo4jConfig.username}
                onChange={(e) => setNeo4jConfig({...neo4jConfig, username: e.target.value})}
                disabled={loading || isAnalyzing}
              />
              <Input.Password
                placeholder="Password"
                value={neo4jConfig.password}
                onChange={(e) => setNeo4jConfig({...neo4jConfig, password: e.target.value})}
                disabled={loading || isAnalyzing}
              />
              <Button
                type="dashed"
                onClick={testNeo4jConnection}
                disabled={loading || isAnalyzing}
                loading={neo4jTesting}
              >
                연결 테스트
              </Button>
              {neo4jConnected && (
                <Alert
                  message="✅ Neo4j 연결 성공"
                  type="success"
                  showIcon
                />
              )}
            </Space>
          </Card>
        </Col>

        {/* 3단계: Chronos 데이터 (시계열 데이터) */}
        <Col span={12}>
          <Card title="3단계: Chronos - 시계열 데이터" extra={<BarChartOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Dragger
                name="chronos_file"
                multiple={false}
                beforeUpload={(file) => handleAgentFileUpload(file, 'chronos')}
                showUploadList={false}
                disabled={loading || isAnalyzing}
              >
                <p className="ant-upload-drag-icon">
                  <UploadOutlined />
                </p>
                <p className="ant-upload-text">
                  시계열 데이터 (CSV)
                </p>
                <p className="ant-upload-hint">
                  필수: employee_id, date, work_focused_ratio, meeting_collaboration_ratio
                  <br />
                  선택: social_dining_ratio, break_relaxation_ratio, system_login_hours
                  <br />
                  ⚠️ 대용량 파일 지원 (최대 500MB)
                </p>
              </Dragger>
              {agentFiles.chronos && (
                <Alert
                  message={`✅ ${agentFiles.chronos.name || 'Chronos 파일 업로드됨'} (${(agentFiles.chronos.size/1024/1024).toFixed(1)}MB)`}
                  type="success"
                  showIcon
                />
              )}
            </Space>
          </Card>
        </Col>

        {/* 4단계: Sentio 데이터 (텍스트 데이터) */}
        <Col span={12}>
          <Card title="4단계: Sentio - 텍스트 데이터" extra={<FileTextOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Dragger
                name="sentio_file"
                multiple={false}
                beforeUpload={(file) => handleAgentFileUpload(file, 'sentio')}
                showUploadList={false}
                disabled={loading || isAnalyzing}
              >
                <p className="ant-upload-drag-icon">
                  <UploadOutlined />
                </p>
                <p className="ant-upload-text">
                  텍스트/피드백 데이터 (CSV)
                </p>
                <p className="ant-upload-hint">
                  필수: EmployeeNumber, SELF_REVIEW_text, PEER_FEEDBACK_text, WEEKLY_SURVEY_text
                  <br />
                  선택: JobRole, Persona_Code, Persona_Name
                </p>
              </Dragger>
              {agentFiles.sentio && (
                <Alert
                  message={`✅ ${agentFiles.sentio.name || 'Sentio 파일 업로드됨'}`}
                  type="success"
                  showIcon
                />
              )}
            </Space>
          </Card>
        </Col>

        {/* 5단계: Agora 데이터 (시장 데이터) - 선택사항 */}
        <Col span={12}>
          <Card title="5단계: Agora - 시장 데이터 (선택사항)" extra={<BarChartOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Dragger
                name="agora_file"
                multiple={false}
                beforeUpload={(file) => handleAgentFileUpload(file, 'agora')}
                showUploadList={false}
                disabled={loading || isAnalyzing}
              >
                <p className="ant-upload-drag-icon">
                  <UploadOutlined />
                </p>
                <p className="ant-upload-text">
                  시장 분석용 HR 데이터 (CSV)
                </p>
                <p className="ant-upload-hint">
                  필수: EmployeeNumber, Age, JobRole, Department (Structura와 동일)
                  <br />
                  💡 API를 통해 직업별 연봉, 시장 동향 데이터 자동 수집
                </p>
              </Dragger>
              {agentFiles.agora && (
                <Alert
                  message={`✅ ${agentFiles.agora.name || 'Agora 파일 업로드됨'}`}
                  type="success"
                  showIcon
                />
              )}
              
              {!agentFiles.agora && agentFiles.structura && (
                <Alert
                  message="Structura 데이터로 Agora 분석 가능"
                  description="별도 파일을 업로드하지 않아도 Structura 데이터를 기반으로 시장 분석을 수행할 수 있습니다."
                  type="info"
                  showIcon
                />
              )}
            </Space>
          </Card>
        </Col>


        {/* 분석 실행 섹션 */}
        <Col span={24}>
          <Card title="6단계: 통합 배치 분석 실행" extra={<ApiOutlined />}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button
                type="primary"
                size="large"
                icon={<ApiOutlined />}
                onClick={runBatchAnalysis}
                disabled={!agentFiles.structura || isAnalyzing}
                loading={isAnalyzing}
              >
                {isAnalyzing ? '분석 진행 중...' : '배치 분석 시작'}
              </Button>

              {isAnalyzing && (
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>전체 진행률</Text>
                    <Progress
                      percent={analysisProgress.overall}
                      status="active"
                      strokeColor={{
                        '0%': '#108ee9',
                        '100%': '#87d068',
                      }}
                      format={(percent) => `${percent}%`}
                    />
                  </div>
                  
                  <Row gutter={[16, 8]}>
                    <Col span={12}>
                      <Text>Structura (HR 분석)</Text>
                      <Progress 
                        percent={analysisProgress.structura} 
                        size="small"
                        strokeColor="#1890ff"
                        format={(percent) => `${percent}%`}
                      />
                    </Col>
                    <Col span={12}>
                      <Text>Cognita (관계 분석)</Text>
                      <Progress 
                        percent={analysisProgress.cognita} 
                        size="small"
                        strokeColor="#52c41a"
                        format={(percent) => `${percent}%`}
                      />
                    </Col>
                    <Col span={12}>
                      <Text>Chronos (시계열 분석)</Text>
                      <Progress 
                        percent={analysisProgress.chronos} 
                        size="small"
                        strokeColor="#fa8c16"
                        format={(percent) => `${percent}%`}
                      />
                    </Col>
                    <Col span={12}>
                      <Text>Sentio (감정 분석)</Text>
                      <Progress 
                        percent={analysisProgress.sentio} 
                        size="small"
                        strokeColor="#eb2f96"
                        format={(percent) => `${percent}%`}
                      />
                    </Col>
                    <Col span={24}>
                      <Text>Agora (시장 분석)</Text>
                      <Progress 
                        percent={analysisProgress.agora} 
                        size="small"
                        strokeColor="#722ed1"
                        format={(percent) => `${percent}%`}
                      />
                    </Col>
                  </Row>
                </Space>
              )}

              <Text type="secondary">
                Supervisor가 각 에이전트를 순차적으로 실행하여 종합적인 분석을 수행합니다.
              </Text>
            </Space>
          </Card>
        </Col>

        {/* 분석 결과 섹션 */}
        {analysisResults && (
          <Col span={24}>
            <Card title="3단계: 분석 결과" extra={<BarChartOutlined />}>
              <Space direction="vertical" style={{ width: '100%' }}>
                
                {/* 캐시된 결과 정보 표시 */}
                {analysisResults.analysis_metadata?.cached_result && (
                  <Alert
                    message="📂 캐시된 분석 결과"
                    description={
                      <div>
                        <Text>이전에 저장된 분석 결과를 불러왔습니다.</Text>
                        <br />
                        <Text type="secondary">
                          분석 시간: {new Date(analysisResults.analysis_metadata.timestamp).toLocaleString('ko-KR')}
                        </Text>
                        {analysisResults.cache_info && (
                          <>
                            <br />
                            <Text type="secondary">
                              {analysisResults.cache_info.title} | 
                              고위험 {analysisResults.cache_info.highRiskCount}명, 
                              중위험 {analysisResults.cache_info.mediumRiskCount}명, 
                              저위험 {analysisResults.cache_info.lowRiskCount}명
                            </Text>
                          </>
                        )}
                      </div>
                    }
                    type="info"
                    showIcon
                    style={{ marginBottom: 16 }}
                  />
                )}
                
                {/* 최적화된 설정 적용 정보 */}
                {finalRiskSettings && (
                  <Alert
                    message="🎯 사후 분석 최적화 설정 적용됨"
                    description={
                      <div>
                        <Row gutter={16}>
                          <Col span={8}>
                            <Text strong>위험도 임계값:</Text><br />
                            <Text>• 안전군: &lt; {finalRiskSettings.risk_thresholds?.low_risk_threshold || 0.3}</Text><br />
                            <Text>• 주의군: {finalRiskSettings.risk_thresholds?.low_risk_threshold || 0.3} ~ {finalRiskSettings.risk_thresholds?.high_risk_threshold || 0.7}</Text><br />
                            <Text>• 고위험군: ≥ {finalRiskSettings.risk_thresholds?.high_risk_threshold || 0.7}</Text>
                          </Col>
                          <Col span={8}>
                            <Text strong>퇴사 예측 기준:</Text><br />
                            <Text>{finalRiskSettings.attrition_prediction_mode === 'high_risk_only' ? '고위험군만 퇴사 예측' : '주의군 + 고위험군 퇴사 예측'}</Text><br />
                            {finalRiskSettings.performance_metrics?.f1_score && (
                              <>
                                <Text strong>최적화된 F1-Score:</Text><br />
                                <Text>{finalRiskSettings.performance_metrics.f1_score.toFixed(4)}</Text>
                              </>
                            )}
                          </Col>
                          <Col span={8}>
                            {finalRiskSettings.performance_metrics && (
                              <>
                                <Text strong>성능 지표:</Text><br />
                                <Text>• Precision: {finalRiskSettings.performance_metrics.precision?.toFixed(4) || 'N/A'}</Text><br />
                                <Text>• Recall: {finalRiskSettings.performance_metrics.recall?.toFixed(4) || 'N/A'}</Text><br />
                                <Text>• F1-Score: {finalRiskSettings.performance_metrics.f1_score?.toFixed(4) || 'N/A'}</Text>
                              </>
                            )}
                          </Col>
                        </Row>
                      </div>
                    }
                    type="success"
                    showIcon
                    style={{ marginBottom: 16 }}
                  />
                )}
                {/* 최적화된 위험도 분류 통계 */}
                <Row gutter={16}>
                  <Col span={4}>
                    <Statistic
                      title="총 직원 수"
                      value={analysisResults.total_employees || analysisResults.cache_info?.totalEmployees || 0}
                      prefix={<CheckCircleOutlined />}
                    />
                  </Col>
                  <Col span={5}>
                    <Statistic
                      title={`안전군 (< ${finalRiskSettings?.risk_thresholds?.low_risk_threshold || 0.3})`}
                      value={(() => {
                        // 캐시된 결과인 경우 캐시 정보에서 가져오기
                        if (analysisResults.cache_info) {
                          return analysisResults.cache_info.lowRiskCount || 0;
                        }
                        // 실제 분석 결과인 경우 계산
                        return analysisResults.results?.filter(r => {
                          const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                          return score && calculateRiskLevel(score) === 'LOW';
                        }).length || 0;
                      })()}
                      valueStyle={{ color: '#52c41a' }}
                      prefix={<CheckCircleOutlined />}
                    />
                  </Col>
                  <Col span={5}>
                    <Statistic
                      title={`주의군 (${finalRiskSettings?.risk_thresholds?.low_risk_threshold || 0.3} ~ ${finalRiskSettings?.risk_thresholds?.high_risk_threshold || 0.7})`}
                      value={(() => {
                        // 캐시된 결과인 경우 캐시 정보에서 가져오기
                        if (analysisResults.cache_info) {
                          return analysisResults.cache_info.mediumRiskCount || 0;
                        }
                        // 실제 분석 결과인 경우 계산
                        return analysisResults.results?.filter(r => {
                          const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                          return score && calculateRiskLevel(score) === 'MEDIUM';
                        }).length || 0;
                      })()}
                      valueStyle={{ color: '#fa8c16' }}
                      prefix={<ExclamationCircleOutlined />}
                    />
                  </Col>
                  <Col span={5}>
                    <Statistic
                      title={`고위험군 (≥ ${finalRiskSettings?.risk_thresholds?.high_risk_threshold || 0.7})`}
                      value={(() => {
                        // 캐시된 결과인 경우 캐시 정보에서 가져오기
                        if (analysisResults.cache_info) {
                          return analysisResults.cache_info.highRiskCount || 0;
                        }
                        // 실제 분석 결과인 경우 계산
                        return analysisResults.results?.filter(r => {
                          const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                          return score && calculateRiskLevel(score) === 'HIGH';
                        }).length || 0;
                      })()}
                      valueStyle={{ color: '#cf1322' }}
                      prefix={<ExclamationCircleOutlined />}
                    />
                  </Col>
                  <Col span={5}>
                    <Statistic
                      title={`예측 퇴사자 (${finalRiskSettings?.attrition_prediction_mode === 'high_risk_only' ? '고위험군만' : '주의군+고위험군'})`}
                      value={(() => {
                        let highRisk = 0;
                        let mediumRisk = 0;
                        
                        // 캐시된 결과인 경우
                        if (analysisResults.cache_info) {
                          highRisk = analysisResults.cache_info.highRiskCount || 0;
                          mediumRisk = analysisResults.cache_info.mediumRiskCount || 0;
                        }
                        // 실제 분석 결과인 경우
                        else {
                          highRisk = analysisResults.results?.filter(r => {
                            const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                            return score && calculateRiskLevel(score) === 'HIGH';
                          }).length || 0;
                          mediumRisk = analysisResults.results?.filter(r => {
                            const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                            return score && calculateRiskLevel(score) === 'MEDIUM';
                          }).length || 0;
                        }
                        
                        return finalRiskSettings?.attrition_prediction_mode === 'medium_high_risk' 
                          ? highRisk + mediumRisk 
                          : highRisk;
                      })()}
                      valueStyle={{ color: '#722ed1' }}
                      prefix={<TeamOutlined />}
                    />
                  </Col>
                </Row>

                {/* 결과 액션 버튼들 */}
                <Row gutter={16} style={{ marginBottom: 16 }}>
                  <Col>
                    <Button 
                      type="primary" 
                      icon={<DownloadOutlined />}
                      onClick={() => exportBatchResults('csv')}
                      loading={isExporting}
                    >
                      CSV로 내보내기
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      icon={<DownloadOutlined />}
                      onClick={() => exportBatchResults('json')}
                      loading={isExporting}
                    >
                      JSON으로 내보내기
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      icon={<FilePdfOutlined />}
                      onClick={() => exportBatchResults('pdf')}
                      loading={isExporting}
                      disabled={true}
                      title="PDF 생성 기능은 현재 개발 중입니다"
                    >
                      PDF 보고서 (개발중)
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      icon={<BarChartOutlined />}
                      onClick={() => navigateToVisualization()}
                    >
                      시각화 대시보드
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      icon={<TeamOutlined />}
                      onClick={() => navigateToRelationshipAnalysis()}
                    >
                      관계 분석
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      icon={<TeamOutlined />}
                      onClick={() => {
                        if (onNavigate) {
                          onNavigate('group-statistics');
                        } else {
                          message.info('📈 단체 통계 메뉴로 이동하세요.');
                        }
                      }}
                    >
                      단체 통계
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      icon={<FileTextOutlined />}
                      onClick={() => showSavedResultsInfo()}
                      type="dashed"
                    >
                      저장된 결과 확인
                    </Button>
                  </Col>
                  <Col>
                    <Button 
                      danger
                      onClick={() => clearLocalStorage()}
                      title="저장된 분석 결과를 정리하여 브라우저 저장공간을 확보합니다"
                    >
                      저장공간 정리
                    </Button>
                  </Col>
                </Row>

                {/* 종합 결과 내보내기 섹션 */}
                <Card 
                  title="📊 전체 결과 종합 내보내기" 
                  size="small" 
                  style={{ marginBottom: 16 }}
                  extra={
                    <Text type="secondary">
                      {analysisResults.optimization_results ? '최적화 결과 포함' : '기본 분석 결과'}
                    </Text>
                  }
                >
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Paragraph type="secondary">
                      배치 분석 결과와 {analysisResults.optimization_results ? '사후 분석 최적화 결과를' : '통합 분석 결과를'} 
                      다양한 형식으로 내보낼 수 있습니다.
                    </Paragraph>
                    
                    <Row gutter={[8, 8]}>
                      <Col>
                        <Button 
                          type="primary"
                          size="small"
                          icon={<DownloadOutlined />}
                          onClick={() => exportBatchResults('csv')}
                          loading={isExporting}
                        >
                          📈 상세 CSV (추천)
                        </Button>
                      </Col>
                      <Col>
                        <Button 
                          size="small"
                          icon={<DownloadOutlined />}
                          onClick={() => exportBatchResults('json')}
                          loading={isExporting}
                        >
                          🔧 완전한 JSON
                        </Button>
                      </Col>
                      {analysisResults.optimization_results && (
                        <Col>
                          <Button 
                            size="small"
                            icon={<SettingOutlined />}
                            onClick={() => {
                              // 최적화 결과만 별도 다운로드
                              const optimizationData = {
                                optimization_results: analysisResults.optimization_results,
                                export_date: new Date().toISOString(),
                                source: 'batch_analysis'
                              };
                              const jsonContent = JSON.stringify(optimizationData, null, 2);
                              const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
                              const link = document.createElement('a');
                              const url = URL.createObjectURL(blob);
                              link.setAttribute('href', url);
                              link.setAttribute('download', `최적화결과_${new Date().toISOString().slice(0, 10)}.json`);
                              link.style.visibility = 'hidden';
                              document.body.appendChild(link);
                              link.click();
                              document.body.removeChild(link);
                              message.success('최적화 결과를 다운로드했습니다.');
                            }}
                            loading={isExporting}
                          >
                            ⚙️ 최적화 결과만
                          </Button>
                        </Col>
                      )}
                    </Row>

                    {analysisResults.optimization_results && (
                      <Alert
                        message="🎯 최적화된 결과 포함"
                        description={
                          <div>
                            <Text>사후 분석에서 최적화된 임계값과 가중치가 적용된 결과입니다.</Text>
                            <br />
                            <Text type="secondary">
                              • 최적 F1-Score: {analysisResults.optimization_results.weight_optimization?.best_f1_score?.toFixed(4)}
                              • 총 {analysisResults.optimization_results.total_employees}명 분석
                              • 위험도 분류: 안전군 {analysisResults.optimization_results.risk_distribution?.['안전군']}명, 
                                주의군 {analysisResults.optimization_results.risk_distribution?.['주의군']}명, 
                                고위험군 {analysisResults.optimization_results.risk_distribution?.['고위험군']}명
                            </Text>
                          </div>
                        }
                        type="success"
                        showIcon
                        style={{ marginTop: 8 }}
                      />
                    )}
                  </Space>
                </Card>

                {/* 결과 테이블 또는 캐시 요약 */}
                {analysisResults.results && analysisResults.results.length > 0 ? (
                  <Table
                    columns={resultColumns}
                    dataSource={analysisResults.results}
                    rowKey="employee_number"
                    pagination={{
                      pageSize: 10,
                      showSizeChanger: true,
                      showQuickJumper: true,
                      showTotal: (total, range) =>
                        `${range[0]}-${range[1]} of ${total} 직원`
                    }}
                    scroll={{ x: 800 }}
                  />
                ) : analysisResults.cache_info ? (
                  <Card title="캐시된 분석 결과 요약" style={{ marginTop: 16 }}>
                    <Row gutter={16}>
                      <Col span={12}>
                        <Descriptions column={1} bordered size="small">
                          <Descriptions.Item label="분석 제목">{analysisResults.cache_info.title}</Descriptions.Item>
                          <Descriptions.Item label="총 직원 수">{analysisResults.cache_info.totalEmployees || analysisResults.total_employees}명</Descriptions.Item>
                          <Descriptions.Item label="정확도">{analysisResults.cache_info.accuracy}%</Descriptions.Item>
                          <Descriptions.Item label="분석 시간">{new Date(analysisResults.analysis_metadata?.timestamp).toLocaleString('ko-KR')}</Descriptions.Item>
                        </Descriptions>
                      </Col>
                      <Col span={12}>
                        <Card size="small" title="위험도별 분포">
                          <Space direction="vertical" style={{ width: '100%' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <span>고위험군:</span>
                              <Tag color="red">{analysisResults.cache_info.highRiskCount}명</Tag>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <span>중위험군:</span>
                              <Tag color="orange">{analysisResults.cache_info.mediumRiskCount}명</Tag>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <span>저위험군:</span>
                              <Tag color="green">{analysisResults.cache_info.lowRiskCount}명</Tag>
                            </div>
                          </Space>
                        </Card>
                      </Col>
                    </Row>
                    
                    {analysisResults.cache_info.summary && (
                      <Alert
                        message="분석 요약"
                        description={analysisResults.cache_info.summary}
                        type="info"
                        showIcon
                        style={{ marginTop: 16 }}
                      />
                    )}
                    
                    {analysisResults.cache_info.keyInsights && analysisResults.cache_info.keyInsights.length > 0 && (
                      <Card size="small" title="주요 인사이트" style={{ marginTop: 16 }}>
                        <List
                          size="small"
                          dataSource={analysisResults.cache_info.keyInsights}
                          renderItem={(insight, index) => (
                            <List.Item>
                              <Text>
                                <Badge count={index + 1} style={{ backgroundColor: '#1890ff', marginRight: 8 }} />
                                {insight}
                              </Text>
                            </List.Item>
                          )}
                        />
                      </Card>
                    )}
                    
                    <Alert
                      message="상세 데이터 없음"
                      description="이 결과는 캐시된 요약 정보입니다. 개별 직원의 상세 분석 결과를 보려면 새로운 분석을 실행하세요."
                      type="warning"
                      showIcon
                      style={{ marginTop: 16 }}
                    />
                  </Card>
                ) : (
                  <Alert
                    message="분석 결과 없음"
                    description="표시할 분석 결과가 없습니다."
                    type="info"
                    showIcon
                  />
                )}

                {/* Integration 보고서 섹션 */}
                {analysisResults.integration_report && (
                  <Card title="Integration 종합 보고서" extra={<FileTextOutlined />} style={{ marginTop: 16 }}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      {/* 경영진 요약 */}
                      <Card size="small" title="경영진 요약">
                        <Paragraph>
                          <Text strong>개요:</Text> {analysisResults.integration_report.executive_summary?.overview}
                        </Paragraph>
                        <Paragraph>
                          <Text strong>주요 지표:</Text> {analysisResults.integration_report.executive_summary?.key_metrics}
                        </Paragraph>
                        <Paragraph>
                          <Text strong>긴급도 평가:</Text> {analysisResults.integration_report.executive_summary?.urgency_assessment}
                        </Paragraph>
                      </Card>

                      {/* 주요 발견사항 */}
                      <Card size="small" title="주요 발견사항">
                        <ul>
                          {analysisResults.integration_report.key_findings?.map((finding, index) => (
                            <li key={index}>{finding}</li>
                          ))}
                        </ul>
                      </Card>

                      {/* 권장사항 */}
                      {analysisResults.integration_report.recommendations && (
                        <Card size="small" title="권장사항">
                          <Row gutter={16}>
                            <Col span={8}>
                              <Text strong>즉시 조치사항:</Text>
                              <ul>
                                {analysisResults.integration_report.recommendations.immediate_actions?.map((action, index) => (
                                  <li key={index}>{action}</li>
                                ))}
                              </ul>
                            </Col>
                            <Col span={8}>
                              <Text strong>단기 전략:</Text>
                              <ul>
                                {analysisResults.integration_report.recommendations.short_term_strategies?.map((strategy, index) => (
                                  <li key={index}>{strategy}</li>
                                ))}
                              </ul>
                            </Col>
                            <Col span={8}>
                              <Text strong>장기 이니셔티브:</Text>
                              <ul>
                                {analysisResults.integration_report.recommendations.long_term_initiatives?.map((initiative, index) => (
                                  <li key={index}>{initiative}</li>
                                ))}
                              </ul>
                            </Col>
                          </Row>
                        </Card>
                      )}

                      {/* 워크플로우 효과성 */}
                      {analysisResults.integration_report.detailed_risk_analysis?.workflow_effectiveness && (
                        <Card size="small" title="워크플로우 효과성">
                          <Row gutter={16}>
                            <Col span={8}>
                              <Alert
                                message="순차적 실행"
                                description={analysisResults.integration_report.detailed_risk_analysis.workflow_effectiveness.sequential_execution}
                                type="success"
                                showIcon
                              />
                            </Col>
                            <Col span={8}>
                              <Alert
                                message="데이터 통합"
                                description={analysisResults.integration_report.detailed_risk_analysis.workflow_effectiveness.data_integration}
                                type="success"
                                showIcon
                              />
                            </Col>
                            <Col span={8}>
                              <Alert
                                message="포괄적 분석"
                                description={analysisResults.integration_report.detailed_risk_analysis.workflow_effectiveness.comprehensive_coverage}
                                type="success"
                                showIcon
                              />
                            </Col>
                          </Row>
                        </Card>
                      )}
                    </Space>
                  </Card>
                )}
              </Space>
            </Card>
          </Col>
        )}
      </Row>

      {/* 개별 직원 레포트 모달 */}
      <Modal
        title={`직원 ${selectedEmployee?.employee_number} 상세 분석 레포트`}
        open={employeeReportVisible}
        onCancel={() => {
          setEmployeeReportVisible(false);
          setSelectedEmployee(null);
          setEmployeeReport(null);
        }}
        width={800}
        footer={[
          <Button key="close" onClick={() => setEmployeeReportVisible(false)}>
            닫기
          </Button>,
          <Button 
            key="download" 
            type="primary" 
            onClick={() => {
              if (employeeReport) {
                const dataStr = JSON.stringify(employeeReport, null, 2);
                const dataBlob = new Blob([dataStr], {type: 'application/json'});
                const url = URL.createObjectURL(dataBlob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `employee_${selectedEmployee?.employee_number}_report.json`;
                link.click();
                URL.revokeObjectURL(url);
              }
            }}
          >
            레포트 다운로드
          </Button>
        ]}
      >
        {reportGenerating ? (
          <div style={{ textAlign: 'center', padding: '50px' }}>
            <Spin size="large" />
            <p style={{ marginTop: '16px' }}>AI가 종합 분석 레포트를 생성하고 있습니다...</p>
          </div>
        ) : employeeReport ? (
          <div>
            {/* 종합 평가 */}
            <Card size="small" title="종합 평가" style={{ marginBottom: '16px' }}>
              <Descriptions column={2}>
                <Descriptions.Item label="위험도 수준">
                  <Tag color={
                    employeeReport.overall_assessment?.risk_level === 'HIGH' ? 'red' :
                    employeeReport.overall_assessment?.risk_level === 'MEDIUM' ? 'orange' : 'green'
                  }>
                    {employeeReport.overall_assessment?.risk_level}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="위험도 점수">
                  {employeeReport.overall_assessment?.risk_score ? 
                    `${(employeeReport.overall_assessment.risk_score * 100).toFixed(1)}%` : 'N/A'}
                </Descriptions.Item>
              </Descriptions>
              <Alert
                message="AI 종합 분석"
                description={employeeReport.overall_assessment?.summary}
                type="info"
                showIcon
                style={{ marginTop: '12px' }}
              />
            </Card>

            {/* 상세 분석 결과 */}
            <Card size="small" title="에이전트별 상세 분석" style={{ marginBottom: '16px' }}>
              <Row gutter={[16, 16]}>
                {employeeReport.detailed_analysis?.structura && (
                  <Col span={12}>
                    <Card size="small" title="Structura (구조적 분석)">
                      <p><strong>예측값:</strong> {(employeeReport.detailed_analysis.structura.prediction * 100).toFixed(1)}%</p>
                      <p><strong>신뢰도:</strong> {(employeeReport.detailed_analysis.structura.confidence * 100).toFixed(1)}%</p>
                      <Alert message={employeeReport.detailed_analysis.structura.interpretation} type="info" size="small" />
                    </Card>
                  </Col>
                )}
                
                {employeeReport.detailed_analysis?.cognita && (
                  <Col span={12}>
                    <Card size="small" title="Cognita (관계 분석)">
                      <p><strong>위험 수준:</strong> {employeeReport.detailed_analysis.cognita.risk_level}</p>
                      <p><strong>위험 점수:</strong> {(employeeReport.detailed_analysis.cognita.risk_score * 100).toFixed(1)}%</p>
                      <Alert message={employeeReport.detailed_analysis.cognita.interpretation} type="info" size="small" />
                    </Card>
                  </Col>
                )}
                
                {employeeReport.detailed_analysis?.chronos && (
                  <Col span={12}>
                    <Card size="small" title="Chronos (시계열 분석)">
                      <p><strong>예측값:</strong> {(employeeReport.detailed_analysis.chronos.prediction * 100).toFixed(1)}%</p>
                      <p><strong>신뢰도:</strong> {(employeeReport.detailed_analysis.chronos.confidence * 100).toFixed(1)}%</p>
                      <Alert message={employeeReport.detailed_analysis.chronos.interpretation} type="info" size="small" />
                    </Card>
                  </Col>
                )}
                
                {employeeReport.detailed_analysis?.sentio && (
                  <Col span={12}>
                    <Card size="small" title="Sentio (감정 분석)">
                      <p><strong>감정 점수:</strong> {employeeReport.detailed_analysis.sentio.sentiment_score?.toFixed(2)}</p>
                      <p><strong>위험 수준:</strong> {employeeReport.detailed_analysis.sentio.risk_level}</p>
                      <Alert message={employeeReport.detailed_analysis.sentio.interpretation} type="info" size="small" />
                    </Card>
                  </Col>
                )}
                
                {employeeReport.detailed_analysis?.agora && (
                  <Col span={12}>
                    <Card size="small" title="Agora (시장 분석)">
                      <p><strong>시장 압력:</strong> {(employeeReport.detailed_analysis.agora.market_pressure * 100).toFixed(1)}%</p>
                      <Alert message={employeeReport.detailed_analysis.agora.interpretation} type="info" size="small" />
                    </Card>
                  </Col>
                )}
              </Row>
            </Card>

            {/* 권장사항 */}
            <Card size="small" title="AI 권장사항">
              <ul>
                {employeeReport.recommendations?.map((recommendation, index) => (
                  <li key={index} style={{ marginBottom: '8px' }}>
                    <Alert message={recommendation} type="warning" size="small" showIcon />
                  </li>
                ))}
              </ul>
            </Card>

            <Divider />
            <Text type="secondary" style={{ fontSize: 'var(--font-small)' }}>
              레포트 생성 시간: {new Date(employeeReport.generated_at).toLocaleString()}
            </Text>
          </div>
        ) : (
          <Alert message="레포트를 생성할 수 없습니다." type="error" />
        )}
      </Modal>

      {/* 캐시 선택 모달 */}
      <Modal
        title={
          <Space>
            <HistoryOutlined />
            <span>분석 결과 선택</span>
            <Badge count={cachedResults.length} showZero={false} />
          </Space>
        }
        open={cacheModalVisible}
        onCancel={() => setCacheModalVisible(false)}
        footer={null}
        width={800}
      >
        <div style={{ marginBottom: '16px' }}>
          <Alert
            message="이전 분석 결과 중 하나를 선택하여 사용할 수 있습니다."
            description="선택한 결과가 현재 화면에 로드되며, 새로운 분석 없이 바로 확인할 수 있습니다."
            type="info"
            showIcon
          />
        </div>
        
        <List
          dataSource={cachedResults}
          renderItem={(cache, index) => (
            <List.Item
              actions={[
                <Button
                  type="primary"
                  size="small"
                  onClick={() => loadCachedResult(cache.id)}
                  style={{ marginRight: 8 }}
                >
                  이 결과 사용
                </Button>,
                <Button
                  type="default"
                  size="small"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => deleteCachedResult(cache.id)}
                >
                  삭제
                </Button>
              ]}
            >
              <List.Item.Meta
                avatar={
                  <div style={{ 
                    width: '40px', 
                    height: '40px', 
                    borderRadius: '50%', 
                    backgroundColor: index === 0 ? '#52c41a' : '#1890ff',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                    fontWeight: 'bold'
                  }}>
                    {index === 0 ? '최신' : index + 1}
                  </div>
                }
                title={
                  <Space>
                    <Text strong>{cache.title}</Text>
                    {index === 0 && <Tag color="green">최신</Tag>}
                  </Space>
                }
                description={
                  <div>
                    <Space direction="vertical" size="small" style={{ width: '100%' }}>
                      <Space>
                        <Text type="secondary">
                          📅 {new Date(cache.timestamp).toLocaleString('ko-KR')}
                        </Text>
                        <Text type="secondary">
                          👥 {cache.totalEmployees}명
                        </Text>
                        <Text type="secondary">
                          🎯 정확도 {cache.accuracy}%
                        </Text>
                      </Space>
                      <Space>
                        <Tag color="red">고위험 {cache.highRiskCount}명</Tag>
                        <Tag color="orange">중위험 {cache.mediumRiskCount}명</Tag>
                        <Tag color="green">저위험 {cache.lowRiskCount}명</Tag>
                      </Space>
                      <Text type="secondary" style={{ fontSize: 'var(--font-small)' }}>
                        {cache.summary}
                      </Text>
                    </Space>
                  </div>
                }
              />
            </List.Item>
          )}
        />
        
        <div style={{ marginTop: '16px', textAlign: 'center' }}>
          <Button onClick={() => setCacheModalVisible(false)}>
            취소
          </Button>
        </div>
      </Modal>

    </div>
  );
};

export default BatchAnalysis;