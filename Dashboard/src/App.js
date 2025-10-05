import React, { useState, useEffect } from 'react';
import { Layout, Menu, Typography, notification } from 'antd';
import {
  HomeOutlined,
  ApiOutlined,
  RobotOutlined,
  BarChartOutlined,
  TeamOutlined,
  FileTextOutlined
} from '@ant-design/icons';

import Home from './components/Home';
import BatchAnalysis from './components/BatchAnalysis';
import PostAnalysis from './components/PostAnalysis';
import ReportGeneration from './components/ReportGeneration';
import RelationshipAnalysis from './components/RelationshipAnalysis';
import GroupStatistics from './components/GroupStatistics';
import { apiService } from './services/apiService';
import storageManager from './utils/storageManager';

const { Header, Sider, Content } = Layout;
const { Title } = Typography;

const App = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [selectedKey, setSelectedKey] = useState('home');
  const [loading, setLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState(null);
  
  // 전역 배치 분석 결과 상태 (페이지 간 공유)
  const [globalBatchResults, setGlobalBatchResults] = useState(null);
  const [lastAnalysisTimestamp, setLastAnalysisTimestamp] = useState(null);
  const [dataLoaded] = useState(true); // 데이터 로딩 상태를 기본적으로 활성화

  // 전역 에러 핸들러
  useEffect(() => {
    const handleUnhandledRejection = (event) => {
      // Chrome 확장 프로그램 관련 오류는 완전히 무시
      if (event.reason && event.reason.message && 
          (event.reason.message.includes('Extension context invalidated') ||
           event.reason.message.includes('message channel closed') ||
           event.reason.message.includes('disconnected port object') ||
           event.reason.message.includes('Attempting to use a disconnected port') ||
           event.reason.message.includes('Could not establish connection') ||
           event.reason.message.includes('SecretSessionError'))) {
        event.preventDefault();
        return;
      }
      
      // Chrome extension URL 관련 오류도 무시
      if (event.reason && event.reason.stack && 
          event.reason.stack.includes('chrome-extension://')) {
        event.preventDefault();
        return;
      }
      
      console.error('Unhandled promise rejection:', event.reason);
      notification.error({
        message: '예상치 못한 오류',
        description: '시스템에서 예상치 못한 오류가 발생했습니다.',
        duration: 4.5,
      });
    };

    const handleError = (event) => {
      // Chrome 확장 프로그램 관련 오류는 완전히 무시
      if (event.error && event.error.message && 
          (event.error.message.includes('Extension context invalidated') ||
           event.error.message.includes('message channel closed') ||
           event.error.message.includes('disconnected port object') ||
           event.error.message.includes('Attempting to use a disconnected port') ||
           event.error.message.includes('Could not establish connection') ||
           event.error.message.includes('SecretSessionError'))) {
        event.preventDefault();
        return;
      }
      
      // Chrome extension URL 관련 오류도 무시
      if (event.error && event.error.stack && 
          event.error.stack.includes('chrome-extension://')) {
        event.preventDefault();
        return;
      }
      
      console.error('Global error:', event.error);
    };

    window.addEventListener('unhandledrejection', handleUnhandledRejection);
    window.addEventListener('error', handleError);

    return () => {
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
      window.removeEventListener('error', handleError);
    };
  }, []);

  // 메뉴 아이템 - 홈 화면을 첫 번째로 배치
  const menuItems = [
    {
      key: 'home',
      icon: <HomeOutlined />,
      label: '🏠 홈',
    },
    {
      key: 'batch',
      icon: <RobotOutlined />,
      label: '🎯 배치 분석',
    },
    {
      key: 'group-statistics',
      icon: <TeamOutlined />,
      label: '📊 단체 통계',
    },
    {
      key: 'cognita',
      icon: <ApiOutlined />,
      label: '🕸️ 개별 관계분석',
    },
    {
      key: 'post-analysis',
      icon: <BarChartOutlined />,
      label: '🔍 사후 분석',
    },
    {
      key: 'report-generation',
      icon: <FileTextOutlined />,
      label: '📄 보고서 출력',
    },
  ];

  // 서버 상태 확인 및 localStorage에서 배치 결과 복원
  useEffect(() => {
    checkServerStatus();
    
    // localStorage에서 배치 분석 결과 복원 (청크 지원)
    try {
      // 먼저 일반 저장 방식 확인
      const storedResults = localStorage.getItem('batchAnalysisResults');
      const storedTimestamp = localStorage.getItem('lastAnalysisTimestamp');
      
      if (storedResults && storedTimestamp) {
        setGlobalBatchResults(JSON.parse(storedResults));
        setLastAnalysisTimestamp(storedTimestamp);
        console.log('배치 분석 결과 복원됨 (일반 저장):', JSON.parse(storedResults).length + '명');
      } else {
        // 청크 저장 방식 확인
        const metadata = localStorage.getItem('batchAnalysisMetadata');
        if (metadata) {
          const meta = JSON.parse(metadata);
          if (meta.storage_type === 'chunked') {
            console.log(`청크 데이터 복원 시작: ${meta.total_chunks}개 청크`);
            
            const allResults = [];
            for (let i = 0; i < meta.total_chunks; i++) {
              const chunkData = localStorage.getItem(`batchAnalysisResults_chunk_${i}`);
              if (chunkData) {
                allResults.push(...JSON.parse(chunkData));
              }
            }
            
            if (allResults.length > 0) {
              setGlobalBatchResults(allResults);
              setLastAnalysisTimestamp(meta.timestamp);
              console.log(`청크 데이터 복원 완료: ${allResults.length}명`);
            }
          }
        } else {
          // 요약 데이터만 있는 경우
          const summaryData = localStorage.getItem('batchAnalysisResultsSummary');
          if (summaryData) {
            const summary = JSON.parse(summaryData);
            console.log('요약 데이터만 복원됨:', summary);
            setLastAnalysisTimestamp(summary.timestamp);
          }
        }
      }
    } catch (error) {
      console.error('배치 결과 복원 실패:', error);
    }
  }, []);

  const checkServerStatus = async () => {
    try {
      const status = await apiService.checkHealth();
      setServerStatus(status);
      notification.success({
        message: '서버 연결 성공',
        description: 'Agentic AI System이 정상적으로 작동 중입니다.',
        duration: 3,
      });
    } catch (error) {
      console.error('서버 상태 확인 실패:', error);
      setServerStatus(null);
      notification.error({
        message: '서버 연결 실패',
        description: 'Agentic AI System 백엔드 서버가 실행되지 않았습니다. 서버를 시작한 후 새로고침하세요.',
        duration: 0,
      });
    }
  };


  // 로딩 상태 관리
  const setGlobalLoading = (isLoading) => {
    setLoading(isLoading);
  };

  // 현재 선택된 컴포넌트 렌더링
  // 배치 분석 결과 업데이트 함수
  const updateBatchResults = (results) => {
    setGlobalBatchResults(results);
    const timestamp = new Date().toISOString();
    setLastAnalysisTimestamp(timestamp);
    
    // localStorage에 원본 데이터 전체 저장 (용량 초과 시 청크 단위로 분할 저장)
    try {
      // 먼저 전체 결과 저장 시도
      localStorage.setItem('batchAnalysisResults', JSON.stringify(results));
      localStorage.setItem('lastAnalysisTimestamp', timestamp);
      console.log('배치 분석 결과 전역 업데이트:', results);
    } catch (error) {
      if (error.name === 'QuotaExceededError') {
        console.warn('LocalStorage 용량 초과 - 청크 단위로 분할 저장합니다.');
        try {
          // 기존 데이터 정리
          localStorage.removeItem('batchAnalysisResults');
          
          // 데이터 구조 확인 및 안전한 청크 분할
          const resultArray = results.results || results.data || (Array.isArray(results) ? results : []);
          
          if (!Array.isArray(resultArray) || resultArray.length === 0) {
            console.error('유효하지 않은 결과 데이터 구조:', results);
            throw new Error('Invalid results structure');
          }
          
          // 동적 청크 크기 계산 (메모리 효율성 고려)
          const chunkSize = Math.max(100, Math.min(500, Math.floor(4000000 / JSON.stringify(resultArray[0] || {}).length)));
          const chunks = [];
          
          for (let i = 0; i < resultArray.length; i += chunkSize) {
            chunks.push(resultArray.slice(i, i + chunkSize));
          }
          
          // 각 청크를 개별 키로 저장 (안전한 저장)
          let savedChunks = 0;
          for (let i = 0; i < chunks.length; i++) {
            try {
              const chunkData = {
                chunk_index: i,
                total_chunks: chunks.length,
                data: chunks[i],
                timestamp: timestamp
              };
              localStorage.setItem(`batchAnalysisResults_chunk_${i}`, JSON.stringify(chunkData));
              savedChunks++;
            } catch (chunkError) {
              console.error(`청크 ${i} 저장 실패:`, chunkError);
              break; // 더 이상 저장할 수 없으면 중단
            }
          }
          
          // 메타데이터 저장
          const metadata = {
            total_employees: resultArray.length,
            saved_employees: Math.min(savedChunks * chunkSize, resultArray.length),
            total_chunks: chunks.length,
            saved_chunks: savedChunks,
            chunk_size: chunkSize,
            timestamp: timestamp,
            storage_type: 'chunked',
            original_structure: {
              has_results: !!results.results,
              has_data: !!results.data,
              is_array: Array.isArray(results)
            }
          };
          
          localStorage.setItem('batchAnalysisMetadata', JSON.stringify(metadata));
          localStorage.setItem('lastAnalysisTimestamp', timestamp);
          console.log(`청크 분할 저장 완료: ${savedChunks}/${chunks.length}개 청크, 총 ${Math.min(savedChunks * chunkSize, resultArray.length)}/${resultArray.length}명`);
        } catch (secondError) {
          console.error('청크 저장도 실패:', secondError);
          // 최후의 수단: 요약 데이터만 저장
          try {
            localStorage.clear();
            
            // 안전한 데이터 추출
            const resultArray = results.results || results.data || (Array.isArray(results) ? results : []);
            
            const summaryResults = {
              total_employees: resultArray.length,
              timestamp: timestamp,
              summary: {
                high_risk: resultArray.filter(r => {
                  const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                  return score && score >= 0.7;
                }).length,
                medium_risk: resultArray.filter(r => {
                  const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                  return score && score >= 0.3 && score < 0.7;
                }).length,
                low_risk: resultArray.filter(r => {
                  const score = r.analysis_result?.combined_analysis?.integrated_assessment?.overall_risk_score;
                  return score && score < 0.3;
                }).length
              },
              departments: [...new Set(resultArray.map(r => 
                r.analysis_result?.employee_data?.Department || 
                r.department || 
                'Unknown'
              ))],
              storage_type: 'summary_only',
              error_info: {
                original_error: secondError.message,
                data_structure: {
                  has_results: !!results.results,
                  has_data: !!results.data,
                  is_array: Array.isArray(results),
                  result_count: resultArray.length
                }
              }
            };
            
            localStorage.setItem('batchAnalysisResultsSummary', JSON.stringify(summaryResults));
            localStorage.setItem('lastAnalysisTimestamp', timestamp);
            console.log('⚠️ 요약 데이터만 저장 완료:', summaryResults);
          } catch (finalError) {
            console.error('❌ 모든 저장 방식 실패:', finalError);
            // 최종 실패 시에도 타임스탬프는 저장 시도
            try {
              localStorage.setItem('lastAnalysisTimestamp', timestamp);
              localStorage.setItem('batchAnalysisError', JSON.stringify({
                error: finalError.message,
                timestamp: timestamp,
                attempted_storage: 'all_methods_failed'
              }));
            } catch (timestampError) {
              console.error('타임스탬프 저장도 실패:', timestampError);
            }
          }
        }
      } else {
        console.error('배치 결과 저장 실패:', error);
      }
    }
  };

  const renderContent = () => {
    const commonProps = {
      loading,
      setLoading: setGlobalLoading,
      serverStatus,
      dataLoaded,
      // 전역 배치 결과 전달
      globalBatchResults,
      lastAnalysisTimestamp,
      updateBatchResults, // 배치 결과 업데이트 함수
    };

    switch (selectedKey) {
      case 'home':
        return (
          <Home
            {...commonProps}
            onNavigate={setSelectedKey}
          />
        );
      case 'batch':
        return (
          <BatchAnalysis
            {...commonProps}
            onNavigate={setSelectedKey}
          />
        );
      case 'cognita':
        return (
          <RelationshipAnalysis
            {...commonProps}
            batchResults={globalBatchResults} // 전역 상태 사용
          />
        );
      case 'post-analysis':
        return (
          <PostAnalysis 
            {...commonProps}
          />
        );
      case 'report-generation':
        return (
          <ReportGeneration 
            {...commonProps}
          />
        );
      case 'group-statistics':
        return (
          <GroupStatistics
            {...commonProps}
          />
        );
      default:
        return (
          <Home
            {...commonProps}
            onNavigate={setSelectedKey}
          />
        );
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {/* 사이드바 */}
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        theme="light"
        width={280}
        style={{
          boxShadow: '2px 0 8px rgba(0,0,0,0.1)',
        }}
      >
        <div style={{ 
          padding: '20px 16px', 
          textAlign: 'center',
          borderBottom: '1px solid #f0f0f0',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          height: '84px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center'
        }}>
          <Title level={4} style={{ margin: 0, color: 'white' }}>
            Retain Sentinel 360
          </Title>
          <div style={{ fontSize: '12px', opacity: 0.9 }}>
            AI 기반 이직 예측 시스템
          </div>
        </div>
        
        <Menu
          mode="inline"
          selectedKeys={[selectedKey]}
          onClick={({ key }) => setSelectedKey(key)}
          items={menuItems}
          style={{ 
            border: 'none',
            fontSize: '14px',
            paddingTop: '8px'
          }}
        />
      </Sider>

      {/* 메인 콘텐츠 */}
      <Layout>
        <Header style={{ 
          background: '#fff', 
          padding: '0 24px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          height: '64px',
          lineHeight: '64px'
        }}>
          <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
            {selectedKey === 'home' ? '홈' : 
             selectedKey === 'batch' ? '배치 분석' :
             selectedKey === 'group-statistics' ? '단체 통계' :
             selectedKey === 'cognita' ? '개별 관계분석' :
             selectedKey === 'post-analysis' ? '사후 분석' :
             selectedKey === 'report-generation' ? '보고서 출력' : '대시보드'}
          </Title>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            {/* 서버 상태 표시 */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: serverStatus ? '#52c41a' : '#ff4d4f'
              }} />
              <span style={{ fontSize: '14px', color: '#666' }}>
                {serverStatus ? '서버 연결됨' : '서버 연결 실패'}
              </span>
            </div>
            
            {/* 새로고침 버튼 */}
            <button
              onClick={checkServerStatus}
              style={{
                border: 'none',
                background: 'transparent',
                cursor: 'pointer',
                fontSize: '16px',
                color: '#1890ff'
              }}
              title="서버 상태 새로고침"
            >
              🔄 다시 확인
            </button>
          </div>
        </Header>
        
        <Content style={{ 
          margin: '16px 24px',
          padding: '24px',
          background: '#fff',
          borderRadius: '8px',
          overflow: 'auto',
          minHeight: 'calc(100vh - 64px - 32px)'
        }}>
          {renderContent()}
        </Content>
      </Layout>
    </Layout>
  );
};

export default App;
