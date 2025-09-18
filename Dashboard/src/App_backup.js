import React, { useState, useEffect } from 'react';
import { Layout, Menu, Typography, notification, Spin } from 'antd';
import {
  DashboardOutlined,
  UploadOutlined,
  CalculatorOutlined,
  SettingOutlined,
  UserOutlined,
  BarChartOutlined,
  FileTextOutlined,
  ApiOutlined,
  RobotOutlined,
  ExperimentOutlined
} from '@ant-design/icons';

import Dashboard from './components/Dashboard';
import FileUpload from './components/FileUpload';
import ExportResults from './components/ExportResults';
import BatchAnalysis from './components/BatchAnalysis';
import PostAnalysis from './components/PostAnalysis';
import { apiService } from './services/apiService';

const { Header, Sider, Content } = Layout;
const { Title } = Typography;

const App = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [selectedKey, setSelectedKey] = useState('batch');
  const [loading, setLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState(null);
  const [dataLoaded, setDataLoaded] = useState(false);

  // 전역 에러 핸들러
  useEffect(() => {
    const handleUnhandledRejection = (event) => {
      console.error('Unhandled promise rejection:', event.reason);
      // Chrome 확장 프로그램 관련 오류는 무시
      if (event.reason && event.reason.message && 
          (event.reason.message.includes('Extension context invalidated') ||
           event.reason.message.includes('message channel closed') ||
           event.reason.message.includes('disconnected port object'))) {
        event.preventDefault();
        return;
      }
      
      notification.error({
        message: '예상치 못한 오류',
        description: '시스템에서 예상치 못한 오류가 발생했습니다.',
        duration: 4.5,
      });
    };

    const handleError = (event) => {
      console.error('Global error:', event.error);
      // Chrome 확장 프로그램 관련 오류는 무시
      if (event.error && event.error.message && 
          (event.error.message.includes('Extension context invalidated') ||
           event.error.message.includes('message channel closed') ||
           event.error.message.includes('disconnected port object'))) {
        event.preventDefault();
        return;
      }
    };

    window.addEventListener('unhandledrejection', handleUnhandledRejection);
    window.addEventListener('error', handleError);

    return () => {
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
      window.removeEventListener('error', handleError);
    };
  }, []);

  // 메뉴 아이템
  const menuItems = [
    {
      key: 'batch',
      icon: <RobotOutlined />,
      label: '🎯 배치 분석 (메인)',
    },
    {
      key: 'dashboard',
      icon: <DashboardOutlined />,
      label: '📊 시스템 현황',
    },
    {
      key: 'cognita',
      icon: <ApiOutlined />,
      label: '🕸️ 개별 관계분석',
    },
    {
      key: 'post-analysis',
      icon: <BarChartOutlined />,
      label: '📈 사후 분석',
    },
    {
      key: 'export',
      icon: <FileTextOutlined />,
      label: '📋 결과 내보내기',
    },
  ];

  // 서버 상태 확인
  useEffect(() => {
    checkServerStatus();
  }, []);

  const checkServerStatus = async () => {
    try {
      const status = await apiService.checkHealth();
      setServerStatus(status);
      notification.success({
        message: '서버 연결 성공',
        description: `${status.service} v${status.version} 연결됨`,
        duration: 3,
      });
    } catch (error) {
      setServerStatus(null);
      notification.error({
        message: '서버 연결 실패',
        description: 'Agentic AI System 서버가 실행 중인지 확인해주세요.',
        duration: 5,
      });
    }
  };

  // 단계 상태 업데이트 함수
  const updateStepStatus = (step, status) => {
    setStepStatuses(prev => ({
      ...prev,
      [step]: status
    }));
  };

  // 다음 단계로 이동
  const moveToNextStep = () => {
    setCurrentStep(prev => Math.min(prev + 1, 6));
  };

  // 데이터 로드 성공 콜백
  const onDataLoaded = (success) => {
    setDataLoaded(success);
    updateStepStatus('upload', success);
    if (success) {
      notification.success({
        message: '데이터 로드 성공',
        description: '데이터가 성공적으로 로드되었습니다.',
      });
      moveToNextStep();
    }
  };

  // 임계값 계산 완료 콜백
  const onThresholdCalculated = (results) => {
    setThresholdResults(results);
    updateStepStatus('threshold', true);
    notification.success({
      message: '임계값 계산 완료',
      description: '모든 Score의 최적 임계값이 계산되었습니다.',
    });
    moveToNextStep();
  };

  // 가중치 최적화 완료 콜백
  const onWeightOptimized = (results) => {
    setWeightResults(results);
    updateStepStatus('weight', true);
    notification.success({
      message: '가중치 최적화 완료',
      description: `${results.method} 방법으로 최적화가 완료되었습니다.`,
    });
    moveToNextStep();
  };

  // Integration 분석 완료 콜백
  const onIntegrationCompleted = (results) => {
    setIntegrationResults(results);
    updateStepStatus('integration', true);
    notification.success({
      message: 'Integration 분석 완료',
      description: '통합 분석이 성공적으로 완료되었습니다.',
    });
    moveToNextStep();
  };

  // Supervisor 워크플로우 완료 콜백
  const onSupervisorCompleted = (results) => {
    setSupervisorResults(results);
    updateStepStatus('supervisor', true);
    notification.success({
      message: 'Supervisor 워크플로우 완료',
      description: '슈퍼바이저 워크플로우가 성공적으로 완료되었습니다.',
    });
    moveToNextStep();
  };

  // XAI 결과 완료 콜백
  const onXAICompleted = (results) => {
    setXAIResults(results);
    updateStepStatus('xai', true);
    notification.success({
      message: 'XAI 분석 완료',
      description: 'XAI 분석이 성공적으로 완료되었습니다.',
    });
  };

  // 로딩 상태 관리
  const setGlobalLoading = (isLoading) => {
    setLoading(isLoading);
  };

  // 현재 선택된 컴포넌트 렌더링
  const renderContent = () => {
    const commonProps = {
      loading,
      setLoading: setGlobalLoading,
      serverStatus,
      dataLoaded,
      thresholdResults,
      weightResults,
      integrationResults,
      supervisorResults,
      xaiResults,
      currentStep,
      stepStatuses,
    };

    switch (selectedKey) {
      case 'batch':
        return (
          <BatchAnalysis
            {...commonProps}
          />
        );
      case 'dashboard':
        return (
          <Dashboard
            {...commonProps}
            onRefreshStatus={checkServerStatus}
          />
        );
      case 'cognita':
        return (
          <FileUpload
            {...commonProps}
            moduleType="cognita"
            onDataLoaded={onDataLoaded}
          />
        );
      case 'post-analysis':
        return (
          <PostAnalysis
            {...commonProps}
          />
        );
      case 'export':
        return (
          <ExportResults
            {...commonProps}
          />
        );
      default:
        return (
          <BatchAnalysis
            {...commonProps}
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
        width={250}
        style={{
          boxShadow: '2px 0 8px rgba(0,0,0,0.1)',
          zIndex: 100,
        }}
      >
        <div style={{ 
          padding: '16px', 
          textAlign: 'center',
          borderBottom: '1px solid #f0f0f0',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white'
        }}>
          <Title level={4} style={{ color: 'white', margin: 0 }}>
            {collapsed ? 'FC' : 'Final_calc'}
          </Title>
          {!collapsed && (
            <div style={{ fontSize: '12px', opacity: 0.8 }}>
              HR Attrition 예측 시스템
            </div>
          )}
        </div>
        
        <Menu
          mode="inline"
          selectedKeys={[selectedKey]}
          items={menuItems}
          onClick={({ key }) => setSelectedKey(key)}
          style={{ border: 'none', paddingTop: '16px' }}
        />
        
        {/* 서버 상태 표시 */}
        {!collapsed && (
          <div style={{ 
            position: 'absolute', 
            bottom: '16px', 
            left: '16px', 
            right: '16px',
            padding: '12px',
            background: serverStatus ? '#f6ffed' : '#fff2f0',
            border: `1px solid ${serverStatus ? '#b7eb8f' : '#ffccc7'}`,
            borderRadius: '6px',
            fontSize: '12px'
          }}>
            <div style={{ 
              color: serverStatus ? '#52c41a' : '#ff4d4f',
              fontWeight: 'bold'
            }}>
              {serverStatus ? '🟢 서버 연결됨' : '🔴 서버 연결 안됨'}
            </div>
            {serverStatus && (
              <div style={{ color: '#666', marginTop: '4px' }}>
                {serverStatus.service} v{serverStatus.version}
              </div>
            )}
          </div>
        )}
      </Sider>

      {/* 메인 레이아웃 */}
      <Layout>
        {/* 헤더 */}
        <Header className="dashboard-header">
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            height: '100%'
          }}>
            <Title level={3} style={{ color: 'white', margin: 0 }}>
              {menuItems.find(item => item.key === selectedKey)?.label || '대시보드'}
            </Title>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              {/* 상태 표시 */}
              <div style={{ display: 'flex', gap: '8px', fontSize: '12px', flexWrap: 'wrap' }}>
                <span style={{ 
                  color: stepStatuses.upload ? '#52c41a' : '#faad14',
                  fontWeight: 'bold'
                }}>
                  📊 데이터: {stepStatuses.upload ? '✓' : '○'}
                </span>
                <span style={{ 
                  color: stepStatuses.threshold ? '#52c41a' : '#faad14',
                  fontWeight: 'bold'
                }}>
                  🎯 임계값: {stepStatuses.threshold ? '✓' : '○'}
                </span>
                <span style={{ 
                  color: stepStatuses.weight ? '#52c41a' : '#faad14',
                  fontWeight: 'bold'
                }}>
                  ⚖️ 가중치: {stepStatuses.weight ? '✓' : '○'}
                </span>
                <span style={{ 
                  color: stepStatuses.integration ? '#52c41a' : '#faad14',
                  fontWeight: 'bold'
                }}>
                  🔗 통합: {stepStatuses.integration ? '✓' : '○'}
                </span>
                <span style={{ 
                  color: stepStatuses.supervisor ? '#52c41a' : '#faad14',
                  fontWeight: 'bold'
                }}>
                  🤖 워크플로우: {stepStatuses.supervisor ? '✓' : '○'}
                </span>
                <span style={{ 
                  color: stepStatuses.xai ? '#52c41a' : '#faad14',
                  fontWeight: 'bold'
                }}>
                  🔬 XAI: {stepStatuses.xai ? '✓' : '○'}
                </span>
              </div>
            </div>
          </div>
        </Header>

        {/* 컨텐츠 */}
        <Content className="dashboard-content">
          {renderContent()}
        </Content>
      </Layout>

      {/* 전역 로딩 오버레이 */}
      {loading && (
        <div className="loading-overlay">
          <Spin size="large" tip="처리 중입니다..." />
        </div>
      )}
    </Layout>
  );
};

export default App;
