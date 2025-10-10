import React, { useState, useEffect, useRef } from 'react';
import {
  Card,
  Button,
  Steps,
  Typography,
  Row,
  Col,
  Timeline,
  Tag,
  Alert,
  Progress,
  Space,
  Divider,
  Input,
  Switch,
  Modal,
  Avatar
} from 'antd';
import {
  RobotOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  SettingOutlined,
  MessageOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  EyeOutlined,
  DownloadOutlined
} from '@ant-design/icons';
import { apiService } from '../services/apiService';

const { Title, Text, Paragraph } = Typography;

// 타이핑 애니메이션 컴포넌트
const TypingText = ({ text, speed = 50, onComplete }) => {
  const [displayText, setDisplayText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (currentIndex < text.length) {
      intervalRef.current = setTimeout(() => {
        setDisplayText(prev => prev + text[currentIndex]);
        setCurrentIndex(prev => prev + 1);
      }, speed);
    } else if (onComplete) {
      onComplete();
    }

    return () => {
      if (intervalRef.current) {
        clearTimeout(intervalRef.current);
      }
    };
  }, [currentIndex, text, speed, onComplete]);

  useEffect(() => {
    setDisplayText('');
    setCurrentIndex(0);
  }, [text]);

  return (
    <div style={{ minHeight: '20px', fontFamily: 'monospace' }}>
      {displayText}
      {currentIndex < text.length && (
        <span style={{ animation: 'blink 1s infinite' }}>|</span>
      )}
    </div>
  );
};

const SupervisorWorkflow = ({
  loading,
  setLoading,
  serverStatus,
  stepStatuses,
  integrationResults,
  onSupervisorCompleted
}) => {
  const [workflowStatus, setWorkflowStatus] = useState('idle'); // idle, running, paused, completed, error
  const [currentStep, setCurrentStep] = useState(0);
  const [workflowLogs, setWorkflowLogs] = useState([]);
  const [agentMessages, setAgentMessages] = useState([]);
  const [progress, setProgress] = useState(0);
  const [sessionId, setSessionId] = useState(null);
  const [workflowConfig, setWorkflowConfig] = useState({
    maxIterations: 10,
    timeout: 300,
    enableLogging: true,
    autoMode: true
  });
  const [currentMessage, setCurrentMessage] = useState('');
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [workflowResults, setWorkflowResults] = useState(null);

  // 워크플로우 단계 정의
  const workflowSteps = [
    {
      title: '초기화',
      description: '슈퍼바이저 에이전트 및 워커 에이전트 초기화',
      icon: <SettingOutlined />
    },
    {
      title: '데이터 분석',
      description: '통합된 데이터에 대한 종합적 분석 수행',
      icon: <RobotOutlined />
    },
    {
      title: '에이전트 협업',
      description: '다중 에이전트 간 협업을 통한 심화 분석',
      icon: <MessageOutlined />
    },
    {
      title: '결과 종합',
      description: '모든 에이전트의 분석 결과를 종합하여 최종 결론 도출',
      icon: <CheckCircleOutlined />
    }
  ];

  // 워크플로우 시작
  const startWorkflow = async () => {
    try {
      setLoading(true);
      setWorkflowStatus('running');
      setCurrentStep(0);
      setProgress(0);
      setWorkflowLogs([]);
      setAgentMessages([]);

      // 세션 시작
      const sessionResponse = await apiService.startSupervisorSession({
        config: workflowConfig,
        integrationData: integrationResults
      });
      
      setSessionId(sessionResponse.session_id);
      addLog('워크플로우 세션이 시작되었습니다.', 'info');
      
      // 워크플로우 실행
      await executeWorkflow(sessionResponse.session_id);
      
    } catch (error) {
      setWorkflowStatus('error');
      addLog(`워크플로우 시작 실패: ${error.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  // 워크플로우 실행
  const executeWorkflow = async (sessionId) => {
    try {
      setCurrentStep(1);
      setProgress(25);
      addLog('데이터 분석을 시작합니다...', 'info');

      // 1단계: 데이터 분석
      const analysisResponse = await apiService.runSupervisorAnalysis(sessionId, {
        step: 'data_analysis',
        data: integrationResults
      });
      
      await simulateAgentMessage('데이터 분석이 완료되었습니다. 주요 패턴을 식별했습니다.');
      setCurrentStep(2);
      setProgress(50);

      // 2단계: 에이전트 협업
      addLog('에이전트 협업 단계를 시작합니다...', 'info');
      const collaborationResponse = await apiService.runAgentCollaboration(sessionId);
      
      await simulateAgentMessage('다중 에이전트 협업을 통해 심화 분석을 수행했습니다.');
      setCurrentStep(3);
      setProgress(75);

      // 3단계: 결과 종합
      addLog('결과를 종합하고 있습니다...', 'info');
      const synthesisResponse = await apiService.synthesizeResults(sessionId);
      
      await simulateAgentMessage('모든 분석이 완료되었습니다. 최종 결과를 생성합니다.');
      setProgress(100);
      setCurrentStep(4);

      // 최종 결과 설정
      const finalResults = {
        sessionId,
        analysis: analysisResponse,
        collaboration: collaborationResponse,
        synthesis: synthesisResponse,
        timestamp: new Date().toISOString()
      };

      setWorkflowResults(finalResults);
      setWorkflowStatus('completed');
      addLog('워크플로우가 성공적으로 완료되었습니다.', 'success');
      
      onSupervisorCompleted(finalResults);

    } catch (error) {
      setWorkflowStatus('error');
      addLog(`워크플로우 실행 중 오류: ${error.message}`, 'error');
    }
  };

  // 에이전트 메시지 시뮬레이션
  const simulateAgentMessage = (message) => {
    return new Promise((resolve) => {
      setCurrentMessage(message);
      setTimeout(() => {
        setAgentMessages(prev => [...prev, {
          id: Date.now(),
          message,
          timestamp: new Date(),
          agent: 'Supervisor',
          type: 'info'
        }]);
        setCurrentMessage('');
        resolve();
      }, message.length * 50 + 1000); // 타이핑 시간 + 여유시간
    });
  };

  // 로그 추가
  const addLog = (message, type = 'info') => {
    setWorkflowLogs(prev => [...prev, {
      id: Date.now(),
      message,
      type,
      timestamp: new Date()
    }]);
  };

  // 워크플로우 일시정지
  const pauseWorkflow = async () => {
    try {
      if (sessionId) {
        await apiService.pauseSupervisorSession(sessionId);
        setWorkflowStatus('paused');
        addLog('워크플로우가 일시정지되었습니다.', 'warning');
      }
    } catch (error) {
      addLog(`일시정지 실패: ${error.message}`, 'error');
    }
  };

  // 워크플로우 중지
  const stopWorkflow = async () => {
    try {
      if (sessionId) {
        await apiService.stopSupervisorSession(sessionId);
        setWorkflowStatus('idle');
        setCurrentStep(0);
        setProgress(0);
        addLog('워크플로우가 중지되었습니다.', 'warning');
      }
    } catch (error) {
      addLog(`중지 실패: ${error.message}`, 'error');
    }
  };

  // 워크플로우 재시작
  const resumeWorkflow = async () => {
    try {
      if (sessionId) {
        await apiService.resumeSupervisorSession(sessionId);
        setWorkflowStatus('running');
        addLog('워크플로우가 재시작되었습니다.', 'info');
      }
    } catch (error) {
      addLog(`재시작 실패: ${error.message}`, 'error');
    }
  };

  // 상태별 색상 반환
  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return '#1890ff';
      case 'completed': return '#52c41a';
      case 'paused': return '#faad14';
      case 'error': return '#ff4d4f';
      default: return '#d9d9d9';
    }
  };

  // 상태별 아이콘 반환
  const getStatusIcon = (status) => {
    switch (status) {
      case 'running': return <PlayCircleOutlined />;
      case 'completed': return <CheckCircleOutlined />;
      case 'paused': return <PauseCircleOutlined />;
      case 'error': return <ExclamationCircleOutlined />;
      default: return <ClockCircleOutlined />;
    }
  };

  return (
    <div style={{ padding: '24px' }}>
      <style>
        {`
          @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
          }
        `}
      </style>
      
      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card>
            <Title level={2}>
              <RobotOutlined style={{ marginRight: 8 }} />
              Supervisor 워크플로우
            </Title>
            <Paragraph>
              슈퍼바이저 에이전트가 다중 에이전트 시스템을 조율하여 
              통합 분석 결과에 대한 심화 분석을 수행합니다.
            </Paragraph>
          </Card>
        </Col>

        <Col span={24}>
          <Card 
            title="워크플로우 상태" 
            extra={
              <Space>
                <Tag color={getStatusColor(workflowStatus)} icon={getStatusIcon(workflowStatus)}>
                  {workflowStatus.toUpperCase()}
                </Tag>
                <Button
                  icon={<SettingOutlined />}
                  onClick={() => setShowConfigModal(true)}
                  disabled={workflowStatus === 'running'}
                >
                  설정
                </Button>
              </Space>
            }
          >
            <Steps current={currentStep} items={workflowSteps} />
            
            {progress > 0 && (
              <div style={{ marginTop: 16 }}>
                <Progress 
                  percent={progress} 
                  status={workflowStatus === 'running' ? 'active' : 
                          workflowStatus === 'completed' ? 'success' : 
                          workflowStatus === 'error' ? 'exception' : 'normal'}
                  strokeColor={{
                    '0%': '#108ee9',
                    '100%': '#87d068',
                  }}
                />
              </div>
            )}

            <div style={{ marginTop: 16 }}>
              <Space>
                {workflowStatus === 'idle' && (
                  <Button
                    type="primary"
                    icon={<PlayCircleOutlined />}
                    onClick={startWorkflow}
                    disabled={!integrationResults}
                    loading={loading}
                  >
                    워크플로우 시작
                  </Button>
                )}
                
                {workflowStatus === 'running' && (
                  <>
                    <Button
                      icon={<PauseCircleOutlined />}
                      onClick={pauseWorkflow}
                    >
                      일시정지
                    </Button>
                    <Button
                      danger
                      icon={<StopOutlined />}
                      onClick={stopWorkflow}
                    >
                      중지
                    </Button>
                  </>
                )}
                
                {workflowStatus === 'paused' && (
                  <>
                    <Button
                      type="primary"
                      icon={<PlayCircleOutlined />}
                      onClick={resumeWorkflow}
                    >
                      재시작
                    </Button>
                    <Button
                      danger
                      icon={<StopOutlined />}
                      onClick={stopWorkflow}
                    >
                      중지
                    </Button>
                  </>
                )}
              </Space>
            </div>
          </Card>
        </Col>

        {!integrationResults && (
          <Col span={24}>
            <Alert
              message="Integration 분석 필요"
              description="Supervisor 워크플로우를 시작하기 전에 Integration 분석을 완료해주세요."
              type="warning"
              showIcon
            />
          </Col>
        )}

        <Col xs={24} lg={12}>
          <Card title="실시간 에이전트 메시지" style={{ height: '400px' }}>
            <div style={{ height: '320px', overflowY: 'auto' }}>
              {currentMessage && (
                <div style={{ 
                  padding: '12px', 
                  backgroundColor: '#f0f2f5', 
                  borderRadius: '8px',
                  marginBottom: '8px'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                    <Avatar size="small" icon={<RobotOutlined />} />
                    <Text strong style={{ marginLeft: '8px' }}>Supervisor Agent</Text>
                  </div>
                  <TypingText text={currentMessage} speed={30} />
                </div>
              )}
              
              {agentMessages.map(msg => (
                <div key={msg.id} style={{ 
                  padding: '12px', 
                  backgroundColor: '#fff', 
                  border: '1px solid #d9d9d9',
                  borderRadius: '8px',
                  marginBottom: '8px'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                      <Avatar size="small" icon={<RobotOutlined />} />
                      <Text strong style={{ marginLeft: '8px' }}>{msg.agent}</Text>
                    </div>
                    <Text type="secondary" style={{ fontSize: 'var(--font-small)' }}>
                      {msg.timestamp.toLocaleTimeString()}
                    </Text>
                  </div>
                  <Text>{msg.message}</Text>
                </div>
              ))}
            </div>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card title="워크플로우 로그" style={{ height: '400px' }}>
            <div style={{ height: '320px', overflowY: 'auto' }}>
              <Timeline
                items={workflowLogs.map(log => ({
                  key: log.id,
                  color: log.type === 'error' ? 'red' :
                         log.type === 'warning' ? 'orange' :
                         log.type === 'success' ? 'green' : 'blue',
                  children: (
                    <div>
                      <Text strong>{log.timestamp.toLocaleTimeString()}</Text>
                      <br />
                      <Text>{log.message}</Text>
                    </div>
                  )
                }))}
              />
            </div>
          </Card>
        </Col>

        {workflowResults && (
          <Col span={24}>
            <Card title="워크플로우 결과">
              <Row gutter={[16, 16]}>
                <Col xs={24} sm={8}>
                  <Card size="small">
                    <div style={{ textAlign: 'center' }}>
                      <Title level={4}>분석 완료</Title>
                      <CheckCircleOutlined style={{ fontSize: 'var(--icon-xlarge)', color: '#52c41a' }} />
                    </div>
                  </Card>
                </Col>
                <Col xs={24} sm={8}>
                  <Card size="small">
                    <div style={{ textAlign: 'center' }}>
                      <Title level={4}>처리 시간</Title>
                      <Text strong>
                        {workflowResults.timestamp ? 
                          `${Math.round((new Date(workflowResults.timestamp) - new Date()) / -1000)}초` : 
                          'N/A'
                        }
                      </Text>
                    </div>
                  </Card>
                </Col>
                <Col xs={24} sm={8}>
                  <Card size="small">
                    <div style={{ textAlign: 'center' }}>
                      <Title level={4}>세션 ID</Title>
                      <Text code>{workflowResults.sessionId}</Text>
                    </div>
                  </Card>
                </Col>
              </Row>

              <Divider />

              <Space>
                <Button
                  type="primary"
                  icon={<EyeOutlined />}
                  onClick={() => {
                    Modal.info({
                      title: '워크플로우 상세 결과',
                      width: 800,
                      content: (
                        <pre style={{ maxHeight: '400px', overflow: 'auto' }}>
                          {JSON.stringify(workflowResults, null, 2)}
                        </pre>
                      )
                    });
                  }}
                >
                  상세 결과 보기
                </Button>
                
                <Button
                  icon={<DownloadOutlined />}
                  onClick={() => {
                    const blob = new Blob([JSON.stringify(workflowResults, null, 2)], {
                      type: 'application/json'
                    });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `supervisor_workflow_${new Date().toISOString().split('T')[0]}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                >
                  결과 다운로드
                </Button>
              </Space>
            </Card>
          </Col>
        )}
      </Row>

      {/* 설정 모달 */}
      <Modal
        title="워크플로우 설정"
        open={showConfigModal}
        onOk={() => setShowConfigModal(false)}
        onCancel={() => setShowConfigModal(false)}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text strong>최대 반복 횟수</Text>
            <Input
              type="number"
              value={workflowConfig.maxIterations}
              onChange={(e) => setWorkflowConfig(prev => ({
                ...prev,
                maxIterations: parseInt(e.target.value) || 10
              }))}
              style={{ marginTop: 8 }}
            />
          </div>
          
          <div>
            <Text strong>타임아웃 (초)</Text>
            <Input
              type="number"
              value={workflowConfig.timeout}
              onChange={(e) => setWorkflowConfig(prev => ({
                ...prev,
                timeout: parseInt(e.target.value) || 300
              }))}
              style={{ marginTop: 8 }}
            />
          </div>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text strong>로깅 활성화</Text>
            <Switch
              checked={workflowConfig.enableLogging}
              onChange={(checked) => setWorkflowConfig(prev => ({
                ...prev,
                enableLogging: checked
              }))}
            />
          </div>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text strong>자동 모드</Text>
            <Switch
              checked={workflowConfig.autoMode}
              onChange={(checked) => setWorkflowConfig(prev => ({
                ...prev,
                autoMode: checked
              }))}
            />
          </div>
        </Space>
      </Modal>
    </div>
  );
};

export default SupervisorWorkflow;
