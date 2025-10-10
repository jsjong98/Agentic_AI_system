import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Button,
  Input,
  List,
  Avatar,
  Spin,
  message,
  Divider,
  Tag,
  Space,
  Modal,
  Descriptions,
  Badge,
  Timeline,
  Statistic,
  Progress
} from 'antd';
import {
  SendOutlined,
  RobotOutlined,
  UserOutlined,
  HistoryOutlined,
  EyeOutlined,
  TrophyOutlined,
  TeamOutlined,
  BarChartOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  DownloadOutlined,
  UploadOutlined,
  DeleteOutlined,
  SettingOutlined,
  VerticalAlignBottomOutlined
} from '@ant-design/icons';
import { predictionService } from '../services/predictionService';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;

// 타이핑 커서 애니메이션을 위한 스타일
const typingCursorStyle = `
  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
  }
`;

// 스타일을 head에 추가
if (typeof document !== 'undefined' && !document.getElementById('typing-cursor-style')) {
  const style = document.createElement('style');
  style.id = 'typing-cursor-style';
  style.textContent = typingCursorStyle;
  document.head.appendChild(style);
}

const Home = ({ globalBatchResults, lastAnalysisTimestamp, onNavigate }) => {
  const [chatMessages, setChatMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [historyManageVisible, setHistoryManageVisible] = useState(false);
  const [typingMessageId, setTypingMessageId] = useState(null);
  const [typingText, setTypingText] = useState('');
  const [isInitialLoad, setIsInitialLoad] = useState(true); // 초기 로드 상태 추가
  const [userHasSentMessage, setUserHasSentMessage] = useState(false); // 사용자 메시지 전송 여부
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const typingIntervalRef = useRef(null);
  const welcomeMessageShown = useRef(false); // 환영 메시지 표시 여부 추적

  // Chrome 확장 프로그램 오류 필터링
  useEffect(() => {
    const originalError = console.error;
    console.error = (...args) => {
      // Chrome 확장 프로그램 관련 오류 필터링
      const errorMessage = args.join(' ');
      if (errorMessage.includes('chrome-extension://') || 
          errorMessage.includes('ERR_FAILED') ||
          errorMessage.includes('net::ERR_')) {
        // Chrome 확장 프로그램 오류는 무시
        return;
      }
      // 실제 애플리케이션 오류만 출력
      originalError.apply(console, args);
    };

    return () => {
      console.error = originalError;
    };
  }, []);

  // 타이핑 효과 함수 (useEffect보다 먼저 정의)
  const startTypingEffect = useCallback((messageId, fullText, onComplete, shouldScroll = true) => {
    // 기존 타이핑 효과가 있다면 중단
    if (typingIntervalRef.current) {
      clearTimeout(typingIntervalRef.current);
    }

    setTypingMessageId(messageId);
    setTypingText('');
    
    let currentIndex = 0;
    const baseTypingSpeed = 25; // 기본 타이핑 속도 (밀리초)
    
    const typeNextChar = () => {
      if (currentIndex < fullText.length) {
        const nextChar = fullText[currentIndex];
        setTypingText(fullText.substring(0, currentIndex + 1));
        currentIndex++;
        
        // 자동 스크롤 (타이핑 중에도) - shouldScroll이 true이고 사용자가 메시지를 보낸 후에만
        if (shouldScroll && userHasSentMessage && !isInitialLoad) {
          setTimeout(() => {
            chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
          }, 10);
        }
        
        // 문장 부호나 줄바꿈에서 약간 더 긴 지연
        let delay = baseTypingSpeed;
        if (nextChar === '.' || nextChar === '!' || nextChar === '?') {
          delay = 300; // 문장 끝에서 더 긴 지연
        } else if (nextChar === '\n') {
          delay = 200; // 줄바꿈에서 중간 지연
        } else if (nextChar === ',') {
          delay = 150; // 쉼표에서 짧은 지연
        }
        
        typingIntervalRef.current = setTimeout(typeNextChar, delay);
      } else {
        // 타이핑 완료
        setTypingMessageId(null);
        setTypingText('');
        
        // 최종 메시지를 채팅에 추가
        if (onComplete) {
          onComplete();
        }
      }
    };
    
    // 타이핑 시작
    typingIntervalRef.current = setTimeout(typeNextChar, baseTypingSpeed);
  }, [userHasSentMessage, isInitialLoad]);

  // 초기 환영 메시지 및 예측 데이터 로드
  useEffect(() => {
    // 예측 결과 히스토리 먼저 로드
    loadPredictionHistory();
  }, []);

  // 히스토리 로드 후 환영 메시지 설정
  useEffect(() => {
    // 환영 메시지가 이미 표시되었다면 실행하지 않음
    if (welcomeMessageShown.current) {
      return;
    }

    const welcomeContent = predictionHistory.length > 0
      ? '안녕하세요! Retain Sentinel 360 AI 어시스턴트입니다. 저장된 분석 결과를 바탕으로 질문에 답변해드리겠습니다.'
      : '안녕하세요! Retain Sentinel 360 AI 어시스턴트입니다. 먼저 배치 분석을 실행하시면 분석 결과에 대해 질문하실 수 있습니다.';

    const welcomeMessage = {
      id: 1,
      type: 'bot',
      content: welcomeContent,
      timestamp: new Date().toISOString()
    };

    // 환영 메시지 표시 플래그 설정
    welcomeMessageShown.current = true;

    // 환영 메시지도 타이핑 효과로 표시 (초기 로드 시에는 스크롤하지 않음)
    setTimeout(() => {
      startTypingEffect(welcomeMessage.id, welcomeContent, () => {
        setChatMessages([welcomeMessage]);
        setIsInitialLoad(false); // 초기 로드 완료
      }, false); // 스크롤하지 않음
    }, 500); // 0.5초 후 타이핑 시작
  }, [predictionHistory, startTypingEffect]); // startTypingEffect 의존성 추가

  // 전역 배치 결과가 업데이트될 때 예측 히스토리도 업데이트
  useEffect(() => {
    if (globalBatchResults && lastAnalysisTimestamp) {
      // 배치 분석 결과를 예측 히스토리로 변환하여 저장
      const predictionData = predictionService.convertBatchResultToPrediction(globalBatchResults);
      if (predictionData) {
        try {
          predictionService.savePredictionResult(predictionData);
          loadPredictionHistory(); // 히스토리 새로고침
          message.success('새로운 분석 결과가 저장되었습니다.');
        } catch (error) {
          console.error('예측 결과 저장 실패:', error);
        }
      }
    }
  }, [globalBatchResults, lastAnalysisTimestamp]);

  // 채팅 스크롤 자동 이동 (스마트 스크롤)
  useEffect(() => {
    // 초기 로드가 완료되고, 사용자가 메시지를 보낸 후에만 스크롤
    if (!isInitialLoad && userHasSentMessage && chatMessages.length > 1) {
      const chatContainer = chatEndRef.current?.parentElement;
      if (chatContainer) {
        const { scrollTop, scrollHeight, clientHeight } = chatContainer;
        const isNearBottom = scrollHeight - clientHeight - scrollTop < 100; // 100px 여유
        
        // 사용자가 맨 아래 근처에 있을 때만 자동 스크롤
        if (isNearBottom) {
          setTimeout(() => scrollToBottom(), 100); // 약간의 지연으로 부드러운 스크롤
        }
      }
    }
  }, [chatMessages, isInitialLoad, userHasSentMessage]);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // 컴포넌트 언마운트 시 타이핑 효과 정리
  useEffect(() => {
    return () => {
      if (typingIntervalRef.current) {
        clearTimeout(typingIntervalRef.current);
      }
    };
  }, []);

  // 예측 결과 히스토리 로드 (comprehensive_report.json 기반 - ReportGeneration.js와 동일!)
  const loadPredictionHistory = async () => {
    try {
      // 1순위: API에서 최신 데이터 로드 (comprehensive_report.json 기반)
      console.log('🔄 API에서 comprehensive_report.json 기반 히스토리 로드...');
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
          
          // predictionHistory 형식으로 변환
          const historyData = [{
            id: `comprehensive_${data.timestamp}`,
            title: `배치 분석 결과 (${data.total_employees}명)`,
            timestamp: data.timestamp,
            totalEmployees: data.total_employees,
            highRiskCount: highRiskCount,
            mediumRiskCount: mediumRiskCount,
            lowRiskCount: lowRiskCount,
            accuracy: 85,
            status: 'completed',
            summary: `${data.total_employees}명 분석 완료 (comprehensive_report.json 기준)`,
            keyInsights: [
              `고위험군 ${highRiskCount}명 (${(highRiskCount/data.total_employees*100).toFixed(1)}%)`,
              `중위험군 ${mediumRiskCount}명 (${(mediumRiskCount/data.total_employees*100).toFixed(1)}%)`,
              `저위험군 ${lowRiskCount}명 (${(lowRiskCount/data.total_employees*100).toFixed(1)}%)`
            ],
            departmentStats: {}
          }];
          
          setPredictionHistory(historyData);
          console.log('✅ Home 히스토리 로드 완료 (comprehensive_report 기준)');
          return;
        }
      }
      
      // 2순위: API 실패 시 localStorage 폴백
      console.log('⚠️ API 실패, localStorage 폴백...');
      const syncHistory = predictionService.getPredictionHistory();
      setPredictionHistory(syncHistory);
      
    } catch (error) {
      console.error('예측 히스토리 로드 실패:', error);
      
      // 에러 발생 시 localStorage 폴백
      try {
        const syncHistory = predictionService.getPredictionHistory();
        setPredictionHistory(syncHistory);
      } catch (fallbackError) {
        console.error('Fallback도 실패:', fallbackError);
        message.error('예측 히스토리를 불러오는데 실패했습니다.');
      }
    }
  };

  // 실제 LLM API 호출
  const handleSendMessage = async () => {
    if (!currentMessage.trim()) return;

    // 사용자가 메시지를 보냈음을 표시
    setUserHasSentMessage(true);

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: currentMessage,
      timestamp: new Date().toISOString()
    };

    setChatMessages(prev => [...prev, userMessage]);
    const messageToSend = currentMessage;
    setCurrentMessage('');
    setChatLoading(true);

    try {
      // 분석 결과 컨텍스트 준비
      const context = predictionHistory.length > 0 ? {
        totalEmployees: predictionHistory[0].totalEmployees,
        highRiskCount: predictionHistory[0].highRiskCount,
        mediumRiskCount: predictionHistory[0].mediumRiskCount,
        lowRiskCount: predictionHistory[0].lowRiskCount,
        accuracy: predictionHistory[0].accuracy,
        departmentStats: predictionHistory[0].departmentStats,
        keyInsights: predictionHistory[0].keyInsights,
        summary: predictionHistory[0].summary
      } : {};

      // Supervisor LLM API 호출 (GPT-5-nano-2025-08-07) - Supervisor 서버를 통해
      const response = await fetch('http://localhost:5006/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: messageToSend,
          context: context
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      // fallback_response가 있으면 사용, 없으면 일반 response 사용
      const responseContent = data.fallback_response || data.response;
      
      const botResponse = {
        id: Date.now() + 1,
        type: 'bot',
        content: responseContent,
        timestamp: new Date().toISOString(),
        model: data.model || 'AI Assistant',
        tokens_used: data.tokens_used || 0,
        isFallback: !!data.fallback_response
      };

      // 타이핑 효과로 응답 표시 (사용자 메시지 후이므로 스크롤 허용)
      startTypingEffect(botResponse.id, responseContent, () => {
        setChatMessages(prev => [...prev, botResponse]);
      }, true);
      
      // fallback 응답인 경우 경고 메시지 표시
      if (data.fallback_response) {
        console.warn('⚠️ AI 서버 연결에 실패했습니다. 기본 응답을 제공합니다.');
      }
      
    } catch (error) {
      console.error('LLM API 호출 오류:', error);
      
      // API 호출 실패 시 fallback으로 기존 로직 사용
      const fallbackResponse = await generateBotResponse(messageToSend);
      const fallbackContent = `⚠️ AI 서버 연결에 실패했습니다. 기본 응답을 제공합니다.\n\n${fallbackResponse.content}`;
      fallbackResponse.content = fallbackContent;
      
      // 타이핑 효과로 fallback 응답 표시 (사용자 메시지 후이므로 스크롤 허용)
      startTypingEffect(fallbackResponse.id, fallbackContent, () => {
        setChatMessages(prev => [...prev, fallbackResponse]);
      }, true);
      
      message.warning('AI 서버에 연결할 수 없습니다. 기본 응답을 제공합니다.');
    } finally {
      setChatLoading(false);
    }
  };

  // AI 응답 생성 (실제 데이터 기반)
  const generateBotResponse = async (userInput) => {
    try {
      // LLM 기반 채팅 사용 (Supervisor의 /api/chat 엔드포인트)
      const latestPrediction = predictionHistory.length > 0 ? predictionHistory[0] : null;
      
      // 컨텍스트 구성 (선택적)
      const context = latestPrediction ? {
        totalEmployees: latestPrediction.totalEmployees,
        highRiskCount: latestPrediction.highRiskCount,
        mediumRiskCount: latestPrediction.mediumRiskCount,
        lowRiskCount: latestPrediction.lowRiskCount,
        accuracy: latestPrediction.accuracy,
        departmentStats: latestPrediction.departmentStats,
        keyInsights: latestPrediction.keyInsights
      } : {};
      
      // Supervisor의 LLM 채팅 API 호출
      const response = await fetch('http://localhost:5006/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: userInput,
          context: context
        })
      });
      
      if (!response.ok) {
        throw new Error('LLM API 호출 실패');
      }
      
      const data = await response.json();
      return data.response;
      
    } catch (error) {
      console.error('LLM 채팅 오류:', error);
      
      // LLM 실패 시 폴백: 간단한 안내 메시지
      return '죄송합니다. AI 서버 연결에 문제가 발생했습니다.\n\n' +
             '다음 기능을 이용하실 수 있습니다:\n' +
             '• "직원 검색", "부서 분석", "위험도 현황"\n' +
             '• "개선 방안", "통계 보기"\n\n' +
             'Supervisor 서버(5006 포트)가 실행 중인지 확인해주세요.';
    }
  };

  // 예측 결과 상세 보기
  const showPredictionDetail = (prediction) => {
    setSelectedPrediction(prediction);
    setModalVisible(true);
  };

  // 키보드 이벤트 처리
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // 히스토리 내보내기
  const handleExportHistory = () => {
    try {
      predictionService.exportHistory();
      message.success('히스토리가 성공적으로 내보내졌습니다.');
    } catch (error) {
      message.error('히스토리 내보내기에 실패했습니다.');
    }
  };

  // 히스토리 가져오기
  const handleImportHistory = (file) => {
    predictionService.importHistory(file)
      .then((mergedHistory) => {
        setPredictionHistory(mergedHistory);
        message.success(`${mergedHistory.length}개의 히스토리가 성공적으로 가져와졌습니다.`);
      })
      .catch((error) => {
        console.error('히스토리 가져오기 실패:', error);
        message.error('히스토리 가져오기에 실패했습니다.');
      });
  };

  // 파일 선택 처리
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      handleImportHistory(file);
    }
    // 파일 입력 초기화
    event.target.value = '';
  };

  // 히스토리 삭제
  const handleClearHistory = () => {
    Modal.confirm({
      title: '히스토리 삭제 확인',
      content: `${predictionHistory.length}개의 모든 예측 히스토리가 삭제됩니다. 이 작업은 되돌릴 수 없습니다.`,
      okText: '삭제',
      okType: 'danger',
      cancelText: '취소',
      onOk() {
        predictionService.clearHistory();
        setPredictionHistory([]);
        message.success('모든 히스토리가 삭제되었습니다.');
      }
    });
  };

  return (
    <div style={{ padding: '0 8px' }}>
      {/* 헤더 섹션 */}
      <Row gutter={[24, 24]} style={{ marginBottom: '32px' }}>
        <Col span={24}>
          <Card
            style={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              border: 'none',
              color: 'white'
            }}
          >
            <Row align="middle" justify="space-between">
              <Col>
                <Title level={2} style={{ color: 'white', margin: 0 }}>
                  🏠 Retain Sentinel 360 홈
                </Title>
                <Text style={{ color: 'rgba(255,255,255,0.9)', fontSize: 'var(--font-medium)' }}>
                  AI 기반 이직 예측 및 분석 시스템에 오신 것을 환영합니다
                </Text>
              </Col>
              <Col>
                <Space>
                  <Badge count={predictionHistory.length} showZero={false}>
                    <Button 
                      type="primary" 
                      ghost 
                      icon={<HistoryOutlined />}
                      size="large"
                      onClick={() => onNavigate('batch')}
                    >
                      분석 시작하기
                    </Button>
                  </Badge>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      <Row gutter={[24, 24]}>
        {/* LLM 채팅 섹션 */}
        <Col xs={24} lg={14}>
          <Card
            title={
              <Space>
                <RobotOutlined style={{ color: '#1890ff' }} />
                <span>AI 어시스턴트와 채팅</span>
              </Space>
            }
            style={{ height: '600px', display: 'flex', flexDirection: 'column' }}
            bodyStyle={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '16px' }}
          >
            {/* 채팅 메시지 영역 */}
            <div
              style={{
                flex: 1,
                overflowY: 'auto',
                marginBottom: '16px',
                padding: '8px',
                border: '1px solid #f0f0f0',
                borderRadius: '8px',
                backgroundColor: '#fafafa'
              }}
            >
              {chatMessages.map((msg) => (
                <div
                  key={msg.id}
                  style={{
                    display: 'flex',
                    justifyContent: msg.type === 'user' ? 'flex-end' : 'flex-start',
                    marginBottom: '12px'
                  }}
                >
                  <div
                    style={{
                      maxWidth: '70%',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '8px',
                      flexDirection: msg.type === 'user' ? 'row-reverse' : 'row'
                    }}
                  >
                    <Avatar
                      icon={msg.type === 'user' ? <UserOutlined /> : <RobotOutlined />}
                      style={{
                        backgroundColor: msg.type === 'user' ? '#1890ff' : '#52c41a',
                        flexShrink: 0
                      }}
                    />
                    <div
                      style={{
                        padding: '12px 16px',
                        borderRadius: '12px',
                        backgroundColor: msg.type === 'user' ? '#1890ff' : '#fff',
                        color: msg.type === 'user' ? 'white' : '#333',
                        border: msg.type === 'bot' ? '1px solid #d9d9d9' : 'none',
                        whiteSpace: 'pre-line'
                      }}
                    >
                      {msg.content}
                    </div>
                  </div>
                </div>
              ))}
              
              {/* 타이핑 효과 중인 메시지 */}
              {typingMessageId && (
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'flex-start',
                    marginBottom: '12px'
                  }}
                >
                  <div
                    style={{
                      maxWidth: '70%',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '8px'
                    }}
                  >
                    <Avatar
                      icon={<RobotOutlined />}
                      style={{
                        backgroundColor: '#52c41a',
                        flexShrink: 0
                      }}
                    />
                    <div
                      style={{
                        padding: '12px 16px',
                        borderRadius: '12px',
                        backgroundColor: '#fff',
                        color: '#333',
                        border: '1px solid #d9d9d9',
                        whiteSpace: 'pre-line',
                        minHeight: '20px',
                        position: 'relative'
                      }}
                    >
                      {typingText}
                      <span 
                        style={{
                          display: 'inline-block',
                          width: '2px',
                          height: '16px',
                          backgroundColor: '#52c41a',
                          marginLeft: '2px',
                          animation: 'blink 1s infinite'
                        }}
                      />
                    </div>
                  </div>
                </div>
              )}
              
              {chatLoading && (
                <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#52c41a' }} />
                    <div style={{ padding: '12px 16px', backgroundColor: '#fff', borderRadius: '12px', border: '1px solid #d9d9d9' }}>
                      <Spin size="small" /> AI가 응답을 생성하고 있습니다...
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={chatEndRef} />
            </div>

            {/* 메시지 입력 영역 */}
            <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-end' }}>
              <TextArea
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="AI 어시스턴트에게 이직 예측 분석에 대해 질문해보세요..."
                autoSize={{ minRows: 1, maxRows: 3 }}
                style={{ flex: 1 }}
              />
              <Button
                icon={<VerticalAlignBottomOutlined />}
                onClick={scrollToBottom}
                title="맨 아래로 스크롤"
                style={{ marginBottom: '0px' }}
              />
              <Button
                type="primary"
                icon={<SendOutlined />}
                onClick={handleSendMessage}
                loading={chatLoading}
                disabled={!currentMessage.trim()}
              >
                전송
              </Button>
            </div>
          </Card>
        </Col>

        {/* 최근 예측 결과 및 기능 안내 */}
        <Col xs={24} lg={10}>
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            {/* 최근 분석 결과 요약 */}
            <Card
              title={
                <Space>
                  <TrophyOutlined style={{ color: '#52c41a' }} />
                  <span>최신 분석 결과</span>
                </Space>
              }
            >
              {predictionHistory.length > 0 ? (
                <div>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Statistic
                        title="전체 직원"
                        value={predictionHistory[0].totalEmployees}
                        suffix="명"
                        valueStyle={{ color: '#1890ff' }}
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic
                        title="고위험군"
                        value={predictionHistory[0].highRiskCount}
                        suffix="명"
                        valueStyle={{ color: '#ff4d4f' }}
                      />
                    </Col>
                  </Row>
                  <Divider />
                  <Progress
                    percent={predictionHistory[0].accuracy}
                    status="active"
                    strokeColor="#52c41a"
                    format={percent => `정확도 ${percent}%`}
                  />
                  <Paragraph style={{ marginTop: '16px', marginBottom: 0 }}>
                    {predictionHistory[0].summary}
                  </Paragraph>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '40px 20px' }}>
                  <BarChartOutlined style={{ fontSize: 'var(--icon-xlarge)', color: '#d9d9d9', marginBottom: '16px' }} />
                  <Title level={4} style={{ color: '#999' }}>
                    아직 분석 결과가 없습니다
                  </Title>
                  <Paragraph style={{ color: '#666', marginBottom: '24px' }}>
                    배치 분석을 실행하면 여기에 최신 결과가 표시됩니다.
                  </Paragraph>
                  <Button 
                    type="primary" 
                    onClick={() => onNavigate('batch')}
                    size="large"
                  >
                    배치 분석 시작하기
                  </Button>
                </div>
              )}
            </Card>

            {/* 예측 결과 히스토리 */}
            <Card
              title={
                <Space>
                  <HistoryOutlined style={{ color: '#722ed1' }} />
                  <span>분석 히스토리</span>
                  <Badge count={predictionHistory.length} showZero={false} />
                </Space>
              }
              extra={
                predictionHistory.length > 0 && (
                  <Space>
                    <Button 
                      type="text" 
                      icon={<DownloadOutlined />} 
                      onClick={handleExportHistory}
                      title="히스토리 내보내기"
                    />
                    <Button 
                      type="text" 
                      icon={<UploadOutlined />} 
                      onClick={() => fileInputRef.current?.click()}
                      title="히스토리 가져오기"
                    />
                    <Button 
                      type="text" 
                      icon={<SettingOutlined />} 
                      onClick={() => setHistoryManageVisible(true)}
                      title="히스토리 관리"
                    />
                  </Space>
                )
              }
            >
              {predictionHistory.length > 0 ? (
                <List
                  dataSource={predictionHistory}
                  renderItem={(item) => (
                    <List.Item
                      actions={[
                        <Button
                          type="link"
                          icon={<EyeOutlined />}
                          onClick={() => showPredictionDetail(item)}
                        >
                          상세보기
                        </Button>
                      ]}
                    >
                      <List.Item.Meta
                        avatar={
                          <Badge status={item.status === 'completed' ? 'success' : 'processing'} />
                        }
                        title={item.title}
                        description={
                          <Space direction="vertical" size="small">
                            <Text type="secondary">
                              <ClockCircleOutlined /> {new Date(item.timestamp).toLocaleDateString('ko-KR')}
                            </Text>
                            <Space>
                              <Tag color="red">고위험 {item.highRiskCount}명</Tag>
                              <Tag color="orange">중위험 {item.mediumRiskCount}명</Tag>
                              <Tag color="green">저위험 {item.lowRiskCount}명</Tag>
                            </Space>
                          </Space>
                        }
                      />
                    </List.Item>
                  )}
                />
              ) : (
                <div style={{ textAlign: 'center', padding: '40px 20px' }}>
                  <HistoryOutlined style={{ fontSize: 'var(--icon-xlarge)', color: '#d9d9d9', marginBottom: '16px' }} />
                  <Title level={4} style={{ color: '#999' }}>
                    분석 히스토리가 없습니다
                  </Title>
                  <Paragraph style={{ color: '#666' }}>
                    배치 분석을 실행하면 히스토리가 자동으로 저장됩니다.
                  </Paragraph>
                </div>
              )}
            </Card>

            {/* 주요 기능 안내 */}
            <Card
              title={
                <Space>
                  <BarChartOutlined style={{ color: '#fa8c16' }} />
                  <span>주요 기능</span>
                </Space>
              }
            >
              <Timeline
                items={[
                  {
                    dot: <TeamOutlined style={{ fontSize: 'var(--icon-small)' }} />,
                    children: (
                      <div>
                        <Text strong>배치 분석</Text>
                        <br />
                        <Text type="secondary">전체 직원의 이직 위험도를 일괄 분석</Text>
                        <br />
                        <Button type="link" size="small" onClick={() => onNavigate('batch')}>
                          바로가기 →
                        </Button>
                      </div>
                    )
                  },
                  {
                    dot: <BarChartOutlined style={{ fontSize: 'var(--icon-small)' }} />,
                    children: (
                      <div>
                        <Text strong>단체 통계</Text>
                        <br />
                        <Text type="secondary">부서별, 팀별 이직 위험 통계 분석</Text>
                        <br />
                        <Button type="link" size="small" onClick={() => onNavigate('group-statistics')}>
                          바로가기 →
                        </Button>
                      </div>
                    )
                  },
                  {
                    dot: <CheckCircleOutlined style={{ fontSize: 'var(--icon-small)' }} />,
                    children: (
                      <div>
                        <Text strong>사후 분석</Text>
                        <br />
                        <Text type="secondary">예측 결과의 정확도 검증 및 개선</Text>
                        <br />
                        <Button type="link" size="small" onClick={() => onNavigate('post-analysis')}>
                          바로가기 →
                        </Button>
                      </div>
                    )
                  }
                ]}
              />
            </Card>
          </Space>
        </Col>
      </Row>

      {/* 예측 결과 상세 모달 */}
      <Modal
        title={selectedPrediction?.title}
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={800}
      >
        {selectedPrediction && (
          <div>
            <Descriptions bordered column={2} style={{ marginBottom: '24px' }}>
              <Descriptions.Item label="분석 일시">
                {new Date(selectedPrediction.timestamp).toLocaleString('ko-KR')}
              </Descriptions.Item>
              <Descriptions.Item label="상태">
                <Badge 
                  status={selectedPrediction.status === 'completed' ? 'success' : 'processing'} 
                  text={selectedPrediction.status === 'completed' ? '완료' : '진행중'}
                />
              </Descriptions.Item>
              <Descriptions.Item label="전체 직원 수">
                {selectedPrediction.totalEmployees.toLocaleString()}명
              </Descriptions.Item>
              <Descriptions.Item label="모델 정확도">
                {selectedPrediction.accuracy}%
              </Descriptions.Item>
            </Descriptions>

            <Row gutter={16} style={{ marginBottom: '24px' }}>
              <Col span={8}>
                <Card>
                  <Statistic
                    title="고위험군"
                    value={selectedPrediction.highRiskCount}
                    suffix="명"
                    valueStyle={{ color: '#ff4d4f' }}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card>
                  <Statistic
                    title="중위험군"
                    value={selectedPrediction.mediumRiskCount}
                    suffix="명"
                    valueStyle={{ color: '#fa8c16' }}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card>
                  <Statistic
                    title="저위험군"
                    value={selectedPrediction.lowRiskCount}
                    suffix="명"
                    valueStyle={{ color: '#52c41a' }}
                  />
                </Card>
              </Col>
            </Row>

            <Card title="분석 요약" style={{ marginBottom: '16px' }}>
              <Paragraph>{selectedPrediction.summary}</Paragraph>
            </Card>

            <Card title="주요 인사이트">
              <ul>
                {selectedPrediction.keyInsights.map((insight, index) => (
                  <li key={index} style={{ marginBottom: '8px' }}>
                    <Text>{insight}</Text>
                  </li>
                ))}
              </ul>
            </Card>
          </div>
        )}
      </Modal>

      {/* 히스토리 관리 모달 */}
      <Modal
        title="히스토리 관리"
        open={historyManageVisible}
        onCancel={() => setHistoryManageVisible(false)}
        footer={null}
        width={600}
      >
        <div>
          <Descriptions bordered column={1} style={{ marginBottom: '24px' }}>
            <Descriptions.Item label="총 히스토리 개수">
              {predictionHistory.length}개
            </Descriptions.Item>
            <Descriptions.Item label="저장 용량">
              {(JSON.stringify(predictionHistory).length / 1024).toFixed(2)} KB
            </Descriptions.Item>
            <Descriptions.Item label="가장 오래된 기록">
              {predictionHistory.length > 0 
                ? new Date(predictionHistory[predictionHistory.length - 1].timestamp).toLocaleString('ko-KR')
                : '없음'
              }
            </Descriptions.Item>
            <Descriptions.Item label="가장 최근 기록">
              {predictionHistory.length > 0 
                ? new Date(predictionHistory[0].timestamp).toLocaleString('ko-KR')
                : '없음'
              }
            </Descriptions.Item>
          </Descriptions>

          <Space direction="vertical" style={{ width: '100%' }} size="large">
            <Card title="백업 및 복원" size="small">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Button 
                  type="primary" 
                  icon={<DownloadOutlined />} 
                  onClick={handleExportHistory}
                  block
                >
                  히스토리 내보내기 (JSON 파일로 백업)
                </Button>
                <Button 
                  icon={<UploadOutlined />} 
                  onClick={() => fileInputRef.current?.click()}
                  block
                >
                  히스토리 가져오기 (JSON 파일에서 복원)
                </Button>
                <Text type="secondary" style={{ fontSize: 'var(--font-small)' }}>
                  * 가져오기 시 기존 히스토리와 병합되며, 중복은 자동으로 제거됩니다.
                </Text>
              </Space>
            </Card>

            <Card title="위험한 작업" size="small">
              <Button 
                danger 
                icon={<DeleteOutlined />} 
                onClick={handleClearHistory}
                block
              >
                모든 히스토리 삭제
              </Button>
              <Text type="secondary" style={{ fontSize: 'var(--font-small)', marginTop: '8px', display: 'block' }}>
                * 이 작업은 되돌릴 수 없습니다. 삭제 전 백업을 권장합니다.
              </Text>
            </Card>
          </Space>
        </div>
      </Modal>

      {/* 숨겨진 파일 입력 */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileSelect}
        accept=".json"
        style={{ display: 'none' }}
      />
    </div>
  );
};

export default Home;
