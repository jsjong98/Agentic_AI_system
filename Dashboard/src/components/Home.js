import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Button,
  Input,
  Avatar,
  Spin,
  message,
  Tag,
  Space,
  Modal,
  Descriptions,
  Badge,
  Statistic,
} from 'antd';
import {
  SendOutlined,
  RobotOutlined,
  UserOutlined,
  CheckCircleOutlined,
  DownloadOutlined,
  UploadOutlined,
  DeleteOutlined,
  VerticalAlignBottomOutlined
} from '@ant-design/icons';
import { predictionService } from '../services/predictionService';

// API Base URLs from environment variables
const SUPERVISOR_URL = process.env.REACT_APP_SUPERVISOR_URL || 'http://localhost:5006';
const INTEGRATION_URL = process.env.REACT_APP_INTEGRATION_URL || 'http://localhost:5007';

const { Text, Paragraph } = Typography;
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
      const response = await fetch(`${INTEGRATION_URL}/api/results/list-all-employees`);
      
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
      const response = await fetch(`${SUPERVISOR_URL}/api/chat`, {
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
      const response = await fetch(`${SUPERVISOR_URL}/api/chat`, {
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
      {/* Agentic AI 아키텍처 배너 */}
      <div style={{
        background: 'linear-gradient(135deg, #2d2d2d, #4a4a4a)',
        borderRadius: 12, padding: '20px 24px', color: '#fff',
        marginBottom: 20, display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', gap: 16, flexWrap: 'wrap',
      }}>
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 4 }}>
            Agentic AI 기반 선제적 퇴사위험 예측 시스템
          </div>
          <div style={{ fontSize: 12, color: '#ccc' }}>
            5개 전문 Worker Agent의 분석 결과를 Supervisor가 종합하여 360도 관점의 퇴사 위험 진단 제공
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {[
            { name: 'Structura', desc: '정형 데이터', bg: 'rgba(217,57,84,.2)', border: 'rgba(217,57,84,.5)' },
            { name: 'Cognita', desc: '관계망', bg: 'rgba(37,99,235,.2)', border: 'rgba(37,99,235,.5)' },
            { name: 'Chronos', desc: '시계열', bg: 'rgba(232,114,26,.2)', border: 'rgba(232,114,26,.5)' },
            { name: 'Sentio', desc: '자연어', bg: 'rgba(124,58,237,.2)', border: 'rgba(124,58,237,.5)' },
            { name: 'Agora', desc: '외부시장', bg: 'rgba(46,164,79,.2)', border: 'rgba(46,164,79,.5)' },
          ].map(a => (
            <div key={a.name} style={{
              padding: '6px 12px', borderRadius: 8, fontSize: 11, fontWeight: 600,
              textAlign: 'center', whiteSpace: 'nowrap',
              background: a.bg, border: `1px solid ${a.border}`,
            }}>
              {a.name}<br/><span style={{ fontSize: 10 }}>{a.desc}</span>
            </div>
          ))}
        </div>
      </div>

      {/* KPI 카드 행 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12, marginBottom: 20 }}>
        {[
          { label: '전사 인원 현황', value: predictionHistory[0]?.totalEmployees || '1,470', unit: '명', color: '#2d2d2d', top: '#d93954' },
          { label: '퇴사 고위험군', value: predictionHistory[0]?.highRiskCount || '77', unit: '명', color: '#d93954', top: '#d93954' },
          { label: '잠재적 위험군', value: predictionHistory[0]?.mediumRiskCount || '202', unit: '명', color: '#e8721a', top: '#e8721a' },
          { label: '안정/양호군', value: predictionHistory[0]?.lowRiskCount || '1,191', unit: '명', color: '#2ea44f', top: '#2ea44f' },
          { label: '평균 위험 점수', value: '0.28', unit: '', color: '#2563eb', top: '#2563eb' },
        ].map((kpi, i) => (
          <div key={i} style={{
            background: 'var(--card, #fff)', borderRadius: 12, padding: 16,
            border: '1px solid var(--border, #eee)', textAlign: 'center',
            position: 'relative', overflow: 'hidden',
            boxShadow: '0 1px 4px rgba(0,0,0,.06)',
          }}>
            <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 4, background: kpi.top }} />
            <div style={{ fontSize: 11, color: 'var(--sub, #888)', marginBottom: 6, fontWeight: 500 }}>{kpi.label}</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: kpi.color }}>
              {kpi.value}<span style={{ fontSize: 13, color: 'var(--sub, #888)' }}>{kpi.unit}</span>
            </div>
          </div>
        ))}
      </div>

      {/* 에이전트 워크플로우 — 세로 구조 */}
      <Card style={{ marginBottom: 20 }} bodyStyle={{ padding: 20 }}>
        <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#d93954', display: 'inline-block' }} />
          에이전트 워크플로우
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0, padding: '10px 0' }}>
          {/* Supervisor Agent — 맨 위 */}
          <div style={{
            background: '#2d2d2d', color: '#fff', padding: '10px 32px',
            borderRadius: 8, fontSize: 13, fontWeight: 700, textAlign: 'center',
          }}>Supervisor Agent</div>

          {/* 화살표 아래로 */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', margin: '4px 0' }}>
            <div style={{ width: 2, height: 16, background: '#d93954' }} />
            <div style={{ width: 0, height: 0, borderLeft: '6px solid transparent', borderRight: '6px solid transparent', borderTop: '8px solid #d93954' }} />
          </div>

          {/* 5 Worker Agents — 가운데 한 줄 */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', justifyContent: 'center' }}>
            {[
              { name: 'Structura', color: '#d93954', icon: '📊' },
              { name: 'Cognita', color: '#2563eb', icon: '🔗' },
              { name: 'Chronos', color: '#e8721a', icon: '📈' },
              { name: 'Sentio', color: '#7c3aed', icon: '💬' },
              { name: 'Agora', color: '#2ea44f', icon: '🎯' },
            ].map(w => (
              <div key={w.name} style={{
                border: `2px solid ${w.color}`, borderRadius: 10,
                padding: '10px 16px', fontSize: 12, fontWeight: 600,
                textAlign: 'center', background: `${w.color}10`, minWidth: 90,
              }}>
                <div style={{ fontSize: 20, marginBottom: 2 }}>{w.icon}</div>
                {w.name}
              </div>
            ))}
          </div>

          {/* 화살표 아래로 */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', margin: '4px 0' }}>
            <div style={{ width: 2, height: 16, background: '#d93954' }} />
            <div style={{ width: 0, height: 0, borderLeft: '6px solid transparent', borderRight: '6px solid transparent', borderTop: '8px solid #d93954' }} />
          </div>

          {/* Synthesize Agent (Integration) — 맨 아래 */}
          <div style={{
            background: '#d93954', color: '#fff', padding: '10px 32px',
            borderRadius: 8, fontSize: 13, fontWeight: 700, textAlign: 'center',
          }}>Synthesize Agent</div>
        </div>

        <div style={{ textAlign: 'center', fontSize: 11, color: 'var(--sub, #888)', marginTop: 10 }}>
          데이터 수집 → 개별 분석 → 종합 평가 → 위험도 산출
        </div>
      </Card>

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

        {/* 에이전트 상태 & 빠른 질문 */}
        <Col xs={24} lg={10}>
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            {/* 에이전트 연결 상태 */}
            <Card
              title={
                <Space>
                  <CheckCircleOutlined style={{ color: '#52c41a' }} />
                  <span>에이전트 연결 상태</span>
                </Space>
              }
              bodyStyle={{ padding: '12px 16px' }}
            >
              {[
                { name: 'Supervisor', desc: '워크플로우 오케스트레이션', color: '#2d2d2d', port: 'supervisor' },
                { name: 'Structura', desc: 'XGBoost 정형 데이터 분석', color: '#d93954', port: 'structura' },
                { name: 'Cognita', desc: 'Neo4j 관계망 분석', color: '#2563eb', port: 'cognita' },
                { name: 'Chronos', desc: 'LSTM 시계열 패턴 분석', color: '#e8721a', port: 'chronos' },
                { name: 'Sentio', desc: 'NLP 감성/심리 분석', color: '#7c3aed', port: 'sentio' },
                { name: 'Agora', desc: '외부 시장 데이터 분석', color: '#2ea44f', port: 'agora' },
                { name: 'Integration', desc: '결과 통합 및 보고서', color: '#666', port: 'integration' },
              ].map(agent => (
                <div key={agent.name} style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '8px 0', borderBottom: '1px solid #f5f5f5',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div style={{
                      width: 8, height: 8, borderRadius: '50%',
                      background: '#52c41a',
                    }} />
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 600, color: agent.color }}>{agent.name}</div>
                      <div style={{ fontSize: 11, color: '#999' }}>{agent.desc}</div>
                    </div>
                  </div>
                  <Tag color="green" style={{ margin: 0, fontSize: 10 }}>연결됨</Tag>
                </div>
              ))}
            </Card>

            {/* 빠른 질문 카드 */}
            <Card
              title={
                <Space>
                  <RobotOutlined style={{ color: '#d93954' }} />
                  <span>빠른 질문</span>
                </Space>
              }
              bodyStyle={{ padding: '12px 16px' }}
            >
              <div style={{ fontSize: 12, color: '#888', marginBottom: 12 }}>
                아래 질문을 클릭하면 AI 어시스턴트가 실제 데이터를 기반으로 답변합니다.
              </div>
              {[
                '전체 고위험군 현황을 알려줘',
                '사번 1인 직원의 퇴사 위험도는?',
                'Sales 부서의 위험 분석 결과는?',
                '가장 위험도가 높은 직원 5명은?',
                'R&D 부서 고위험군 현황을 알려줘',
              ].map((q) => (
                <div
                  key={q}
                  onClick={() => {
                    setCurrentMessage(q);
                    setTimeout(() => {
                      handleSendMessage();
                    }, 100);
                  }}
                  style={{
                    padding: '10px 14px', marginBottom: 8,
                    background: '#f8f8f9', borderRadius: 8,
                    fontSize: 13, cursor: 'pointer',
                    border: '1px solid #eee',
                    transition: 'all 0.2s',
                  }}
                  onMouseEnter={(e) => { e.target.style.background = '#fef7f8'; e.target.style.borderColor = '#d93954'; }}
                  onMouseLeave={(e) => { e.target.style.background = '#f8f8f9'; e.target.style.borderColor = '#eee'; }}
                >
                  💬 {q}
                </div>
              ))}
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
