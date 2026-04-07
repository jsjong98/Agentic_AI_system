import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Button, Input, Avatar, Spin } from 'antd';
import { SendOutlined, RobotOutlined, UserOutlined } from '@ant-design/icons';

const SUPERVISOR_URL = process.env.REACT_APP_SUPERVISOR_URL || 'http://localhost:5006';
const INTEGRATION_URL = process.env.REACT_APP_INTEGRATION_URL || 'http://localhost:5007';
const { TextArea } = Input;

// ───────────────────────────── style injection ─────────────────────────────
if (typeof document !== 'undefined' && !document.getElementById('home-main-style')) {
  const s = document.createElement('style');
  s.id = 'home-main-style';
  s.textContent = `
    @keyframes blink{0%,50%{opacity:1}51%,100%{opacity:0}}
    @keyframes flowPulse{0%{opacity:1;transform:scaleY(1)}50%{opacity:0.5;transform:scaleY(0.8)}100%{opacity:1;transform:scaleY(1)}}
    @keyframes agentGlow{0%,100%{box-shadow:0 0 8px rgba(217,57,84,.4)}50%{box-shadow:0 0 20px rgba(217,57,84,.8)}}
    @keyframes dotFlow{0%{transform:translateY(0);opacity:1}100%{transform:translateY(30px);opacity:0}}
    @keyframes fadeSlide{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
  `;
  document.head.appendChild(s);
}

// ─────────────────────────── Animated Workflow ──────────────────────────────
const AGENTS = [
  { n: 'Structura', c: '#d93954', i: '📊', desc: '정형 데이터' },
  { n: 'Cognita',   c: '#2563eb', i: '🔗', desc: '관계망' },
  { n: 'Chronos',   c: '#e8721a', i: '📈', desc: '시계열' },
  { n: 'Sentio',    c: '#7c3aed', i: '💬', desc: '감성' },
  { n: 'Agora',     c: '#2ea44f', i: '🎯', desc: '외부시장' },
];

const STATUS_MSGS = [
  '시스템 대기 중...',
  'Supervisor Agent: 분석 요청 수신',
  'Worker Agents에게 작업 분배 중...',
  'Structura: 정형 데이터 분석 완료 ✓',
  'Cognita: 관계망 분석 완료 ✓',
  'Chronos: 시계열 분석 완료 ✓',
  'Sentio: 감성 분석 완료 ✓',
  'Agora: 외부 시장 분석 완료 ✓',
  '분석 결과 종합 중...',
  'Synthesize Agent: 위험도 산출 완료 ✓',
  '분석 사이클 완료 — 재시작 중...',
];

const AgentWorkflow = ({ isDark }) => {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const steps = [
      [0, 1500], [1, 1500], [2, 900],
      [3, 900], [4, 900], [5, 900], [6, 900], [7, 900],
      [8, 900], [9, 2500], [10, 1500],
    ];
    const run = async () => {
      while (!cancelled) {
        for (const [p, d] of steps) {
          if (cancelled) return;
          setPhase(p);
          await new Promise(r => setTimeout(r, d));
        }
      }
    };
    run();
    return () => { cancelled = true; };
  }, []);

  const supActive  = phase >= 1;
  const dispFlow   = phase >= 2;
  const agDone     = (i) => phase >= 3 + i;
  const resultFlow = phase >= 8;
  const synActive  = phase >= 9;
  const done       = phase >= 9;

  const card = { borderRadius: 10, padding: '0 8px', fontFamily: 'inherit', transition: 'all 0.45s' };
  const C = isDark ? { bg: '#151c2c', sub: '#6b7280' } : { bg: '#fff', sub: '#888' };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', userSelect: 'none' }}>
      {/* Status badge */}
      <div style={{
        marginBottom: 14, fontSize: 12, fontWeight: 600,
        color: done ? '#2ea44f' : '#d93954',
        background: done ? '#e6f6ec' : '#fde8ec',
        padding: '4px 14px', borderRadius: 20,
        transition: 'all 0.4s',
        animation: 'fadeSlide 0.3s ease',
        minWidth: 260, textAlign: 'center',
      }}>
        {STATUS_MSGS[phase]}
      </div>

      {/* Supervisor */}
      <div style={{
        ...card,
        padding: '10px 36px', fontWeight: 700, fontSize: 13,
        background: supActive ? '#2d2d2d' : (isDark ? '#1e2a3a' : '#f0f0f0'),
        color: supActive ? '#fff' : C.sub,
        border: supActive ? '2px solid #d93954' : `2px solid ${isDark ? '#2a3a4a' : '#e0e0e0'}`,
        boxShadow: supActive ? '0 0 20px rgba(217,57,84,.45)' : 'none',
      }}>
        🤖 Supervisor Agent
      </div>

      {/* Arrow: Supervisor → Agents */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', height: 36, justifyContent: 'center' }}>
        <div style={{
          width: 2, height: 24,
          background: dispFlow ? '#d93954' : (isDark ? '#2a3a4a' : '#e0e0e0'),
          animation: dispFlow ? 'flowPulse 0.9s ease-in-out infinite' : 'none',
          transition: 'background 0.4s',
        }} />
        <div style={{
          width: 0, height: 0,
          borderLeft: '6px solid transparent', borderRight: '6px solid transparent',
          borderTop: `8px solid ${dispFlow ? '#d93954' : (isDark ? '#2a3a4a' : '#e0e0e0')}`,
          transition: 'border-color 0.4s',
        }} />
      </div>

      {/* Agents row */}
      <div style={{ display: 'flex', gap: 8, justifyContent: 'center', flexWrap: 'wrap' }}>
        {AGENTS.map((a, i) => (
          <div key={a.n} style={{
            border: `2px solid ${agDone(i) ? a.c : (isDark ? '#2a3a4a' : '#e0e0e0')}`,
            borderRadius: 10, padding: '8px 10px', textAlign: 'center', minWidth: 68,
            background: agDone(i) ? `${a.c}1A` : (isDark ? '#1e2a3a' : '#fafafa'),
            color: agDone(i) ? a.c : C.sub,
            boxShadow: agDone(i) ? `0 0 14px ${a.c}55` : 'none',
            transition: 'all 0.45s',
          }}>
            <div style={{ fontSize: 20, marginBottom: 2 }}>{a.i}</div>
            <div style={{ fontSize: 11, fontWeight: 700 }}>{a.n}</div>
            <div style={{ fontSize: 10, marginTop: 1, color: agDone(i) ? a.c : (isDark ? '#3a4a5a' : '#ccc') }}>
              {a.desc}
            </div>
            <div style={{ fontSize: 11, color: '#2ea44f', marginTop: 2, minHeight: 14 }}>
              {agDone(i) ? '✓' : ''}
            </div>
          </div>
        ))}
      </div>

      {/* Arrow: Agents → Synthesize */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', height: 36, justifyContent: 'center' }}>
        <div style={{
          width: 0, height: 0,
          borderLeft: '6px solid transparent', borderRight: '6px solid transparent',
          borderBottom: `8px solid ${resultFlow ? '#d93954' : (isDark ? '#2a3a4a' : '#e0e0e0')}`,
          transition: 'border-color 0.4s',
        }} />
        <div style={{
          width: 2, height: 24,
          background: resultFlow ? '#d93954' : (isDark ? '#2a3a4a' : '#e0e0e0'),
          animation: resultFlow ? 'flowPulse 0.9s ease-in-out infinite' : 'none',
          transition: 'background 0.4s',
        }} />
      </div>

      {/* Synthesize */}
      <div style={{
        ...card,
        padding: '10px 36px', fontWeight: 700, fontSize: 13,
        background: synActive ? '#d93954' : (isDark ? '#1e2a3a' : '#f0f0f0'),
        color: synActive ? '#fff' : C.sub,
        border: synActive ? '2px solid #d93954' : `2px solid ${isDark ? '#2a3a4a' : '#e0e0e0'}`,
        boxShadow: synActive ? '0 0 24px rgba(217,57,84,.55)' : 'none',
      }}>
        ⚡ Synthesize Agent
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 16, marginTop: 14, flexWrap: 'wrap', justifyContent: 'center' }}>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 11, color: C.sub }}>
          <div style={{ width: 24, height: 3, background: '#d93954', borderRadius: 2 }} /> 데이터 흐름
        </div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 11, color: C.sub }}>
          <div style={{ width: 10, height: 10, background: '#2ea44f', borderRadius: '50%' }} /> 완료
        </div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 11, color: C.sub }}>
          <div style={{ width: 10, height: 10, background: '#e0e0e0', borderRadius: '50%', border: '1px solid #ccc' }} /> 대기
        </div>
      </div>
    </div>
  );
};

// ──────────────────────────── HomeMain Component ─────────────────────────────
const HomeMain = ({ globalBatchResults, lastAnalysisTimestamp }) => {
  const [chatMessages, setChatMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [typingMessageId, setTypingMessageId] = useState(null);
  const [typingText, setTypingText] = useState('');
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  const [userHasSentMessage, setUserHasSentMessage] = useState(false);
  const chatEndRef = useRef(null);
  const typingIntervalRef = useRef(null);
  const welcomeMessageShown = useRef(false);

  // Detect dark mode from CSS variable
  const [isDark, setIsDark] = useState(false);
  useEffect(() => {
    const check = () => {
      const bg = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim();
      setIsDark(bg === '#0e1525' || bg.startsWith('#0'));
    };
    check();
    const obs = new MutationObserver(check);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['style'] });
    return () => obs.disconnect();
  }, []);

  const C = {
    card: 'var(--card,#fff)', border: 'var(--border,#eee)',
    sub: 'var(--sub,#888)', text: 'var(--text,#2d2d2d)', bg: 'var(--bg,#fafafa)',
  };
  const cardS = {
    background: C.card, borderRadius: 12, padding: 20,
    border: `1px solid ${C.border}`, boxShadow: '0 1px 4px rgba(0,0,0,.06)',
  };
  const secTitle = (icon, title) => (
    <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 14 }}>
      <span style={{ color: '#d93954' }}>{icon}</span> {title}
    </div>
  );

  // ── typing effect ──
  const startTypingEffect = useCallback((messageId, fullText, onComplete, shouldScroll = true) => {
    if (typingIntervalRef.current) clearTimeout(typingIntervalRef.current);
    setTypingMessageId(messageId);
    setTypingText('');
    let idx = 0;
    const type = () => {
      if (idx < fullText.length) {
        setTypingText(fullText.substring(0, idx + 1));
        idx++;
        if (shouldScroll && userHasSentMessage && !isInitialLoad)
          setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 10);
        const ch = fullText[idx - 1];
        const d = ch === '.' || ch === '!' || ch === '?' ? 300 : ch === '\n' ? 200 : ch === ',' ? 150 : 25;
        typingIntervalRef.current = setTimeout(type, d);
      } else {
        setTypingMessageId(null);
        setTypingText('');
        if (onComplete) onComplete();
      }
    };
    typingIntervalRef.current = setTimeout(type, 25);
  }, [userHasSentMessage, isInitialLoad]);

  useEffect(() => { loadPredictionHistory(); }, []);

  useEffect(() => {
    if (welcomeMessageShown.current) return;
    const content = '안녕하세요! Retain Sentinel 360 AI 어시스턴트입니다. 분석 결과에 대해 질문하실 수 있습니다.';
    const msg = { id: 1, type: 'bot', content, timestamp: new Date().toISOString() };
    welcomeMessageShown.current = true;
    setTimeout(() => startTypingEffect(msg.id, content, () => { setChatMessages([msg]); setIsInitialLoad(false); }, false), 500);
  }, [predictionHistory, startTypingEffect]);

  useEffect(() => { return () => { if (typingIntervalRef.current) clearTimeout(typingIntervalRef.current); }; }, []);

  const loadPredictionHistory = async () => {
    try {
      const response = await fetch(`${INTEGRATION_URL}/api/results/list-all-employees`);
      if (response.ok) {
        const data = await response.json();
        if (data.success && data.results && data.results.length > 0) {
          const h = data.results.filter(r => r.risk_level === 'HIGH').length;
          const m = data.results.filter(r => r.risk_level === 'MEDIUM').length;
          const l = data.results.filter(r => r.risk_level === 'LOW').length;
          setPredictionHistory([{ id: 'latest', totalEmployees: data.total_employees, highRiskCount: h, mediumRiskCount: m, lowRiskCount: l }]);
          return;
        }
      }
      setPredictionHistory([]);
    } catch { setPredictionHistory([]); }
  };

  const handleSendMessage = async () => {
    if (!currentMessage.trim()) return;
    setUserHasSentMessage(true);
    const userMsg = { id: Date.now(), type: 'user', content: currentMessage, timestamp: new Date().toISOString() };
    setChatMessages(prev => [...prev, userMsg]);
    const msg = currentMessage;
    setCurrentMessage('');
    setChatLoading(true);
    try {
      const ctx = predictionHistory[0] || {};
      const resp = await fetch(`${SUPERVISOR_URL}/api/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, context: ctx }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const content = data.fallback_response || data.response;
      const bot = { id: Date.now() + 1, type: 'bot', content, timestamp: new Date().toISOString() };
      startTypingEffect(bot.id, content, () => setChatMessages(prev => [...prev, bot]), true);
    } catch {
      const bot = { id: Date.now() + 1, type: 'bot', content: 'AI 서버에 연결할 수 없습니다.', timestamp: new Date().toISOString() };
      setChatMessages(prev => [...prev, bot]);
    } finally { setChatLoading(false); }
  };

  const handleKeyPress = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); } };

  return (
    <div>
      {/* ── Program Description Banner ── */}
      <div style={{
        background: 'linear-gradient(135deg, #2d2d2d 0%, #1a1a2e 60%, #16213e 100%)',
        borderRadius: 14, padding: '24px 28px', marginBottom: 20, color: '#fff',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 20, flexWrap: 'wrap',
        boxShadow: '0 4px 20px rgba(0,0,0,.18)',
      }}>
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 6, display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ background: '#d93954', borderRadius: 8, padding: '4px 12px', fontSize: 13 }}>
              Retain Sentinel 360
            </span>
          </div>
          <div style={{ fontSize: 14, color: '#e0e0e0', fontWeight: 600, marginBottom: 6 }}>
            Agentic AI 기반 선제적 퇴사위험 예측 및 관리 시스템
          </div>
          <div style={{ fontSize: 12, color: '#aaa', lineHeight: 1.7 }}>
            5개 전문 Worker Agent(Structura · Cognita · Chronos · Sentio · Agora)의
            다차원 분석을 통해 조직 인재 이탈 위험을 360도 관점으로 진단하고 선제적 개입 전략을 제시합니다.
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {[
            { n: 'Structura', c: 'rgba(217,57,84,.25)', b: 'rgba(217,57,84,.6)', d: '정형 데이터' },
            { n: 'Cognita',   c: 'rgba(37,99,235,.25)',  b: 'rgba(37,99,235,.6)',  d: '관계망' },
            { n: 'Chronos',   c: 'rgba(232,114,26,.25)', b: 'rgba(232,114,26,.6)', d: '시계열' },
            { n: 'Sentio',    c: 'rgba(124,58,237,.25)', b: 'rgba(124,58,237,.6)', d: '자연어' },
            { n: 'Agora',     c: 'rgba(46,164,79,.25)',  b: 'rgba(46,164,79,.6)',  d: '외부시장' },
          ].map(a => (
            <div key={a.n} style={{
              padding: '6px 14px', borderRadius: 8, fontSize: 11, fontWeight: 600,
              background: a.c, border: `1px solid ${a.b}`, textAlign: 'center', color: '#fff',
            }}>
              {a.n}<br /><span style={{ fontSize: 10, fontWeight: 400, color: '#ccc' }}>{a.d}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Workflow + Chat ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.6fr', gap: 16, marginBottom: 20 }}>

        {/* Workflow card */}
        <div style={cardS}>
          {secTitle('⚡', '에이전트 워크플로우 (실시간)')}
          <AgentWorkflow isDark={isDark} />
          <div style={{ marginTop: 16, padding: '10px 12px', background: isDark ? '#1e2a3a' : '#f8f8f9', borderRadius: 8, fontSize: 11, color: C.sub, lineHeight: 1.6 }}>
            <strong style={{ color: '#d93954' }}>데이터 흐름:</strong>&nbsp;
            데이터 수집 → Supervisor 분배 → 개별 Agent 분석 → Synthesize 종합 → 위험도 산출
          </div>
        </div>

        {/* Chat card */}
        <div style={cardS}>
          {secTitle('🤖', 'AI 어시스턴트')}
          <div style={{
            height: 340, overflowY: 'auto', marginBottom: 12, padding: 8,
            border: `1px solid ${C.border}`, borderRadius: 8, background: C.bg,
          }}>
            {chatMessages.map(msg => (
              <div key={msg.id} style={{ display: 'flex', justifyContent: msg.type === 'user' ? 'flex-end' : 'flex-start', marginBottom: 12 }}>
                <div style={{ maxWidth: '72%', display: 'flex', alignItems: 'flex-start', gap: 8, flexDirection: msg.type === 'user' ? 'row-reverse' : 'row' }}>
                  <Avatar
                    icon={msg.type === 'user' ? <UserOutlined /> : <RobotOutlined />}
                    style={{ backgroundColor: msg.type === 'user' ? '#d93954' : '#2ea44f', flexShrink: 0 }}
                    size="small"
                  />
                  <div style={{
                    padding: '10px 14px', borderRadius: 12, fontSize: 13,
                    backgroundColor: msg.type === 'user' ? '#d93954' : C.card,
                    color: msg.type === 'user' ? '#fff' : C.text,
                    border: msg.type === 'bot' ? `1px solid ${C.border}` : 'none',
                    whiteSpace: 'pre-line',
                  }}>
                    {msg.content}
                  </div>
                </div>
              </div>
            ))}
            {typingMessageId && (
              <div style={{ display: 'flex', marginBottom: 12 }}>
                <div style={{ maxWidth: '72%', display: 'flex', alignItems: 'flex-start', gap: 8 }}>
                  <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#2ea44f', flexShrink: 0 }} size="small" />
                  <div style={{ padding: '10px 14px', borderRadius: 12, backgroundColor: C.card, border: `1px solid ${C.border}`, whiteSpace: 'pre-line', fontSize: 13 }}>
                    {typingText}
                    <span style={{ display: 'inline-block', width: 2, height: 14, background: '#d93954', marginLeft: 2, animation: 'blink 1s infinite' }} />
                  </div>
                </div>
              </div>
            )}
            {chatLoading && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#2ea44f' }} size="small" />
                <div style={{ padding: '10px 14px', background: C.card, borderRadius: 12, border: `1px solid ${C.border}`, fontSize: 13 }}>
                  <Spin size="small" /> 응답 생성 중...
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <TextArea
              value={currentMessage}
              onChange={e => setCurrentMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="AI 어시스턴트에게 질문하세요... (예: 고위험 직원 현황은?)"
              autoSize={{ minRows: 1, maxRows: 3 }}
              style={{ flex: 1 }}
            />
            <Button
              type="primary" icon={<SendOutlined />}
              onClick={handleSendMessage} loading={chatLoading}
              disabled={!currentMessage.trim()}
              style={{ background: '#d93954', borderColor: '#d93954' }}
            >
              전송
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomeMain;
