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
    @keyframes fadeSlide{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
    @keyframes scanPulse{0%{opacity:.6}50%{opacity:1}100%{opacity:.6}}
    @keyframes progressGlow{0%{box-shadow:0 0 4px rgba(217,57,84,.3)}50%{box-shadow:0 0 12px rgba(217,57,84,.6)}100%{box-shadow:0 0 4px rgba(217,57,84,.3)}}
    @keyframes barStripe{0%{background-position:0 0}100%{background-position:20px 0}}
  `;
  document.head.appendChild(s);
}

// ─────────────────────────── Animated Workflow ──────────────────────────────
const TOTAL_EMPLOYEES = 1470;
const AGENTS = [
  { n: 'Structura', c: '#d93954', i: '📊', desc: '정형 데이터' },
  { n: 'Cognita',   c: '#2563eb', i: '🔗', desc: '관계망' },
  { n: 'Chronos',   c: '#e8721a', i: '📈', desc: '시계열' },
  { n: 'Sentio',    c: '#7c3aed', i: '💬', desc: '감성' },
  { n: 'Agora',     c: '#2ea44f', i: '🎯', desc: '외부시장' },
];

/* ── phases: 'idle' | 'init' | 'dispatch' | 'scanning' | 'synthesize' | 'done' ── */
const AgentWorkflow = ({ isDark }) => {
  const [phase, setPhase] = useState('idle');
  const [agentProgress, setAgentProgress] = useState([0, 0, 0, 0, 0]);
  const [agentEmpId, setAgentEmpId] = useState([0, 0, 0, 0, 0]);
  const [synProgress, setSynProgress] = useState(0);
  const [cycleCount, setCycleCount] = useState(0);
  const cancelRef = useRef(false);
  const frameRef = useRef(null);

  useEffect(() => {
    cancelRef.current = false;
    const wait = (ms) => new Promise(r => { const t = setTimeout(r, ms); return () => clearTimeout(t); });

    const runCycle = async () => {
      while (!cancelRef.current) {
        // Phase: idle
        setPhase('idle');
        setAgentProgress([0, 0, 0, 0, 0]);
        setAgentEmpId([0, 0, 0, 0, 0]);
        setSynProgress(0);
        await wait(1200);
        if (cancelRef.current) return;

        // Phase: init
        setPhase('init');
        await wait(1000);
        if (cancelRef.current) return;

        // Phase: dispatch
        setPhase('dispatch');
        await wait(800);
        if (cancelRef.current) return;

        // Phase: scanning — all 5 agents scan 1470 employees simultaneously
        setPhase('scanning');
        const speeds = [1.0, 0.85, 0.92, 0.78, 0.95]; // each agent slightly different speed
        const prog = [0, 0, 0, 0, 0];
        const empIds = [0, 0, 0, 0, 0];
        const sampleIds = Array.from({ length: TOTAL_EMPLOYEES }, (_, i) => i + 1);
        // shuffle for random-looking employee IDs
        for (let k = sampleIds.length - 1; k > 0; k--) {
          const j = Math.floor(Math.random() * (k + 1));
          [sampleIds[k], sampleIds[j]] = [sampleIds[j], sampleIds[k]];
        }

        await new Promise((resolve) => {
          const start = performance.now();
          const SCAN_DURATION = 6000; // 6 seconds to scan all
          const tick = (now) => {
            if (cancelRef.current) { resolve(); return; }
            const elapsed = now - start;
            const baseT = Math.min(elapsed / SCAN_DURATION, 1);
            // ease-in-out for smoother feel
            const eased = baseT < 0.5
              ? 2 * baseT * baseT
              : 1 - Math.pow(-2 * baseT + 2, 2) / 2;

            let allDone = true;
            for (let i = 0; i < 5; i++) {
              const agentT = Math.min(eased * (1 / speeds[i]), 1);
              prog[i] = Math.floor(agentT * TOTAL_EMPLOYEES);
              empIds[i] = prog[i] > 0 ? sampleIds[Math.min(prog[i] - 1, TOTAL_EMPLOYEES - 1)] : 0;
              if (prog[i] < TOTAL_EMPLOYEES) allDone = false;
            }
            setAgentProgress([...prog]);
            setAgentEmpId([...empIds]);

            if (allDone) {
              resolve();
            } else {
              frameRef.current = requestAnimationFrame(tick);
            }
          };
          frameRef.current = requestAnimationFrame(tick);
        });
        if (cancelRef.current) return;

        // small pause after all agents done
        await wait(600);
        if (cancelRef.current) return;

        // Phase: synthesize — combining results
        setPhase('synthesize');
        await new Promise((resolve) => {
          const start = performance.now();
          const SYN_DURATION = 2500;
          const tick = (now) => {
            if (cancelRef.current) { resolve(); return; }
            const t = Math.min((now - start) / SYN_DURATION, 1);
            const count = Math.floor(t * TOTAL_EMPLOYEES);
            setSynProgress(count);
            if (count >= TOTAL_EMPLOYEES) { resolve(); return; }
            frameRef.current = requestAnimationFrame(tick);
          };
          frameRef.current = requestAnimationFrame(tick);
        });
        if (cancelRef.current) return;

        // Phase: done
        setPhase('done');
        setCycleCount(c => c + 1);
        await wait(4000);
      }
    };

    runCycle();
    return () => {
      cancelRef.current = true;
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    };
  }, []);

  const supActive = phase !== 'idle';
  const dispFlow = phase === 'dispatch' || phase === 'scanning' || phase === 'synthesize' || phase === 'done';
  const isScanning = phase === 'scanning';
  const resultFlow = phase === 'synthesize' || phase === 'done';
  const synActive = phase === 'synthesize' || phase === 'done';
  const allDone = phase === 'done';

  const totalScanned = agentProgress.reduce((s, v) => s + v, 0);
  const overallPct = Math.min(totalScanned / (TOTAL_EMPLOYEES * 5) * 100, 100);

  const card = { borderRadius: 10, padding: '0 8px', fontFamily: 'inherit', transition: 'all 0.45s' };
  const C = isDark ? { bg: '#151c2c', sub: '#6b7280' } : { bg: '#fff', sub: '#888' };

  // Dynamic status message
  const getStatusMsg = () => {
    switch (phase) {
      case 'idle': return '시스템 대기 중...';
      case 'init': return `Supervisor Agent: ${TOTAL_EMPLOYEES}명 직원 분석 요청 수신`;
      case 'dispatch': return 'Worker Agents에게 작업 분배 중...';
      case 'scanning': {
        const done = agentProgress.filter(p => p >= TOTAL_EMPLOYEES).length;
        const scanning = agentProgress.filter(p => p > 0 && p < TOTAL_EMPLOYEES).length;
        if (done === 5) return '모든 Agent 분석 완료 ✓';
        return `분석 진행 중 — ${done}/5 완료, ${scanning}개 스캔 중...`;
      }
      case 'synthesize':
        return `Synthesize: 결과 종합 중... ${synProgress.toLocaleString()}/${TOTAL_EMPLOYEES.toLocaleString()}`;
      case 'done':
        return `✓ ${TOTAL_EMPLOYEES.toLocaleString()}명 분석 완료 — 사이클 #${cycleCount}`;
      default: return '';
    }
  };

  // Progress bar component
  const ProgressBar = ({ pct, color, height = 3 }) => (
    <div style={{
      width: '100%', height, borderRadius: height / 2,
      background: isDark ? '#1a2333' : '#e8e8e8', overflow: 'hidden',
    }}>
      <div style={{
        width: `${pct}%`, height: '100%', borderRadius: height / 2,
        background: pct >= 100
          ? color
          : `repeating-linear-gradient(90deg,${color},${color} 8px,${color}cc 8px,${color}cc 16px)`,
        backgroundSize: pct < 100 ? '20px 100%' : 'auto',
        animation: pct > 0 && pct < 100 ? 'barStripe 0.6s linear infinite' : 'none',
        transition: 'width 0.15s linear',
      }} />
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', userSelect: 'none' }}>
      {/* Status badge */}
      <div style={{
        marginBottom: 10, fontSize: 11, fontWeight: 600,
        color: allDone ? '#2ea44f' : '#d93954',
        background: allDone ? '#e6f6ec' : '#fde8ec',
        padding: '4px 14px', borderRadius: 20,
        transition: 'all 0.4s',
        animation: isScanning ? 'scanPulse 1.5s ease-in-out infinite' : 'fadeSlide 0.3s ease',
        minWidth: 280, textAlign: 'center',
      }}>
        {getStatusMsg()}
      </div>

      {/* Overall progress bar */}
      {(isScanning || phase === 'synthesize') && (
        <div style={{ width: '90%', marginBottom: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: C.sub, marginBottom: 3 }}>
            <span>전체 진행률</span>
            <span style={{ fontWeight: 700, color: '#d93954', fontVariantNumeric: 'tabular-nums' }}>
              {phase === 'synthesize'
                ? `종합 ${synProgress.toLocaleString()} / ${TOTAL_EMPLOYEES.toLocaleString()}`
                : `${Math.round(overallPct)}%`}
            </span>
          </div>
          <ProgressBar pct={phase === 'synthesize' ? (synProgress / TOTAL_EMPLOYEES * 100) : overallPct} color="#d93954" height={4} />
        </div>
      )}

      {/* Supervisor */}
      <div style={{
        ...card,
        padding: '8px 28px', fontWeight: 700, fontSize: 12,
        background: supActive ? '#2d2d2d' : (isDark ? '#1e2a3a' : '#f0f0f0'),
        color: supActive ? '#fff' : C.sub,
        border: supActive ? '2px solid #d93954' : `2px solid ${isDark ? '#2a3a4a' : '#e0e0e0'}`,
        boxShadow: supActive ? '0 0 20px rgba(217,57,84,.45)' : 'none',
        animation: (phase === 'init' || phase === 'dispatch') ? 'progressGlow 1.5s ease-in-out infinite' : 'none',
      }}>
        🤖 Supervisor Agent
        {supActive && (
          <span style={{ fontSize: 10, fontWeight: 400, marginLeft: 8, opacity: 0.8 }}>
            ({TOTAL_EMPLOYEES.toLocaleString()}명)
          </span>
        )}
      </div>

      {/* Arrow: Supervisor → Agents */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', height: 28, justifyContent: 'center' }}>
        <div style={{
          width: 2, height: 18,
          background: dispFlow ? '#d93954' : (isDark ? '#2a3a4a' : '#e0e0e0'),
          animation: dispFlow ? 'flowPulse 0.9s ease-in-out infinite' : 'none',
          transition: 'background 0.4s',
        }} />
        <div style={{
          width: 0, height: 0,
          borderLeft: '5px solid transparent', borderRight: '5px solid transparent',
          borderTop: `6px solid ${dispFlow ? '#d93954' : (isDark ? '#2a3a4a' : '#e0e0e0')}`,
          transition: 'border-color 0.4s',
        }} />
      </div>

      {/* Agents row */}
      <div style={{ display: 'flex', gap: 6, justifyContent: 'center', flexWrap: 'wrap', width: '100%' }}>
        {AGENTS.map((a, i) => {
          const pct = (agentProgress[i] / TOTAL_EMPLOYEES) * 100;
          const done = agentProgress[i] >= TOTAL_EMPLOYEES;
          const active = agentProgress[i] > 0;
          return (
            <div key={a.n} style={{
              border: `2px solid ${done ? a.c : active ? `${a.c}88` : (isDark ? '#2a3a4a' : '#e0e0e0')}`,
              borderRadius: 10, padding: '6px 6px 8px', textAlign: 'center', minWidth: 72, flex: '1 1 0',
              background: done ? `${a.c}1A` : active ? `${a.c}0A` : (isDark ? '#1e2a3a' : '#fafafa'),
              color: active ? a.c : C.sub,
              boxShadow: done ? `0 0 14px ${a.c}55` : active ? `0 0 6px ${a.c}22` : 'none',
              transition: 'all 0.3s',
            }}>
              <div style={{ fontSize: 16, marginBottom: 1 }}>{a.i}</div>
              <div style={{ fontSize: 10, fontWeight: 700 }}>{a.n}</div>
              <div style={{ fontSize: 9, marginTop: 1, color: active ? a.c : (isDark ? '#3a4a5a' : '#ccc') }}>
                {a.desc}
              </div>
              {/* Progress bar per agent */}
              <div style={{ margin: '4px 0 2px', padding: '0 2px' }}>
                <ProgressBar pct={pct} color={a.c} height={3} />
              </div>
              {/* Count / Employee ID */}
              <div style={{ fontSize: 9, fontWeight: 600, minHeight: 14, fontVariantNumeric: 'tabular-nums' }}>
                {done ? (
                  <span style={{ color: '#2ea44f' }}>✓ {TOTAL_EMPLOYEES.toLocaleString()}</span>
                ) : active ? (
                  <span style={{ color: a.c, animation: 'scanPulse 0.8s ease infinite' }}>
                    #{agentEmpId[i]} ({agentProgress[i].toLocaleString()}/{TOTAL_EMPLOYEES.toLocaleString()})
                  </span>
                ) : (
                  <span style={{ color: C.sub }}>대기</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Arrow: Agents → Synthesize */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', height: 28, justifyContent: 'center' }}>
        <div style={{
          width: 0, height: 0,
          borderLeft: '5px solid transparent', borderRight: '5px solid transparent',
          borderBottom: `6px solid ${resultFlow ? '#d93954' : (isDark ? '#2a3a4a' : '#e0e0e0')}`,
          transition: 'border-color 0.4s',
        }} />
        <div style={{
          width: 2, height: 18,
          background: resultFlow ? '#d93954' : (isDark ? '#2a3a4a' : '#e0e0e0'),
          animation: resultFlow ? 'flowPulse 0.9s ease-in-out infinite' : 'none',
          transition: 'background 0.4s',
        }} />
      </div>

      {/* Synthesize */}
      <div style={{
        ...card,
        padding: '8px 28px', fontWeight: 700, fontSize: 12,
        background: synActive ? (allDone ? '#2ea44f' : '#d93954') : (isDark ? '#1e2a3a' : '#f0f0f0'),
        color: synActive ? '#fff' : C.sub,
        border: synActive ? `2px solid ${allDone ? '#2ea44f' : '#d93954'}` : `2px solid ${isDark ? '#2a3a4a' : '#e0e0e0'}`,
        boxShadow: synActive ? `0 0 24px ${allDone ? 'rgba(46,164,79,.55)' : 'rgba(217,57,84,.55)'}` : 'none',
        transition: 'all 0.5s',
      }}>
        ⚡ Synthesize Agent
        {synActive && (
          <span style={{ fontSize: 10, fontWeight: 400, marginLeft: 8, opacity: 0.8 }}>
            {allDone ? '✓ 완료' : `${synProgress.toLocaleString()}/${TOTAL_EMPLOYEES.toLocaleString()}`}
          </span>
        )}
      </div>

      {/* Legend + stats */}
      <div style={{ display: 'flex', gap: 12, marginTop: 10, flexWrap: 'wrap', justifyContent: 'center', alignItems: 'center' }}>
        <div style={{ display: 'flex', gap: 5, alignItems: 'center', fontSize: 10, color: C.sub }}>
          <div style={{ width: 20, height: 3, background: '#d93954', borderRadius: 2 }} /> 데이터 흐름
        </div>
        <div style={{ display: 'flex', gap: 5, alignItems: 'center', fontSize: 10, color: C.sub }}>
          <div style={{ width: 8, height: 8, background: '#2ea44f', borderRadius: '50%' }} /> 완료
        </div>
        <div style={{ display: 'flex', gap: 5, alignItems: 'center', fontSize: 10, color: C.sub }}>
          <div style={{ width: 8, height: 8, background: isDark ? '#2a3a4a' : '#e0e0e0', borderRadius: '50%', border: '1px solid #ccc' }} /> 대기
        </div>
        {cycleCount > 0 && (
          <div style={{ fontSize: 10, color: C.sub, marginLeft: 4, fontVariantNumeric: 'tabular-nums' }}>
            | 완료 사이클: {cycleCount}
          </div>
        )}
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
