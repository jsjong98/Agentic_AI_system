import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Button, Input, Avatar, Spin, message, Modal } from 'antd';
import { SendOutlined, RobotOutlined, UserOutlined } from '@ant-design/icons';
import { predictionService } from '../services/predictionService';

const SUPERVISOR_URL = process.env.REACT_APP_SUPERVISOR_URL || 'http://localhost:5006';
const INTEGRATION_URL = process.env.REACT_APP_INTEGRATION_URL || 'http://localhost:5007';

const { TextArea } = Input;

if (typeof document !== 'undefined' && !document.getElementById('typing-cursor-style')) {
  const s = document.createElement('style');
  s.id = 'typing-cursor-style';
  s.textContent = '@keyframes blink{0%,50%{opacity:1}51%,100%{opacity:0}}';
  document.head.appendChild(s);
}

const Home = ({ globalBatchResults, lastAnalysisTimestamp }) => {
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

  const startTypingEffect = useCallback((messageId, fullText, onComplete, shouldScroll = true) => {
    if (typingIntervalRef.current) clearTimeout(typingIntervalRef.current);
    setTypingMessageId(messageId);
    setTypingText('');
    let idx = 0;
    const type = () => {
      if (idx < fullText.length) {
        setTypingText(fullText.substring(0, idx + 1));
        idx++;
        if (shouldScroll && userHasSentMessage && !isInitialLoad) {
          setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 10);
        }
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

  useEffect(() => {
    if (globalBatchResults && lastAnalysisTimestamp) {
      const p = predictionService.convertBatchResultToPrediction(globalBatchResults);
      if (p) { try { predictionService.savePredictionResult(p); loadPredictionHistory(); } catch {} }
    }
  }, [globalBatchResults, lastAnalysisTimestamp]);

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
      setPredictionHistory(predictionService.getPredictionHistory());
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
      const resp = await fetch(`${SUPERVISOR_URL}/api/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: msg, context: ctx }) });
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

  // 데이터
  const depts = [
    { n: 'Research & Development', c: 961, h: 46, m: 111, l: 804 },
    { n: 'Sales', c: 446, h: 26, m: 81, l: 339 },
    { n: 'Human Resources', c: 63, h: 5, m: 10, l: 48 },
  ];
  const totalEmp = 1470, totalHigh = 77, totalMed = 202, totalLow = 1191;
  const C = { card: 'var(--card,#fff)', border: 'var(--border,#eee)', sub: 'var(--sub,#888)', text: 'var(--text,#2d2d2d)', bg: 'var(--bg,#fafafa)' };
  const cardS = { background: C.card, borderRadius: 12, padding: 20, border: `1px solid ${C.border}`, boxShadow: '0 1px 4px rgba(0,0,0,.06)' };
  const secTitle = (icon, title) => <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 14 }}><span style={{ color: '#d93954' }}>{icon}</span> {title}</div>;

  return (
    <div>
      {/* KPI */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 12, marginBottom: 20 }}>
        {[
          { label: '전사 인원 현황', val: totalEmp.toLocaleString(), unit: '명', color: C.text, top: '#d93954' },
          { label: '퇴사 고위험군', val: totalHigh, unit: '명', color: '#d93954', top: '#d93954' },
          { label: '잠재적 위험군', val: totalMed, unit: '명', color: '#e8721a', top: '#e8721a' },
          { label: '안정/양호군', val: totalLow.toLocaleString(), unit: '명', color: '#2ea44f', top: '#2ea44f' },
          { label: '평균 위험 점수', val: '0.28', unit: '', color: '#2563eb', top: '#2563eb' },
        ].map((k, i) => (
          <div key={i} style={{ ...cardS, textAlign: 'center', position: 'relative', overflow: 'hidden' }}>
            <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 4, background: k.top }} />
            <div style={{ fontSize: 11, color: C.sub, marginBottom: 6, fontWeight: 500 }}>{k.label}</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: k.color }}>{k.val}<span style={{ fontSize: 13, color: C.sub }}>{k.unit}</span></div>
          </div>
        ))}
      </div>

      {/* 필터 */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center' }}>
        <span style={{ fontSize: 12, color: C.sub, fontWeight: 600 }}>필터:</span>
        {['전체 부서', '전체 직급', '전체 위험등급'].map((f, i) => (
          <select key={i} style={{ padding: '7px 10px', border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12, fontFamily: 'inherit', background: C.card, color: C.text }}><option>{f}</option></select>
        ))}
      </div>

      {/* 부서 테이블 + Persona */}
      <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: 16, marginBottom: 20 }}>
        <div style={cardS}>
          {secTitle('☰', '전사 인원 현황 (부서별)')}
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead><tr>{['Department','인원','위험 분포','고위험','위험률'].map(h => <th key={h} style={{ textAlign: 'left', padding: '10px 12px', background: C.bg, borderBottom: `2px solid ${C.border}`, fontWeight: 600, color: C.sub, fontSize: 12 }}>{h}</th>)}</tr></thead>
            <tbody>{depts.map(d => { const r = Math.round(d.h/d.c*100); const bc = r>=10?'#d93954':r>=5?'#e8721a':'#2ea44f'; return (
              <tr key={d.n} style={{ borderBottom: `1px solid ${C.border}` }}>
                <td style={{ padding: '10px 12px', fontWeight: 500 }}>{d.n}</td>
                <td style={{ padding: '10px 12px' }}>{d.c}명</td>
                <td style={{ padding: '10px 12px' }}><div style={{ display: 'flex', alignItems: 'center', gap: 8 }}><div style={{ height: 8, borderRadius: 4, width: r*5, minWidth: 4, background: bc }} /><span style={{ fontSize: 11, color: C.sub }}>{r}%</span></div></td>
                <td style={{ padding: '10px 12px' }}><strong style={{ color: '#d93954' }}>{d.h}명</strong></td>
                <td style={{ padding: '10px 12px' }}><span style={{ padding: '3px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600, background: r>=10?'#fde8ec':'#e6f6ec', color: bc }}>{r}%</span></td>
              </tr>);})}</tbody>
          </table>
        </div>
        <div style={cardS}>
          {secTitle('◉', 'Persona 분포')}
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr>{['유형','상세','분포','비율'].map(h => <th key={h} style={{ textAlign: 'left', padding: 8, background: C.bg, borderBottom: `2px solid ${C.border}`, fontWeight: 600, color: C.sub }}>{h}</th>)}</tr></thead>
            <tbody>{[
              { t: 'P01', l: '번아웃 직전', c: 38, p: '49%', bg: '#fde8ec', cl: '#d93954' },
              { t: 'P02', l: '보상 실망', c: 8, p: '10%', bg: '#e8f0fe', cl: '#2563eb' },
              { t: 'P03', l: '성장 정체', c: 20, p: '26%', bg: '#fef3e2', cl: '#e8721a' },
              { t: 'P04', l: '보상체감 낮음', c: 11, p: '14%', bg: '#f3e8fd', cl: '#7c3aed' },
            ].map(p => (
              <tr key={p.t} style={{ borderBottom: `1px solid ${C.border}` }}>
                <td style={{ padding: 8 }}><span style={{ padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600, background: p.bg, color: p.cl }}>{p.t}</span></td>
                <td style={{ padding: 8 }}>{p.l}</td><td style={{ padding: 8 }}>{p.c}명</td><td style={{ padding: 8 }}>{p.p}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </div>

      {/* Agent 기여도 + 워크플로우 */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
        <div style={cardS}>
          {secTitle('◉', 'Agent별 위험 탐지 기여도')}
          {[{ n: 'Structura', p: 32, c: '#d93954' }, { n: 'Cognita', p: 22, c: '#2563eb' }, { n: 'Chronos', p: 21, c: '#e8721a' }, { n: 'Sentio', p: 15, c: '#7c3aed' }, { n: 'Agora', p: 10, c: '#2ea44f' }].map(a => (
            <div key={a.n} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
              <span style={{ width: 70, fontSize: 12, fontWeight: 600, textAlign: 'right' }}>{a.n}</span>
              <div style={{ flex: 1, height: 22, background: C.border, borderRadius: 6, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${a.p * 2.5}%`, background: a.c, borderRadius: 6, display: 'flex', alignItems: 'center', paddingLeft: 8, fontSize: 11, color: '#fff', fontWeight: 600 }}>{a.p}%</div>
              </div>
            </div>
          ))}
        </div>
        <div style={cardS}>
          {secTitle('◉', '에이전트 워크플로우')}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ background: '#2d2d2d', color: '#fff', padding: '8px 28px', borderRadius: 8, fontSize: 12, fontWeight: 700 }}>Supervisor Agent</div>
            <div style={{ width: 2, height: 12, background: '#d93954' }} /><div style={{ width: 0, height: 0, borderLeft: '5px solid transparent', borderRight: '5px solid transparent', borderTop: '7px solid #d93954' }} />
            <div style={{ display: 'flex', gap: 8, margin: '6px 0', flexWrap: 'wrap', justifyContent: 'center' }}>
              {[{ n: 'Structura', c: '#d93954', i: '📊' }, { n: 'Cognita', c: '#2563eb', i: '🔗' }, { n: 'Chronos', c: '#e8721a', i: '📈' }, { n: 'Sentio', c: '#7c3aed', i: '💬' }, { n: 'Agora', c: '#2ea44f', i: '🎯' }].map(w => (
                <div key={w.n} style={{ border: `2px solid ${w.c}`, borderRadius: 8, padding: '6px 10px', fontSize: 10, fontWeight: 600, textAlign: 'center', background: `${w.c}10` }}><div style={{ fontSize: 14 }}>{w.i}</div>{w.n}</div>
              ))}
            </div>
            <div style={{ width: 0, height: 0, borderLeft: '5px solid transparent', borderRight: '5px solid transparent', borderTop: '7px solid #d93954' }} /><div style={{ width: 2, height: 12, background: '#d93954' }} />
            <div style={{ background: '#d93954', color: '#fff', padding: '8px 28px', borderRadius: 8, fontSize: 12, fontWeight: 700 }}>Synthesize Agent</div>
          </div>
          <div style={{ textAlign: 'center', fontSize: 10, color: C.sub, marginTop: 8 }}>데이터 수집 → 개별 분석 → 종합 평가 → 위험도 산출</div>
        </div>
      </div>

      {/* AI 채팅 */}
      <div style={cardS}>
        {secTitle('◉', 'AI 어시스턴트')}
        <div style={{ maxHeight: 400, overflowY: 'auto', marginBottom: 12, padding: 8, border: `1px solid ${C.border}`, borderRadius: 8, background: C.bg }}>
          {chatMessages.map(msg => (
            <div key={msg.id} style={{ display: 'flex', justifyContent: msg.type === 'user' ? 'flex-end' : 'flex-start', marginBottom: 12 }}>
              <div style={{ maxWidth: '70%', display: 'flex', alignItems: 'flex-start', gap: 8, flexDirection: msg.type === 'user' ? 'row-reverse' : 'row' }}>
                <Avatar icon={msg.type === 'user' ? <UserOutlined /> : <RobotOutlined />} style={{ backgroundColor: msg.type === 'user' ? '#d93954' : '#2ea44f', flexShrink: 0 }} size="small" />
                <div style={{ padding: '10px 14px', borderRadius: 12, backgroundColor: msg.type === 'user' ? '#d93954' : C.card, color: msg.type === 'user' ? '#fff' : C.text, border: msg.type === 'bot' ? `1px solid ${C.border}` : 'none', whiteSpace: 'pre-line', fontSize: 13 }}>{msg.content}</div>
              </div>
            </div>
          ))}
          {typingMessageId && (
            <div style={{ display: 'flex', marginBottom: 12 }}>
              <div style={{ maxWidth: '70%', display: 'flex', alignItems: 'flex-start', gap: 8 }}>
                <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#2ea44f', flexShrink: 0 }} size="small" />
                <div style={{ padding: '10px 14px', borderRadius: 12, backgroundColor: C.card, border: `1px solid ${C.border}`, whiteSpace: 'pre-line', fontSize: 13 }}>{typingText}<span style={{ display: 'inline-block', width: 2, height: 14, background: '#d93954', marginLeft: 2, animation: 'blink 1s infinite' }} /></div>
              </div>
            </div>
          )}
          {chatLoading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#2ea44f' }} size="small" />
              <div style={{ padding: '10px 14px', background: C.card, borderRadius: 12, border: `1px solid ${C.border}`, fontSize: 13 }}><Spin size="small" /> 응답 생성 중...</div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <TextArea value={currentMessage} onChange={e => setCurrentMessage(e.target.value)} onKeyPress={handleKeyPress} placeholder="AI 어시스턴트에게 질문하세요... (예: 사번 1 직원의 위험도는?)" autoSize={{ minRows: 1, maxRows: 3 }} style={{ flex: 1 }} />
          <Button type="primary" icon={<SendOutlined />} onClick={handleSendMessage} loading={chatLoading} disabled={!currentMessage.trim()} style={{ background: '#d93954', borderColor: '#d93954' }}>전송</Button>
        </div>
      </div>
    </div>
  );
};

export default Home;
