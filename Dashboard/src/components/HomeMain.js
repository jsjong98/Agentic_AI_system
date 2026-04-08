import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import ReactFlow, {
  Background,
  useNodesState, useEdgesState,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';

const SUPERVISOR_URL = process.env.REACT_APP_SUPERVISOR_URL || 'http://localhost:5000';

/* ── colour tokens ── */
const C = {
  red:    '#d93954',
  orange: '#e8721a',
  blue:   '#2563eb',
  purple: '#7c3aed',
  green:  '#2ea44f',
};

const AGENTS = [
  { id: 'structura', label: 'Structura',  sub: '정형 데이터 분석',  color: C.red    },
  { id: 'cognita',   label: 'Cognita',    sub: '관계망 분석',       color: C.blue   },
  { id: 'chronos',   label: 'Chronos',    sub: '시계열 행동 분석',  color: C.orange },
  { id: 'sentio',    label: 'Sentio',     sub: '자연어 감성 분석',  color: C.purple },
  { id: 'agora',     label: 'Agora',      sub: '외부 시장 분석',    color: C.green  },
];

/* ── custom node: Supervisor ── */
const SupervisorNode = ({ data }) => {
  const { phase } = data;
  const pulse = phase === 'supervisor' || phase === 'dispatching';
  return (
    <div style={{
      background: pulse ? '#1e293b' : '#0f172a',
      border: `2px solid ${pulse ? '#60a5fa' : '#334155'}`,
      borderRadius: 12, padding: '12px 24px', color: '#fff',
      minWidth: 180, textAlign: 'center',
      boxShadow: pulse ? '0 0 18px rgba(96,165,250,0.45)' : '0 2px 8px rgba(0,0,0,.3)',
      transition: 'all 0.4s ease',
    }}>
      <div style={{ fontSize: 18, marginBottom: 4 }}>🧠</div>
      <div style={{ fontWeight: 700, fontSize: 13 }}>Supervisor Agent</div>
      <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>오케스트레이션</div>
      {pulse && (
        <div style={{ fontSize: 10, color: '#60a5fa', marginTop: 6 }}>
          ● 워커 에이전트 배분 중...
        </div>
      )}
    </div>
  );
};

/* ── custom node: Worker Agent ── */
const AgentNode = ({ data }) => {
  const { label, sub, color, status } = data;
  const isActive = status === 'active';
  const isDone   = status === 'done';

  const bg     = isActive ? `${color}22` : isDone ? `${color}11` : '#f8fafc';
  const border = isActive ? color : isDone ? `${color}88` : '#e2e8f0';
  const tc     = isActive || isDone ? color : '#64748b';

  return (
    <div style={{
      background: bg, border: `2px solid ${border}`,
      borderRadius: 12, padding: '10px 18px',
      minWidth: 130, textAlign: 'center',
      boxShadow: isActive ? `0 0 14px ${color}55` : '0 1px 4px rgba(0,0,0,.08)',
      transition: 'all 0.3s ease',
    }}>
      <div style={{ fontWeight: 700, fontSize: 12, color: tc }}>{label}</div>
      <div style={{ fontSize: 10, color: isActive ? color : '#94a3b8', marginTop: 2 }}>{sub}</div>
      {isActive && <div style={{ fontSize: 9, color, marginTop: 5 }}>● 분석 중...</div>}
      {isDone   && <div style={{ fontSize: 9, color, marginTop: 5 }}>✓ 완료</div>}
    </div>
  );
};

/* ── custom node: Synthesize ── */
const SynthesizeNode = ({ data }) => {
  const { active, done } = data;
  const color = active || done ? '#f59e0b' : '#94a3b8';
  return (
    <div style={{
      background: active ? '#fffbeb' : done ? '#fefce8' : '#f8fafc',
      border: `2px solid ${active ? '#f59e0b' : done ? '#fde68a' : '#e2e8f0'}`,
      borderRadius: 12, padding: '12px 24px',
      minWidth: 180, textAlign: 'center',
      boxShadow: active ? '0 0 18px rgba(245,158,11,0.4)' : '0 1px 4px rgba(0,0,0,.06)',
      transition: 'all 0.4s ease',
    }}>
      <div style={{ fontSize: 18, marginBottom: 4 }}>⚡</div>
      <div style={{ fontWeight: 700, fontSize: 13, color }}>Synthesize Agent</div>
      <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>통합 분석 & 최종 판단</div>
      {active && <div style={{ fontSize: 10, color: '#f59e0b', marginTop: 6 }}>● 종합 분석 중...</div>}
      {done   && <div style={{ fontSize: 10, color: '#16a34a', marginTop: 6 }}>✓ 분석 완료</div>}
    </div>
  );
};

const NODE_TYPES = {
  supervisor: SupervisorNode,
  agent:      AgentNode,
  synthesize: SynthesizeNode,
};

/* ── build nodes ── */
const buildNodes = (phase, agentStatuses) => {
  const isSynth = phase === 'synthesizing';
  const isDone  = phase === 'done';
  return [
    {
      id: 'supervisor', type: 'supervisor',
      position: { x: 310, y: 20 },
      data: { phase },
      draggable: false,
    },
    ...AGENTS.map((ag, i) => ({
      id: ag.id, type: 'agent',
      position: { x: i * 160, y: 190 },
      data: { label: ag.label, sub: ag.sub, color: ag.color, status: agentStatuses[ag.id] || 'idle' },
      draggable: false,
    })),
    {
      id: 'synthesize', type: 'synthesize',
      position: { x: 310, y: 360 },
      data: { active: isSynth, done: isDone },
      draggable: false,
    },
  ];
};

/* ── build edges ── */
const buildEdges = (phase, agentStatuses) => {
  const dispatching = ['dispatching', 'analyzing', 'synthesizing', 'done'].includes(phase);
  const synthActive = phase === 'synthesizing' || phase === 'done';

  return [
    ...AGENTS.map(ag => ({
      id: `sup-${ag.id}`, source: 'supervisor', target: ag.id,
      animated: dispatching,
      style: { stroke: dispatching ? ag.color : '#cbd5e1', strokeWidth: dispatching ? 2 : 1 },
      markerEnd: { type: MarkerType.ArrowClosed, color: dispatching ? ag.color : '#cbd5e1' },
    })),
    ...AGENTS.map(ag => {
      const active = synthActive && agentStatuses[ag.id] === 'done';
      return {
        id: `${ag.id}-syn`, source: ag.id, target: 'synthesize',
        animated: active,
        style: { stroke: active ? ag.color : '#cbd5e1', strokeWidth: active ? 2 : 1 },
        markerEnd: { type: MarkerType.ArrowClosed, color: active ? ag.color : '#cbd5e1' },
      };
    }),
  ];
};

/* ── chat message ── */
const ChatMsg = ({ msg }) => {
  const isUser = msg.role === 'user';
  return (
    <div style={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start', marginBottom: 12 }}>
      {!isUser && (
        <div style={{
          width: 28, height: 28, borderRadius: '50%',
          background: 'linear-gradient(135deg,#0f172a,#2563eb)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 11, color: '#fff', flexShrink: 0, marginRight: 8, marginTop: 2,
        }}>AI</div>
      )}
      <div style={{
        maxWidth: '78%',
        background: isUser ? '#2563eb' : 'var(--card,#fff)',
        color: isUser ? '#fff' : 'var(--text,#1e293b)',
        borderRadius: isUser ? '16px 16px 4px 16px' : '4px 16px 16px 16px',
        padding: '10px 14px', fontSize: 13, lineHeight: 1.65,
        boxShadow: '0 1px 4px rgba(0,0,0,.08)',
        border: isUser ? 'none' : '1px solid var(--border,#e2e8f0)',
        whiteSpace: 'pre-wrap',
      }}>
        {msg.content}
        {msg.typing && (
          <span style={{
            display: 'inline-block', width: 6, height: 14,
            background: '#2563eb', marginLeft: 3,
            animation: 'cursor-blink 1s steps(1) infinite',
          }} />
        )}
      </div>
    </div>
  );
};

/* ── main component ── */
const HomeMain = () => {
  const [messages, setMessages]    = useState([
    { role: 'assistant', content: '안녕하세요! HR Agentic AI 시스템입니다.\n분석하고 싶은 직원 정보나 퇴사 위험 현황에 대해 질문해 주세요.' },
  ]);
  const [input, setInput]          = useState('');
  const [loading, setLoading]      = useState(false);
  const [phase, setPhase]          = useState('idle');
  const [agentStatuses, setAgents] = useState(
    Object.fromEntries(AGENTS.map(a => [a.id, 'idle']))
  );

  const chatEndRef  = useRef(null);
  const timerIds    = useRef([]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const clearTimers = () => {
    timerIds.current.forEach(clearTimeout);
    timerIds.current = [];
  };

  const schedule = (fn, delay) => {
    const id = setTimeout(fn, delay);
    timerIds.current.push(id);
  };

  const runWorkflow = useCallback(() => {
    clearTimers();
    setPhase('supervisor');
    setAgents(Object.fromEntries(AGENTS.map(a => [a.id, 'idle'])));

    schedule(() => setPhase('dispatching'), 600);

    AGENTS.forEach((ag, i) => {
      schedule(() => setAgents(prev => ({ ...prev, [ag.id]: 'active' })), 1200 + i * 280);
    });

    AGENTS.forEach((ag, i) => {
      schedule(() => setAgents(prev => ({ ...prev, [ag.id]: 'done' })), 2800 + i * 240);
    });

    schedule(() => setPhase('synthesizing'), 4100);
    schedule(() => setPhase('done'), 5700);
  }, []);

  const typeResponse = useCallback((text) => {
    const id = Date.now();
    let i = 0;
    setMessages(prev => [...prev, { role: 'assistant', content: '', typing: true, id }]);
    const ticker = setInterval(() => {
      i++;
      setMessages(prev =>
        prev.map(m => m.id === id ? { ...m, content: text.slice(0, i), typing: i < text.length } : m)
      );
      if (i >= text.length) clearInterval(ticker);
    }, 18);
  }, []);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setLoading(true);
    runWorkflow();

    try {
      const res  = await fetch(`${SUPERVISOR_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();
      typeResponse(data.response || data.message || '분석을 완료했습니다.');
    } catch {
      typeResponse('서버 연결에 실패했습니다. 잠시 후 다시 시도해 주세요.');
    } finally {
      setLoading(false);
    }
  }, [input, loading, runWorkflow, typeResponse]);

  const handleKey = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  }, [handleSend]);

  /* ReactFlow state synced from phase/agentStatuses */
  const rfNodes = useMemo(() => buildNodes(phase, agentStatuses), [phase, agentStatuses]);
  const rfEdges = useMemo(() => buildEdges(phase, agentStatuses), [phase, agentStatuses]);
  const [nodes, setNodes, onNodesChange] = useNodesState(rfNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(rfEdges);

  useEffect(() => setNodes(rfNodes), [rfNodes, setNodes]);
  useEffect(() => setEdges(rfEdges), [rfEdges, setEdges]);

  const phaseLabel = {
    supervisor:  '🧠 Supervisor 요청 수신...',
    dispatching: '📡 Worker Agent 배분 중...',
    analyzing:   '🔍 Agent 분석 진행 중...',
    synthesizing:'⚡ Synthesize Agent 통합 분석 중...',
    done:        '✅ 분석 완료',
  }[phase] || '';

  const QUICK_PROMPTS = [
    '고위험 직원 현황을 알려줘',
    'R&D 부서 퇴사 위험 분석해줘',
    'P01 번아웃 직원 개입 전략은?',
    '최근 이상 행동 패턴 보고해줘',
  ];

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 64px)', overflow: 'hidden' }}>
      <style>{`
        @keyframes cursor-blink { 0%,100%{opacity:1} 50%{opacity:0} }
        .react-flow__attribution { display:none !important; }
      `}</style>

      {/* ── LEFT: workflow diagram ── */}
      <div style={{
        flex: '0 0 55%', display: 'flex', flexDirection: 'column',
        borderRight: '1px solid var(--border,#e2e8f0)',
        background: 'var(--bg,#f8fafc)',
      }}>
        {/* header bar */}
        <div style={{
          padding: '14px 20px',
          background: 'linear-gradient(135deg,#0f172a 0%,#1a1a2e 100%)',
          color: '#fff', flexShrink: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14 }}>Agentic AI Workflow</div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
              5개 Worker Agent 기반 퇴사위험 분석 파이프라인
            </div>
          </div>
          {phase !== 'idle' && (
            <div style={{
              fontSize: 11, color: '#60a5fa',
              background: 'rgba(96,165,250,0.15)',
              border: '1px solid rgba(96,165,250,0.3)',
              padding: '4px 12px', borderRadius: 20,
            }}>
              {phaseLabel}
            </div>
          )}
        </div>

        {/* canvas */}
        <div style={{ flex: 1, position: 'relative' }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={NODE_TYPES}
            fitView
            fitViewOptions={{ padding: 0.3 }}
            panOnDrag={false}
            zoomOnScroll={false}
            zoomOnPinch={false}
            zoomOnDoubleClick={false}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={false}
          >
            <Background color="#e2e8f0" gap={20} size={1} />
          </ReactFlow>
        </div>

        {/* agent status chips */}
        <div style={{
          padding: '10px 16px', flexShrink: 0,
          borderTop: '1px solid var(--border,#e2e8f0)',
          background: 'var(--card,#fff)',
          display: 'flex', gap: 6, flexWrap: 'wrap',
        }}>
          {AGENTS.map(ag => {
            const st = agentStatuses[ag.id];
            return (
              <span key={ag.id} style={{
                padding: '3px 10px', borderRadius: 20, fontSize: 11, fontWeight: 600,
                display: 'inline-flex', alignItems: 'center', gap: 5,
                background: st !== 'idle' ? `${ag.color}15` : '#f1f5f9',
                border: `1px solid ${st !== 'idle' ? ag.color + '55' : '#e2e8f0'}`,
                color: st !== 'idle' ? ag.color : '#94a3b8',
                transition: 'all 0.3s',
              }}>
                <span style={{
                  width: 6, height: 6, borderRadius: '50%',
                  background: st !== 'idle' ? ag.color : '#cbd5e1',
                  display: 'inline-block',
                }} />
                {ag.label}
                {st === 'active' && ' ···'}
                {st === 'done'   && ' ✓'}
              </span>
            );
          })}
        </div>
      </div>

      {/* ── RIGHT: chat panel ── */}
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column',
        background: 'var(--bg,#f8fafc)', minWidth: 0,
      }}>
        {/* chat header */}
        <div style={{
          padding: '14px 20px', flexShrink: 0,
          background: 'var(--card,#fff)',
          borderBottom: '1px solid var(--border,#e2e8f0)',
        }}>
          <div style={{ fontWeight: 700, fontSize: 14, color: 'var(--text,#1e293b)' }}>
            HR AI 어시스턴트
          </div>
          <div style={{ fontSize: 11, color: 'var(--sub,#94a3b8)', marginTop: 2 }}>
            퇴사위험 분석 · 개입 전략 · 인사이트
          </div>
        </div>

        {/* message list */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px' }}>
          {messages.map((msg, i) => <ChatMsg key={i} msg={msg} />)}
          {loading && (
            <div style={{ display: 'flex', gap: 4, paddingLeft: 36, paddingBottom: 8 }}>
              {[0, 1, 2].map(i => (
                <div key={i} style={{
                  width: 7, height: 7, borderRadius: '50%', background: '#94a3b8',
                  animation: `cursor-blink 1.2s ${i * 0.2}s steps(1) infinite`,
                }} />
              ))}
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* quick prompts */}
        {messages.length <= 1 && (
          <div style={{ padding: '0 20px 12px', display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {QUICK_PROMPTS.map(q => (
              <button
                key={q}
                onClick={() => setInput(q)}
                style={{
                  background: 'var(--card,#fff)',
                  border: '1px solid var(--border,#e2e8f0)',
                  borderRadius: 20, padding: '5px 12px', fontSize: 11,
                  color: 'var(--sub,#64748b)', cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = '#2563eb'; e.currentTarget.style.color = '#2563eb'; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border,#e2e8f0)'; e.currentTarget.style.color = 'var(--sub,#64748b)'; }}
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {/* input bar */}
        <div style={{
          padding: '12px 20px', flexShrink: 0,
          background: 'var(--card,#fff)',
          borderTop: '1px solid var(--border,#e2e8f0)',
          display: 'flex', gap: 10, alignItems: 'flex-end',
        }}>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="분석하고 싶은 내용을 입력하세요... (Enter로 전송)"
            rows={1}
            style={{
              flex: 1, resize: 'none',
              border: '1px solid var(--border,#e2e8f0)',
              borderRadius: 12, padding: '10px 14px',
              fontSize: 13, lineHeight: 1.5,
              background: 'var(--bg,#f8fafc)',
              color: 'var(--text,#1e293b)',
              outline: 'none', fontFamily: 'inherit',
              maxHeight: 120, overflowY: 'auto',
              transition: 'border-color 0.2s',
            }}
            onFocus={e => { e.target.style.borderColor = '#2563eb'; }}
            onBlur={e => { e.target.style.borderColor = 'var(--border,#e2e8f0)'; }}
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            style={{
              background: loading || !input.trim() ? '#94a3b8' : '#2563eb',
              color: '#fff', border: 'none', borderRadius: 12,
              padding: '10px 20px', fontSize: 13, fontWeight: 600,
              cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
              transition: 'background 0.2s', flexShrink: 0,
              whiteSpace: 'nowrap',
            }}
          >
            {loading ? '분석 중...' : '전송'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default HomeMain;
