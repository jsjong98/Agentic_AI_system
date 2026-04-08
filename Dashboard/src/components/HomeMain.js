import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import ReactFlow, {
  Background,
  useNodesState, useEdgesState,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';

const SUPERVISOR_URL = process.env.REACT_APP_SUPERVISOR_URL || 'http://localhost:5000';
const TOTAL_EMP = 1470;
const BATCH     = 50;   // display step size

const C = {
  red:    '#d93954',
  orange: '#e8721a',
  blue:   '#2563eb',
  purple: '#7c3aed',
  green:  '#2ea44f',
};

const AGENTS = [
  { id: 'structura', label: 'Structura',  sub: '정형 데이터',  color: C.red    },
  { id: 'cognita',   label: 'Cognita',    sub: '관계망',       color: C.blue   },
  { id: 'chronos',   label: 'Chronos',    sub: '시계열',       color: C.orange },
  { id: 'sentio',    label: 'Sentio',     sub: '자연어',       color: C.purple },
  { id: 'agora',     label: 'Agora',      sub: '외부시장',     color: C.green  },
];

/* ── Supervisor node ── */
const SupervisorNode = ({ data }) => {
  const { phase } = data;
  const pulse = phase === 'supervisor' || phase === 'dispatching';
  return (
    <div style={{
      background: pulse ? '#1e3a5f' : '#0f172a',
      border: `2px solid ${pulse ? '#60a5fa' : '#334155'}`,
      borderRadius: 14, padding: '14px 28px', color: '#fff',
      minWidth: 200, textAlign: 'center',
      boxShadow: pulse ? '0 0 24px rgba(96,165,250,0.55)' : '0 2px 12px rgba(0,0,0,.4)',
      transition: 'all 0.4s ease',
    }}>
      <div style={{ fontSize: 22, marginBottom: 6 }}>🧠</div>
      <div style={{ fontWeight: 700, fontSize: 14 }}>Supervisor Agent</div>
      <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 3 }}>오케스트레이션 · 태스크 배분</div>
      {pulse && (
        <div style={{
          fontSize: 10, color: '#60a5fa', marginTop: 8,
          animation: 'pulse-text 1.2s ease-in-out infinite',
        }}>
          ▶ 1,470명 분석 태스크 배분 중...
        </div>
      )}
    </div>
  );
};

/* ── Agent node ── */
const AgentNode = ({ data }) => {
  const { label, sub, color, status, scanCount } = data;
  const isActive = status === 'active';
  const isDone   = status === 'done';
  const tc = isActive || isDone ? color : '#64748b';

  /* compute displayed range */
  const rangeEnd   = Math.min(scanCount, TOTAL_EMP);
  const rangeStart = Math.max(1, rangeEnd - BATCH + 1);

  return (
    <div style={{
      background: isActive ? `${color}20` : isDone ? `${color}0d` : '#f8fafc',
      border: `2px solid ${isActive ? color : isDone ? color + '77' : '#e2e8f0'}`,
      borderRadius: 12, padding: '10px 16px',
      minWidth: 140, textAlign: 'center',
      boxShadow: isActive ? `0 0 18px ${color}66` : '0 1px 4px rgba(0,0,0,.07)',
      transition: 'all 0.35s ease',
    }}>
      <div style={{ fontWeight: 700, fontSize: 12, color: tc }}>{label}</div>
      <div style={{ fontSize: 10, color: isActive ? color : '#94a3b8', marginTop: 2 }}>{sub}</div>

      {isActive && scanCount > 0 && (
        <div style={{
          marginTop: 7, padding: '4px 8px',
          background: `${color}18`, borderRadius: 6,
          border: `1px solid ${color}44`,
        }}>
          <div style={{ fontSize: 9, color, fontWeight: 600 }}>
            직원 #{String(rangeStart).padStart(3,'0')} ~ #{String(rangeEnd).padStart(3,'0')}
          </div>
          <div style={{ fontSize: 8, color: '#94a3b8', marginTop: 1 }}>스캔 중...</div>
        </div>
      )}

      {isActive && scanCount === 0 && (
        <div style={{ fontSize: 9, color, marginTop: 6 }}>● 초기화 중...</div>
      )}

      {isDone && (
        <div style={{
          marginTop: 7, padding: '3px 8px',
          background: `${color}12`, borderRadius: 6,
          fontSize: 9, color, fontWeight: 600,
        }}>
          ✓ 1,470명 분석 완료
        </div>
      )}
    </div>
  );
};

/* ── Synthesize node ── */
const SynthesizeNode = ({ data }) => {
  const { active, done } = data;
  const color = active || done ? '#f59e0b' : '#94a3b8';
  return (
    <div style={{
      background: active ? '#fffbeb' : done ? '#fefce8' : '#f8fafc',
      border: `2px solid ${active ? '#f59e0b' : done ? '#fde68a' : '#e2e8f0'}`,
      borderRadius: 14, padding: '14px 28px',
      minWidth: 200, textAlign: 'center',
      boxShadow: active ? '0 0 24px rgba(245,158,11,0.45)' : '0 1px 4px rgba(0,0,0,.06)',
      transition: 'all 0.4s ease',
    }}>
      <div style={{ fontSize: 22, marginBottom: 6 }}>⚡</div>
      <div style={{ fontWeight: 700, fontSize: 14, color }}>Synthesize Agent</div>
      <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 3 }}>통합 분석 · 최종 위험도 판정</div>
      {active && (
        <div style={{ fontSize: 10, color: '#f59e0b', marginTop: 8, animation: 'pulse-text 1.2s ease-in-out infinite' }}>
          ▶ 5개 Agent 결과 통합 중...
        </div>
      )}
      {done && (
        <div style={{ fontSize: 10, color: '#16a34a', marginTop: 8 }}>
          ✓ 분석 완료 · 보고서 생성됨
        </div>
      )}
    </div>
  );
};

const NODE_TYPES = {
  supervisor: SupervisorNode,
  agent:      AgentNode,
  synthesize: SynthesizeNode,
};

/* ── Node layout ── */
/*
  5 agents at y=200, evenly spaced across 840px wide canvas
  Supervisor at x=320, y=20
  Synthesize at x=320, y=390
*/
const buildNodes = (phase, agentStatuses, scanCounts) => {
  const isSynth = phase === 'synthesizing';
  const isDone  = phase === 'done';
  return [
    {
      id: 'supervisor', type: 'supervisor',
      position: { x: 270, y: 20 },
      data: { phase },
      draggable: false,
    },
    ...AGENTS.map((ag, i) => ({
      id: ag.id, type: 'agent',
      position: { x: i * 175, y: 200 },
      data: {
        label: ag.label,
        sub:   ag.sub,
        color: ag.color,
        status:    agentStatuses[ag.id] || 'idle',
        scanCount: scanCounts[ag.id]    || 0,
      },
      draggable: false,
    })),
    {
      id: 'synthesize', type: 'synthesize',
      position: { x: 270, y: 395 },
      data: { active: isSynth, done: isDone },
      draggable: false,
    },
  ];
};

/* ── Edge styles ── */
const buildEdges = (phase, agentStatuses) => {
  const dispatching = ['dispatching', 'analyzing', 'synthesizing', 'done'].includes(phase);
  const synthActive = phase === 'synthesizing' || phase === 'done';

  const supEdges = AGENTS.map(ag => ({
    id: `sup-${ag.id}`,
    source: 'supervisor', target: ag.id,
    type: 'smoothstep',
    animated: dispatching,
    style: {
      stroke: dispatching ? ag.color : '#cbd5e1',
      strokeWidth: dispatching ? 3 : 1.5,
      filter: dispatching ? `drop-shadow(0 0 4px ${ag.color}88)` : 'none',
    },
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: dispatching ? ag.color : '#cbd5e1',
      width: 16, height: 16,
    },
  }));

  const synthEdges = AGENTS.map(ag => {
    const active = synthActive && agentStatuses[ag.id] === 'done';
    return {
      id: `${ag.id}-syn`,
      source: ag.id, target: 'synthesize',
      type: 'smoothstep',
      animated: active,
      style: {
        stroke: active ? ag.color : '#cbd5e1',
        strokeWidth: active ? 3 : 1.5,
        filter: active ? `drop-shadow(0 0 4px ${ag.color}88)` : 'none',
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: active ? ag.color : '#cbd5e1',
        width: 16, height: 16,
      },
    };
  });

  return [...supEdges, ...synthEdges];
};

/* ── Chat message ── */
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

/* ── Scan log ticker (bottom of canvas) ── */
const ScanLog = ({ phase, agentStatuses, scanCounts }) => {
  const active = AGENTS.filter(ag => agentStatuses[ag.id] === 'active');
  if (!active.length && phase !== 'synthesizing') return null;

  return (
    <div style={{
      position: 'absolute', bottom: 8, left: 12, right: 12,
      background: 'rgba(15,23,42,0.85)', borderRadius: 8,
      padding: '8px 12px', backdropFilter: 'blur(4px)',
      border: '1px solid rgba(255,255,255,0.08)',
      zIndex: 10,
    }}>
      {phase === 'synthesizing' ? (
        <div style={{ fontSize: 10, color: '#fbbf24', display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ animation: 'pulse-text 1s infinite' }}>▶</span>
          Synthesize Agent: 5개 분석 결과 통합 처리 중...
        </div>
      ) : (
        active.map(ag => {
          const cnt   = scanCounts[ag.id] || 0;
          const end   = Math.min(cnt, TOTAL_EMP);
          const start = Math.max(1, end - BATCH + 1);
          return (
            <div key={ag.id} style={{
              fontSize: 10, color: ag.color,
              display: 'flex', alignItems: 'center', gap: 6,
              marginBottom: 2,
            }}>
              <span style={{ fontWeight: 700, minWidth: 60 }}>{ag.label}</span>
              <span style={{ color: '#94a3b8' }}>
                직원 #{String(start).padStart(3,'0')} ~ #{String(end).padStart(3,'0')} 분석 중
              </span>
              <span style={{ color: '#64748b', marginLeft: 'auto' }}>
                {end}/{TOTAL_EMP}
              </span>
            </div>
          );
        })
      )}
    </div>
  );
};

/* ── Main component ── */
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
  const [scanCounts, setScanCounts] = useState(
    Object.fromEntries(AGENTS.map(a => [a.id, 0]))
  );

  const chatEndRef  = useRef(null);
  const timerIds    = useRef([]);
  const scanTimers  = useRef({});

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const clearTimers = () => {
    timerIds.current.forEach(clearTimeout);
    timerIds.current = [];
    Object.values(scanTimers.current).forEach(clearInterval);
    scanTimers.current = {};
  };

  const schedule = (fn, delay) => {
    const id = setTimeout(fn, delay);
    timerIds.current.push(id);
  };

  /* start scan counter for one agent */
  const startScan = useCallback((agId) => {
    if (scanTimers.current[agId]) clearInterval(scanTimers.current[agId]);
    setScanCounts(prev => ({ ...prev, [agId]: 0 }));

    /* tick every 80ms, increment by BATCH per tick → ~2.35s to reach 1470 */
    const ticker = setInterval(() => {
      setScanCounts(prev => {
        const next = prev[agId] + BATCH;
        if (next >= TOTAL_EMP) {
          clearInterval(scanTimers.current[agId]);
          delete scanTimers.current[agId];
          return { ...prev, [agId]: TOTAL_EMP };
        }
        return { ...prev, [agId]: next };
      });
    }, 80);
    scanTimers.current[agId] = ticker;
  }, []);

  const runWorkflow = useCallback(() => {
    clearTimers();
    setPhase('supervisor');
    setAgents(Object.fromEntries(AGENTS.map(a => [a.id, 'idle'])));
    setScanCounts(Object.fromEntries(AGENTS.map(a => [a.id, 0])));

    schedule(() => setPhase('dispatching'), 700);

    /* activate agents & start scans staggered */
    AGENTS.forEach((ag, i) => {
      schedule(() => {
        setAgents(prev => ({ ...prev, [ag.id]: 'active' }));
        startScan(ag.id);
      }, 1300 + i * 300);
    });

    /* mark done staggered (after scan finishes ~2.4s) */
    AGENTS.forEach((ag, i) => {
      schedule(() => {
        setAgents(prev => ({ ...prev, [ag.id]: 'done' }));
        setScanCounts(prev => ({ ...prev, [ag.id]: TOTAL_EMP }));
      }, 1300 + i * 300 + 2500);
    });

    /* synthesize (all agents done by ~1300+4*300+2500 = 5000ms) */
    schedule(() => setPhase('synthesizing'), 5200);
    schedule(() => setPhase('done'), 6800);
  }, [startScan]);

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

  /* sync ReactFlow */
  const rfNodes = useMemo(() => buildNodes(phase, agentStatuses, scanCounts), [phase, agentStatuses, scanCounts]);
  const rfEdges = useMemo(() => buildEdges(phase, agentStatuses),             [phase, agentStatuses]);
  const [nodes, setNodes, onNodesChange] = useNodesState(rfNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(rfEdges);

  useEffect(() => setNodes(rfNodes), [rfNodes, setNodes]);
  useEffect(() => setEdges(rfEdges), [rfEdges, setEdges]);

  const phaseLabel = {
    supervisor:  '🧠 요청 수신',
    dispatching: '📡 에이전트 배분',
    analyzing:   '🔍 병렬 분석 중',
    synthesizing:'⚡ 결과 통합 중',
    done:        '✅ 분석 완료',
  }[phase] || '';

  const QUICK_PROMPTS = [
    '고위험 직원 현황을 알려줘',
    'R&D 부서 퇴사 위험 분석해줘',
    'P01 번아웃 직원 개입 전략은?',
    '최근 이상 행동 패턴 보고해줘',
  ];

  const anyActive = Object.values(agentStatuses).some(s => s === 'active');

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 64px)', overflow: 'hidden' }}>
      <style>{`
        @keyframes cursor-blink  { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes pulse-text    { 0%,100%{opacity:1} 50%{opacity:0.4} }

        /* Make ReactFlow animated edges glow visibly */
        .react-flow__edge.animated .react-flow__edge-path {
          stroke-dasharray: 10 6 !important;
          animation: rf-dash 0.45s linear infinite !important;
        }
        @keyframes rf-dash { to { stroke-dashoffset: -16; } }

        .react-flow__attribution { display: none !important; }
        .react-flow__handle       { display: none !important; }
      `}</style>

      {/* ── LEFT: workflow canvas ── */}
      <div style={{
        flex: '0 0 56%', display: 'flex', flexDirection: 'column',
        borderRight: '1px solid var(--border,#e2e8f0)',
        background: 'var(--bg,#f8fafc)',
      }}>
        {/* header */}
        <div style={{
          padding: '13px 20px', flexShrink: 0,
          background: 'linear-gradient(135deg,#0f172a 0%,#1a1a2e 100%)',
          color: '#fff',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14 }}>Agentic AI Workflow</div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
              전체 1,470명 · 5개 Worker Agent 병렬 분석
            </div>
          </div>
          {phase !== 'idle' && (
            <div style={{
              fontSize: 11, color: '#60a5fa',
              background: 'rgba(96,165,250,0.15)',
              border: '1px solid rgba(96,165,250,0.3)',
              padding: '4px 12px', borderRadius: 20,
              animation: phase !== 'done' ? 'pulse-text 1.5s ease-in-out infinite' : 'none',
            }}>
              {phaseLabel}
            </div>
          )}
        </div>

        {/* ReactFlow canvas */}
        <div style={{ flex: 1, position: 'relative' }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={NODE_TYPES}
            fitView
            fitViewOptions={{ padding: 0.22 }}
            panOnDrag={false}
            zoomOnScroll={false}
            zoomOnPinch={false}
            zoomOnDoubleClick={false}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={false}
          >
            <Background color="#dde1e9" gap={24} size={1} />
          </ReactFlow>

          {/* scan log overlay */}
          {(anyActive || phase === 'synthesizing') && (
            <ScanLog phase={phase} agentStatuses={agentStatuses} scanCounts={scanCounts} />
          )}
        </div>

        {/* agent status chips */}
        <div style={{
          padding: '9px 14px', flexShrink: 0,
          borderTop: '1px solid var(--border,#e2e8f0)',
          background: 'var(--card,#fff)',
          display: 'flex', gap: 6, flexWrap: 'wrap',
        }}>
          {AGENTS.map(ag => {
            const st  = agentStatuses[ag.id];
            const cnt = scanCounts[ag.id];
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
                  animation: st === 'active' ? 'pulse-text 1s infinite' : 'none',
                  display: 'inline-block',
                }} />
                {ag.label}
                {st === 'active' && cnt > 0 && (
                  <span style={{ fontSize: 10, fontWeight: 400 }}>
                    {cnt}/{TOTAL_EMP}
                  </span>
                )}
                {st === 'done' && ' ✓'}
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
        <div style={{
          padding: '13px 20px', flexShrink: 0,
          background: 'var(--card,#fff)',
          borderBottom: '1px solid var(--border,#e2e8f0)',
        }}>
          <div style={{ fontWeight: 700, fontSize: 14, color: 'var(--text,#1e293b)' }}>HR AI 어시스턴트</div>
          <div style={{ fontSize: 11, color: 'var(--sub,#94a3b8)', marginTop: 2 }}>
            퇴사위험 분석 · 개입 전략 · 인사이트
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px' }}>
          {messages.map((msg, i) => <ChatMsg key={i} msg={msg} />)}
          {loading && (
            <div style={{ display: 'flex', gap: 4, paddingLeft: 36, paddingBottom: 8 }}>
              {[0,1,2].map(i => (
                <div key={i} style={{
                  width: 7, height: 7, borderRadius: '50%', background: '#94a3b8',
                  animation: `cursor-blink 1.2s ${i*0.2}s steps(1) infinite`,
                }} />
              ))}
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {messages.length <= 1 && (
          <div style={{ padding: '0 20px 12px', display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {QUICK_PROMPTS.map(q => (
              <button key={q} onClick={() => setInput(q)} style={{
                background: 'var(--card,#fff)',
                border: '1px solid var(--border,#e2e8f0)',
                borderRadius: 20, padding: '5px 12px', fontSize: 11,
                color: 'var(--sub,#64748b)', cursor: 'pointer', transition: 'all 0.2s',
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = '#2563eb'; e.currentTarget.style.color = '#2563eb'; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border,#e2e8f0)'; e.currentTarget.style.color = 'var(--sub,#64748b)'; }}
              >{q}</button>
            ))}
          </div>
        )}

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
              transition: 'background 0.2s', flexShrink: 0, whiteSpace: 'nowrap',
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
