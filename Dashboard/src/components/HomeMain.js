import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import ReactFlow, {
  Background, useNodesState, useEdgesState, MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';

const SUPERVISOR_URL = process.env.REACT_APP_SUPERVISOR_URL || 'http://localhost:5000';
const TOTAL_EMP  = 1470;
const SCAN_TICK  = 50;   // ms per tick
const SCAN_BATCH = 37;   // employees per tick → ~40 ticks = 2 s

const C = {
  red:    '#d93954',
  blue:   '#2563eb',
  orange: '#e8721a',
  purple: '#7c3aed',
  green:  '#2ea44f',
  amber:  '#f59e0b',
};

const AGENTS = [
  { id: 'structura', label: 'Structura', sub: '정형 데이터 분석',  color: C.red    },
  { id: 'cognita',   label: 'Cognita',   sub: '관계망 분석',       color: C.blue   },
  { id: 'chronos',   label: 'Chronos',   sub: '시계열 행동 분석',  color: C.orange },
  { id: 'sentio',    label: 'Sentio',    sub: '자연어 감성 분석',  color: C.purple },
  { id: 'agora',     label: 'Agora',     sub: '외부 시장 분석',    color: C.green  },
];

/* ─────────────────────────────────────────
   Dark-mode detection (CSS-var compatible)
───────────────────────────────────────── */
function useDarkMode() {
  const check = () =>
    document.documentElement.classList.contains('dark') ||
    document.body.classList.contains('dark') ||
    document.documentElement.getAttribute('data-theme') === 'dark' ||
    window.matchMedia('(prefers-color-scheme: dark)').matches;

  const [isDark, setIsDark] = useState(check);

  useEffect(() => {
    const update = () => setIsDark(check());
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener('change', update);
    const ob = new MutationObserver(update);
    ob.observe(document.documentElement, { attributes: true });
    ob.observe(document.body, { attributes: true });
    return () => { mq.removeEventListener('change', update); ob.disconnect(); };
  }, []);

  return isDark;
}

/* ─────────────────────────────────────────
   Custom node: Supervisor
───────────────────────────────────────── */
const SupervisorNode = ({ data }) => {
  const { phase } = data;
  const active = phase === 'supervisor' || phase === 'dispatching';
  const done   = ['synthesizing', 'reporting', 'done'].includes(phase);

  return (
    <div style={{
      minWidth: 190, textAlign: 'center',
      borderRadius: 14, padding: '14px 24px',
      background: active
        ? 'linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%)'
        : done
          ? 'linear-gradient(135deg,#0f172a 0%,#163460 100%)'
          : 'var(--card,#fff)',
      border: `2px solid ${active ? '#60a5fa' : done ? '#3b82f688' : 'var(--border,#e2e8f0)'}`,
      color: active || done ? '#fff' : 'var(--sub,#94a3b8)',
      boxShadow: active
        ? '0 0 30px rgba(96,165,250,0.55), 0 4px 16px rgba(0,0,0,.25)'
        : done
          ? '0 0 16px rgba(96,165,250,0.2)'
          : '0 1px 4px rgba(0,0,0,.08)',
      transition: 'all 0.5s ease',
    }}>
      <div style={{ fontSize: 24, marginBottom: 6, filter: active || done ? 'none' : 'grayscale(0.6) opacity(0.6)' }}>🧠</div>
      <div style={{ fontWeight: 700, fontSize: 13 }}>Supervisor Agent</div>
      <div style={{ fontSize: 10, opacity: 0.65, marginTop: 3 }}>오케스트레이션 · 태스크 배분</div>

      {phase === 'idle' && (
        <div style={{ fontSize: 10, marginTop: 8, color: 'var(--sub,#cbd5e1)', letterSpacing: 1 }}>
          ○ 대기 중
        </div>
      )}
      {active && (
        <div style={{ fontSize: 10, color: '#93c5fd', marginTop: 8, animation: 'wf-blink 1.1s ease-in-out infinite' }}>
          ▶ 1,470명 분석 태스크 배분 중...
        </div>
      )}
      {done && !active && (
        <div style={{ fontSize: 10, color: '#86efac', marginTop: 8 }}>✓ 배분 완료</div>
      )}
    </div>
  );
};

/* ─────────────────────────────────────────
   Custom node: Worker Agent
───────────────────────────────────────── */
const AgentNode = ({ data }) => {
  const { label, sub, color, status, scanCount } = data;
  const isActive = status === 'active';
  const isDone   = status === 'done';
  const isIdle   = status === 'idle';

  const end   = Math.min(scanCount, TOTAL_EMP);
  const start = Math.max(1, end - SCAN_BATCH + 1);
  const pct   = Math.round((end / TOTAL_EMP) * 100);

  return (
    <div style={{
      minWidth: 135, textAlign: 'center',
      borderRadius: 12, padding: '10px 14px',
      background: isActive
        ? `${color}1a`
        : isDone
          ? `${color}0d`
          : 'var(--card,#fff)',
      border: `2px solid ${isActive ? color : isDone ? color + '77' : 'var(--border,#e2e8f0)'}`,
      color: isActive ? color : isDone ? color + 'cc' : 'var(--sub,#94a3b8)',
      boxShadow: isActive
        ? `0 0 20px ${color}55, 0 2px 8px rgba(0,0,0,.1)`
        : isDone
          ? `0 0 8px ${color}22`
          : '0 1px 3px rgba(0,0,0,.06)',
      transition: 'all 0.4s ease',
    }}>
      {/* header */}
      <div style={{ fontWeight: 700, fontSize: 12 }}>{label}</div>
      <div style={{ fontSize: 9, opacity: 0.75, marginTop: 2 }}>{sub}</div>

      {/* idle */}
      {isIdle && (
        <div style={{ fontSize: 9, marginTop: 6, color: 'var(--sub,#cbd5e1)' }}>○ 대기</div>
      )}

      {/* active: scan counter */}
      {isActive && (
        <div style={{ marginTop: 7 }}>
          <div style={{
            fontSize: 9, fontWeight: 600, color,
            background: `${color}15`, border: `1px solid ${color}44`,
            borderRadius: 5, padding: '3px 6px', marginBottom: 4,
          }}>
            #{String(start).padStart(4,'0')} ~ #{String(end).padStart(4,'0')}
          </div>
          {/* progress bar */}
          <div style={{ height: 3, borderRadius: 2, background: `${color}25`, overflow: 'hidden' }}>
            <div style={{
              height: '100%', borderRadius: 2,
              background: color,
              width: `${pct}%`,
              transition: 'width 0.05s linear',
            }} />
          </div>
          <div style={{ fontSize: 8, color, marginTop: 3 }}>{pct}% · 분석 중</div>
        </div>
      )}

      {/* done */}
      {isDone && (
        <div style={{
          marginTop: 6, fontSize: 9, fontWeight: 600,
          background: `${color}12`, borderRadius: 5, padding: '3px 6px',
        }}>
          ✓ 1,470명 완료
        </div>
      )}
    </div>
  );
};

/* ─────────────────────────────────────────
   Custom node: Synthesize
───────────────────────────────────────── */
const SynthesizeNode = ({ data }) => {
  const { active, done } = data;
  const color = C.amber;

  return (
    <div style={{
      minWidth: 190, textAlign: 'center',
      borderRadius: 14, padding: '14px 24px',
      background: active
        ? '#fffbeb'
        : done ? '#fefce8' : 'var(--card,#fff)',
      border: `2px solid ${active ? color : done ? color + '88' : 'var(--border,#e2e8f0)'}`,
      color: active || done ? '#92400e' : 'var(--sub,#94a3b8)',
      boxShadow: active
        ? `0 0 28px ${color}55, 0 4px 16px rgba(0,0,0,.12)`
        : done ? `0 0 10px ${color}22` : '0 1px 4px rgba(0,0,0,.06)',
      transition: 'all 0.5s ease',
    }}>
      <div style={{ fontSize: 24, marginBottom: 6, filter: active || done ? 'none' : 'grayscale(0.6) opacity(0.6)' }}>⚡</div>
      <div style={{ fontWeight: 700, fontSize: 13 }}>Synthesize Agent</div>
      <div style={{ fontSize: 10, opacity: 0.65, marginTop: 3 }}>통합 분석 · 최종 위험도 판정</div>

      {!active && !done && (
        <div style={{ fontSize: 10, marginTop: 8, color: 'var(--sub,#cbd5e1)' }}>○ 대기 중</div>
      )}
      {active && (
        <div style={{ fontSize: 10, color, marginTop: 8, animation: 'wf-blink 1.1s ease-in-out infinite' }}>
          ▶ 5개 Agent 결과 통합 분석 중...
        </div>
      )}
      {done && !active && (
        <div style={{ fontSize: 10, color: '#d97706', marginTop: 8 }}>✓ 통합 완료</div>
      )}
    </div>
  );
};

/* ─────────────────────────────────────────
   Custom node: Report (new!)
───────────────────────────────────────── */
const ReportNode = ({ data }) => {
  const { phase } = data;
  const generating = phase === 'reporting';
  const done       = phase === 'done';
  const active     = generating || done;
  const color      = '#16a34a';

  return (
    <div style={{
      minWidth: 210, textAlign: 'center',
      borderRadius: 14, padding: '14px 24px',
      background: generating
        ? '#f0fdf4'
        : done ? '#f0fdf4' : 'var(--card,#fff)',
      border: `2px solid ${active ? color + (done ? 'bb' : 'ff') : 'var(--border,#e2e8f0)'}`,
      color: active ? '#14532d' : 'var(--sub,#94a3b8)',
      boxShadow: generating
        ? `0 0 28px ${color}44, 0 4px 16px rgba(0,0,0,.1)`
        : done ? `0 0 12px ${color}22` : '0 1px 4px rgba(0,0,0,.06)',
      transition: 'all 0.5s ease',
    }}>
      <div style={{ fontSize: 24, marginBottom: 6, filter: active ? 'none' : 'grayscale(0.6) opacity(0.6)' }}>📋</div>
      <div style={{ fontWeight: 700, fontSize: 13 }}>리포트 생성</div>
      <div style={{ fontSize: 10, opacity: 0.65, marginTop: 3 }}>퇴사위험 분석 보고서</div>

      {!active && (
        <div style={{ fontSize: 10, marginTop: 8, color: 'var(--sub,#cbd5e1)' }}>○ 대기 중</div>
      )}
      {generating && (
        <div style={{ fontSize: 10, color, marginTop: 8, animation: 'wf-blink 1.1s ease-in-out infinite' }}>
          ▶ 보고서 생성 중...
        </div>
      )}
      {done && (
        <div style={{ marginTop: 8 }}>
          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, marginBottom: 4,
          }}>
            {[
              { label: '고위험군', value: '486명', sub: '33.1%' },
              { label: '즉시 개입', value: '101명', sub: '번아웃' },
            ].map(s => (
              <div key={s.label} style={{
                background: color + '18', borderRadius: 6, padding: '4px 6px',
                border: `1px solid ${color}33`,
              }}>
                <div style={{ fontSize: 8, color, opacity: 0.8 }}>{s.label}</div>
                <div style={{ fontSize: 11, fontWeight: 700, color }}>{s.value}</div>
                <div style={{ fontSize: 8, color, opacity: 0.6 }}>{s.sub}</div>
              </div>
            ))}
          </div>
          <div style={{ fontSize: 9, color: '#16a34a', fontWeight: 600 }}>✓ 분석 보고서 완료</div>
        </div>
      )}
    </div>
  );
};

/* ─────────────────────────────────────────
   ReactFlow helpers
───────────────────────────────────────── */
const NODE_TYPES = {
  supervisor: SupervisorNode,
  agent:      AgentNode,
  synthesize: SynthesizeNode,
  report:     ReportNode,
};

const buildNodes = (phase, agentStatuses, scanCounts) => {
  const isSynth    = ['synthesizing', 'reporting', 'done'].includes(phase);
  const synthDone  = ['reporting', 'done'].includes(phase);
  // 5 agents: x = 0,160,320,480,640 (160px spacing, ~135px wide)
  // Center of span: (640+135)/2 = 387.5 → supervisor/synthesize/report at x=292 (390-98)
  const CX = 297;
  return [
    { id: 'supervisor', type: 'supervisor', position: { x: CX, y: 20 },  data: { phase },        draggable: false },
    ...AGENTS.map((ag, i) => ({
      id: ag.id, type: 'agent',
      position: { x: i * 162, y: 185 },
      data: { label: ag.label, sub: ag.sub, color: ag.color, status: agentStatuses[ag.id] || 'idle', scanCount: scanCounts[ag.id] || 0 },
      draggable: false,
    })),
    { id: 'synthesize', type: 'synthesize', position: { x: CX, y: 360 }, data: { active: isSynth && !synthDone, done: synthDone }, draggable: false },
    { id: 'report',     type: 'report',     position: { x: CX, y: 535 }, data: { phase },        draggable: false },
  ];
};

/* Active edge style — uses inline strokeDasharray+animation so it works
   regardless of ReactFlow CSS load order. animated:false to avoid conflicts. */
const activeEdgeStyle = (color) => ({
  stroke: color,
  strokeWidth: 3.5,
  strokeDasharray: '10 5',
  animation: 'wf-dash 0.4s linear infinite',
  filter: `drop-shadow(0 0 6px ${color}bb)`,
});
const idleEdgeStyle = () => ({
  stroke: '#cbd5e1',
  strokeWidth: 1.5,
});

const buildEdges = (phase, agentStatuses) => {
  const dispatched  = ['dispatching','synthesizing','reporting','done'].includes(phase);
  const synthActive = ['synthesizing','reporting','done'].includes(phase);
  const repActive   = ['reporting','done'].includes(phase);

  return [
    /* Supervisor → each Agent */
    ...AGENTS.map(ag => ({
      id: `sup-${ag.id}`, source: 'supervisor', target: ag.id,
      type: 'smoothstep', animated: false,
      style: dispatched ? activeEdgeStyle(ag.color) : idleEdgeStyle(),
      markerEnd: { type: MarkerType.ArrowClosed, color: dispatched ? ag.color : '#cbd5e1', width: 14, height: 14 },
    })),
    /* each Agent → Synthesize */
    ...AGENTS.map(ag => {
      const flow = synthActive && agentStatuses[ag.id] === 'done';
      return {
        id: `${ag.id}-syn`, source: ag.id, target: 'synthesize',
        type: 'smoothstep', animated: false,
        style: flow ? activeEdgeStyle(ag.color) : idleEdgeStyle(),
        markerEnd: { type: MarkerType.ArrowClosed, color: flow ? ag.color : '#cbd5e1', width: 14, height: 14 },
      };
    }),
    /* Synthesize → Report */
    {
      id: 'syn-rep', source: 'synthesize', target: 'report',
      type: 'smoothstep', animated: false,
      style: repActive ? activeEdgeStyle('#16a34a') : idleEdgeStyle(),
      markerEnd: { type: MarkerType.ArrowClosed, color: repActive ? '#16a34a' : '#cbd5e1', width: 14, height: 14 },
    },
  ];
};

/* ─────────────────────────────────────────
   Scan log overlay inside canvas
───────────────────────────────────────── */
const ScanLog = ({ agentStatuses, scanCounts, phase }) => {
  const activeAgents = AGENTS.filter(ag => agentStatuses[ag.id] === 'active');
  if (!activeAgents.length && phase !== 'synthesizing' && phase !== 'reporting') return null;

  return (
    <div style={{
      position: 'absolute', bottom: 10, left: 12, right: 12, zIndex: 10,
      background: 'rgba(15,23,42,0.82)', borderRadius: 8, padding: '8px 12px',
      border: '1px solid rgba(255,255,255,0.08)', backdropFilter: 'blur(6px)',
    }}>
      {phase === 'reporting' ? (
        <div style={{ fontSize: 10, color: '#86efac', display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ animation: 'wf-blink 1s infinite' }}>▶</span> 보고서 생성 중... (퇴사위험 분석 리포트)
        </div>
      ) : phase === 'synthesizing' && !activeAgents.length ? (
        <div style={{ fontSize: 10, color: '#fbbf24', display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ animation: 'wf-blink 1s infinite' }}>▶</span> Synthesize Agent: 5개 분석 결과 통합 중...
        </div>
      ) : (
        activeAgents.map(ag => {
          const cnt   = scanCounts[ag.id] || 0;
          const end   = Math.min(cnt, TOTAL_EMP);
          const start = Math.max(1, end - SCAN_BATCH + 1);
          const pct   = Math.round((end / TOTAL_EMP) * 100);
          return (
            <div key={ag.id} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
              <span style={{ fontSize: 10, fontWeight: 700, color: ag.color, minWidth: 65 }}>{ag.label}</span>
              <div style={{ flex: 1, height: 3, background: `${ag.color}25`, borderRadius: 2 }}>
                <div style={{ height: '100%', width: `${pct}%`, background: ag.color, borderRadius: 2, transition: 'width 0.05s' }} />
              </div>
              <span style={{ fontSize: 9, color: '#94a3b8', minWidth: 90, textAlign: 'right' }}>
                #{String(start).padStart(4,'0')}~#{String(end).padStart(4,'0')} ({pct}%)
              </span>
            </div>
          );
        })
      )}
    </div>
  );
};

/* ─────────────────────────────────────────
   Chat message bubble
───────────────────────────────────────── */
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
            display: 'inline-block', width: 6, height: 14, background: '#2563eb',
            marginLeft: 3, animation: 'wf-blink 1s steps(1) infinite',
          }} />
        )}
      </div>
    </div>
  );
};

/* ─────────────────────────────────────────
   Phase label map
───────────────────────────────────────── */
const PHASE_LABEL = {
  idle:        '',
  supervisor:  '🧠 요청 수신',
  dispatching: '📡 에이전트 배분 중',
  synthesizing:'⚡ 결과 통합 중',
  reporting:   '📋 보고서 생성 중',
  done:        '✅ 분석 완료',
};

/* ─────────────────────────────────────────
   Main component
───────────────────────────────────────── */
const HomeMain = () => {
  const isDark = useDarkMode();

  const [messages, setMessages]     = useState([{
    role: 'assistant',
    content: '안녕하세요! HR Agentic AI 시스템입니다.\n분석하고 싶은 직원 정보나 퇴사 위험 현황에 대해 질문해 주세요.',
  }]);
  const [input, setInput]           = useState('');
  const [loading, setLoading]       = useState(false);
  const [phase, setPhase]           = useState('idle');
  const [agentStatuses, setAgents]  = useState(
    Object.fromEntries(AGENTS.map(a => [a.id, 'idle']))
  );
  const [scanCounts, setScanCounts] = useState(
    Object.fromEntries(AGENTS.map(a => [a.id, 0]))
  );

  const chatEndRef = useRef(null);
  const timers     = useRef([]);
  const scanInts   = useRef({});

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const clearAll = () => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
    Object.values(scanInts.current).forEach(clearInterval);
    scanInts.current = {};
  };

  const after = (fn, ms) => {
    const id = setTimeout(fn, ms);
    timers.current.push(id);
  };

  const startScan = useCallback((agId) => {
    if (scanInts.current[agId]) clearInterval(scanInts.current[agId]);
    setScanCounts(prev => ({ ...prev, [agId]: 0 }));
    scanInts.current[agId] = setInterval(() => {
      setScanCounts(prev => {
        const next = prev[agId] + SCAN_BATCH;
        if (next >= TOTAL_EMP) {
          clearInterval(scanInts.current[agId]);
          delete scanInts.current[agId];
          return { ...prev, [agId]: TOTAL_EMP };
        }
        return { ...prev, [agId]: next };
      });
    }, SCAN_TICK);
  }, []);

  /* ── 10-second workflow ──────────────────────────
     0.0s  supervisor
     0.5s  dispatching
     1.0s  agent[0] active
     1.2s  agent[1] active
     1.4s  agent[2] active
     1.6s  agent[3] active
     1.8s  agent[4] active
     3.0s  agent[0] done   (2 s scan each)
     3.2s  agent[1] done
     3.4s  agent[2] done
     3.6s  agent[3] done
     3.8s  agent[4] done
     4.0s  synthesizing
     6.0s  reporting
     7.5s  done
    10.0s  idle (auto-reset)
  ─────────────────────────────────────────────── */
  const runWorkflow = useCallback(() => {
    clearAll();
    setPhase('supervisor');
    setAgents(Object.fromEntries(AGENTS.map(a => [a.id, 'idle'])));
    setScanCounts(Object.fromEntries(AGENTS.map(a => [a.id, 0])));

    after(() => setPhase('dispatching'), 500);

    AGENTS.forEach((ag, i) => {
      after(() => {
        setAgents(prev => ({ ...prev, [ag.id]: 'active' }));
        startScan(ag.id);
      }, 1000 + i * 200);

      after(() => {
        setAgents(prev => ({ ...prev, [ag.id]: 'done' }));
        setScanCounts(prev => ({ ...prev, [ag.id]: TOTAL_EMP }));
      }, 3000 + i * 200);
    });

    after(() => setPhase('synthesizing'), 4000);
    after(() => setPhase('reporting'),    6000);
    after(() => setPhase('done'),         7500);
    after(() => {
      setPhase('idle');
      setAgents(Object.fromEntries(AGENTS.map(a => [a.id, 'idle'])));
      setScanCounts(Object.fromEntries(AGENTS.map(a => [a.id, 0])));
    }, 10000);
  }, [startScan]);

  const typeResponse = useCallback((text) => {
    const id = Date.now();
    let i = 0;
    setMessages(prev => [...prev, { role: 'assistant', content: '', typing: true, id }]);
    const t = setInterval(() => {
      i++;
      setMessages(prev => prev.map(m => m.id === id ? { ...m, content: text.slice(0, i), typing: i < text.length } : m));
      if (i >= text.length) clearInterval(t);
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
        method: 'POST', headers: { 'Content-Type': 'application/json' },
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

  const anyActive   = Object.values(agentStatuses).some(s => s === 'active');
  const showScanLog = anyActive || phase === 'synthesizing' || phase === 'reporting';
  const phaseLabel  = PHASE_LABEL[phase] || '';

  const QUICK = [
    '고위험 직원 현황을 알려줘',
    'R&D 부서 퇴사 위험 분석해줘',
    'P01 번아웃 직원 개입 전략은?',
    '전체 직원 퇴사 위험도 요약해줘',
  ];

  /* canvas bg adapts to dark mode */
  const canvasBg = isDark ? '#0d1117' : '#f1f5f9';
  const bgDotColor = isDark ? '#1e293b' : '#dde1e9';

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 64px)', overflow: 'hidden' }}>
      <style>{`
        @keyframes wf-blink { 0%,100%{opacity:1} 50%{opacity:0.25} }
        @keyframes wf-dash  { to { stroke-dashoffset: -15; } }

        .react-flow__attribution { display: none !important; }
        /* Hide handle dots visually but keep in DOM for edge routing */
        .react-flow__handle {
          opacity: 0 !important;
          pointer-events: none !important;
          width: 4px !important; height: 4px !important;
        }
        .react-flow__node { cursor: default !important; }
      `}</style>

      {/* ══ LEFT: Workflow canvas ══════════════════════════════════ */}
      <div style={{
        flex: '0 0 56%', display: 'flex', flexDirection: 'column',
        borderRight: '1px solid var(--border,#e2e8f0)',
      }}>
        {/* header */}
        <div style={{
          padding: '13px 20px', flexShrink: 0,
          background: 'linear-gradient(135deg,#0f172a 0%,#1a1a2e 100%)',
          color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14 }}>Agentic AI Workflow</div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
              전체 1,470명 · 5개 Worker Agent 병렬 분석 파이프라인
            </div>
          </div>
          {phase !== 'idle' ? (
            <div style={{
              fontSize: 11, color: '#60a5fa',
              background: 'rgba(96,165,250,0.15)',
              border: '1px solid rgba(96,165,250,0.3)',
              padding: '4px 14px', borderRadius: 20,
              animation: phase !== 'done' ? 'wf-blink 1.5s ease-in-out infinite' : 'none',
            }}>
              {phaseLabel}
            </div>
          ) : (
            <div style={{
              fontSize: 11, color: '#475569',
              background: 'rgba(71,85,105,0.15)',
              border: '1px solid rgba(71,85,105,0.3)',
              padding: '4px 14px', borderRadius: 20,
            }}>
              ○ 대기 중
            </div>
          )}
        </div>

        {/* canvas */}
        <div style={{ flex: 1, position: 'relative', background: canvasBg }}>
          <ReactFlow
            nodes={nodes} edges={edges}
            onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
            nodeTypes={NODE_TYPES}
            fitView fitViewOptions={{ padding: 0.18 }}
            panOnDrag={false} zoomOnScroll={false}
            zoomOnPinch={false} zoomOnDoubleClick={false}
            nodesDraggable={false} nodesConnectable={false} elementsSelectable={false}
          >
            <Background color={bgDotColor} gap={24} size={1.5} />
          </ReactFlow>
          {showScanLog && (
            <ScanLog agentStatuses={agentStatuses} scanCounts={scanCounts} phase={phase} />
          )}
        </div>

        {/* agent status chips */}
        <div style={{
          padding: '8px 14px', flexShrink: 0,
          borderTop: '1px solid var(--border,#e2e8f0)',
          background: 'var(--card,#fff)',
          display: 'flex', gap: 5, flexWrap: 'wrap', alignItems: 'center',
        }}>
          {AGENTS.map(ag => {
            const st  = agentStatuses[ag.id];
            const cnt = scanCounts[ag.id];
            const pct = cnt ? Math.round((cnt / TOTAL_EMP) * 100) : 0;
            return (
              <span key={ag.id} style={{
                padding: '3px 10px', borderRadius: 20, fontSize: 10, fontWeight: 600,
                display: 'inline-flex', alignItems: 'center', gap: 4,
                background: st !== 'idle' ? `${ag.color}15` : 'var(--bg,#f8fafc)',
                border: `1px solid ${st !== 'idle' ? ag.color + '55' : 'var(--border,#e2e8f0)'}`,
                color: st !== 'idle' ? ag.color : 'var(--sub,#94a3b8)',
                transition: 'all 0.35s',
              }}>
                <span style={{
                  width: 5, height: 5, borderRadius: '50%', flexShrink: 0,
                  background: st !== 'idle' ? ag.color : 'var(--border,#e2e8f0)',
                  animation: st === 'active' ? 'wf-blink 0.8s infinite' : 'none',
                  display: 'inline-block',
                }} />
                {ag.label}
                {st === 'active' && <span style={{ fontWeight: 400 }}> {pct}%</span>}
                {st === 'done'   && ' ✓'}
              </span>
            );
          })}
        </div>
      </div>

      {/* ══ RIGHT: Chat panel ════════════════════════════════════ */}
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column',
        background: 'var(--bg,#f8fafc)', minWidth: 0,
      }}>
        {/* chat header */}
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

        {/* messages */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px' }}>
          {messages.map((msg, i) => <ChatMsg key={i} msg={msg} />)}
          {loading && (
            <div style={{ display: 'flex', gap: 4, paddingLeft: 36, paddingBottom: 8 }}>
              {[0,1,2].map(i => (
                <div key={i} style={{
                  width: 7, height: 7, borderRadius: '50%', background: '#94a3b8',
                  animation: `wf-blink 1.2s ${i*0.2}s steps(1) infinite`,
                }} />
              ))}
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* quick prompts */}
        {messages.length <= 1 && (
          <div style={{ padding: '0 20px 10px', display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {QUICK.map(q => (
              <button key={q} onClick={() => setInput(q)} style={{
                background: 'var(--card,#fff)', border: '1px solid var(--border,#e2e8f0)',
                borderRadius: 20, padding: '5px 12px', fontSize: 11,
                color: 'var(--sub,#64748b)', cursor: 'pointer', transition: 'all 0.2s',
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = C.blue; e.currentTarget.style.color = C.blue; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border,#e2e8f0)'; e.currentTarget.style.color = 'var(--sub,#64748b)'; }}
              >{q}</button>
            ))}
          </div>
        )}

        {/* input bar */}
        <div style={{
          padding: '12px 20px', flexShrink: 0,
          background: 'var(--card,#fff)', borderTop: '1px solid var(--border,#e2e8f0)',
          display: 'flex', gap: 10, alignItems: 'flex-end',
        }}>
          <textarea value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKey}
            placeholder="분석하고 싶은 내용을 입력하세요... (Enter로 전송)" rows={1}
            style={{
              flex: 1, resize: 'none', border: '1px solid var(--border,#e2e8f0)',
              borderRadius: 12, padding: '10px 14px', fontSize: 13, lineHeight: 1.5,
              background: 'var(--bg,#f8fafc)', color: 'var(--text,#1e293b)',
              outline: 'none', fontFamily: 'inherit', maxHeight: 120, overflowY: 'auto',
              transition: 'border-color 0.2s',
            }}
            onFocus={e => { e.target.style.borderColor = C.blue; }}
            onBlur={e => { e.target.style.borderColor = 'var(--border,#e2e8f0)'; }}
          />
          <button onClick={handleSend} disabled={loading || !input.trim()} style={{
            background: loading || !input.trim() ? '#94a3b8' : C.blue,
            color: '#fff', border: 'none', borderRadius: 12,
            padding: '10px 20px', fontSize: 13, fontWeight: 600,
            cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
            transition: 'background 0.2s', flexShrink: 0, whiteSpace: 'nowrap',
          }}>
            {loading ? '분석 중...' : '전송'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default HomeMain;
