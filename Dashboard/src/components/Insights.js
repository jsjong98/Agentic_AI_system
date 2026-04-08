import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts';

const INTEGRATION_URL = process.env.REACT_APP_INTEGRATION_URL || 'http://localhost:5007';

/* ── colour tokens ── */
const C = {
  red:    '#d93954',
  orange: '#e8721a',
  yellow: '#e5a100',
  green:  '#2ea44f',
  blue:   '#2563eb',
  purple: '#7c3aed',
};

const AGENT_STYLE = {
  Structura: { bg: '#fde8ec', color: '#d93954' },
  Cognita:   { bg: '#e8f0fe', color: '#2563eb' },
  Chronos:   { bg: '#fef3e2', color: '#e8721a' },
  Sentio:    { bg: '#f3e8fd', color: '#7c3aed' },
  Agora:     { bg: '#e6f6ec', color: '#2ea44f' },
};

/* agent field mapping from the integration API response */
const AGENT_KEYS   = ['Structura', 'Cognita', 'Chronos', 'Sentio', 'Agora'];
const AGENT_FIELDS = ['structura_score', 'cognita_score', 'chronos_score', 'sentio_score', 'agora_score'];
const AGENT_COLORS = [C.red, C.blue, C.orange, C.purple, C.green];

/* fallback — matches attrition-dashboard.html static data */
const FALLBACK_AVG = [
  { name: 'Structura', avg: 0.78, color: C.red    },
  { name: 'Cognita',   avg: 0.63, color: C.blue   },
  { name: 'Chronos',   avg: 0.76, color: C.orange  },
  { name: 'Sentio',    avg: 0.71, color: C.purple  },
  { name: 'Agora',     avg: 0.58, color: C.green   },
];

/* ── tiny sub-components ── */
const Chip = ({ name, score }) => {
  const s = AGENT_STYLE[name] || { bg: '#eee', color: '#333' };
  return (
    <span style={{
      padding: '3px 8px', borderRadius: 4, fontSize: 10, fontWeight: 600,
      display: 'inline-flex', background: s.bg, color: s.color,
      marginRight: 6, marginTop: 4,
    }}>
      {name}: {score}
    </span>
  );
};

const InsightCard = ({ emoji, title, body, chips, borderColor }) => (
  <div style={{
    background: 'var(--card,#fff)', borderRadius: 12, padding: 16,
    borderLeft: `4px solid ${borderColor}`,
    boxShadow: '0 1px 4px rgba(0,0,0,.06)', marginBottom: 14,
  }}>
    <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 6 }}>
      {emoji} {title}
    </div>
    <div style={{ fontSize: 13, color: 'var(--sub,#555)', lineHeight: 1.6, marginBottom: 8 }}>
      {body}
    </div>
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
      {chips.map(c => <Chip key={c.name} name={c.name} score={c.score} />)}
    </div>
  </div>
);

/* ── static insight data ── */
const LEFT_INSIGHTS = [
  {
    emoji: '🚨', borderColor: C.red,
    title: '번아웃에 직면한 직원(P01) 그룹 확산',
    body: '고위험군 486명 중 P01(번아웃) 101명, P04(저평가 전문가) 192명이 다수. R&D 부서 집중. 평균 초과근무시간 전사 대비 2.3배, 직무만족도 하위 15%.',
    chips: [{ name: 'Structura', score: 0.87 }, { name: 'Sentio', score: 0.82 }, { name: 'Chronos', score: 0.76 }],
  },
  {
    emoji: '🔗', borderColor: C.blue,
    title: '관계망 단절 패턴 감지',
    body: '최근 6개월간 신규 협업 관계 미형성 직원이 고위험군에서 72%. 프로젝트 참여 다양성 감소, 사회적 고립 지수 지속 상승.',
    chips: [{ name: 'Cognita', score: 0.80 }, { name: 'Chronos', score: 0.65 }],
  },
  {
    emoji: '📈', borderColor: C.orange,
    title: '행동 패턴 이상 징후 증가',
    body: '최근 3주간 로그인 시간 불규칙성 전분기 대비 38% 증가. 이메일 발송량 감소, 시스템 접속 빈도 저하 등 이직 준비 행동 패턴 관찰.',
    chips: [{ name: 'Chronos', score: 0.76 }, { name: 'Agora', score: 0.58 }],
  },
];

const RIGHT_INSIGHTS = [
  {
    emoji: '💬', borderColor: C.purple,
    title: '감성 분석: 부정 감정 키워드 급증',
    body: "코칭 면담 및 자기평가 텍스트에서 '소진', '불확실', '답답함' 등 부정 키워드 전년 대비 45% 증가. JobLevel 1~2 (Research Scientist, Laboratory Technician) 집중.",
    chips: [{ name: 'Sentio', score: 0.85 }],
  },
  {
    emoji: '🎯', borderColor: C.green,
    title: '외부 시장 Pull Factor 강화',
    body: 'Technology, D&AI 부서 LinkedIn 등 접속 빈도 전분기 대비 62% 증가. AI/데이터 직군 시장 보상 현 수준 대비 15~25% 높음.',
    chips: [{ name: 'Agora', score: 0.72 }, { name: 'Structura', score: 0.55 }],
  },
];

/* ── custom tooltip ── */
const AgentTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#fff', border: '1px solid #eee', borderRadius: 8, padding: '8px 12px', fontSize: 12, boxShadow: '0 2px 8px rgba(0,0,0,.1)' }}>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>{label}</div>
      <div style={{ color: payload[0].fill }}>평균 위험 점수: <strong>{payload[0].value.toFixed(2)}</strong></div>
    </div>
  );
};

/* ── main component ── */
const Insights = () => {
  const [agentAvg, setAgentAvg]       = useState(FALLBACK_AVG);
  const [highRiskCount, setHighRisk]  = useState(null);   // null = still loading
  const [usingFallback, setFallback]  = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const resp = await fetch(`${INTEGRATION_URL}/api/results/list-all-employees`);
        if (!resp.ok) throw new Error('fetch failed');
        const data = await resp.json();
        if (!data.success || !data.results?.length) throw new Error('no data');

        const highRisk = data.results.filter(r => r.risk_level === 'HIGH');
        setHighRisk(highRisk.length);

        if (highRisk.length === 0) return;   // keep fallback values

        const computed = AGENT_KEYS.map((name, i) => {
          const field = AGENT_FIELDS[i];
          const vals  = highRisk.map(r => Number(r[field]) || 0).filter(v => v > 0);
          const avg   = vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : FALLBACK_AVG[i].avg;
          return { name, avg: Math.round(avg * 1000) / 1000, color: AGENT_COLORS[i] };
        });
        setAgentAvg(computed);
        setFallback(false);
      } catch {
        setHighRisk(0);   // show fallback quietly
      }
    };
    load();
  }, []);

  const cardS = {
    background: 'var(--card,#fff)', borderRadius: 12, padding: 20,
    border: '1px solid var(--border,#eee)', boxShadow: '0 1px 4px rgba(0,0,0,.06)',
  };

  return (
    <div>
      {/* ── Architecture banner ── */}
      <div style={{
        background: 'linear-gradient(135deg, #2d2d2d 0%, #1a1a2e 60%, #16213e 100%)',
        borderRadius: 12, padding: '20px 24px', marginBottom: 20, color: '#fff',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        gap: 16, flexWrap: 'wrap',
      }}>
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>
            Agentic AI 기반 선제적 퇴사위험 예측 시스템
          </div>
          <div style={{ fontSize: 12, color: '#ccc' }}>
            5개 전문 Worker Agent의 분석 결과를 종합하여 360도 관점의 퇴사 위험 진단 제공
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {Object.entries(AGENT_STYLE).map(([name, s]) => (
            <span key={name} style={{
              padding: '6px 12px', borderRadius: 8, fontSize: 11, fontWeight: 600, textAlign: 'center',
              background: `${s.color}22`, border: `1px solid ${s.color}66`, color: s.color,
            }}>
              {name}<br /><span style={{ fontSize: 10, fontWeight: 400, color: '#aaa' }}>
                {{ Structura:'정형 데이터', Cognita:'관계망', Chronos:'시계열', Sentio:'자연어', Agora:'외부시장' }[name]}
              </span>
            </span>
          ))}
        </div>
      </div>

      {/* ── Section title ── */}
      <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#d93954', display: 'inline-block', flexShrink: 0 }} />
        AI 핵심 인사이트 (Synthesize Agent)
      </div>

      {/* ── Two-column insight grid ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

        {/* Left: 3 insight cards */}
        <div>
          {LEFT_INSIGHTS.map(ins => <InsightCard key={ins.title} {...ins} />)}
        </div>

        {/* Right: 2 insight cards + agent avg chart */}
        <div>
          {RIGHT_INSIGHTS.map(ins => <InsightCard key={ins.title} {...ins} />)}

          {/* ── Agent avg bar chart ── */}
          <div style={cardS}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
              <span style={{ color: C.red }}>⚙</span>
              <span style={{ fontWeight: 700, fontSize: 14 }}>Agent별 평균 위험 점수 (고위험군)</span>
              {highRiskCount !== null && (
                <span style={{ fontSize: 11, color: 'var(--sub,#888)', marginLeft: 4 }}>
                  {highRiskCount > 0 ? `실제 ${highRiskCount}명 기준` : '(데이터 없음 — 기준값)'}
                </span>
              )}
              {usingFallback && highRiskCount === null && (
                <span style={{ fontSize: 10, color: '#aaa' }}>로딩 중...</span>
              )}
              {usingFallback && highRiskCount !== null && (
                <span style={{ fontSize: 10, color: '#aaa', fontStyle: 'italic' }}>기준값</span>
              )}
            </div>

            <ResponsiveContainer width="100%" height={190}>
              <BarChart data={agentAvg} margin={{ top: 18, right: 16, bottom: 4, left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 11, fontFamily: 'inherit', fontWeight: 600 }}
                  axisLine={false} tickLine={false}
                />
                <YAxis
                  domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
                  tick={{ fontSize: 10, fontFamily: 'inherit' }}
                  axisLine={false} tickLine={false}
                />
                <Tooltip content={<AgentTooltip />} />
                <Bar
                  dataKey="avg" radius={[6, 6, 0, 0]}
                  label={{ position: 'top', fontSize: 12, fontWeight: 500, fill: '#555', formatter: v => v.toFixed(2) }}
                >
                  {agentAvg.map((entry, i) => (
                    <Cell key={i} fill={`${entry.color}CC`} stroke={entry.color} strokeWidth={1.5} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>

            <div style={{ fontSize: 11, color: 'var(--sub,#888)', marginTop: 4, textAlign: 'center' }}>
              * 각 Agent가 고위험 직원에게 부여한 위험 점수의 평균값 (0~1 스케일)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Insights;
