import React from 'react';

/* ── colour tokens ── */
const C = {
  red: '#d93954',
  orange: '#e8721a',
  yellow: '#e5a100',
  green: '#2ea44f',
  blue: '#2563eb',
  purple: '#7c3aed',
};

/* ── agent chip palette ── */
const AGENT_STYLE = {
  Structura: { bg: '#fde8ec', color: '#d93954' },
  Cognita:   { bg: '#e8f0fe', color: '#2563eb' },
  Chronos:   { bg: '#fef3e2', color: '#e8721a' },
  Sentio:    { bg: '#f3e8fd', color: '#7c3aed' },
  Agora:     { bg: '#e6f6ec', color: '#2ea44f' },
};

/* ── tiny helpers ── */
const Chip = ({ name, score }) => {
  const s = AGENT_STYLE[name] || { bg: '#eee', color: '#333' };
  return (
    <span
      style={{
        padding: '3px 8px',
        borderRadius: 4,
        fontSize: 10,
        fontWeight: 600,
        display: 'inline-flex',
        background: s.bg,
        color: s.color,
        marginRight: 6,
        marginTop: 4,
      }}
    >
      {name} {score}
    </span>
  );
};

const KpiCard = ({ label, value, accent }) => (
  <div
    style={{
      background: 'var(--card, #fff)',
      borderRadius: 12,
      padding: 16,
      border: '1px solid var(--border, #eee)',
      textAlign: 'center',
      position: 'relative',
      overflow: 'hidden',
      boxShadow: '0 1px 4px rgba(0,0,0,.06)',
      flex: '1 1 0',
      minWidth: 140,
    }}
  >
    {/* coloured top bar */}
    <div
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: 4,
        background: accent,
      }}
    />
    <div style={{ fontSize: 12, color: '#888', marginBottom: 6 }}>{label}</div>
    <div style={{ fontSize: 22, fontWeight: 700, color: accent }}>{value}</div>
  </div>
);

const InsightCard = ({ emoji, title, body, chips, borderColor }) => (
  <div
    style={{
      background: 'var(--card, #fff)',
      borderRadius: 12,
      padding: 16,
      borderLeft: `4px solid ${borderColor}`,
      boxShadow: '0 1px 4px rgba(0,0,0,.06)',
      marginBottom: 14,
    }}
  >
    <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 6 }}>
      {emoji} {title}
    </div>
    <div style={{ fontSize: 13, color: '#555', lineHeight: 1.6, marginBottom: 8 }}>
      {body}
    </div>
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
      {chips.map((c) => (
        <Chip key={c.name + c.score} name={c.name} score={c.score} />
      ))}
    </div>
  </div>
);

/* ── data ── */
const KPI = [
  { label: '전사 인원 현황', value: '1,470명', accent: C.blue },
  { label: '퇴사 고위험군', value: '77명', accent: C.red },
  { label: '잠재적 위험군', value: '202명', accent: C.orange },
  { label: '안정/양호군', value: '1,191명', accent: C.green },
  { label: '평균 위험 점수', value: '0.28', accent: C.yellow },
];

const AGENTS = ['Structura', 'Cognita', 'Chronos', 'Sentio', 'Agora'];

const LEFT_INSIGHTS = [
  {
    emoji: '🚨',
    title: '번아웃 직전 그룹의 급속 확산',
    body: '고위험군 중 50%(77명)이 번아웃 직전 상태. 평균 초과근무시간 전사 대비 2.3배, 직무만족도 하위 15%.',
    borderColor: C.red,
    chips: [
      { name: 'Structura', score: 0.87 },
      { name: 'Sentio', score: 0.82 },
      { name: 'Chronos', score: 0.76 },
    ],
  },
  {
    emoji: '🔗',
    title: '관계망 단절 패턴 감지',
    body: '최근 6개월간 신규 협업 관계 미형성 직원이 고위험군에서 72%. 사회적 고립 지수 지속 상승.',
    borderColor: C.blue,
    chips: [
      { name: 'Cognita', score: 0.80 },
      { name: 'Chronos', score: 0.65 },
    ],
  },
  {
    emoji: '📈',
    title: '행동 패턴 이상 징후 증가',
    body: '최근 3주간 로그인 시간 불규칙성 전분기 대비 38% 증가. 이직 준비 행동 패턴 관찰.',
    borderColor: C.orange,
    chips: [
      { name: 'Chronos', score: 0.76 },
      { name: 'Agora', score: 0.58 },
    ],
  },
];

const RIGHT_INSIGHTS = [
  {
    emoji: '💬',
    title: '감성 분석: 부정 감정 키워드 급증',
    body: "코칭 면담 및 자기평가 텍스트에서 '소진', '불확실', '답답함' 등 부정 키워드 45% 증가.",
    borderColor: C.purple,
    chips: [{ name: 'Sentio', score: 0.85 }],
  },
  {
    emoji: '🎯',
    title: '외부 시장 Pull Factor 강화',
    body: 'Technology, R&D 부서 LinkedIn 접속 빈도 전분기 대비 62% 증가. 시장 보상 15~25% 높음.',
    borderColor: C.green,
    chips: [
      { name: 'Agora', score: 0.72 },
      { name: 'Structura', score: 0.55 },
    ],
  },
];

/* ── main component ── */
const Insights = () => (
  <div style={{ padding: 24 }}>
    {/* ── KPI row ── */}
    <div style={{ display: 'flex', gap: 14, marginBottom: 24, flexWrap: 'wrap' }}>
      {KPI.map((k) => (
        <KpiCard key={k.label} {...k} />
      ))}
    </div>

    {/* ── Architecture banner ── */}
    <div
      style={{
        background: 'linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%)',
        borderRadius: 12,
        padding: '18px 24px',
        marginBottom: 24,
        color: '#fff',
      }}
    >
      <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 10 }}>
        Agentic AI 기반 선제적 퇴사위험 예측 시스템
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
        {AGENTS.map((a) => {
          const s = AGENT_STYLE[a];
          return (
            <span
              key={a}
              style={{
                padding: '5px 14px',
                borderRadius: 20,
                fontSize: 12,
                fontWeight: 600,
                background: s.bg,
                color: s.color,
              }}
            >
              {a} Agent
            </span>
          );
        })}
      </div>
    </div>

    {/* ── Section title ── */}
    <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 16 }}>
      AI 핵심 인사이트 (Synthesize Agent)
    </div>

    {/* ── Two-column insight grid ── */}
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
      <div>
        {LEFT_INSIGHTS.map((ins) => (
          <InsightCard key={ins.title} {...ins} />
        ))}
      </div>
      <div>
        {RIGHT_INSIGHTS.map((ins) => (
          <InsightCard key={ins.title} {...ins} />
        ))}
      </div>
    </div>
  </div>
);

export default Insights;
