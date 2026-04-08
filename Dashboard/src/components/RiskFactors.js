import React, { useState, useEffect } from 'react';
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Radar, Legend, ResponsiveContainer, Tooltip,
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

/* ── static data ── */
const topFactors = [
  { name: '초과근무시간',    value: 0.82, color: C.red    },
  { name: '직무만족도',      value: 0.74, color: C.red    },
  { name: '연봉(동료대비)',  value: 0.68, color: C.red    },
  { name: '승진후 경과기간', value: 0.62, color: C.orange  },
  { name: '사회적 고립지수', value: 0.58, color: C.orange  },
  { name: '감성 부정지수',   value: 0.55, color: C.orange  },
  { name: '로그인 불규칙성', value: 0.51, color: C.yellow  },
  { name: '외부 플랫폼 접속', value: 0.47, color: C.yellow },
  { name: 'PM 경험 부족',   value: 0.42, color: C.yellow  },
  { name: '연봉 상승률',     value: 0.38, color: C.yellow  },
];

/* Radar chart data — 5 risk dimensions × 4 personas */
const radarData = [
  { factor: '구조적 불만족', P01: 85, P02: 70, P03: 55, P04: 80 },
  { factor: '관계적 단절',   P01: 45, P02: 35, P03: 60, P04: 30 },
  { factor: '행동적 이탈',   P01: 72, P02: 40, P03: 50, P04: 35 },
  { factor: '심리적 소진',   P01: 90, P02: 50, P03: 65, P04: 40 },
  { factor: '외부 Pull',    P01: 30, P02: 80, P03: 45, P04: 70 },
];

const personas = [
  {
    id: 'P01', title: '번아웃 직전',   count: '77명', avg: '0.84',
    tagBg: '#fde8ec', tagColor: C.red,
    factors: [
      { name: '초과근무시간', pct: 78, color: C.red    },
      { name: '직무만족도',   pct: 45, color: C.red    },
      { name: '감성 부정지수', pct: 62, color: C.orange },
    ],
  },
  {
    id: 'P02', title: '보상 실망',     count: '12명', avg: '0.79',
    tagBg: '#e8f0fe', tagColor: C.blue,
    factors: [
      { name: '보상 격차 (vs 동료)', pct: 72, color: C.red    },
      { name: '외부 보상 비교',      pct: 65, color: C.orange },
    ],
  },
  {
    id: 'P03', title: '성장 정체',     count: '40명', avg: '0.73',
    tagBg: '#fef3e2', tagColor: C.orange,
    factors: [
      { name: '승진후 경과기간', pct: 68, color: C.orange },
      { name: 'PM 경험 부족',   pct: 55, color: C.orange },
    ],
  },
  {
    id: 'P04', title: '보상체감 낮음', count: '26명', avg: '0.68',
    tagBg: '#f3e8fd', tagColor: C.purple,
    factors: [
      { name: '인센티브 미수령',  pct: 58, color: C.orange },
      { name: '연봉 상승률 정체', pct: 42, color: C.yellow },
    ],
  },
];

const matrixData = [
  {
    label: '고위험',
    cells: [
      { text: '자연감소',    count: '8명',  bg: '#fde8ec', color: C.red    },
      { text: '선별적 개입', count: '14명', bg: '#fef3e2', color: C.orange },
      { text: '적극 개입',   count: '22명', bg: '#fef3e2', color: C.orange },
      { text: '최우선 개입', count: '11명', bg: '#fde8ec', color: C.red    },
    ],
  },
  {
    label: '잠재적',
    cells: [
      { text: '성과개선',    count: '25명',  bg: '#fff9e6', color: '#b8860b' },
      { text: '잠재력 평가', count: '45명',  bg: '#fff9e6', color: '#b8860b' },
      { text: '역량 강화',   count: '38명',  bg: '#e8f0fe', color: C.blue   },
      { text: '선제적 관리', count: '19명',  bg: '#fef3e2', color: C.orange  },
    ],
  },
  {
    label: '저위험',
    cells: [
      { text: '유지 관리', count: '120명', bg: '#e6f6ec', color: C.green },
      { text: '안정 유지', count: '245명', bg: '#e6f6ec', color: C.green },
      { text: '성장 지원', count: '210명', bg: '#e8f0fe', color: C.blue  },
      { text: '핵심인재',  count: '177명', bg: '#e8f0fe', color: C.blue  },
    ],
  },
];

/* ── shared styles ── */
const cardS = {
  background: 'var(--card,#fff)',
  borderRadius: 12, padding: 20,
  border: '1px solid var(--border,#eee)',
  boxShadow: '0 1px 4px rgba(0,0,0,.06)',
};
const secTitle = (icon, text) => (
  <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 14 }}>
    <span style={{ color: C.red }}>{icon}</span> {text}
  </div>
);
/* ── radar legend colours ── */
const RADAR_COLORS = [C.red, C.blue, C.orange, C.purple];

function RiskFactors() {
  const [radarReal, setRadarReal] = useState(null);

  useEffect(() => {
    fetch(`${INTEGRATION_URL}/api/results/list-all-employees`)
      .then(r => r.json())
      .then(data => {
        if (!data.success || !data.results?.length) return;
        const all = data.results;
        const depts = ['Research & Development', 'Sales', 'Human Resources'];
        const keys  = ['structura_score', 'cognita_score', 'chronos_score', 'sentio_score', 'agora_score'];
        const dims  = ['구조적 불만족', '관계적 단절', '행동적 이탈', '심리적 소진', '외부 Pull'];
        const labels = { 'Research & Development': 'R&D', 'Sales': 'Sales', 'Human Resources': 'HR' };

        const avgFor = (pool, key) => {
          const vals = pool.map(e => (e[key] || 0)).filter(v => v > 0);
          return vals.length ? Math.round((vals.reduce((s, v) => s + v, 0) / vals.length) * 100) : 0;
        };

        const highAll = all.filter(e => e.risk_level === 'HIGH');
        const computed = dims.map((factor, i) => {
          const row = { factor };
          row['전사'] = avgFor(highAll, keys[i]);
          depts.forEach(d => {
            const pool = all.filter(e => e.department === d && e.risk_level === 'HIGH');
            row[labels[d]] = avgFor(pool, keys[i]);
          });
          return row;
        });
        setRadarReal(computed);
      })
      .catch(() => {});
  }, []);

  const activeRadarData   = radarReal || radarData;
  const activeRadarKeys   = radarReal ? ['전사', 'R&D', 'Sales', 'HR'] : ['P01', 'P02', 'P03', 'P04'];
  const activeLegendLabels = radarReal
    ? { '전사': '전사 고위험', 'R&D': 'R&D', 'Sales': 'Sales', 'HR': 'HR' }
    : { P01: 'P01 번아웃', P02: 'P02 보상실망', P03: 'P03 성장정체', P04: 'P04 보상체감' };

  return (
    <div>
      {/* ── Section title ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 20 }}>
        <span style={{ width: 10, height: 10, borderRadius: '50%', background: C.red, display: 'inline-block' }} />
        <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700 }}>퇴사 위험 요인 분석 (SHAP 기반)</h2>
      </div>

      {/* ── Row 1: Top 10 요인 + Agent 레이더 ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>

        {/* Left: Top 10 위험 요인 */}
        <div style={cardS}>
          {secTitle('☰', '전사 Top 10 위험 요인')}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
            {topFactors.map((f, i) => (
              <div key={f.name} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ width: 18, fontSize: 12, color: 'var(--sub,#999)', textAlign: 'right', flexShrink: 0 }}>
                  {i + 1}
                </span>
                <span style={{ width: 114, fontSize: 12, color: 'var(--text,#444)', flexShrink: 0 }}>{f.name}</span>
                <div style={{ flex: 1, height: 22, background: 'var(--border,#f0f0f0)', borderRadius: 6, overflow: 'hidden' }}>
                  <div style={{
                    width: `${f.value * 100}%`, height: '100%', background: f.color,
                    borderRadius: 6, display: 'flex', alignItems: 'center',
                    justifyContent: 'flex-end', paddingRight: 8,
                  }}>
                    <span style={{ fontSize: 11, color: '#fff', fontWeight: 600 }}>{f.value.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Agent별 위험 요인 기여도 (Radar) — 실제 데이터 */}
        <div style={cardS}>
          {secTitle('◉', radarReal ? '부서별 위험 요인 기여도 (실제)' : 'Agent별 위험 요인 기여도')}
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={activeRadarData}>
              <PolarGrid stroke="var(--border,#e5e7eb)" />
              <PolarAngleAxis
                dataKey="factor"
                tick={{ fontSize: 11, fontFamily: 'inherit', fill: 'var(--text,#555)' }}
              />
              <PolarRadiusAxis
                angle={90} domain={[0, 100]}
                tick={{ fontSize: 9, fill: 'var(--sub,#999)' }}
                tickCount={4}
              />
              <Tooltip
                formatter={(v, name) => [`${v}점`, activeLegendLabels[name] || name]}
                contentStyle={{ fontSize: 12, borderRadius: 8 }}
              />
              {activeRadarKeys.map((p, i) => (
                <Radar
                  key={p} name={p} dataKey={p}
                  stroke={RADAR_COLORS[i]}
                  fill={RADAR_COLORS[i]}
                  fillOpacity={0.12}
                  strokeWidth={2}
                />
              ))}
              <Legend
                wrapperStyle={{ fontSize: 11, fontFamily: 'inherit', paddingTop: 8 }}
                formatter={(value) => activeLegendLabels[value] || value}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Row 2: Persona별 위험 프로필 + 성과-위험 매트릭스 ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

        {/* Left: Persona별 위험 프로필 */}
        <div style={cardS}>
          {secTitle('★', 'Persona별 위험 프로필')}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
            {personas.map((p) => (
              <div key={p.id}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                  <span style={{
                    padding: '3px 12px', borderRadius: 12, fontSize: 12, fontWeight: 700,
                    background: p.tagBg, color: p.tagColor,
                  }}>
                    {p.id} {p.title}
                  </span>
                  <span style={{ fontSize: 12, color: 'var(--sub,#888)' }}>
                    {p.count} | 평균 {p.avg}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, paddingLeft: 4 }}>
                  {p.factors.map((f) => (
                    <div key={f.name}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3, fontSize: 12 }}>
                        <span style={{ color: 'var(--text,#444)' }}>{f.name}</span>
                        <span style={{ fontWeight: 700, color: f.color }}>{f.pct}%</span>
                      </div>
                      <div style={{ height: 20, background: 'var(--border,#f0f0f0)', borderRadius: 5, overflow: 'hidden' }}>
                        <div style={{
                          width: `${f.pct}%`, height: '100%', background: f.color, borderRadius: 5,
                          display: 'flex', alignItems: 'center', paddingLeft: 8,
                          fontSize: 10, color: '#fff', fontWeight: 600,
                          transition: 'width 0.8s ease',
                        }}>
                          {f.pct}%
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: 성과-위험 매트릭스 */}
        <div style={cardS}>
          {secTitle('▣', '성과 - 위험 매트릭스')}
          <div style={{ fontSize: 12, color: 'var(--sub,#666)', marginBottom: 10 }}>
            퇴사 위험과 성과를 결합하여 전략적 선별
          </div>
          <div style={{ overflowX: 'auto' }}>
            <div style={{
              display: 'grid',
              gridTemplateColumns: '72px 1fr 1fr 1fr 1fr',
              gap: 5, minWidth: 360,
            }}>
              {/* Header */}
              <div />
              {['C', 'B', 'A', 'S'].map(h => (
                <div key={h} style={{ textAlign: 'center', fontWeight: 700, fontSize: 14, color: 'var(--sub,#555)', padding: '6px 0' }}>
                  {h}
                </div>
              ))}

              {/* Data rows */}
              {matrixData.map((row) => (
                <React.Fragment key={row.label}>
                  <div style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontWeight: 700, fontSize: 12, color: 'var(--text,#444)',
                    padding: '4px 0',
                  }}>
                    {row.label}
                  </div>
                  {row.cells.map((cell, ci) => (
                    <div key={ci} style={{
                      background: cell.bg, color: cell.color, borderRadius: 8,
                      padding: '12px 6px', textAlign: 'center',
                    }}>
                      <div style={{ fontWeight: 600, fontSize: 11 }}>{cell.text}</div>
                      <div style={{ fontSize: 15, fontWeight: 800, marginTop: 3 }}>{cell.count}</div>
                    </div>
                  ))}
                </React.Fragment>
              ))}
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 10, fontSize: 11, color: 'var(--sub,#999)', padding: '0 72px' }}>
            <span>← 성과 낮음</span>
            <span>성과 높음 →</span>
          </div>

          <div style={{
            marginTop: 14, padding: '10px 14px',
            background: '#fef7f8', borderRadius: 8,
            borderLeft: `4px solid ${C.red}`,
            fontSize: 13, color: C.red, fontWeight: 600,
          }}>
            핵심: S등급 고위험군 11명은 최우선 개입 필요. A등급 잠재위험 38명은 역량 강화 우선 권장.
          </div>
        </div>
      </div>
    </div>
  );
}

export default RiskFactors;
