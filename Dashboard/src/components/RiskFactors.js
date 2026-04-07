import React from 'react';

const kpiData = [
  { label: '전사', value: '1,470명', color: '#2563eb', bg: '#e8f0fe' },
  { label: '고위험군', value: '77명', color: '#d93954', bg: '#fde8ec' },
  { label: '잠재위험', value: '202명', color: '#e8721a', bg: '#fef3e2' },
  { label: '안정', value: '1,191명', color: '#2ea44f', bg: '#e6f6ec' },
  { label: '평균점수', value: '0.28', color: '#7c3aed', bg: '#f3e8fd' },
];

const topFactors = [
  { name: '초과근무시간', value: 0.82, color: '#d93954' },
  { name: '직무만족도', value: 0.74, color: '#d93954' },
  { name: '연봉(동료대비)', value: 0.68, color: '#d93954' },
  { name: '승진후 경과기간', value: 0.62, color: '#e8721a' },
  { name: '사회적 고립지수', value: 0.58, color: '#e8721a' },
  { name: '감성 부정지수', value: 0.55, color: '#e8721a' },
  { name: '로그인 불규칙성', value: 0.51, color: '#e5a100' },
  { name: '외부 플랫폼 접속', value: 0.47, color: '#e5a100' },
  { name: 'PM 경험 부족', value: 0.42, color: '#e5a100' },
  { name: '연봉 상승률', value: 0.38, color: '#e5a100' },
];

const personas = [
  {
    id: 'P01', title: '번아웃 직전', count: '77명', avg: '0.84',
    tagBg: '#fde8ec', tagColor: '#d93954',
    factors: [
      { name: '초과근무시간', pct: 78, color: '#d93954' },
      { name: '직무만족도', pct: 45, color: '#d93954' },
      { name: '감성 부정지수', pct: 62, color: '#e8721a' },
    ],
  },
  {
    id: 'P02', title: '보상 실망', count: '12명', avg: '0.79',
    tagBg: '#e8f0fe', tagColor: '#2563eb',
    factors: [
      { name: '보상 격차', pct: 72, color: '#d93954' },
      { name: '외부 보상 비교', pct: 65, color: '#e8721a' },
    ],
  },
  {
    id: 'P03', title: '성장 정체', count: '40명', avg: '0.73',
    tagBg: '#fef3e2', tagColor: '#e8721a',
    factors: [
      { name: '승진후 경과기간', pct: 68, color: '#e8721a' },
      { name: 'PM 경험 부족', pct: 55, color: '#e8721a' },
    ],
  },
  {
    id: 'P04', title: '보상체감 낮음', count: '26명', avg: '0.68',
    tagBg: '#f3e8fd', tagColor: '#7c3aed',
    factors: [
      { name: '인센티브 미수령', pct: 58, color: '#e8721a' },
      { name: '연봉 상승률 정체', pct: 42, color: '#e5a100' },
    ],
  },
];

const matrixData = [
  {
    label: '고위험',
    cells: [
      { text: '자연감소', count: '8명', bg: '#fde8ec', color: '#d93954' },
      { text: '선별적 개입', count: '14명', bg: '#fef3e2', color: '#e8721a' },
      { text: '적극 개입', count: '22명', bg: '#fef3e2', color: '#e8721a' },
      { text: '최우선 개입', count: '11명', bg: '#fde8ec', color: '#d93954' },
    ],
  },
  {
    label: '잠재적',
    cells: [
      { text: '성과개선', count: '25명', bg: '#fff9e6', color: '#b8860b' },
      { text: '잠재력 평가', count: '45명', bg: '#fff9e6', color: '#b8860b' },
      { text: '역량 강화', count: '38명', bg: '#e8f0fe', color: '#2563eb' },
      { text: '선제적 관리', count: '19명', bg: '#fef3e2', color: '#e8721a' },
    ],
  },
  {
    label: '저위험',
    cells: [
      { text: '유지 관리', count: '120명', bg: '#e6f6ec', color: '#2ea44f' },
      { text: '안정 유지', count: '245명', bg: '#e6f6ec', color: '#2ea44f' },
      { text: '성장 지원', count: '210명', bg: '#e8f0fe', color: '#2563eb' },
      { text: '핵심인재', count: '177명', bg: '#e8f0fe', color: '#2563eb' },
    ],
  },
];

const cardStyle = {
  background: '#fff',
  borderRadius: 12,
  padding: 24,
  boxShadow: '0 1px 4px rgba(0,0,0,0.07)',
};

function RiskFactors() {
  return (
    <div style={{ padding: '24px 32px', background: '#f5f6fa', minHeight: '100vh' }}>
      {/* KPI Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16, marginBottom: 28 }}>
        {kpiData.map((k) => (
          <div key={k.label} style={{
            ...cardStyle,
            padding: '18px 20px',
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
          }}>
            <span style={{ fontSize: 13, color: '#888' }}>{k.label}</span>
            <span style={{ fontSize: 26, fontWeight: 700, color: k.color }}>{k.value}</span>
          </div>
        ))}
      </div>

      {/* Section Title */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 20 }}>
        <span style={{
          display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
          background: '#d93954',
        }} />
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#222' }}>
          퇴사 위험 요인 분석 (SHAP 기반)
        </h2>
      </div>

      {/* First two-column grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}>
        {/* Left: Top 10 */}
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 18px', fontSize: 16, fontWeight: 700, color: '#333' }}>
            전사 Top 10 위험 요인
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {topFactors.map((f, i) => (
              <div key={f.name} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ width: 18, fontSize: 13, color: '#999', textAlign: 'right', flexShrink: 0 }}>
                  {i + 1}
                </span>
                <span style={{ width: 110, fontSize: 13, color: '#444', flexShrink: 0 }}>{f.name}</span>
                <div style={{
                  flex: 1, height: 22, background: '#f0f0f0', borderRadius: 6, position: 'relative',
                  overflow: 'hidden',
                }}>
                  <div style={{
                    width: `${f.value * 100}%`, height: '100%', background: f.color,
                    borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'flex-end',
                    paddingRight: 8, boxSizing: 'border-box',
                  }}>
                    <span style={{ fontSize: 12, color: '#fff', fontWeight: 600 }}>{f.value.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Persona profiles */}
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 18px', fontSize: 16, fontWeight: 700, color: '#333' }}>
            Persona별 위험 프로필
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {personas.map((p) => (
              <div key={p.id}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                  <span style={{
                    display: 'inline-block', padding: '3px 10px', borderRadius: 12,
                    fontSize: 13, fontWeight: 700, background: p.tagBg, color: p.tagColor,
                  }}>
                    {p.id} {p.title}
                  </span>
                  <span style={{ fontSize: 12, color: '#888' }}>{p.count}, 평균 {p.avg}</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, paddingLeft: 8 }}>
                  {p.factors.map((f) => (
                    <div key={f.name} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ width: 100, fontSize: 12, color: '#555', flexShrink: 0 }}>{f.name}</span>
                      <div style={{
                        flex: 1, height: 18, background: '#f0f0f0', borderRadius: 5,
                        overflow: 'hidden',
                      }}>
                        <div style={{
                          width: `${f.pct}%`, height: '100%', background: f.color,
                          borderRadius: 5, display: 'flex', alignItems: 'center', justifyContent: 'flex-end',
                          paddingRight: 6, boxSizing: 'border-box',
                        }}>
                          <span style={{ fontSize: 11, color: '#fff', fontWeight: 600 }}>{f.pct}%</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Second row: Performance-Risk Matrix */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 20 }}>
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 18px', fontSize: 16, fontWeight: 700, color: '#333' }}>
            성과 - 위험 매트릭스
          </h3>

          <div style={{
            display: 'grid',
            gridTemplateColumns: '80px 1fr 1fr 1fr 1fr',
            gap: 6,
          }}>
            {/* Header row */}
            <div />
            {['C', 'B', 'A', 'S'].map((h) => (
              <div key={h} style={{
                textAlign: 'center', fontWeight: 700, fontSize: 14, color: '#555',
                padding: '6px 0',
              }}>
                {h}
              </div>
            ))}

            {/* Data rows */}
            {matrixData.map((row) => (
              <React.Fragment key={row.label}>
                <div style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontWeight: 700, fontSize: 13, color: '#444',
                }}>
                  {row.label}
                </div>
                {row.cells.map((cell, ci) => (
                  <div key={ci} style={{
                    background: cell.bg, color: cell.color, borderRadius: 8,
                    padding: '14px 8px', textAlign: 'center',
                  }}>
                    <div style={{ fontWeight: 700, fontSize: 13 }}>{cell.text}</div>
                    <div style={{ fontSize: 15, fontWeight: 800, marginTop: 2 }}>{cell.count}</div>
                  </div>
                ))}
              </React.Fragment>
            ))}
          </div>

          {/* Bottom axis label */}
          <div style={{
            textAlign: 'center', fontSize: 13, color: '#999', marginTop: 12,
          }}>
            &larr; 성과 낮음 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 성과 높음 &rarr;
          </div>

          {/* Highlight box */}
          <div style={{
            marginTop: 16, padding: '12px 16px', background: '#fde8ec',
            borderRadius: 8, borderLeft: '4px solid #d93954',
            fontSize: 14, color: '#d93954', fontWeight: 600,
          }}>
            핵심: S등급 고위험군 11명은 최우선 개입 필요.
          </div>
        </div>
      </div>
    </div>
  );
}

export default RiskFactors;
