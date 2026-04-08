import { useState, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, ResponsiveContainer,
} from 'recharts';

// ─────────────────────── 부서별 전체 데이터 ───────────────────────────────────
// 실제 IBM HR 데이터셋 3개 부서 기준
// viewMode: 'all' | 'Research & Development' | 'Sales' | 'Human Resources'

const DEPT_DB = {
  'all': {
    kpi: {
      total: '1,174', high: 155, med: 387, low: 632,
      avgScore: '0.42', highChange: '▲ 12%', medChange: '▲ 30%',
    },
    depts: [
      { n: 'Research & Development', c: 610, h: 80, m: 201, l: 329, r: 13 },
      { n: 'Sales',                  c: 446, h: 66, m: 148, l: 232, r: 15 },
      { n: 'Human Resources',        c: 118, h: 9,  m: 38,  l: 71,  r: 8  },
    ],
    persona: [
      { t: 'P01', l: '번아웃 직전',   c: 77, p: '50%', bg: '#fde8ec', cl: '#d93954' },
      { t: 'P02', l: '보상 실망',     c: 12, p: '8%',  bg: '#e8f0fe', cl: '#2563eb' },
      { t: 'P03', l: '성장 정체',     c: 40, p: '26%', bg: '#fef3e2', cl: '#e8721a' },
      { t: 'P04', l: '보상체감 낮음', c: 26, p: '16%', bg: '#f3e8fd', cl: '#7c3aed' },
    ],
    perfRisk: [
      { grade: 'EP', 고위험: 20, 잠재위험: 15, 안정: 45 },
      { grade: 'HP', 고위험: 36, 잠재위험: 28, 안정: 62 },
      { grade: 'ME', 고위험: 17, 잠재위험: 22, 안정: 55 },
      { grade: 'IP', 고위험: 32, 잠재위험: 25, 안정: 70 },
      { grade: 'BE', 고위험: 12, 잠재위험: 18, 안정: 48 },
    ],
    riskDetail: [
      { dept: 'R&D',    grade: 'Senior Eng.', reason: '번아웃, 직무불만족',      score: 0.91, level: '고위험',   lc: '#d93954', lbg: '#fde8ec' },
      { dept: 'Sales',  grade: 'Account Mgr', reason: '관계 단절, 보상 불만',    score: 0.88, level: '고위험',   lc: '#d93954', lbg: '#fde8ec' },
      { dept: 'R&D',    grade: 'Lead Eng.',   reason: '성장기회 부족',           score: 0.85, level: '고위험',   lc: '#d93954', lbg: '#fde8ec' },
      { dept: 'Sales',  grade: 'Sales Rep.',  reason: '외부 시장 보상비교',      score: 0.79, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
      { dept: 'R&D',    grade: 'Engineer',    reason: '로그인 불규칙, 이직준비', score: 0.76, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
      { dept: 'HR',     grade: 'HR Specialist', reason: '관리구조 불안정',      score: 0.72, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
      { dept: 'Sales',  grade: 'Sales Mgr',   reason: '행동패턴 이상',          score: 0.69, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
      { dept: 'R&D',    grade: 'Researcher',  reason: '초과근무, 감성 부정',    score: 0.66, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
    ],
    trend: [
      { m: '24.10', h: 98,  med: 310 }, { m: '24.11', h: 105, med: 320 },
      { m: '24.12', h: 112, med: 330 }, { m: '25.01', h: 118, med: 340 },
      { m: '25.02', h: 125, med: 350 }, { m: '25.03', h: 130, med: 355 },
      { m: '25.04', h: 128, med: 360 }, { m: '25.05', h: 135, med: 365 },
      { m: '25.06', h: 140, med: 370 }, { m: '25.07', h: 145, med: 375 },
      { m: '25.08', h: 150, med: 380 }, { m: '25.09', h: 155, med: 387 },
    ],
    agentContrib: [
      { n: 'Structura', p: 32, c: '#d93954' },
      { n: 'Cognita',   p: 22, c: '#2563eb' },
      { n: 'Chronos',   p: 21, c: '#e8721a' },
      { n: 'Sentio',    p: 15, c: '#7c3aed' },
      { n: 'Agora',     p: 10, c: '#2ea44f' },
    ],
  },

  'Research & Development': {
    kpi: {
      total: '610', high: 80, med: 201, low: 329,
      avgScore: '0.40', highChange: '▲ 9%', medChange: '▲ 27%',
    },
    depts: [
      { n: 'Research & Development', c: 610, h: 80, m: 201, l: 329, r: 13 },
    ],
    persona: [
      { t: 'P01', l: '번아웃 직전',   c: 40, p: '51%', bg: '#fde8ec', cl: '#d93954' },
      { t: 'P02', l: '보상 실망',     c: 6,  p: '8%',  bg: '#e8f0fe', cl: '#2563eb' },
      { t: 'P03', l: '성장 정체',     c: 24, p: '30%', bg: '#fef3e2', cl: '#e8721a' },
      { t: 'P04', l: '보상체감 낮음', c: 8,  p: '10%', bg: '#f3e8fd', cl: '#7c3aed' },
    ],
    perfRisk: [
      { grade: 'EP', 고위험: 10, 잠재위험: 8,  안정: 24 },
      { grade: 'HP', 고위험: 19, 잠재위험: 14, 안정: 33 },
      { grade: 'ME', 고위험: 9,  잠재위험: 12, 안정: 28 },
      { grade: 'IP', 고위험: 17, 잠재위험: 13, 안정: 37 },
      { grade: 'BE', 고위험: 6,  잠재위험: 9,  안정: 24 },
    ],
    riskDetail: [
      { dept: 'R&D Core',    grade: 'Senior Eng.', reason: '번아웃, 직무불만족',      score: 0.91, level: '고위험',   lc: '#d93954', lbg: '#fde8ec' },
      { dept: 'R&D Applied', grade: 'Lead Eng.',   reason: '성장기회 부족',           score: 0.85, level: '고위험',   lc: '#d93954', lbg: '#fde8ec' },
      { dept: 'R&D Core',    grade: 'Engineer',    reason: '로그인 불규칙, 이직준비', score: 0.76, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
      { dept: 'R&D Applied', grade: 'Researcher',  reason: '초과근무, 감성 부정',    score: 0.66, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
    ],
    trend: [
      { m: '24.10', h: 51,  med: 161 }, { m: '24.11', h: 54,  med: 166 },
      { m: '24.12', h: 58,  med: 171 }, { m: '25.01', h: 61,  med: 176 },
      { m: '25.02', h: 65,  med: 182 }, { m: '25.03', h: 67,  med: 184 },
      { m: '25.04', h: 66,  med: 187 }, { m: '25.05', h: 70,  med: 190 },
      { m: '25.06', h: 73,  med: 193 }, { m: '25.07', h: 75,  med: 195 },
      { m: '25.08', h: 78,  med: 198 }, { m: '25.09', h: 80,  med: 201 },
    ],
    agentContrib: [
      { n: 'Chronos',   p: 34, c: '#e8721a' },
      { n: 'Structura', p: 28, c: '#d93954' },
      { n: 'Sentio',    p: 18, c: '#7c3aed' },
      { n: 'Cognita',   p: 14, c: '#2563eb' },
      { n: 'Agora',     p: 6,  c: '#2ea44f' },
    ],
  },

  'Sales': {
    kpi: {
      total: '446', high: 66, med: 148, low: 232,
      avgScore: '0.44', highChange: '▲ 18%', medChange: '▲ 35%',
    },
    depts: [
      { n: 'Sales', c: 446, h: 66, m: 148, l: 232, r: 15 },
    ],
    persona: [
      { t: 'P01', l: '번아웃 직전',   c: 28, p: '42%', bg: '#fde8ec', cl: '#d93954' },
      { t: 'P02', l: '보상 실망',     c: 5,  p: '8%',  bg: '#e8f0fe', cl: '#2563eb' },
      { t: 'P03', l: '성장 정체',     c: 12, p: '18%', bg: '#fef3e2', cl: '#e8721a' },
      { t: 'P04', l: '보상체감 낮음', c: 21, p: '32%', bg: '#f3e8fd', cl: '#7c3aed' },
    ],
    perfRisk: [
      { grade: 'EP', 고위험: 8,  잠재위험: 6,  안정: 16 },
      { grade: 'HP', 고위험: 14, 잠재위험: 11, 안정: 24 },
      { grade: 'ME', 고위험: 7,  잠재위험: 8,  안정: 20 },
      { grade: 'IP', 고위험: 13, 잠재위험: 9,  안정: 26 },
      { grade: 'BE', 고위험: 5,  잠재위험: 6,  안정: 18 },
    ],
    riskDetail: [
      { dept: 'Inside Sales',  grade: 'Account Mgr', reason: '관계 단절, 보상 불만',  score: 0.88, level: '고위험',   lc: '#d93954', lbg: '#fde8ec' },
      { dept: 'Field Sales',   grade: 'Sales Rep.',  reason: '외부 시장 보상비교',    score: 0.79, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
      { dept: 'Inside Sales',  grade: 'Sales Mgr',   reason: '행동패턴 이상',        score: 0.69, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
      { dept: 'Field Sales',   grade: 'Senior Rep.', reason: '보상체감 저하, 목표초과',score: 0.62, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
    ],
    trend: [
      { m: '24.10', h: 37,  med: 118 }, { m: '24.11', h: 40,  med: 122 },
      { m: '24.12', h: 42,  med: 125 }, { m: '25.01', h: 45,  med: 130 },
      { m: '25.02', h: 47,  med: 133 }, { m: '25.03', h: 49,  med: 135 },
      { m: '25.04', h: 48,  med: 137 }, { m: '25.05', h: 51,  med: 139 },
      { m: '25.06', h: 53,  med: 141 }, { m: '25.07', h: 55,  med: 143 },
      { m: '25.08', h: 57,  med: 145 }, { m: '25.09', h: 66,  med: 148 },
    ],
    agentContrib: [
      { n: 'Agora',     p: 38, c: '#2ea44f' },
      { n: 'Structura', p: 26, c: '#d93954' },
      { n: 'Sentio',    p: 18, c: '#7c3aed' },
      { n: 'Chronos',   p: 12, c: '#e8721a' },
      { n: 'Cognita',   p: 6,  c: '#2563eb' },
    ],
  },

  'Human Resources': {
    kpi: {
      total: '118', high: 9, med: 38, low: 71,
      avgScore: '0.34', highChange: '▲ 5%', medChange: '▲ 18%',
    },
    depts: [
      { n: 'Human Resources', c: 118, h: 9, m: 38, l: 71, r: 8 },
    ],
    persona: [
      { t: 'P01', l: '번아웃 직전',   c: 5,  p: '56%', bg: '#fde8ec', cl: '#d93954' },
      { t: 'P02', l: '보상 실망',     c: 1,  p: '11%', bg: '#e8f0fe', cl: '#2563eb' },
      { t: 'P03', l: '성장 정체',     c: 2,  p: '22%', bg: '#fef3e2', cl: '#e8721a' },
      { t: 'P04', l: '보상체감 낮음', c: 1,  p: '11%', bg: '#f3e8fd', cl: '#7c3aed' },
    ],
    perfRisk: [
      { grade: 'EP', 고위험: 2,  잠재위험: 2,  안정: 6  },
      { grade: 'HP', 고위험: 3,  잠재위험: 4,  안정: 9  },
      { grade: 'ME', 고위험: 1,  잠재위험: 2,  안정: 7  },
      { grade: 'IP', 고위험: 2,  잠재위험: 3,  안정: 10 },
      { grade: 'BE', 고위험: 1,  잠재위험: 2,  안정: 6  },
    ],
    riskDetail: [
      { dept: 'HR Ops',   grade: 'HR Specialist', reason: '관리구조 불안정',  score: 0.72, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
      { dept: 'HR BP',    grade: 'HR Manager',    reason: '번아웃, 과도한 감정노동', score: 0.68, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
      { dept: 'HR Ops',   grade: 'Recruiter',     reason: '감성 부정 키워드', score: 0.61, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
    ],
    trend: [
      { m: '24.10', h: 5,  med: 24 }, { m: '24.11', h: 5,  med: 25 },
      { m: '24.12', h: 6,  med: 26 }, { m: '25.01', h: 6,  med: 27 },
      { m: '25.02', h: 6,  med: 28 }, { m: '25.03', h: 7,  med: 28 },
      { m: '25.04', h: 7,  med: 29 }, { m: '25.05', h: 7,  med: 30 },
      { m: '25.06', h: 8,  med: 32 }, { m: '25.07', h: 8,  med: 34 },
      { m: '25.08', h: 8,  med: 36 }, { m: '25.09', h: 9,  med: 38 },
    ],
    agentContrib: [
      { n: 'Sentio',    p: 36, c: '#7c3aed' },
      { n: 'Structura', p: 28, c: '#d93954' },
      { n: 'Cognita',   p: 20, c: '#2563eb' },
      { n: 'Chronos',   p: 10, c: '#e8721a' },
      { n: 'Agora',     p: 6,  c: '#2ea44f' },
    ],
  },
};

// ──────────────────────────── Custom Tooltip ─────────────────────────────────
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#fff', border: '1px solid #eee', borderRadius: 8, padding: '8px 12px', fontSize: 12, boxShadow: '0 2px 8px rgba(0,0,0,.1)' }}>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || p.fill, marginBottom: 2 }}>
          {p.name}: <strong>{p.value}명</strong>
        </div>
      ))}
    </div>
  );
};

// ──────────────────────────── Home (인원현황) ─────────────────────────────────
const Home = ({ viewMode = 'all' }) => {
  const [deptFilter, setDeptFilter]   = useState('전체 부서');
  const [gradeFilter, setGradeFilter] = useState('전체 직급');
  const [riskFilter, setRiskFilter]   = useState('전체 위험등급');

  // 현재 view에 맞는 데이터 선택
  const data = useMemo(() => DEPT_DB[viewMode] || DEPT_DB['all'], [viewMode]);
  const { kpi, depts, persona, perfRisk, riskDetail, trend, agentContrib } = data;

  // 부서 레이블
  const deptLabel = viewMode === 'all' ? '전사' : viewMode;

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
  const th = { textAlign: 'left', padding: '10px 12px', background: C.bg, borderBottom: `2px solid ${C.border}`, fontWeight: 600, color: C.sub, fontSize: 12 };
  const td = { padding: '10px 12px', borderBottom: `1px solid ${C.border}` };

  // trend 데이터에 레이블 key 맞추기
  const trendData = trend.map(t => ({ month: t.m, 고위험: t.h, 잠재위험: t.med }));

  return (
    <div>
      {/* ── KPI Row ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 12, marginBottom: 20 }}>
        {[
          { label: `${deptLabel} 인원 현황`, val: kpi.total,                  unit: '명', color: C.text,    top: '#d93954', sub: '' },
          { label: '퇴사 고위험군',          val: kpi.high,                   unit: '명', color: '#d93954', top: '#d93954', sub: kpi.highChange },
          { label: '잠재적 위험군',          val: kpi.med,                    unit: '명', color: '#e8721a', top: '#e8721a', sub: kpi.medChange },
          { label: '안정/양호군',            val: kpi.low,                    unit: '명', color: '#2ea44f', top: '#2ea44f', sub: `${Math.round(kpi.low/parseInt(kpi.total.replace(',',''))*100)}%` },
          { label: '평균 위험 점수',         val: kpi.avgScore,               unit: '',   color: '#2563eb', top: '#2563eb', sub: '▲ vs 전분기' },
        ].map((k, i) => (
          <div key={i} style={{ ...cardS, textAlign: 'center', position: 'relative', overflow: 'hidden', padding: '16px 12px' }}>
            <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 4, background: k.top }} />
            <div style={{ fontSize: 11, color: C.sub, marginBottom: 6, fontWeight: 500 }}>{k.label}</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: k.color }}>
              {k.val}<span style={{ fontSize: 13, color: C.sub }}>{k.unit}</span>
            </div>
            <div style={{ fontSize: 10, color: '#aaa', marginTop: 3 }}>{k.sub}</div>
          </div>
        ))}
      </div>

      {/* ── Filter Bar ── */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center' }}>
        <span style={{ fontSize: 12, color: C.sub, fontWeight: 600 }}>필터:</span>
        {[
          { val: deptFilter,  set: setDeptFilter,  opts: ['전체 부서', 'Research & Development', 'Sales', 'Human Resources'] },
          { val: gradeFilter, set: setGradeFilter, opts: ['전체 직급', 'EP', 'HP', 'ME', 'IP', 'BE'] },
          { val: riskFilter,  set: setRiskFilter,  opts: ['전체 위험등급', '고위험', '잠재적 위험', '저위험'] },
        ].map((f, i) => (
          <select key={i} value={f.val} onChange={e => f.set(e.target.value)}
            style={{ padding: '7px 10px', border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12, fontFamily: 'inherit', background: C.card, color: C.text }}>
            {f.opts.map(o => <option key={o}>{o}</option>)}
          </select>
        ))}
      </div>

      {/* ── 부서별 현황 + Persona 분포 ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: 16, marginBottom: 20 }}>

        {/* 전사/부서 인원 현황 */}
        <div style={cardS}>
          {secTitle('☰', `${viewMode === 'all' ? '전사' : ''} 인원 현황 (부서별)`)}
          <div style={{ maxHeight: 240, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, minWidth: 440 }}>
              <thead>
                <tr>{['Department', '인원', '위험 분포', '고위험', '위험률'].map(h => <th key={h} style={th}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {depts.map(d => {
                  const bc     = d.r >= 15 ? '#d93954' : d.r >= 10 ? '#e8721a' : '#2ea44f';
                  const badgeBg = d.r >= 15 ? '#fde8ec' : d.r >= 10 ? '#fef3e2' : '#e6f6ec';
                  return (
                    <tr key={d.n} style={{ borderBottom: `1px solid ${C.border}` }}>
                      <td style={{ ...td, fontWeight: 500 }}>{d.n}</td>
                      <td style={td}>{d.c.toLocaleString()}명</td>
                      <td style={td}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ height: 8, borderRadius: 4, width: Math.min(d.r * 7, 120), minWidth: 4, background: bc }} />
                          <span style={{ fontSize: 11, color: C.sub }}>{d.r}%</span>
                        </div>
                      </td>
                      <td style={td}><strong style={{ color: '#d93954' }}>{d.h}명</strong></td>
                      <td style={td}>
                        <span style={{ padding: '3px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600, background: badgeBg, color: bc }}>{d.r}%</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* 위험 분포 요약 바 */}
          {viewMode !== 'all' && (
            <div style={{ marginTop: 14 }}>
              <div style={{ fontSize: 12, color: C.sub, marginBottom: 6 }}>위험 분포 (고위험 / 잠재 / 안정)</div>
              <div style={{ display: 'flex', height: 12, borderRadius: 6, overflow: 'hidden' }}>
                {[
                  { val: kpi.high, color: '#d93954' },
                  { val: kpi.med,  color: '#e8721a' },
                  { val: kpi.low,  color: '#2ea44f' },
                ].map((s, i) => {
                  const pct = Math.round(s.val / parseInt(kpi.total.replace(',','')) * 100);
                  return <div key={i} style={{ flex: pct, background: s.color, transition: 'flex 0.6s ease' }} title={`${pct}%`} />;
                })}
              </div>
              <div style={{ display: 'flex', gap: 12, marginTop: 6, fontSize: 11 }}>
                {[{label:'고위험',val:kpi.high,c:'#d93954'},{label:'잠재위험',val:kpi.med,c:'#e8721a'},{label:'안정',val:kpi.low,c:'#2ea44f'}].map(s => (
                  <span key={s.label} style={{ color: s.c, fontWeight: 600 }}>{s.label}: {s.val}명</span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Persona 분포 */}
        <div style={cardS}>
          {secTitle('◉', 'Persona 분포')}
          {/* Mini donut (SVG) */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 14 }}>
            <div style={{ position: 'relative', width: 90, height: 90, flexShrink: 0 }}>
              <svg viewBox="0 0 36 36" style={{ width: '100%', height: '100%', transform: 'rotate(-90deg)' }}>
                {(() => {
                  const colors  = ['#d93954', '#2563eb', '#e8721a', '#7c3aed'];
                  const total   = persona.reduce((s, p) => s + p.c, 0);
                  let offset    = 0;
                  return persona.map((p, i) => {
                    const pct = (p.c / total) * 100;
                    const el  = (
                      <circle key={i} cx="18" cy="18" r="15.9155" fill="transparent"
                        stroke={colors[i]} strokeWidth="3.8"
                        strokeDasharray={`${pct} ${100 - pct}`}
                        strokeDashoffset={-offset}
                      />
                    );
                    offset += pct;
                    return el;
                  });
                })()}
              </svg>
              <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#d93954' }}>
                  {Math.round(kpi.high / parseInt(kpi.total.replace(',','')) * 100)}%
                </div>
                <div style={{ fontSize: 9, color: C.sub }}>고위험</div>
              </div>
            </div>
            <div style={{ flex: 1 }}>
              {persona.map(p => (
                <div key={p.t} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 7 }}>
                  <span style={{ padding: '2px 7px', borderRadius: 4, fontSize: 10, fontWeight: 700, background: p.bg, color: p.cl, flexShrink: 0 }}>{p.t}</span>
                  <span style={{ fontSize: 11, flex: 1, color: C.text }}>{p.l}</span>
                  <span style={{ fontSize: 11, fontWeight: 600 }}>{p.c}명</span>
                  <span style={{ fontSize: 10, color: C.sub, width: 30, textAlign: 'right' }}>{p.p}</span>
                </div>
              ))}
            </div>
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr>{['유형','상세','분포','비율'].map(h => <th key={h} style={{ ...th, padding: 7 }}>{h}</th>)}</tr></thead>
            <tbody>
              {persona.map(p => (
                <tr key={p.t} style={{ borderBottom: `1px solid ${C.border}` }}>
                  <td style={{ padding: 7 }}><span style={{ padding: '2px 7px', borderRadius: 4, fontSize: 10, fontWeight: 600, background: p.bg, color: p.cl }}>{p.t}</span></td>
                  <td style={{ padding: 7, fontSize: 11 }}>{p.l}</td>
                  <td style={{ padding: 7, fontSize: 11 }}>{p.c}명</td>
                  <td style={{ padding: 7, fontSize: 11 }}>{p.p}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── 성과등급별 분포 + 퇴사위험자 상세 ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>

        {/* 성과등급별 퇴사위험자 분포 */}
        <div style={cardS}>
          {secTitle('▣', '성과등급별 퇴사위험자 분포')}
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={perfRisk} margin={{ top: 4, right: 8, bottom: 4, left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border,#f0f0f0)" vertical={false} />
              <XAxis dataKey="grade" tick={{ fontSize: 12, fontFamily: 'inherit' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fontFamily: 'inherit' }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'inherit', paddingTop: 6 }} />
              <Bar dataKey="고위험"  stackId="a" fill="#d93954" />
              <Bar dataKey="잠재위험" stackId="a" fill="#e8721a" />
              <Bar dataKey="안정"    stackId="a" fill="#e0e0e0" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* 성과등급별 퇴사위험자 상세 */}
        <div style={cardS}>
          {secTitle('⚠', '성과등급별 퇴사위험자 상세')}
          <div style={{ maxHeight: 250, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, minWidth: 380 }}>
              <thead>
                <tr>{['부서/팀', '직급', '예상 사유', '위험점수', '상태'].map(h => <th key={h} style={{ ...th, padding: '8px 10px' }}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {riskDetail.map((r, i) => (
                  <tr key={i} style={{ borderBottom: `1px solid ${C.border}` }}>
                    <td style={{ padding: '8px 10px', fontWeight: 500 }}>{r.dept}</td>
                    <td style={{ padding: '8px 10px' }}>{r.grade}</td>
                    <td style={{ padding: '8px 10px', color: C.sub, fontSize: 11 }}>{r.reason}</td>
                    <td style={{ padding: '8px 10px' }}><strong style={{ color: r.lc }}>{r.score.toFixed(2)}</strong></td>
                    <td style={{ padding: '8px 10px' }}>
                      <span style={{ padding: '3px 8px', borderRadius: 12, fontSize: 11, fontWeight: 600, background: r.lbg, color: r.lc }}>{r.level}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* ── 월별 추이 + Agent 기여도 ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

        {/* 월별 퇴사위험 추이 */}
        <div style={cardS}>
          {secTitle('▬', '월별 퇴사위험 추이')}
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={trendData} margin={{ top: 4, right: 8, bottom: 4, left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border,#f0f0f0)" />
              <XAxis dataKey="month" tick={{ fontSize: 10, fontFamily: 'inherit' }} axisLine={false} tickLine={false} angle={-30} textAnchor="end" height={40} />
              <YAxis tick={{ fontSize: 11, fontFamily: 'inherit' }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'inherit', paddingTop: 6 }} />
              <Line type="monotone" dataKey="고위험"  stroke="#d93954" strokeWidth={2} dot={{ r: 3, fill: '#d93954' }} activeDot={{ r: 5 }} />
              <Line type="monotone" dataKey="잠재위험" stroke="#e8721a" strokeWidth={1.5} strokeDasharray="5 5" dot={{ r: 2, fill: '#e8721a' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Agent별 위험 탐지 기여도 */}
        <div style={cardS}>
          {secTitle('⚙', 'Agent별 위험 탐지 기여도')}
          {viewMode !== 'all' && (
            <div style={{ fontSize: 11, color: '#888', marginBottom: 10, fontStyle: 'italic' }}>
              {deptLabel} 특화 — {
                viewMode === 'Research & Development' ? '행동 이상 탐지(Chronos) 비중 높음' :
                viewMode === 'Sales'                  ? '외부 시장 이탈(Agora) 비중 높음' :
                                                        '감성 분석(Sentio) 비중 높음'
              }
            </div>
          )}
          {agentContrib.map(a => (
            <div key={a.n} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 13 }}>
              <span style={{ width: 72, fontSize: 12, fontWeight: 700, textAlign: 'right', color: a.c }}>{a.n}</span>
              <div style={{ flex: 1, height: 24, background: C.border, borderRadius: 6, overflow: 'hidden' }}>
                <div style={{
                  height: '100%', width: `${a.p * 2.5}%`, background: a.c, borderRadius: 6,
                  display: 'flex', alignItems: 'center', paddingLeft: 8,
                  fontSize: 11, color: '#fff', fontWeight: 700, transition: 'width 0.8s ease',
                }}>
                  {a.p}%
                </div>
              </div>
              <span style={{ fontSize: 11, color: C.sub, width: 32 }}>{a.p}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Home;
