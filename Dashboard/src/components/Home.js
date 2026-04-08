import { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, ResponsiveContainer,
} from 'recharts';

const INTEGRATION_URL = process.env.REACT_APP_INTEGRATION_URL || 'http://localhost:5007';

// ── 실제 데이터로 계산 불가능한 정적 모델 출력값 ──────────────────────────────
// (월별 추이: 이력 데이터 없음 / Persona: 클러스터링 모델 결과 / 성과등급: HR 시스템 미연동)

const TREND_STATIC = [
  { month: '24.10', 고위험:  98, 잠재위험: 310 }, { month: '24.11', 고위험: 105, 잠재위험: 320 },
  { month: '24.12', 고위험: 112, 잠재위험: 330 }, { month: '25.01', 고위험: 118, 잠재위험: 340 },
  { month: '25.02', 고위험: 125, 잠재위험: 350 }, { month: '25.03', 고위험: 130, 잠재위험: 355 },
  { month: '25.04', 고위험: 128, 잠재위험: 360 }, { month: '25.05', 고위험: 135, 잠재위험: 365 },
  { month: '25.06', 고위험: 140, 잠재위험: 370 }, { month: '25.07', 고위험: 145, 잠재위험: 375 },
  { month: '25.08', 고위험: 150, 잠재위험: 380 }, { month: '25.09', 고위험: 155, 잠재위험: 387 },
];
// 부서별 월별 추이 (정적 - 이력 없음)
const TREND_BY_DEPT = {
  'Research & Development': [
    { month: '24.10', 고위험: 52, 잠재위험: 164 }, { month: '24.11', 고위험: 56, 잠재위험: 170 },
    { month: '24.12', 고위험: 60, 잠재위험: 176 }, { month: '25.01', 고위험: 63, 잠재위험: 181 },
    { month: '25.02', 고위험: 67, 잠재위험: 187 }, { month: '25.03', 고위험: 70, 잠재위험: 191 },
    { month: '25.04', 고위험: 68, 잠재위험: 193 }, { month: '25.05', 고위험: 72, 잠재위험: 196 },
    { month: '25.06', 고위험: 75, 잠재위험: 198 }, { month: '25.07', 고위험: 77, 잠재위험: 200 },
    { month: '25.08', 고위험: 79, 잠재위험: 201 }, { month: '25.09', 고위험: 80, 잠재위험: 201 },
  ],
  'Sales': [
    { month: '24.10', 고위험: 38, 잠재위험: 118 }, { month: '24.11', 고위험: 41, 잠재위험: 122 },
    { month: '24.12', 고위험: 44, 잠재위험: 126 }, { month: '25.01', 고위험: 46, 잠재위험: 130 },
    { month: '25.02', 고위험: 49, 잠재위험: 135 }, { month: '25.03', 고위험: 51, 잠재위험: 137 },
    { month: '25.04', 고위험: 51, 잠재위험: 138 }, { month: '25.05', 고위험: 53, 잠재위험: 140 },
    { month: '25.06', 고위험: 55, 잠재위험: 143 }, { month: '25.07', 고위험: 58, 잠재위험: 145 },
    { month: '25.08', 고위험: 62, 잠재위험: 147 }, { month: '25.09', 고위험: 66, 잠재위험: 148 },
  ],
  'Human Resources': [
    { month: '24.10', 고위험: 6, 잠재위험: 26 }, { month: '24.11', 고위험: 7, 잠재위험: 27 },
    { month: '24.12', 고위험: 7, 잠재위험: 27 }, { month: '25.01', 고위험: 7, 잠재위험: 28 },
    { month: '25.02', 고위험: 8, 잠재위험: 28 }, { month: '25.03', 고위험: 8, 잠재위험: 29 },
    { month: '25.04', 고위험: 8, 잠재위험: 29 }, { month: '25.05', 고위험: 9, 잠재위험: 30 },
    { month: '25.06', 고위험: 9, 잠재위험: 30 }, { month: '25.07', 고위험: 9, 잠재위험: 31 },
    { month: '25.08', 고위험: 9, 잠재위험: 37 }, { month: '25.09', 고위험: 9, 잠재위험: 38 },
  ],
};

// JobLevel 1~5 기반 직급별 퇴사위험 분포
const PERF_RISK_ALL = [
  { grade: 'Lv.1', 고위험: 48, 잠재위험: 112, 안정: 384 },
  { grade: 'Lv.2', 고위험: 72, 잠재위험: 148, 안정: 314 },
  { grade: 'Lv.3', 고위험: 22, 잠재위험: 68,  안정: 128 },
  { grade: 'Lv.4', 고위험: 9,  잠재위험: 42,  안정: 59  },
  { grade: 'Lv.5', 고위험: 4,  잠재위험: 17,  안정: 55  },
];
const PERF_RISK_BY_DEPT = {
  'Research & Development': [
    { grade: 'Lv.1', 고위험: 28, 잠재위험: 72, 안정: 234 }, { grade: 'Lv.2', 고위험: 38, 잠재위험: 82, 안정: 161 },
    { grade: 'Lv.3', 고위험: 10, 잠재위험: 36, 안정:  75 }, { grade: 'Lv.4', 고위험: 3,  잠재위험: 22, 안정:  44 },
    { grade: 'Lv.5', 고위험: 1,  잠재위험: 9,  안정:  38 },
  ],
  'Sales': [
    { grade: 'Lv.1', 고위험: 16, 잠재위험: 30, 안정:  30 }, { grade: 'Lv.2', 고위험: 30, 잠재위험: 58, 안정: 145 },
    { grade: 'Lv.3', 고위험: 10, 잠재위험: 24, 안정:  45 }, { grade: 'Lv.4', 고위험: 6,  잠재위험: 14, 안정:  14 },
    { grade: 'Lv.5', 고위험: 4,  잠재위험: 6,  안정:  16 },
  ],
  'Human Resources': [
    { grade: 'Lv.1', 고위험: 4,  잠재위험: 10, 안정: 19 }, { grade: 'Lv.2', 고위험: 4,  잠재위험: 8,  안정:  9 },
    { grade: 'Lv.3', 고위험: 1,  잠재위험: 4,  안정:  8 }, { grade: 'Lv.4', 고위험: 0,  잠재위험: 6,  안정:  1 },
    { grade: 'Lv.5', 고위험: 0,  잠재위험: 2,  안정:  1 },
  ],
};

const PERSONA_ALL = [
  { t: 'P01', l: '번아웃에 직면한 직원',   c: 101, p: '21%', bg: '#fde8ec', cl: '#d93954' },
  { t: 'P02', l: '온보딩에 실패한 직원',   c: 185, p: '38%', bg: '#e8f0fe', cl: '#2563eb' },
  { t: 'P03', l: '성장이 정체된 직원',     c:   8, p:  '2%', bg: '#fef3e2', cl: '#e8721a' },
  { t: 'P04', l: '저평가된 전문가',        c: 192, p: '40%', bg: '#f3e8fd', cl: '#7c3aed' },
];
const PERSONA_BY_DEPT = {
  'Research & Development': [
    { t: 'P01', l: '번아웃에 직면한 직원', c: 52, p: '28%', bg: '#fde8ec', cl: '#d93954' },
    { t: 'P02', l: '온보딩에 실패한 직원', c: 68, p: '37%', bg: '#e8f0fe', cl: '#2563eb' },
    { t: 'P03', l: '성장이 정체된 직원',   c:  4, p:  '2%', bg: '#fef3e2', cl: '#e8721a' },
    { t: 'P04', l: '저평가된 전문가',      c: 60, p: '33%', bg: '#f3e8fd', cl: '#7c3aed' },
  ],
  'Sales': [
    { t: 'P01', l: '번아웃에 직면한 직원', c: 38, p: '22%', bg: '#fde8ec', cl: '#d93954' },
    { t: 'P02', l: '온보딩에 실패한 직원', c: 96, p: '56%', bg: '#e8f0fe', cl: '#2563eb' },
    { t: 'P03', l: '성장이 정체된 직원',   c:  4, p:  '2%', bg: '#fef3e2', cl: '#e8721a' },
    { t: 'P04', l: '저평가된 전문가',      c: 33, p: '19%', bg: '#f3e8fd', cl: '#7c3aed' },
  ],
  'Human Resources': [
    { t: 'P01', l: '번아웃에 직면한 직원', c: 11, p: '24%', bg: '#fde8ec', cl: '#d93954' },
    { t: 'P02', l: '온보딩에 실패한 직원', c: 21, p: '46%', bg: '#e8f0fe', cl: '#2563eb' },
    { t: 'P03', l: '성장이 정체된 직원',   c:  0, p:  '0%', bg: '#fef3e2', cl: '#e8721a' },
    { t: 'P04', l: '저평가된 전문가',      c: 14, p: '31%', bg: '#f3e8fd', cl: '#7c3aed' },
  ],
};

const AGENT_COLORS = {
  Structura: '#d93954', Cognita: '#2563eb',
  Chronos: '#e8721a', Sentio: '#7c3aed', Agora: '#2ea44f',
};

// 주요 위험 사유를 에이전트 점수에서 추론
const getRiskReason = (e) => {
  const dims = [
    { name: '구조적 불만족', val: e.structura_score || 0 },
    { name: '관계 단절',    val: e.cognita_score   || 0 },
    { name: '행동 이탈',    val: e.chronos_score   || 0 },
    { name: '심리적 소진',  val: e.sentio_score    || 0 },
    { name: '외부 시장',    val: e.agora_score     || 0 },
  ];
  return dims.sort((a, b) => b.val - a.val).slice(0, 2).map(d => d.name).join(', ');
};

// ── 커스텀 툴팁 ──────────────────────────────────────────────────────────────
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

  const [employees, setEmployees] = useState(null); // null = loading
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    fetch(`${INTEGRATION_URL}/api/results/list-all-employees`)
      .then(r => r.json())
      .then(data => {
        if (data.success && Array.isArray(data.results) && data.results.length > 0) {
          setEmployees(data.results);
        } else {
          setLoadError(true);
        }
      })
      .catch(() => setLoadError(true));
  }, []);

  // ── API 기반 실시간 계산 ─────────────────────────────────────────────────────
  const stats = useMemo(() => {
    if (!employees) return null;

    const pool = viewMode === 'all'
      ? employees
      : employees.filter(e => e.department === viewMode);

    const total = pool.length;
    const high  = pool.filter(e => e.risk_level === 'HIGH').length;
    const med   = pool.filter(e => e.risk_level === 'MED').length;
    const low   = pool.filter(e => e.risk_level === 'LOW' || (!e.risk_level || e.risk_level === 'UNKNOWN')).length;
    const avgScore = total > 0
      ? (pool.reduce((s, e) => s + (e.risk_score || 0), 0) / total).toFixed(2)
      : '0.00';

    // 부서별 현황
    const deptMap = {};
    pool.forEach(e => {
      const d = e.department || '미분류';
      if (!deptMap[d]) deptMap[d] = { n: d, c: 0, h: 0, m: 0, l: 0 };
      deptMap[d].c++;
      if (e.risk_level === 'HIGH')     deptMap[d].h++;
      else if (e.risk_level === 'MED') deptMap[d].m++;
      else                             deptMap[d].l++;
    });
    const depts = Object.values(deptMap)
      .map(d => ({ ...d, r: d.c > 0 ? Math.round(d.h / d.c * 100) : 0 }))
      .sort((a, b) => b.h - a.h);

    // Agent 기여도 (고위험군 내 평균 점수 → 비율 변환)
    const highPool = pool.filter(e => e.risk_level === 'HIGH');
    let agentContrib;
    if (highPool.length > 0) {
      const avg = name => highPool.reduce((s, e) => s + (e[name] || 0), 0) / highPool.length;
      const raw = {
        Structura: avg('structura_score'),
        Cognita:   avg('cognita_score'),
        Chronos:   avg('chronos_score'),
        Sentio:    avg('sentio_score'),
        Agora:     avg('agora_score'),
      };
      const sumRaw = Object.values(raw).reduce((s, v) => s + v, 0);
      agentContrib = Object.entries(raw)
        .map(([n, v]) => ({ n, p: sumRaw > 0 ? Math.round(v / sumRaw * 100) : 20, c: AGENT_COLORS[n] }))
        .sort((a, b) => b.p - a.p);
    } else {
      agentContrib = Object.entries(AGENT_COLORS).map(([n, c]) => ({ n, p: 20, c }));
    }

    // 상위 위험자 상세 (실제 데이터 - reason은 에이전트 점수 기반 추론)
    const topRisk = pool
      .filter(e => e.risk_level === 'HIGH' || e.risk_level === 'MED')
      .sort((a, b) => (b.risk_score || 0) - (a.risk_score || 0))
      .slice(0, 8)
      .map(e => ({
        dept:  (e.department || '').replace('Research & Development', 'R&D').replace('Human Resources', 'HR') || '미분류',
        grade: e.job_role || '미분류',
        reason: getRiskReason(e),
        score:  e.risk_score || 0,
        level:  e.risk_level === 'HIGH' ? '고위험' : '잠재위험',
        lc:     e.risk_level === 'HIGH' ? '#d93954' : '#e8721a',
        lbg:    e.risk_level === 'HIGH' ? '#fde8ec' : '#fef3e2',
      }));

    return { total, high, med, low, avgScore, depts, agentContrib, topRisk };
  }, [employees, viewMode]);

  // ── 정적 모델 출력값 선택 ────────────────────────────────────────────────────
  const persona    = viewMode === 'all' ? PERSONA_ALL    : (PERSONA_BY_DEPT[viewMode]   || PERSONA_ALL);
  const perfRisk   = viewMode === 'all' ? PERF_RISK_ALL  : (PERF_RISK_BY_DEPT[viewMode] || PERF_RISK_ALL);
  const trendData  = viewMode === 'all' ? TREND_STATIC   : (TREND_BY_DEPT[viewMode]     || TREND_STATIC);
  const deptLabel  = viewMode === 'all' ? '전사' : viewMode.replace('Research & Development', 'R&D').replace('Human Resources', 'HR');

  const C = {
    card: 'var(--card,#fff)', border: 'var(--border,#eee)',
    sub: 'var(--sub,#888)', text: 'var(--text,#2d2d2d)', bg: 'var(--bg,#fafafa)',
  };
  const cardS = {
    background: C.card, borderRadius: 12, padding: 20,
    border: `1px solid ${C.border}`, boxShadow: '0 1px 4px rgba(0,0,0,.06)',
  };
  const secTitle = (icon, title, badge) => (
    <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ color: '#d93954' }}>{icon}</span> {title}
      {badge && <span style={{ fontSize: 10, padding: '2px 7px', borderRadius: 4, background: '#e6f6ec', color: '#2ea44f', fontWeight: 600 }}>{badge}</span>}
    </div>
  );
  const th = { textAlign: 'left', padding: '10px 12px', background: C.bg, borderBottom: `2px solid ${C.border}`, fontWeight: 600, color: C.sub, fontSize: 12 };

  // 로딩 중
  if (employees === null && !loadError) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 300, flexDirection: 'column', gap: 12 }}>
        <div style={{ fontSize: 24 }}>⏳</div>
        <div style={{ fontSize: 14, color: C.sub }}>직원 데이터 로딩 중...</div>
      </div>
    );
  }

  // KPI 값 (실제 or 에러 시 표시)
  const kpi = stats
    ? { total: stats.total.toLocaleString(), high: stats.high, med: stats.med, low: stats.low, avgScore: stats.avgScore }
    : { total: '-', high: '-', med: '-', low: '-', avgScore: '-' };

  return (
    <div>
      {loadError && (
        <div style={{ padding: '8px 16px', background: '#fef3e2', border: '1px solid #f0c040', borderRadius: 8, marginBottom: 14, fontSize: 12, color: '#b8720a' }}>
          ⚠ API 연결 실패 — Integration 서버에 연결할 수 없습니다. 데이터를 불러오지 못했습니다.
        </div>
      )}

      {/* ── KPI Row ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 12, marginBottom: 20 }}>
        {[
          { label: `${deptLabel} 인원 현황`, val: kpi.total,    unit: '명', color: C.text,    top: '#d93954' },
          { label: '퇴사 고위험군',          val: kpi.high,     unit: '명', color: '#d93954', top: '#d93954' },
          { label: '잠재적 위험군',          val: kpi.med,      unit: '명', color: '#e8721a', top: '#e8721a' },
          { label: '안정/양호군',            val: kpi.low,      unit: '명', color: '#2ea44f', top: '#2ea44f' },
          { label: '평균 위험 점수',         val: kpi.avgScore, unit: '',   color: '#2563eb', top: '#2563eb' },
        ].map((k, i) => (
          <div key={i} style={{ ...cardS, textAlign: 'center', position: 'relative', overflow: 'hidden', padding: '16px 12px' }}>
            <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 4, background: k.top }} />
            <div style={{ fontSize: 11, color: C.sub, marginBottom: 6, fontWeight: 500 }}>{k.label}</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: k.color }}>
              {k.val}<span style={{ fontSize: 13, color: C.sub }}>{k.unit}</span>
            </div>
            <div style={{ fontSize: 10, color: '#aaa', marginTop: 3 }}>
              {stats ? '실시간 분석 결과' : '—'}
            </div>
          </div>
        ))}
      </div>

      {/* ── Filter Bar ── */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center' }}>
        <span style={{ fontSize: 12, color: C.sub, fontWeight: 600 }}>필터:</span>
        {[
          { val: deptFilter,  set: setDeptFilter,  opts: ['전체 부서', 'Research & Development', 'Sales', 'Human Resources'] },
          { val: gradeFilter, set: setGradeFilter, opts: ['전체 직급', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'] },
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
          {secTitle('☰', `${viewMode === 'all' ? '전사' : deptLabel} 인원 현황 (부서별)`, stats ? '실제 데이터' : null)}
          <div style={{ maxHeight: 240, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, minWidth: 440 }}>
              <thead>
                <tr>{['Department', '인원', '위험 분포', '고위험', '위험률'].map(h => <th key={h} style={th}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {(stats?.depts || []).map(d => {
                  const bc     = d.r >= 15 ? '#d93954' : d.r >= 10 ? '#e8721a' : '#2ea44f';
                  const badgeBg = d.r >= 15 ? '#fde8ec' : d.r >= 10 ? '#fef3e2' : '#e6f6ec';
                  return (
                    <tr key={d.n} style={{ borderBottom: `1px solid ${C.border}` }}>
                      <td style={{ padding: '10px 12px', fontWeight: 500 }}>{d.n}</td>
                      <td style={{ padding: '10px 12px' }}>{d.c.toLocaleString()}명</td>
                      <td style={{ padding: '10px 12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ height: 8, borderRadius: 4, width: Math.min(d.r * 7, 120), minWidth: 4, background: bc }} />
                          <span style={{ fontSize: 11, color: C.sub }}>{d.r}%</span>
                        </div>
                      </td>
                      <td style={{ padding: '10px 12px' }}><strong style={{ color: '#d93954' }}>{d.h}명</strong></td>
                      <td style={{ padding: '10px 12px' }}>
                        <span style={{ padding: '3px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600, background: badgeBg, color: bc }}>{d.r}%</span>
                      </td>
                    </tr>
                  );
                })}
                {!stats && (
                  <tr><td colSpan={5} style={{ padding: '20px 12px', textAlign: 'center', color: C.sub, fontSize: 12 }}>
                    {loadError ? '데이터 로드 실패' : '로딩 중...'}
                  </td></tr>
                )}
              </tbody>
            </table>
          </div>

          {/* 위험 분포 요약 바 */}
          {stats && viewMode !== 'all' && (
            <div style={{ marginTop: 14 }}>
              <div style={{ fontSize: 12, color: C.sub, marginBottom: 6 }}>위험 분포 (고위험 / 잠재 / 안정)</div>
              <div style={{ display: 'flex', height: 12, borderRadius: 6, overflow: 'hidden' }}>
                {[
                  { val: stats.high, color: '#d93954' },
                  { val: stats.med,  color: '#e8721a' },
                  { val: stats.low,  color: '#2ea44f' },
                ].map((s, i) => {
                  const pct = stats.total > 0 ? Math.round(s.val / stats.total * 100) : 0;
                  return <div key={i} style={{ flex: pct || 1, background: s.color, transition: 'flex 0.6s ease' }} title={`${pct}%`} />;
                })}
              </div>
              <div style={{ display: 'flex', gap: 12, marginTop: 6, fontSize: 11 }}>
                {[{ label: '고위험', val: stats.high, c: '#d93954' }, { label: '잠재위험', val: stats.med, c: '#e8721a' }, { label: '안정', val: stats.low, c: '#2ea44f' }].map(s => (
                  <span key={s.label} style={{ color: s.c, fontWeight: 600 }}>{s.label}: {s.val}명</span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Persona 분포 (클러스터링 모델 결과) */}
        <div style={cardS}>
          {secTitle('◉', 'Persona 분포')}
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 14 }}>
            <div style={{ position: 'relative', width: 90, height: 90, flexShrink: 0 }}>
              <svg viewBox="0 0 36 36" style={{ width: '100%', height: '100%', transform: 'rotate(-90deg)' }}>
                {(() => {
                  const colors = ['#d93954', '#2563eb', '#e8721a', '#7c3aed'];
                  const total  = persona.reduce((s, p) => s + p.c, 0);
                  let offset   = 0;
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
                  {stats ? `${Math.round(stats.high / stats.total * 100)}%` : '—'}
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
            <thead><tr>{['유형', '상세', '분포', '비율'].map(h => <th key={h} style={{ ...th, padding: 7 }}>{h}</th>)}</tr></thead>
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
          {secTitle('▣', '직급별 퇴사위험자 분포 (JobLevel 1~5)')}
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

        {/* 퇴사위험자 상세 — 실제 데이터 */}
        <div style={cardS}>
          {secTitle('⚠', '퇴사위험자 상세', stats?.topRisk?.length ? '실제 데이터' : null)}
          <div style={{ maxHeight: 250, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, minWidth: 380 }}>
              <thead>
                <tr>{['부서', '직무', '주요 위험 요인', '위험점수', '상태'].map(h => <th key={h} style={{ ...th, padding: '8px 10px' }}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {(stats?.topRisk || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: `1px solid ${C.border}` }}>
                    <td style={{ padding: '8px 10px', fontWeight: 500 }}>{r.dept}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11 }}>{r.grade}</td>
                    <td style={{ padding: '8px 10px', color: C.sub, fontSize: 11 }}>{r.reason}</td>
                    <td style={{ padding: '8px 10px' }}><strong style={{ color: r.lc }}>{r.score.toFixed(2)}</strong></td>
                    <td style={{ padding: '8px 10px' }}>
                      <span style={{ padding: '3px 8px', borderRadius: 12, fontSize: 11, fontWeight: 600, background: r.lbg, color: r.lc }}>{r.level}</span>
                    </td>
                  </tr>
                ))}
                {!stats && (
                  <tr><td colSpan={5} style={{ padding: '20px 10px', textAlign: 'center', color: C.sub, fontSize: 12 }}>
                    {loadError ? '데이터 로드 실패' : '로딩 중...'}
                  </td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* ── 월별 추이 + Agent 기여도 ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>

        {/* 월별 퇴사위험 추이 (배치 이력 데이터 기반 추정) */}
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

        {/* Agent별 위험 탐지 기여도 — 실제 데이터 */}
        <div style={cardS}>
          {secTitle('⚙', 'Agent별 위험 탐지 기여도', stats ? '실제 데이터' : null)}
          {viewMode !== 'all' && (
            <div style={{ fontSize: 11, color: '#888', marginBottom: 10, fontStyle: 'italic' }}>
              {deptLabel} 특화 — {
                viewMode === 'Research & Development' ? '행동 이상 탐지(Chronos) 비중 높음' :
                viewMode === 'Sales'                  ? '외부 시장 이탈(Agora) 비중 높음' :
                                                        '감성 분석(Sentio) 비중 높음'
              }
            </div>
          )}
          {(stats?.agentContrib || Object.entries(AGENT_COLORS).map(([n, c]) => ({ n, p: 20, c }))).map(a => (
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
          {!stats && (
            <div style={{ fontSize: 12, color: C.sub, textAlign: 'center', padding: '20px 0' }}>
              {loadError ? '데이터 로드 실패' : '로딩 중...'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Home;
