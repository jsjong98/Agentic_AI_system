import { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, ResponsiveContainer,
} from 'recharts';

// ──────────────────────────── Static Data ────────────────────────────────────
const DEPTS = [
  { n: 'Consulting LT',        c: 35,  h: 6,  r: 17 },
  { n: 'Strategy&',            c: 125, h: 28, r: 22 },
  { n: 'Fin. Transformation',  c: 145, h: 15, r: 10 },
  { n: 'Operation',            c: 125, h: 32, r: 26 },
  { n: 'Industry Solution',    c: 126, h: 14, r: 11 },
  { n: 'Customer',             c: 110, h: 18, r: 16 },
  { n: 'Technology',           c: 240, h: 19, r: 8  },
  { n: 'D&AI',                 c: 130, h: 12, r: 9  },
  { n: 'Financial Services',   c: 108, h: 8,  r: 7  },
  { n: 'Risk & Cyber',         c: 30,  h: 3,  r: 10 },
];

const PERSONA = [
  { t: 'P01', l: '번아웃 직전',   c: 77,  p: '50%', bg: '#fde8ec', cl: '#d93954' },
  { t: 'P02', l: '보상 실망',     c: 12,  p: '8%',  bg: '#e8f0fe', cl: '#2563eb' },
  { t: 'P03', l: '성장 정체',     c: 40,  p: '26%', bg: '#fef3e2', cl: '#e8721a' },
  { t: 'P04', l: '보상체감 낮음', c: 26,  p: '16%', bg: '#f3e8fd', cl: '#7c3aed' },
];

const PERF_RISK = [
  { grade: 'EP', 고위험: 20, 잠재위험: 15, 안정: 45 },
  { grade: 'HP', 고위험: 36, 잠재위험: 28, 안정: 62 },
  { grade: 'ME', 고위험: 17, 잠재위험: 22, 안정: 55 },
  { grade: 'IP', 고위험: 32, 잠재위험: 25, 안정: 70 },
  { grade: 'BE', 고위험: 12, 잠재위험: 18, 안정: 48 },
];

const RISK_DETAIL = [
  { dept: 'Strategy&',         grade: 'S&',      reason: '업무 소진, 경력 정체',    score: 0.91, level: '고위험',   lc: '#d93954', lbg: '#fde8ec' },
  { dept: 'Operation',         grade: 'Manager',  reason: '관계 단절, 동기저하',     score: 0.88, level: '고위험',   lc: '#d93954', lbg: '#fde8ec' },
  { dept: 'Customer',          grade: 'Director', reason: '외부 시장 보상비교',      score: 0.85, level: '고위험',   lc: '#d93954', lbg: '#fde8ec' },
  { dept: 'Technology',        grade: 'S&',       reason: '번아웃, 직무불만족',      score: 0.79, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
  { dept: 'D&AI',              grade: 'Manager',  reason: '성장기회 부족',           score: 0.76, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
  { dept: 'Consulting LT',     grade: 'S&',       reason: '보상 불만, 이직준비',     score: 0.74, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
  { dept: 'Fin. Transformation', grade: 'Director', reason: '관리구조 불안정',      score: 0.71, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
  { dept: 'Risk & Cyber',      grade: 'Manager',  reason: '행동패턴 이상',           score: 0.68, level: '잠재위험', lc: '#e8721a', lbg: '#fef3e2' },
];

const MONTHS = ['24.10','24.11','24.12','25.01','25.02','25.03','25.04','25.05','25.06','25.07','25.08','25.09'];
const RISK_TREND = MONTHS.map((m, i) => ({
  month: m,
  고위험:    [98,105,112,118,125,130,128,135,140,145,150,155][i],
  잠재위험:  [310,320,330,340,350,355,360,365,370,375,380,387][i],
}));

const AGENT_CONTRIB = [
  { n: 'Structura', p: 32, c: '#d93954' },
  { n: 'Cognita',   p: 22, c: '#2563eb' },
  { n: 'Chronos',   p: 21, c: '#e8721a' },
  { n: 'Sentio',    p: 15, c: '#7c3aed' },
  { n: 'Agora',     p: 10, c: '#2ea44f' },
];

// ──────────────────────────── Custom Tooltip ─────────────────────────────────
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#fff', border: '1px solid #eee', borderRadius: 8, padding: '8px 12px', fontSize: 12, boxShadow: '0 2px 8px rgba(0,0,0,.1)' }}>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, marginBottom: 2 }}>
          {p.name}: <strong>{p.value}명</strong>
        </div>
      ))}
    </div>
  );
};

// ──────────────────────────── Home (인원현황) ─────────────────────────────────
const Home = () => {
  const [deptFilter, setDeptFilter] = useState('전체 부서');
  const [gradeFilter, setGradeFilter] = useState('전체 직급');
  const [riskFilter, setRiskFilter] = useState('전체 위험등급');

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

  return (
    <div>
      {/* ── KPI Row ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 12, marginBottom: 20 }}>
        {[
          { label: '전사 인원 현황', val: '1,174', unit: '명', color: C.text, top: '#d93954', sub: '전년 대비 +32명' },
          { label: '퇴사 고위험군', val: '155',   unit: '명', color: '#d93954', top: '#d93954', sub: '▲ 12%' },
          { label: '잠재적 위험군', val: '387',   unit: '명', color: '#e8721a', top: '#e8721a', sub: '▲ 30%' },
          { label: '안정/양호군',   val: '752',   unit: '명', color: '#2ea44f', top: '#2ea44f', sub: '58%' },
          { label: '평균 위험 점수', val: '0.42', unit: '',   color: '#2563eb', top: '#2563eb', sub: '▲ 0.05 vs 전분기' },
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
          { val: deptFilter,  set: setDeptFilter,  opts: ['전체 부서','Consulting LT','Strategy&','Fin. Transformation','Operation','Industry Solution','Customer','Technology','D&AI','Financial Services','Risk & Cyber'] },
          { val: gradeFilter, set: setGradeFilter, opts: ['전체 직급','EP','HP','ME','IP','BE'] },
          { val: riskFilter,  set: setRiskFilter,  opts: ['전체 위험등급','고위험','잠재적 위험','저위험'] },
        ].map((f, i) => (
          <select key={i} value={f.val} onChange={e => f.set(e.target.value)}
            style={{ padding: '7px 10px', border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12, fontFamily: 'inherit', background: C.card, color: C.text }}>
            {f.opts.map(o => <option key={o}>{o}</option>)}
          </select>
        ))}
      </div>

      {/* ── 부서별 현황 + Persona 분포 ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: 16, marginBottom: 20 }}>

        {/* 전사 인원 현황 (부서별) */}
        <div style={cardS}>
          {secTitle('☰', '전사 인원 현황 (부서별)')}
          <div style={{ maxHeight: 320, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, minWidth: 480 }}>
              <thead>
                <tr>{['Department','인원','위험 분포','고위험','위험률'].map(h => <th key={h} style={th}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {DEPTS.map(d => {
                  const bc = d.r >= 16 ? '#d93954' : d.r >= 11 ? '#e8721a' : '#2ea44f';
                  const badgeBg = d.r >= 16 ? '#fde8ec' : d.r >= 11 ? '#fef3e2' : '#e6f6ec';
                  return (
                    <tr key={d.n} style={{ borderBottom: `1px solid ${C.border}` }}>
                      <td style={{ ...td, fontWeight: 500 }}>{d.n}</td>
                      <td style={td}>{d.c}명</td>
                      <td style={td}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ height: 8, borderRadius: 4, width: d.r * 6, minWidth: 4, background: bc }} />
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
        </div>

        {/* Persona 분포 */}
        <div style={cardS}>
          {secTitle('◉', 'Persona 분포')}
          {/* Donut (CSS) */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 14 }}>
            <div style={{ position: 'relative', width: 100, height: 100, flexShrink: 0 }}>
              <svg viewBox="0 0 36 36" style={{ width: '100%', height: '100%', transform: 'rotate(-90deg)' }}>
                {(() => {
                  const total = PERSONA.reduce((s, p) => s + p.c, 0);
                  const colors = ['#d93954', '#2563eb', '#e8721a', '#7c3aed'];
                  let offset = 0;
                  return PERSONA.map((p, i) => {
                    const pct = (p.c / total) * 100;
                    const el = (
                      <circle key={i} cx="18" cy="18" r="15.9155"
                        fill="transparent" stroke={colors[i]} strokeWidth="3.5"
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
                <div style={{ fontSize: 16, fontWeight: 700, color: '#d93954' }}>13%</div>
                <div style={{ fontSize: 9, color: C.sub }}>고위험 비율</div>
              </div>
            </div>
            <div style={{ flex: 1 }}>
              {PERSONA.map(p => (
                <div key={p.t} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <span style={{ padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600, background: p.bg, color: p.cl, flexShrink: 0 }}>{p.t}</span>
                  <span style={{ fontSize: 12, flex: 1 }}>{p.l}</span>
                  <span style={{ fontSize: 12, fontWeight: 600 }}>{p.c}명</span>
                  <span style={{ fontSize: 11, color: C.sub }}>{p.p}</span>
                </div>
              ))}
            </div>
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr>{['유형','상세','분포','비율'].map(h => <th key={h} style={{ ...th, padding: 8 }}>{h}</th>)}</tr></thead>
            <tbody>
              {PERSONA.map(p => (
                <tr key={p.t} style={{ borderBottom: `1px solid ${C.border}` }}>
                  <td style={{ padding: 8 }}><span style={{ padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600, background: p.bg, color: p.cl }}>{p.t}</span></td>
                  <td style={{ padding: 8 }}>{p.l}</td>
                  <td style={{ padding: 8 }}>{p.c}명</td>
                  <td style={{ padding: 8 }}>{p.p}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── 성과등급별 분포 + 퇴사위험자 상세 ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>

        {/* 성과등급별 퇴사위험자 분포 (Stacked Bar) */}
        <div style={cardS}>
          {secTitle('▣', '성과등급별 퇴사위험자 분포')}
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={PERF_RISK} margin={{ top: 4, right: 8, bottom: 4, left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
              <XAxis dataKey="grade" tick={{ fontSize: 12, fontFamily: 'inherit' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fontFamily: 'inherit' }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'inherit', paddingTop: 8 }} />
              <Bar dataKey="고위험"  stackId="a" fill="#d93954" radius={[0,0,0,0]} />
              <Bar dataKey="잠재위험" stackId="a" fill="#e8721a" radius={[0,0,0,0]} />
              <Bar dataKey="안정"    stackId="a" fill="#e0e0e0" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* 성과등급별 퇴사위험자 상세 (Table) */}
        <div style={cardS}>
          {secTitle('⚠', '성과등급별 퇴사위험자 상세')}
          <div style={{ maxHeight: 260, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, minWidth: 420 }}>
              <thead>
                <tr>{['부서','직급','예상 사유','위험점수','상태'].map(h => <th key={h} style={{ ...th, padding: '8px 10px' }}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {RISK_DETAIL.map((r, i) => (
                  <tr key={i} style={{ borderBottom: `1px solid ${C.border}` }}>
                    <td style={{ padding: '8px 10px', fontWeight: 500 }}>{r.dept}</td>
                    <td style={{ padding: '8px 10px' }}>{r.grade}</td>
                    <td style={{ padding: '8px 10px', color: C.sub }}>{r.reason}</td>
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

        {/* 월별 퇴사위험 추이 (Line Chart) */}
        <div style={cardS}>
          {secTitle('▬', '월별 퇴사위험 추이')}
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={RISK_TREND} margin={{ top: 4, right: 8, bottom: 4, left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="month" tick={{ fontSize: 10, fontFamily: 'inherit' }} axisLine={false} tickLine={false} angle={-30} textAnchor="end" height={40} />
              <YAxis tick={{ fontSize: 11, fontFamily: 'inherit' }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'inherit', paddingTop: 8 }} />
              <Line type="monotone" dataKey="고위험"  stroke="#d93954" strokeWidth={2} dot={{ r: 3, fill: '#d93954' }} activeDot={{ r: 5 }} />
              <Line type="monotone" dataKey="잠재위험" stroke="#e8721a" strokeWidth={1.5} strokeDasharray="5 5" dot={{ r: 2, fill: '#e8721a' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Agent별 위험 탐지 기여도 */}
        <div style={cardS}>
          {secTitle('⚙', 'Agent별 위험 탐지 기여도')}
          <div style={{ marginTop: 8 }}>
            {AGENT_CONTRIB.map(a => (
              <div key={a.n} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                <span style={{ width: 72, fontSize: 12, fontWeight: 700, textAlign: 'right', color: a.c }}>{a.n}</span>
                <div style={{ flex: 1, height: 26, background: C.border, borderRadius: 6, overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', width: `${a.p * 2.8}%`, background: a.c,
                    borderRadius: 6, display: 'flex', alignItems: 'center',
                    paddingLeft: 8, fontSize: 11, color: '#fff', fontWeight: 700,
                    transition: 'width 0.8s ease',
                  }}>
                    {a.p}%
                  </div>
                </div>
                <span style={{ fontSize: 11, color: C.sub, width: 32 }}>{a.p}%</span>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 8, padding: '8px 12px', background: '#fef7f8', borderRadius: 8, fontSize: 11, color: C.sub }}>
            Structura(정형 데이터)가 가장 높은 기여도, Agora(외부시장)이 보조적 역할
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
