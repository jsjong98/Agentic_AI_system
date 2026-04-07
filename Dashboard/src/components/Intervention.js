import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts';

/* ── data ── */
const strategies = [
  {
    title: '구조적 불만족 (Structura)',
    iconBg: '#fde8ec', iconColor: '#d93954', icon: '▦',
    bullets: [
      '총체적 보상 검토 및 시장 대비 경쟁력 확보',
      '투명한 경력 경로 제시 및 승진 기준 명확화',
      '업무량 분배 재검토 및 초과근무 관리',
    ],
  },
  {
    title: '관계적 단절 (Cognita)',
    iconBg: '#e8f0fe', iconColor: '#2563eb', icon: '🔗',
    bullets: [
      '관리자 주도 1:1 Communication 체계 강화',
      '프로젝트 페어링을 통한 협업 기회 확대',
      '멘토링 프로그램 연계 및 네트워크 복원',
    ],
  },
  {
    title: '행동적 이탈 (Chronos)',
    iconBg: '#fef3e2', iconColor: '#e8721a', icon: '📈',
    bullets: [
      '자율성 부여 및 유의미한 업무 재할당',
      '업무량 및 기대 수준 재조정',
      '시의적절한 인정과 격려 제공',
    ],
  },
  {
    title: '심리적 소진 (Sentio)',
    iconBg: '#f3e8fd', iconColor: '#7c3aed', icon: '💬',
    bullets: [
      'JD-R 모델 기반 관리적 개입',
      'Job Crafting 기법 도입',
      '웰니스 프로그램 지원 및 EAP 연계',
    ],
  },
  {
    title: '외부 시장 요인 (Agora)',
    iconBg: '#e6f6ec', iconColor: '#2ea44f', icon: '🎯',
    bullets: [
      "내부 '탤런트 마켓플레이스' 관점의 관리",
      '전략적 보상 조정 (시장 벤치마크 기반)',
      '경쟁 우위 EVP 강화',
    ],
  },
];

const priorities = [
  {
    badge: '최우선', badgeColor: '#d93954',
    label: 'S등급 고위험군 (11명)',
    items: ['72시간 내 관리자-HR BP 공동 면담', '개인별 보상 패키지 재검토', '경력 개발 로드맵 수립', '주간 체크인 모니터링 시작'],
  },
  {
    badge: '적극 개입', badgeColor: '#e8721a',
    label: 'A등급 고위험군 (22명)',
    items: ['1주 내 Retention Interview', 'Persona별 맞춤 개입 적용', '프로젝트 배치 재조정', '격주 모니터링'],
  },
  {
    badge: '선제 관리', badgeColor: '#e5a100',
    label: '잠재적 위험 고성과군 (57명)',
    items: ['월간 1:1 코칭 면담 정기화', '성장 기회 우선 배정', '동료 네트워크 활성화'],
  },
];

/* simulation line chart data — matches attrition-dashboard.html interventionChart */
const simData = [
  { label: '현재',   미실행: 155, 실행: 155 },
  { label: '1개월',  미실행: 162, 실행: 148 },
  { label: '2개월',  미실행: 170, 실행: 135 },
  { label: '3개월',  미실행: 180, 실행: 120 },
  { label: '4개월',  미실행: 192, 실행: 108 },
  { label: '5개월',  미실행: 205, 실행: 98  },
  { label: '6개월',  미실행: 218, 실행: 90  },
];

/* ── shared styles ── */
const cardS = {
  background: 'var(--card,#fff)',
  borderRadius: 12, padding: 20,
  border: '1px solid var(--border,#eee)',
  boxShadow: '0 1px 4px rgba(0,0,0,.06)',
};

const SimTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#fff', border: '1px solid #eee', borderRadius: 8, padding: '8px 12px', fontSize: 12, boxShadow: '0 2px 8px rgba(0,0,0,.1)' }}>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, marginBottom: 2 }}>
          {p.name}: <strong>{p.value}명</strong>
        </div>
      ))}
      {payload.length === 2 && (
        <div style={{ borderTop: '1px solid #f0f0f0', marginTop: 4, paddingTop: 4, color: '#2ea44f', fontWeight: 700 }}>
          감소 효과: {payload[0].value - payload[1].value}명
        </div>
      )}
    </div>
  );
};

/* ── main component ── */
const Intervention = () => (
  <div>
    {/* Section header */}
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
      <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#d93954', display: 'inline-block' }} />
      <h2 style={{ fontSize: 18, fontWeight: 700, margin: 0 }}>맞춤 개입 전략 프레임워크</h2>
    </div>
    <p style={{ fontSize: 13, color: 'var(--sub,#888)', marginBottom: 20, marginLeft: 18 }}>
      각 Agent의 분석 결과를 기반으로, Persona별 맞춤형 개입 전략을 제시합니다.
    </p>

    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, alignItems: 'start' }}>

      {/* ── LEFT: WHAT ── */}
      <div style={cardS}>
        <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 16 }}>
          <span style={{ color: '#d93954' }}>?</span> WHAT: 위험 프로필별 맞춤 개입
        </div>
        {strategies.map((s, i) => (
          <div key={i} style={{
            border: '1px solid var(--border,#eee)', borderRadius: 10,
            padding: 14, marginBottom: 10,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
              <span style={{
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                width: 32, height: 32, borderRadius: 8,
                background: s.iconBg, color: s.iconColor, fontSize: 15, flexShrink: 0,
              }}>
                {s.icon}
              </span>
              <span style={{ fontWeight: 600, fontSize: 13 }}>{s.title}</span>
            </div>
            <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none' }}>
              {s.bullets.map((b, j) => (
                <li key={j} style={{ display: 'flex', alignItems: 'flex-start', gap: 8, fontSize: 12, color: 'var(--text,#444)', marginBottom: 3 }}>
                  <span style={{
                    display: 'inline-block', width: 5, height: 5, borderRadius: '50%',
                    background: s.iconColor, marginTop: 6, flexShrink: 0,
                  }} />
                  {b}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {/* ── RIGHT: HOW + Simulation chart ── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

        {/* HOW card */}
        <div style={cardS}>
          <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 16 }}>
            <span style={{ color: '#d93954' }}>⚙</span> HOW: 우선순위 기반 개입 실행 계획
          </div>
          {priorities.map((p, i) => (
            <div key={i} style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6, flexWrap: 'wrap' }}>
                <span style={{
                  display: 'inline-block', padding: '2px 10px', borderRadius: 4,
                  fontSize: 11, fontWeight: 700, color: '#fff', background: p.badgeColor,
                }}>
                  {p.badge}
                </span>
                <span style={{ fontWeight: 600, fontSize: 13 }}>{p.label}</span>
              </div>
              <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none' }}>
                {p.items.map((item, j) => (
                  <li key={j} style={{ display: 'flex', alignItems: 'flex-start', gap: 8, fontSize: 12, color: 'var(--text,#555)', marginBottom: 3 }}>
                    <span style={{ color: p.badgeColor, marginTop: 2 }}>•</span> {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Simulation line chart card */}
        <div style={cardS}>
          <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>
            <span style={{ color: '#d93954' }}>📊</span> 개입 효과 예측 시뮬레이션
          </div>
          <div style={{ fontSize: 12, color: 'var(--sub,#888)', marginBottom: 14 }}>
            개입 실행 시 예상 고위험군 감소 추이
          </div>

          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={simData} margin={{ top: 8, right: 20, bottom: 4, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border,#f0f0f0)" />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 12, fontFamily: 'inherit' }}
                axisLine={false} tickLine={false}
              />
              <YAxis
                tick={{ fontSize: 11, fontFamily: 'inherit' }}
                axisLine={false} tickLine={false}
                domain={[60, 240]}
                label={{ value: '고위험군 (명)', angle: -90, position: 'insideLeft', fontSize: 11, fill: 'var(--sub,#888)', dy: 40 }}
              />
              <Tooltip content={<SimTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: 12, fontFamily: 'inherit', paddingTop: 8 }}
              />
              <ReferenceLine y={155} stroke="#d93954" strokeDasharray="4 4" strokeOpacity={0.4} />
              <Line
                type="monotone" dataKey="미실행"
                stroke="#bbb" strokeWidth={2} strokeDasharray="6 4"
                dot={{ r: 3, fill: '#bbb' }} activeDot={{ r: 5 }}
              />
              <Line
                type="monotone" dataKey="실행"
                stroke="#d93954" strokeWidth={2.5}
                dot={{ r: 4, fill: '#d93954' }} activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>

          {/* Summary row */}
          <div style={{ display: 'flex', gap: 10, marginTop: 12 }}>
            {[
              { label: '6개월 후 (미실행)', val: '218명', color: '#bbb', bg: '#f5f5f5' },
              { label: '6개월 후 (개입 시)', val: '90명', color: '#d93954', bg: '#fde8ec' },
              { label: '예상 감소 효과', val: '−128명', color: '#2ea44f', bg: '#e6f6ec' },
            ].map((s, i) => (
              <div key={i} style={{ flex: 1, background: s.bg, borderRadius: 8, padding: '10px 12px', textAlign: 'center' }}>
                <div style={{ fontSize: 10, color: 'var(--sub,#888)', marginBottom: 4 }}>{s.label}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: s.color }}>{s.val}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  </div>
);

export default Intervention;
