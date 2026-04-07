import React from 'react';

const kpiData = [
  { label: '전사', value: '1,470명', bg: '#f0f5ff', color: '#2563eb' },
  { label: '고위험군', value: '77명', bg: '#fde8ec', color: '#d93954' },
  { label: '잠재위험', value: '202명', bg: '#fef3e2', color: '#e8721a' },
  { label: '안정', value: '1,191명', bg: '#e6f6ec', color: '#2ea44f' },
  { label: '평균점수', value: '0.28', bg: '#f3e8fd', color: '#7c3aed' },
];

const strategies = [
  {
    title: '구조적 불만족 (Structura)',
    iconBg: '#fde8ec',
    iconColor: '#d93954',
    icon: '▦',
    bullets: [
      '총체적 보상 검토 및 시장 대비 경쟁력 확보',
      '투명한 경력 경로 제시 및 승진 기준 명확화',
      '업무량 분배 재검토 및 초과근무 관리',
    ],
  },
  {
    title: '관계적 단절 (Cognita)',
    iconBg: '#e8f0fe',
    iconColor: '#2563eb',
    icon: '🔗',
    bullets: [
      '관리자 주도 1:1 Communication 체계 강화',
      '프로젝트 페어링을 통한 협업 기회 확대',
      '멘토링 프로그램 연계 및 네트워크 복원',
    ],
  },
  {
    title: '행동적 이탈 (Chronos)',
    iconBg: '#fef3e2',
    iconColor: '#e8721a',
    icon: '📈',
    bullets: [
      '자율성 부여 및 유의미한 업무 재할당',
      '업무량 및 기대 수준 재조정',
      '시의적절한 인정과 격려 제공',
    ],
  },
  {
    title: '심리적 소진 (Sentio)',
    iconBg: '#f3e8fd',
    iconColor: '#7c3aed',
    icon: '💬',
    bullets: [
      'JD-R 모델 기반 관리적 개입',
      'Job Crafting 기법 도입',
      '웰니스 프로그램 지원 및 EAP 연계',
    ],
  },
  {
    title: '외부 시장 요인 (Agora)',
    iconBg: '#e6f6ec',
    iconColor: '#2ea44f',
    icon: '🎯',
    bullets: [
      "내부 '탤런트 마켓플레이스' 관점의 관리",
      '전략적 보상 조정 (시장 벤치마크 기반)',
      '경쟁 우위 EVP 강화',
    ],
  },
];

const priorities = [
  {
    badge: '최우선',
    badgeColor: '#d93954',
    label: 'S등급 고위험군 (11명)',
    items: [
      '72시간 내 관리자-HR BP 공동 면담',
      '개인별 보상 패키지 재검토',
      '경력 개발 로드맵 수립',
      '주간 체크인 모니터링 시작',
    ],
  },
  {
    badge: '적극 개입',
    badgeColor: '#e8721a',
    label: 'A등급 고위험군 (22명)',
    items: [
      '1주 내 Retention Interview',
      'Persona별 맞춤 개입 적용',
      '프로젝트 배치 재조정',
      '격주 모니터링',
    ],
  },
  {
    badge: '선제 관리',
    badgeColor: '#e5a100',
    label: '잠재적 위험 고성과군 (57명)',
    items: [
      '월간 1:1 코칭 면담 정기화',
      '성장 기회 우선 배정',
      '동료 네트워크 활성화',
    ],
  },
];

const simulationData = [
  { label: '현재', without: '155명', with: '155명' },
  { label: '1개월', without: '162명', with: '148명' },
  { label: '3개월', without: '180명', with: '120명' },
  { label: '6개월', without: '218명', with: '90명' },
];

const cardStyle = {
  background: '#fff',
  borderRadius: 12,
  padding: 24,
  boxShadow: '0 1px 4px rgba(0,0,0,0.07)',
};

const sectionTitle = {
  fontSize: 18,
  fontWeight: 700,
  margin: 0,
};

const Intervention = () => {
  return (
    <div style={{ padding: '24px 32px', background: '#f5f6fa', minHeight: '100vh' }}>
      {/* KPI Row */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 28 }}>
        {kpiData.map((k, i) => (
          <div
            key={i}
            style={{
              flex: 1,
              background: '#fff',
              borderRadius: 10,
              padding: '18px 20px',
              boxShadow: '0 1px 4px rgba(0,0,0,0.07)',
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: 13, color: '#888', marginBottom: 6 }}>{k.label}</div>
            <div style={{ fontSize: 26, fontWeight: 700, color: k.color }}>{k.value}</div>
          </div>
        ))}
      </div>

      {/* Section Title */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <span
          style={{
            display: 'inline-block',
            width: 10,
            height: 10,
            borderRadius: '50%',
            background: '#d93954',
          }}
        />
        <h2 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>맞춤 개입 전략 프레임워크</h2>
      </div>
      <p style={{ fontSize: 14, color: '#888', marginBottom: 24, marginLeft: 18 }}>
        각 Agent의 분석 결과를 기반으로, Persona별 맞춤형 개입 전략을 제시합니다.
      </p>

      {/* Two-column layout */}
      <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>
        {/* LEFT: WHAT */}
        <div style={{ flex: 1, ...cardStyle }}>
          <h3 style={sectionTitle}>WHAT: 위험 프로필별 맞춤 개입</h3>
          <div style={{ marginTop: 18 }}>
            {strategies.map((s, i) => (
              <div
                key={i}
                style={{
                  border: '1px solid #eee',
                  borderRadius: 10,
                  padding: 14,
                  marginBottom: 12,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                  <span
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: 32,
                      height: 32,
                      borderRadius: 8,
                      background: s.iconBg,
                      color: s.iconColor,
                      fontSize: 16,
                    }}
                  >
                    {s.icon}
                  </span>
                  <span style={{ fontWeight: 600, fontSize: 14, color: '#222' }}>{s.title}</span>
                </div>
                <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none' }}>
                  {s.bullets.map((b, j) => (
                    <li
                      key={j}
                      style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: 8,
                        fontSize: 13,
                        color: '#444',
                        marginBottom: 4,
                        paddingLeft: 4,
                      }}
                    >
                      <span
                        style={{
                          display: 'inline-block',
                          width: 5,
                          height: 5,
                          borderRadius: '50%',
                          background: s.iconColor,
                          marginTop: 6,
                          flexShrink: 0,
                        }}
                      />
                      {b}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        {/* RIGHT: HOW + Simulation */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 24 }}>
          {/* HOW Card */}
          <div style={cardStyle}>
            <h3 style={sectionTitle}>HOW: 우선순위 기반 개입 실행 계획</h3>
            <div style={{ marginTop: 18 }}>
              {priorities.map((p, i) => (
                <div key={i} style={{ marginBottom: 18 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                    <span
                      style={{
                        display: 'inline-block',
                        padding: '2px 10px',
                        borderRadius: 4,
                        fontSize: 11,
                        fontWeight: 700,
                        color: '#fff',
                        background: p.badgeColor,
                      }}
                    >
                      {p.badge}
                    </span>
                    <span style={{ fontWeight: 600, fontSize: 14, color: '#222' }}>{p.label}</span>
                  </div>
                  <ul style={{ margin: 0, paddingLeft: 22, listStyle: 'disc', color: '#555' }}>
                    {p.items.map((item, j) => (
                      <li key={j} style={{ fontSize: 13, marginBottom: 3 }}>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          {/* Simulation Card */}
          <div style={cardStyle}>
            <h3 style={sectionTitle}>개입 효과 예측 시뮬레이션</h3>
            <div style={{ marginTop: 18 }}>
              <table
                style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  fontSize: 13,
                }}
              >
                <thead>
                  <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#666', fontWeight: 600 }}>
                      시점
                    </th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#d93954', fontWeight: 600 }}>
                      미실행(예상)
                    </th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#2ea44f', fontWeight: 600 }}>
                      실행(예상)
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {simulationData.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f0f0f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.label}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center', color: '#d93954' }}>
                        {row.without}
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'center', color: '#2ea44f' }}>
                        {row.with}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p style={{ fontSize: 12, color: '#999', marginTop: 12, marginBottom: 0 }}>
                * 개입 실행 시 예상 고위험군 감소 추이
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Intervention;
