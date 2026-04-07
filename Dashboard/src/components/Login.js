import React, { useState } from 'react';

const USERS = [
  { email: '***@redacted', password: '***REDACTED***', role: 'admin', name: 'Admin User 1', initials: 'JO' },
  { email: '***@redacted', password: '***REDACTED***', role: 'admin', name: 'Admin User 2', initials: 'CC' },
  { email: '***@redacted', password: '***REDACTED***', role: 'admin', name: 'Admin User 3', initials: 'JK' },
  { email: '***@redacted', password: '***REDACTED***', role: 'hr', name: 'HR Manager', initials: 'HR' },
];

const PWC_LOGO = `data:image/svg+xml,${encodeURIComponent(`<?xml version="1.0" encoding="UTF-8"?><svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 70 53" width="70" height="53"><defs><style>.cls-1{fill:#fd5108;}</style></defs><g><g><g><path d="M51.7,30.3c-2.5.4-3.7,2.2-3.7,5.4s1.7,5.4,4.2,5.4,2.3-.4,4.6-1.5v2.6c-2.7,1.3-4.3,1.6-6.6,1.6s-4.1-.6-5.4-2c-1.4-1.4-2.1-3.3-2.1-5.3,0-4.6,3.4-7.7,8.4-7.7s5.6,1.5,5.6,3.7-1.1,2.4-2.6,2.4-1.5-.2-2.3-.7v-3.9h0ZM39.6,36.4c2.2-2.8,3-3.9,3-5.3s-1.1-2.5-2.5-2.5-1.7.4-2.1.9v5.7l-3.6,4.8v-11h-3.4l-5.7,9.5v-9.5h-2l-5.2,1.3v1.3l2.8.3v11.6h3.7l5.5-9v9h4l5.6-7.1h-.1ZM7.2,32c.8,0,1.3-.2,1.7-.2,2.4,0,3.7,1.6,3.7,4.6s-1.6,5.4-4.5,5.4-.4,0-.8,0v-9.7h0ZM7.2,43.4c.9,0,1.9,0,2.4,0,4.9,0,8-3.1,8-7.8s-2.3-6.9-5.5-6.9-2.3.3-4.9,1.9v-1.9h-1.5l-5.7,1.7v1.4h2.4v16.2l-2.1.5v1.3h9.3v-1.3l-2.4-.5v-4.7h0Z"/><path class="cls-1" d="M49.1,24.8h-15.5l2.6-4.4h15.5l-2.6,4.4ZM69.9,16h-15.5l-2.6,4.4h15.5l2.6-4.4Z"/></g></g></g></svg>`)}`;

export function authenticate(email, password) {
  return USERS.find(u => u.email === email && u.password === password) || null;
}

export function getStoredUser() {
  try {
    const data = localStorage.getItem('pwc_user');
    return data ? JSON.parse(data) : null;
  } catch {
    return null;
  }
}

export function logout() {
  localStorage.removeItem('pwc_user');
}

export { PWC_LOGO };

const Login = ({ onLogin }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    setTimeout(() => {
      const user = authenticate(email, password);
      if (user) {
        const userData = { email: user.email, role: user.role, name: user.name, initials: user.initials };
        localStorage.setItem('pwc_user', JSON.stringify(userData));
        onLogin(userData);
      } else {
        setError('이메일 또는 비밀번호가 올바르지 않습니다.');
      }
      setLoading(false);
    }, 400);
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <div style={styles.logoSection}>
          <img src={PWC_LOGO} alt="PwC" style={{ height: 40 }} />
          <div style={styles.divider} />
          <h1 style={styles.title}>조직 퇴사위험 대시보드</h1>
          <p style={styles.subtitle}>Agentic AI 기반 선제적 퇴사위험 예측 및 관리시스템</p>
        </div>

        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={styles.inputGroup}>
            <label style={styles.label}>이메일</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="name@example.com"
              style={styles.input}
              required
              autoFocus
            />
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>비밀번호</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="비밀번호 입력"
              style={styles.input}
              required
            />
          </div>

          {error && <div style={styles.error}>{error}</div>}

          <button
            type="submit"
            disabled={loading}
            style={{
              ...styles.button,
              opacity: loading ? 0.7 : 1,
            }}
          >
            {loading ? '로그인 중...' : '로그인'}
          </button>
        </form>

        <div style={styles.footer}>
          &copy; 2025 PwC Consulting. All rights reserved.
        </div>
      </div>
    </div>
  );
};

const styles = {
  container: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%)',
    fontFamily: "'Noto Sans KR', system-ui, -apple-system, sans-serif",
  },
  card: {
    background: '#fff',
    borderRadius: 16,
    boxShadow: '0 4px 24px rgba(0,0,0,0.08)',
    padding: '48px 40px 32px',
    width: 400,
    maxWidth: '90vw',
    borderTop: '4px solid #d93954',
  },
  logoSection: {
    textAlign: 'center',
    marginBottom: 32,
  },
  divider: {
    width: 40,
    height: 3,
    background: '#d93954',
    margin: '16px auto',
    borderRadius: 2,
  },
  title: {
    fontSize: 20,
    fontWeight: 700,
    color: '#2d2d2d',
    margin: '0 0 6px',
  },
  subtitle: {
    fontSize: 12,
    color: '#888',
    margin: 0,
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: 18,
  },
  inputGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  label: {
    fontSize: 13,
    fontWeight: 600,
    color: '#555',
  },
  input: {
    padding: '10px 14px',
    border: '1px solid #ddd',
    borderRadius: 8,
    fontSize: 14,
    fontFamily: 'inherit',
    outline: 'none',
    transition: 'border-color 0.2s',
  },
  error: {
    background: '#fde8ec',
    color: '#d93954',
    padding: '10px 14px',
    borderRadius: 8,
    fontSize: 13,
    fontWeight: 500,
  },
  button: {
    padding: '12px',
    background: '#d93954',
    color: '#fff',
    border: 'none',
    borderRadius: 8,
    fontSize: 15,
    fontWeight: 600,
    cursor: 'pointer',
    fontFamily: 'inherit',
    marginTop: 4,
  },
  footer: {
    textAlign: 'center',
    fontSize: 11,
    color: '#aaa',
    marginTop: 28,
  },
};

export default Login;
