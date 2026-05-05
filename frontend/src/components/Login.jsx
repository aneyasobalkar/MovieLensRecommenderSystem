import { useState } from 'react'
import { api } from '../api.js'

const S = {
  wrap: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'linear-gradient(135deg, #0f0f13 0%, #1a1a2e 100%)',
  },
  card: {
    background: '#16213e',
    borderRadius: 16,
    padding: '40px 48px',
    width: 380,
    boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
  },
  logo: {
    textAlign: 'center',
    marginBottom: 32,
  },
  title: {
    fontSize: 28,
    fontWeight: 700,
    color: '#e94560',
    letterSpacing: '-0.5px',
  },
  sub: { color: '#8888a0', fontSize: 13, marginTop: 4 },
  tabs: { display: 'flex', marginBottom: 28, borderRadius: 8, overflow: 'hidden' },
  tab: (active) => ({
    flex: 1,
    padding: '10px 0',
    textAlign: 'center',
    cursor: 'pointer',
    fontSize: 14,
    fontWeight: 600,
    background: active ? '#e94560' : '#0f3460',
    color: active ? '#fff' : '#8888a0',
    border: 'none',
    transition: 'all 0.2s',
  }),
  field: { marginBottom: 16 },
  label: { display: 'block', fontSize: 12, color: '#8888a0', marginBottom: 6 },
  input: {
    width: '100%',
    padding: '10px 14px',
    background: '#0f3460',
    border: '1px solid #1e4a8a',
    borderRadius: 8,
    color: '#e8e8f0',
    fontSize: 14,
    outline: 'none',
  },
  btn: {
    width: '100%',
    padding: '12px 0',
    background: '#e94560',
    color: '#fff',
    border: 'none',
    borderRadius: 8,
    fontSize: 15,
    fontWeight: 700,
    cursor: 'pointer',
    marginTop: 8,
  },
  error: {
    color: '#ff6b7a',
    fontSize: 13,
    marginTop: 12,
    textAlign: 'center',
  },
}

export default function Login({ onAuth }) {
  const [tab, setTab] = useState('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function submit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const data =
        tab === 'login'
          ? await api.login(username, password)
          : await api.register(username, password)
      onAuth(data.access_token, data.username, data.variant)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={S.wrap}>
      <div style={S.card}>
        <div style={S.logo}>
          <div style={S.title}>CineMatch</div>
          <div style={S.sub}>Personalised movie recommendations</div>
        </div>

        <div style={S.tabs}>
          <button style={S.tab(tab === 'login')} onClick={() => setTab('login')}>
            Sign in
          </button>
          <button style={S.tab(tab === 'register')} onClick={() => setTab('register')}>
            Register
          </button>
        </div>

        <form onSubmit={submit}>
          <div style={S.field}>
            <label style={S.label}>Username</label>
            <input
              style={S.input}
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="your_username"
              autoFocus
            />
          </div>
          <div style={S.field}>
            <label style={S.label}>Password</label>
            <input
              style={S.input}
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
            />
          </div>
          <button style={S.btn} disabled={loading}>
            {loading ? 'Please wait…' : tab === 'login' ? 'Sign in' : 'Create account'}
          </button>
          {error && <div style={S.error}>{error}</div>}
        </form>
      </div>
    </div>
  )
}
