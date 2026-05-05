import { useEffect, useState, useCallback } from 'react'
import { api } from '../api.js'
import MovieCard from './MovieCard.jsx'

const VARIANT_LABEL = {
  A: 'Item-Item CF',
  B: 'SVD Matrix Factorization',
}

export default function MovieGrid({ token, variant, username, onLogout }) {
  const [recs, setRecs] = useState([])
  const [fromCache, setFromCache] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const data = await api.getRecommendations(token, 12)
      setRecs(data.recommendations)
      setFromCache(data.from_cache)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [token])

  useEffect(() => { load() }, [load])

  return (
    <div style={{ minHeight: '100vh', background: '#0f0f13' }}>
      {/* Header */}
      <div
        style={{
          background: '#16213e',
          borderBottom: '1px solid #1e2a4a',
          padding: '14px 32px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={{ fontSize: 20, fontWeight: 700, color: '#e94560' }}>CineMatch</span>
          <span
            style={{
              background: variant === 'A' ? '#0f3460' : '#1a3a2a',
              color: variant === 'A' ? '#5b9ef5' : '#4caf88',
              fontSize: 11,
              fontWeight: 700,
              padding: '3px 10px',
              borderRadius: 20,
              letterSpacing: '0.5px',
            }}
          >
            Variant {variant} · {VARIANT_LABEL[variant]}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={{ color: '#8888a0', fontSize: 13 }}>
            Hi, <strong style={{ color: '#e8e8f0' }}>{username}</strong>
          </span>
          <button
            onClick={onLogout}
            style={{
              background: 'transparent',
              border: '1px solid #2a2a4a',
              color: '#8888a0',
              padding: '6px 14px',
              borderRadius: 6,
              cursor: 'pointer',
              fontSize: 13,
            }}
          >
            Sign out
          </button>
        </div>
      </div>

      {/* Main */}
      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '36px 24px' }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: 28,
          }}
        >
          <div>
            <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>
              Your Recommendations
            </h1>
            <div style={{ fontSize: 12, color: '#6666a0' }}>
              {fromCache ? '⚡ Served from cache' : '🔄 Fresh from model'} · click a card to track it
            </div>
          </div>
          <button
            onClick={load}
            disabled={loading}
            style={{
              background: '#e94560',
              color: '#fff',
              border: 'none',
              borderRadius: 8,
              padding: '9px 20px',
              cursor: loading ? 'default' : 'pointer',
              fontWeight: 600,
              fontSize: 13,
              opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? 'Loading…' : 'Refresh'}
          </button>
        </div>

        {error && (
          <div style={{ color: '#ff6b7a', marginBottom: 20, fontSize: 14 }}>{error}</div>
        )}

        {loading ? (
          <div style={{ color: '#6666a0', textAlign: 'center', paddingTop: 80 }}>
            Loading recommendations…
          </div>
        ) : (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
              gap: 16,
            }}
          >
            {recs.map((movie) => (
              <MovieCard
                key={movie.movie_id}
                movie={movie}
                token={token}
                onClicked={load}
              />
            ))}
          </div>
        )}

        <ABPanel />
      </div>
    </div>
  )
}


function ABPanel() {
  const [metrics, setMetrics] = useState(null)
  const [open, setOpen] = useState(false)

  async function load() {
    const data = await api.abMetrics()
    setMetrics(data)
    setOpen(true)
  }

  const VARIANT_COLOR = { A: '#5b9ef5', B: '#4caf88' }

  return (
    <div style={{ marginTop: 48 }}>
      <button
        onClick={open ? () => setOpen(false) : load}
        style={{
          background: 'transparent',
          border: '1px solid #2a2a4a',
          color: '#8888a0',
          padding: '8px 18px',
          borderRadius: 8,
          cursor: 'pointer',
          fontSize: 13,
        }}
      >
        {open ? 'Hide A/B metrics' : 'Show A/B test metrics'}
      </button>

      {open && metrics && (
        <div
          style={{
            marginTop: 20,
            background: '#16213e',
            borderRadius: 12,
            padding: 24,
          }}
        >
          <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 20 }}>
            A/B Test — Click-Through Rate
          </h2>
          <div style={{ display: 'flex', gap: 20 }}>
            {Object.entries(metrics).map(([variant, m]) => (
              <div
                key={variant}
                style={{
                  flex: 1,
                  background: '#0f3460',
                  borderRadius: 10,
                  padding: '16px 20px',
                  borderLeft: `3px solid ${VARIANT_COLOR[variant]}`,
                }}
              >
                <div
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    color: VARIANT_COLOR[variant],
                    marginBottom: 6,
                    letterSpacing: '0.5px',
                  }}
                >
                  VARIANT {variant}
                </div>
                <div style={{ fontSize: 13, color: '#c0c0d8', marginBottom: 12 }}>
                  {m.model}
                </div>
                <div style={{ display: 'flex', gap: 24 }}>
                  <Stat label="Impressions" value={m.impressions} />
                  <Stat label="Clicks" value={m.clicks} />
                  <Stat
                    label="CTR"
                    value={`${(m.ctr * 100).toFixed(1)}%`}
                    accent={VARIANT_COLOR[variant]}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function Stat({ label, value, accent }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: '#6666a0', marginBottom: 2 }}>{label}</div>
      <div
        style={{
          fontSize: 22,
          fontWeight: 700,
          color: accent || '#e8e8f0',
        }}
      >
        {value}
      </div>
    </div>
  )
}
