import { useState } from 'react'
import { api } from '../api.js'

const SCORE_MAX = 5.0

function ScoreBar({ score }) {
  const pct = Math.min(100, (score / SCORE_MAX) * 100)
  return (
    <div style={{ background: '#0f3460', borderRadius: 4, height: 4, marginBottom: 10 }}>
      <div
        style={{
          width: `${pct}%`,
          height: '100%',
          background: 'linear-gradient(90deg, #e94560, #f5a623)',
          borderRadius: 4,
          transition: 'width 0.4s ease',
        }}
      />
    </div>
  )
}

function StarPicker({ onRate }) {
  const [hover, setHover] = useState(0)
  const [chosen, setChosen] = useState(0)

  function pick(v) {
    setChosen(v)
    onRate(v)
  }

  return (
    <div style={{ display: 'flex', gap: 3, marginTop: 6 }}>
      {[1, 2, 3, 4, 5].map((v) => (
        <span
          key={v}
          style={{
            fontSize: 18,
            cursor: 'pointer',
            color: v <= (hover || chosen) ? '#f5a623' : '#2a2a4a',
            transition: 'color 0.15s',
          }}
          onMouseEnter={() => setHover(v)}
          onMouseLeave={() => setHover(0)}
          onClick={() => pick(v)}
        >
          ★
        </span>
      ))}
    </div>
  )
}

export default function MovieCard({ movie, token, onClicked }) {
  const [rated, setRated] = useState(false)
  const [clicked, setClicked] = useState(false)

  async function handleClick() {
    if (clicked) return
    setClicked(true)
    try {
      await api.recordClick(token, movie.movie_id)
    } catch (_) {}
  }

  async function handleRate(stars) {
    try {
      await api.rateMovie(token, movie.movie_id, stars)
      setRated(true)
      onClicked?.()  // only reload recs after rating (cache bust)
    } catch (_) {}
  }

  return (
    <div
      style={{
        background: '#16213e',
        borderRadius: 12,
        padding: '18px 20px',
        cursor: 'pointer',
        border: clicked ? '1px solid #e94560' : '1px solid transparent',
        transition: 'border-color 0.2s, transform 0.15s',
      }}
      onClick={handleClick}
      onMouseEnter={(e) => (e.currentTarget.style.transform = 'translateY(-3px)')}
      onMouseLeave={(e) => (e.currentTarget.style.transform = 'translateY(0)')}
    >
      <div
        style={{
          fontSize: 14,
          fontWeight: 600,
          color: '#e8e8f0',
          marginBottom: 8,
          lineHeight: 1.4,
          minHeight: 40,
        }}
      >
        {movie.title}
      </div>

      <ScoreBar score={movie.score} />

      <div style={{ fontSize: 11, color: '#6666a0', marginBottom: 8 }}>
        Match score: {(movie.score).toFixed(2)}
      </div>

      <div onClick={(e) => e.stopPropagation()}>
        {rated ? (
          <div style={{ fontSize: 12, color: '#4caf88' }}>Rated ✓</div>
        ) : (
          <StarPicker onRate={handleRate} />
        )}
      </div>
    </div>
  )
}
