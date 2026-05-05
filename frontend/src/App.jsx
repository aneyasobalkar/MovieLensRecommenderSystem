import { useState, useEffect } from 'react'
import Login from './components/Login.jsx'
import MovieGrid from './components/MovieGrid.jsx'

const TOKEN_KEY = 'cinematch_token'
const USER_KEY  = 'cinematch_user'
const VAR_KEY   = 'cinematch_variant'

export default function App() {
  const [token, setToken]     = useState(() => localStorage.getItem(TOKEN_KEY))
  const [username, setUsername] = useState(() => localStorage.getItem(USER_KEY) || '')
  const [variant, setVariant] = useState(() => localStorage.getItem(VAR_KEY) || '')

  function handleAuth(tok, user, vrnt) {
    setToken(tok)
    setUsername(user)
    setVariant(vrnt)
    localStorage.setItem(TOKEN_KEY, tok)
    localStorage.setItem(USER_KEY, user)
    localStorage.setItem(VAR_KEY, vrnt)
  }

  function handleLogout() {
    setToken(null)
    setUsername('')
    setVariant('')
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
    localStorage.removeItem(VAR_KEY)
  }

  if (!token) {
    return <Login onAuth={handleAuth} />
  }

  return (
    <MovieGrid
      token={token}
      variant={variant}
      username={username}
      onLogout={handleLogout}
    />
  )
}
