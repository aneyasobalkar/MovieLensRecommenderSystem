const BASE = '/api'

function authHeaders(token) {
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function request(method, path, { token, body } = {}) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(token),
    },
    body: body != null ? JSON.stringify(body) : undefined,
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`)
  return data
}

// Auth endpoints use form-encoded bodies (OAuth2PasswordRequestForm)
async function postForm(path, fields) {
  const body = new URLSearchParams(fields)
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`)
  return data
}

export const api = {
  register: (username, password) =>
    request('POST', '/auth/register', { body: { username, password } }),

  login: (username, password) =>
    postForm('/auth/login', { username, password }),

  me: (token) =>
    request('GET', '/auth/me', { token }),

  getRecommendations: (token, n = 10) =>
    request('GET', `/recommendations?n=${n}`, { token }),

  rateMovie: (token, movie_id, rating) =>
    request('POST', '/ratings', { token, body: { movie_id, rating } }),

  myRatings: (token) =>
    request('GET', '/ratings/me', { token }),

  recordClick: (token, movie_id) =>
    request('POST', '/events/click', { token, body: { movie_id } }),

  abMetrics: () =>
    request('GET', '/ab_test/metrics'),
}
