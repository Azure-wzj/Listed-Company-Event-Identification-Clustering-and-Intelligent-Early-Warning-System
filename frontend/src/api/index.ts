import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 60000,
})

export async function fetchEvents(params?: any) {
  const res = await api.get('/events/', { params })
  return res.data
}

export async function fetchEventDetail(groupId: string) {
  const res = await api.get(`/events/${groupId}`)
  return res.data
}

export async function fetchTimeline(params?: any) {
  const res = await api.get('/timeline/', { params })
  return res.data
}

export async function fetchTimelineStats() {
  const res = await api.get('/timeline/stats')
  return res.data
}

export async function fetchGraph(params?: any) {
  const res = await api.get('/graph/', { params })
  return res.data
}

export async function fetchGraphCompanies(limit = 50) {
  const res = await api.get('/graph/companies', { params: { limit } })
  return res.data
}

export async function fetchCompanyGraph(code: string) {
  const res = await api.get(`/graph/company/${code}`)
  return res.data
}

export async function fetchAlerts(params?: any) {
  const res = await api.get('/alert/', { params })
  return res.data
}

export async function fetchAlertStats() {
  const res = await api.get('/alert/stats')
  return res.data
}

export async function fetchAlertConfig() {
  const res = await api.get('/alert/config')
  return res.data
}

// ── 增量接入 ──
export async function ingestPost(payload: any) {
  const res = await api.post('/ingest', payload)
  return res.data
}

export async function ingestStatus() {
  const res = await api.get('/ingest/status')
  return res.data
}

export async function ingestReset() {
  const res = await api.post('/ingest/reset')
  return res.data
}

export async function ingestNl(payload: any) {
  const res = await api.post('/ingest/nl', payload)
  return res.data
}

export async function ingestNlStatus() {
  const res = await api.get('/ingest/nl/status')
  return res.data
}

export async function ingestNlTest(payload: any) {
  const res = await api.post('/ingest/nl/test', payload)
  return res.data
}

export default api
