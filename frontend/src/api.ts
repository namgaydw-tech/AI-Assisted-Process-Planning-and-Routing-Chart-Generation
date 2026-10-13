import type {Request, Response, GeometryResponse} from './types'
async function checked<T>(res: globalThis.Response): Promise<T> {
  if (!res.ok) {
    const payload = await res.json().catch(() => null)
    const details = payload?.detail
    const message = typeof details === 'string' ? details : JSON.stringify(details ?? {})
    throw new Error(message || `Request failed with HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}
export const api = {
  health: () => fetch('/api/health').then(checked<{status:string;training_status:string}>),
  plan: (data: Request) => fetch('/api/plan', {
    method:'POST', headers: {'Content-Type':'application/json'}, body:JSON.stringify(data)
  }).then(checked<Response>),
  step: (file: File) => {
    const body = new FormData(); body.append('file',file)
    return fetch('/api/analyze-step', { method:'POST', body }).then(checked<GeometryResponse>)
  }
}
