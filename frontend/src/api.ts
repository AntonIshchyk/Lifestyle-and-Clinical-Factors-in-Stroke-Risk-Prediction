export async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url)

  if (!response.ok) {
    throw new Error(await responseMessage(response))
  }

  return response.json() as Promise<T>
}

export async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    throw new Error(await responseMessage(response))
  }

  return response.json() as Promise<T>
}

async function responseMessage(response: Response): Promise<string> {
  const fallback = `Request failed with status ${response.status}`
  const text = await response.text()
  if (!text) return fallback

  try {
    const parsed = JSON.parse(text) as { message?: string; error?: string }
    return parsed.message || parsed.error || fallback
  } catch {
    const match = text.match(/<p>(.*?)<\/p>/)
    return match?.[1] || fallback
  }
}
