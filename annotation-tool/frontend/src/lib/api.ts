import type { AuthStatus, LabelsResponse, Pair } from "@/types"

function getToken(): string {
  return localStorage.getItem("session_token") || ""
}

function headers(): HeadersInit {
  return {
    Authorization: `Bearer ${getToken()}`,
    "Content-Type": "application/json",
  }
}

async function request<T>(url: string, opts: RequestInit = {}): Promise<T> {
  const res = await fetch(url, { headers: headers(), ...opts })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || res.statusText)
  }
  return res.json()
}

export const api = {
  login: (token: string) =>
    request<{ session_token: string; annotator_name: string; submitted: boolean }>(
      "/auth/login",
      { method: "POST", body: JSON.stringify({ token }) },
    ),

  status: () => request<AuthStatus>("/auth/status"),

  allPairs: () => request<Pair[]>("/pairs/all"),

  labels: () => request<LabelsResponse>("/pairs/labels"),

  saveLabel: (pair_id: string, label: string) =>
    request<{ success: boolean }>("/pairs/label", {
      method: "POST",
      body: JSON.stringify({ pair_id, label }),
    }),

  toggleFlag: (pair_id: string, flagged: boolean) =>
    request<{ success: boolean }>("/pairs/flag", {
      method: "POST",
      body: JSON.stringify({ pair_id, flagged }),
    }),

  submit: () =>
    request<{ success: boolean; submitted_at: string }>("/pairs/submit", {
      method: "POST",
    }),
}
