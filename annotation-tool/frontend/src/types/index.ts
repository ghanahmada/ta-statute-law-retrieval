export interface Pair {
  pair_id: string
  case_id: string
  variant: "humanized" | "summarized"
  query_text: string
  article_id: string
  article_title: string
  article_text: string
  kuhperdata_book: string
}

export interface LabelData {
  label: "RELEVANT" | "NOT_RELEVANT"
  confidence: "low" | "medium" | "high"
}

export interface LabelsResponse {
  labels: Record<string, LabelData>
  flagged: string[]
}

export interface AuthStatus {
  authenticated: boolean
  name?: string
  submitted?: boolean
}
