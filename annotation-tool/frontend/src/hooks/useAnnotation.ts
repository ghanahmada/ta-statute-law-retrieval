import { useCallback, useState } from "react"
import { api } from "@/lib/api"
import type { LabelData, Pair } from "@/types"

export interface AnnotationState {
  pairs: Pair[]
  labels: Record<string, LabelData>
  flagged: Set<string>
  currentIndex: number
  loading: boolean
  submitted: boolean
  error: string | null
}

export function useAnnotation() {
  const [pairs, setPairs] = useState<Pair[]>([])
  const [labels, setLabels] = useState<Record<string, LabelData>>({})
  const [flagged, setFlagged] = useState<Set<string>>(new Set())
  const [currentIndex, setCurrentIndex] = useState(0)
  const [loading, setLoading] = useState(true)
  const [submitted, setSubmitted] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const init = useCallback(async () => {
    setLoading(true)
    try {
      const [pairsData, labelsData] = await Promise.all([
        api.allPairs(),
        api.labels(),
      ])
      setPairs(pairsData)
      setLabels(labelsData.labels)
      setFlagged(new Set(labelsData.flagged))
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load data")
    } finally {
      setLoading(false)
    }
  }, [])

  const currentPair = pairs[currentIndex] ?? null
  const totalAnswered = Object.keys(labels).length
  const totalFlagged = flagged.size
  const totalPairs = pairs.length

  const goTo = useCallback(
    (index: number) => {
      if (index >= 0 && index < pairs.length) {
        setCurrentIndex(index)
      }
    },
    [pairs.length],
  )

  const next = useCallback(() => goTo(currentIndex + 1), [currentIndex, goTo])
  const prev = useCallback(() => goTo(currentIndex - 1), [currentIndex, goTo])

  const saveLabel = useCallback(
    async (label: "RELEVANT" | "NOT_RELEVANT") => {
      if (!currentPair) return
      const pairId = currentPair.pair_id
      setLabels((prev) => ({ ...prev, [pairId]: { label } }))
      try {
        await api.saveLabel(pairId, label)
      } catch (e) {
        setLabels((prev) => {
          const copy = { ...prev }
          delete copy[pairId]
          return copy
        })
        setError(e instanceof Error ? e.message : "Failed to save")
      }
    },
    [currentPair],
  )

  const toggleFlag = useCallback(
    async (pairId: string) => {
      const nowFlagged = !flagged.has(pairId)
      setFlagged((prev) => {
        const next = new Set(prev)
        if (nowFlagged) next.add(pairId)
        else next.delete(pairId)
        return next
      })
      try {
        await api.toggleFlag(pairId, nowFlagged)
      } catch {
        setFlagged((prev) => {
          const next = new Set(prev)
          if (nowFlagged) next.delete(pairId)
          else next.add(pairId)
          return next
        })
      }
    },
    [flagged],
  )

  const submitAll = useCallback(async () => {
    try {
      await api.submit()
      setSubmitted(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Submit failed")
      throw e
    }
  }, [])

  const getStatus = useCallback(
    (pairId: string) => {
      const answered = pairId in labels
      const isFlagged = flagged.has(pairId)
      if (answered && isFlagged) return "flagged-answered" as const
      if (answered) return "answered" as const
      if (isFlagged) return "flagged" as const
      return "unanswered" as const
    },
    [labels, flagged],
  )

  return {
    pairs,
    labels,
    flagged,
    currentIndex,
    currentPair,
    loading,
    submitted,
    error,
    totalAnswered,
    totalFlagged,
    totalPairs,
    init,
    goTo,
    next,
    prev,
    saveLabel,
    toggleFlag,
    submitAll,
    getStatus,
    setSubmitted,
    clearError: () => setError(null),
  }
}
