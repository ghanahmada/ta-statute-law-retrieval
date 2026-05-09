import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  Flag,
  Send,
} from "lucide-react"
import type { useAnnotation } from "@/hooks/useAnnotation"

interface Props {
  annotation: ReturnType<typeof useAnnotation>
  onBack: () => void
  onSubmitted: () => void
}

const statusColors = {
  unanswered: "bg-white border border-slate-300",
  answered: "bg-navy-600",
  flagged: "bg-amber-100 border border-amber-400",
  "flagged-answered": "bg-amber-400 ring-1 ring-navy-600",
}

export function ConfirmationPage({ annotation, onBack, onSubmitted }: Props) {
  const {
    pairs,
    totalAnswered,
    totalFlagged,
    totalPairs,
    getStatus,
    goTo,
    submitAll,
  } = annotation

  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const remaining = totalPairs - totalAnswered
  const canSubmit = remaining === 0

  const flaggedPairs = pairs.filter((p) => {
    const s = getStatus(p.pair_id)
    return s === "flagged" || s === "flagged-answered"
  })

  const handleSubmit = async () => {
    setSubmitting(true)
    setError(null)
    try {
      await submitAll()
      onSubmitted()
    } catch (e) {
      setError(e instanceof Error ? e.message : "Submission failed")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="flex min-h-dvh items-center justify-center bg-navy-50 p-6">
      <Card className="w-full max-w-3xl border-navy-200 shadow-lg">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl font-semibold text-navy-800">
            Review & Submit
          </CardTitle>
          <p className="text-sm text-navy-500">
            Review your progress before final submission
          </p>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-3 gap-4">
            <div className="rounded-lg border border-navy-200 bg-white p-4 text-center">
              <CheckCircle2 className="mx-auto mb-2 h-6 w-6 text-navy-600" />
              <p className="text-2xl font-bold text-navy-800">{totalAnswered}</p>
              <p className="text-xs text-navy-500">Answered</p>
            </div>
            <div className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-center">
              <Flag className="mx-auto mb-2 h-6 w-6 text-amber-600" />
              <p className="text-2xl font-bold text-amber-700">{totalFlagged}</p>
              <p className="text-xs text-amber-600">Flagged</p>
            </div>
            <div className={cn(
              "rounded-lg border p-4 text-center",
              remaining > 0
                ? "border-red-300 bg-red-50"
                : "border-green-300 bg-green-50",
            )}>
              {remaining > 0 ? (
                <AlertTriangle className="mx-auto mb-2 h-6 w-6 text-red-500" />
              ) : (
                <CheckCircle2 className="mx-auto mb-2 h-6 w-6 text-green-600" />
              )}
              <p className={cn(
                "text-2xl font-bold",
                remaining > 0 ? "text-red-700" : "text-green-700",
              )}>
                {remaining}
              </p>
              <p className={cn(
                "text-xs",
                remaining > 0 ? "text-red-500" : "text-green-600",
              )}>
                Remaining
              </p>
            </div>
          </div>

          {remaining > 0 && (
            <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              <AlertTriangle className="mr-2 inline h-4 w-4" />
              You must answer all {totalPairs} pairs before submitting.
            </div>
          )}

          {flaggedPairs.length > 0 && (
            <>
              <Separator className="bg-navy-100" />
              <div>
                <h3 className="mb-2 text-sm font-semibold text-navy-700">
                  Flagged Items ({flaggedPairs.length})
                </h3>
                <div className="max-h-40 space-y-1 overflow-y-auto">
                  {flaggedPairs.map((p) => {
                    const idx = pairs.indexOf(p)
                    return (
                      <button
                        key={p.pair_id}
                        onClick={() => { goTo(idx); onBack() }}
                        className="flex w-full items-center gap-2 rounded px-2 py-1 text-left text-sm text-navy-600 hover:bg-navy-50"
                      >
                        <Flag className="h-3 w-3 fill-amber-400 text-amber-500" />
                        <span className="font-mono text-xs">{p.pair_id}</span>
                        <span className="truncate text-navy-400">
                          {p.query_text.slice(0, 60)}...
                        </span>
                      </button>
                    )
                  })}
                </div>
              </div>
            </>
          )}

          <Separator className="bg-navy-100" />

          <div>
            <h3 className="mb-2 text-sm font-semibold text-navy-700">All Pairs</h3>
            <div className="grid grid-cols-10 gap-1.5">
              {pairs.map((p, i) => {
                const status = getStatus(p.pair_id)
                return (
                  <button
                    key={p.pair_id}
                    onClick={() => { goTo(i); onBack() }}
                    title={`${p.pair_id} — ${status}`}
                    className={cn(
                      "flex h-6 w-6 items-center justify-center rounded text-[10px] font-medium",
                      statusColors[status],
                      status === "answered" && "text-white",
                      status === "flagged-answered" && "text-navy-900",
                    )}
                  >
                    {i + 1}
                  </button>
                )
              })}
            </div>
          </div>

          {error && (
            <p className="text-sm text-red-600">{error}</p>
          )}

          <div className="flex justify-between pt-2">
            <Button
              variant="outline"
              onClick={onBack}
              className="border-navy-200 text-navy-600 hover:bg-navy-50"
            >
              <ArrowLeft className="mr-1.5 h-4 w-4" />
              Back to Annotating
            </Button>
            <Button
              onClick={handleSubmit}
              disabled={!canSubmit || submitting}
              className="bg-navy-700 text-white hover:bg-navy-600 disabled:opacity-50"
            >
              <Send className="mr-1.5 h-4 w-4" />
              {submitting ? "Submitting..." : "Submit All Answers"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
