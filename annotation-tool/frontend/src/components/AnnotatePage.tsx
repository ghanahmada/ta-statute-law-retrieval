import { useEffect } from "react"
import { useAnnotation } from "@/hooks/useAnnotation"
import { Button } from "@/components/ui/button"
import { ProgressHeader } from "./ProgressHeader"
import { PairView } from "./PairView"
import { LabelControls } from "./LabelControls"
import { Sidebar } from "./Sidebar"
import { Separator } from "@/components/ui/separator"
import { ChevronLeft, ChevronRight } from "lucide-react"

interface Props {
  annotatorName: string
  onReview: () => void
  annotation: ReturnType<typeof useAnnotation>
}

export function AnnotatePage({ annotatorName, onReview, annotation }: Props) {
  const {
    pairs,
    labels,
    flagged,
    currentIndex,
    currentPair,
    loading,
    submitted,
    totalAnswered,
    totalPairs,
    init,
    goTo,
    next,
    prev,
    saveLabel,
    toggleFlag,
    getStatus,
  } = annotation

  useEffect(() => {
    if (pairs.length === 0 && !loading) {
      init()
    }
  }, [pairs.length, loading, init])

  useEffect(() => {
    if (submitted) return
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      switch (e.key.toLowerCase()) {
        case "r": {
          const existing = currentPair ? labels[currentPair.pair_id] : null
          if (currentPair) saveLabel("RELEVANT", existing?.reasoning)
          break
        }
        case "n": {
          const existing = currentPair ? labels[currentPair.pair_id] : null
          if (currentPair) saveLabel("NOT_RELEVANT", existing?.reasoning)
          break
        }
        case "f":
          if (currentPair) toggleFlag(currentPair.pair_id)
          break
        case "arrowleft":
          e.preventDefault()
          prev()
          break
        case "arrowright":
          e.preventDefault()
          next()
          break
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [submitted, currentPair, labels, saveLabel, toggleFlag, prev, next])

  if (loading || (pairs.length === 0)) {
    return (
      <div className="flex min-h-dvh items-center justify-center bg-navy-50">
        <p className="text-navy-500">Loading annotation data...</p>
      </div>
    )
  }

  if (!currentPair) {
    return (
      <div className="flex min-h-dvh items-center justify-center bg-navy-50">
        <p className="text-navy-500">No pairs available.</p>
      </div>
    )
  }

  const existingLabel = labels[currentPair.pair_id] ?? null
  const isFlagged = flagged.has(currentPair.pair_id)

  return (
    <div className="flex h-dvh bg-navy-50">
      <main className="flex flex-1 flex-col overflow-hidden">
        <ProgressHeader
          annotatorName={annotatorName}
          answered={totalAnswered}
          total={totalPairs}
        />
        <div className="flex-1 overflow-y-auto p-6">
          <PairView
            pair={currentPair}
            index={currentIndex}
            total={totalPairs}
            isFlagged={isFlagged}
            submitted={submitted}
            onToggleFlag={() => toggleFlag(currentPair.pair_id)}
          />
          <Separator className="my-6 bg-navy-100" />
          <LabelControls
            pairId={currentPair.pair_id}
            existingLabel={existingLabel}
            submitted={submitted}
            onSave={saveLabel}
          />
        </div>

        <div className="flex items-center justify-between border-t border-navy-200 bg-white px-6 py-3">
          <Button
            variant="outline"
            onClick={prev}
            disabled={currentIndex === 0}
            className="border-navy-200 text-navy-600 hover:bg-navy-50"
          >
            <ChevronLeft className="mr-1 h-4 w-4" />
            Previous
          </Button>
          <span className="text-sm text-navy-400">
            {currentIndex + 1} / {totalPairs}
          </span>
          <Button
            onClick={next}
            disabled={currentIndex === totalPairs - 1}
            className="bg-navy-700 text-white hover:bg-navy-600"
          >
            Next
            <ChevronRight className="ml-1 h-4 w-4" />
          </Button>
        </div>
      </main>

      <Sidebar
        total={totalPairs}
        currentIndex={currentIndex}
        submitted={submitted}
        getStatus={(i) => getStatus(pairs[i]?.pair_id ?? "")}
        getPairId={(i) => pairs[i]?.pair_id ?? ""}
        onGoTo={goTo}
        onReview={onReview}
      />
    </div>
  )
}
