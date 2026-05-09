import { useAnnotation } from "@/hooks/useAnnotation"
import { ProgressHeader } from "./ProgressHeader"
import { PairView } from "./PairView"
import { LabelControls } from "./LabelControls"
import { Sidebar } from "./Sidebar"
import { Separator } from "@/components/ui/separator"

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
    goTo,
    next,
    prev,
    saveLabel,
    toggleFlag,
    getStatus,
  } = annotation

  if (loading) {
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
          <PairView pair={currentPair} index={currentIndex} total={totalPairs} />
          <Separator className="my-6 bg-navy-100" />
          <LabelControls
            pairId={currentPair.pair_id}
            existingLabel={existingLabel}
            isFlagged={isFlagged}
            isFirst={currentIndex === 0}
            isLast={currentIndex === totalPairs - 1}
            submitted={submitted}
            onSave={saveLabel}
            onToggleFlag={() => toggleFlag(currentPair.pair_id)}
            onPrev={prev}
            onNext={next}
          />
        </div>
      </main>

      <Sidebar
        total={totalPairs}
        currentIndex={currentIndex}
        getStatus={(i) => getStatus(pairs[i]?.pair_id ?? "")}
        getPairId={(i) => pairs[i]?.pair_id ?? ""}
        onGoTo={goTo}
        onReview={onReview}
      />
    </div>
  )
}
