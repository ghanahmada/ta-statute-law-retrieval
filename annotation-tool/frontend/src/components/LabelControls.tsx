import { useCallback, useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import type { LabelData } from "@/types"
import {
  ChevronLeft,
  ChevronRight,
  Flag,
  Save,
} from "lucide-react"
import { cn } from "@/lib/utils"

interface Props {
  pairId: string
  existingLabel: LabelData | null
  isFlagged: boolean
  isFirst: boolean
  isLast: boolean
  submitted: boolean
  onSave: (label: "RELEVANT" | "NOT_RELEVANT", confidence: "low" | "medium" | "high") => void
  onToggleFlag: () => void
  onPrev: () => void
  onNext: () => void
}

const confidenceLevels = ["low", "medium", "high"] as const

export function LabelControls({
  pairId,
  existingLabel,
  isFlagged,
  isFirst,
  isLast,
  submitted,
  onSave,
  onToggleFlag,
  onPrev,
  onNext,
}: Props) {
  const [label, setLabel] = useState<"RELEVANT" | "NOT_RELEVANT" | null>(
    existingLabel?.label ?? null,
  )
  const [confidence, setConfidence] = useState<"low" | "medium" | "high">(
    existingLabel?.confidence ?? "high",
  )

  useEffect(() => {
    setLabel(existingLabel?.label ?? null)
    setConfidence(existingLabel?.confidence ?? "high")
  }, [pairId, existingLabel])

  const handleSaveAndNext = useCallback(() => {
    if (!label) return
    onSave(label, confidence)
    if (!isLast) onNext()
  }, [label, confidence, onSave, onNext, isLast])

  useEffect(() => {
    if (submitted) return
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      switch (e.key.toLowerCase()) {
        case "r":
          setLabel("RELEVANT")
          break
        case "n":
          setLabel("NOT_RELEVANT")
          break
        case "f":
          onToggleFlag()
          break
        case "arrowleft":
          e.preventDefault()
          onPrev()
          break
        case "arrowright":
          e.preventDefault()
          onNext()
          break
        case "enter":
          e.preventDefault()
          handleSaveAndNext()
          break
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [submitted, onToggleFlag, onPrev, onNext, handleSaveAndNext])

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Button
          onClick={() => setLabel("RELEVANT")}
          disabled={submitted}
          className={cn(
            "min-w-[140px] text-sm font-semibold transition-all",
            label === "RELEVANT"
              ? "bg-navy-600 text-white shadow-md hover:bg-navy-700"
              : "border border-navy-300 bg-white text-navy-600 hover:bg-navy-50",
          )}
        >
          RELEVANT
        </Button>
        <Button
          onClick={() => setLabel("NOT_RELEVANT")}
          disabled={submitted}
          className={cn(
            "min-w-[140px] text-sm font-semibold transition-all",
            label === "NOT_RELEVANT"
              ? "bg-slate-500 text-white shadow-md hover:bg-slate-600"
              : "border border-slate-300 bg-white text-slate-500 hover:bg-slate-50",
          )}
        >
          NOT RELEVANT
        </Button>
        <Button
          variant="outline"
          onClick={onToggleFlag}
          disabled={submitted}
          className={cn(
            "transition-all",
            isFlagged
              ? "border-amber-400 bg-amber-100 text-amber-700 hover:bg-amber-200"
              : "border-navy-200 text-navy-400 hover:bg-navy-50 hover:text-amber-600",
          )}
        >
          <Flag className={cn("h-4 w-4", isFlagged && "fill-amber-500")} />
        </Button>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-navy-500">Confidence:</span>
        {confidenceLevels.map((level) => (
          <button
            key={level}
            onClick={() => setConfidence(level)}
            disabled={submitted}
            className={cn(
              "rounded-md px-3 py-1 text-xs font-medium capitalize transition-all",
              confidence === level
                ? "bg-navy-700 text-white"
                : "border border-navy-200 bg-white text-navy-500 hover:bg-navy-50",
            )}
          >
            {level}
          </button>
        ))}
      </div>

      <div className="flex items-center justify-between pt-2">
        <Button
          variant="outline"
          onClick={onPrev}
          disabled={isFirst}
          className="border-navy-200 text-navy-600 hover:bg-navy-50"
        >
          <ChevronLeft className="mr-1 h-4 w-4" />
          Previous
        </Button>
        <Button
          onClick={handleSaveAndNext}
          disabled={!label || submitted}
          className="bg-navy-700 text-white hover:bg-navy-600"
        >
          <Save className="mr-1.5 h-4 w-4" />
          {existingLabel ? "Update" : "Save"}
          {!isLast && " & Next"}
          {!isLast && <ChevronRight className="ml-1 h-4 w-4" />}
        </Button>
      </div>
    </div>
  )
}
