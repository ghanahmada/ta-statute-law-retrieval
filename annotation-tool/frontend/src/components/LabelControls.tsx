import { useCallback, useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import type { LabelData } from "@/types"
import {
  ChevronLeft,
  ChevronRight,
  Flag,
} from "lucide-react"
import { cn } from "@/lib/utils"

interface Props {
  pairId: string
  existingLabel: LabelData | null
  isFlagged: boolean
  isFirst: boolean
  isLast: boolean
  submitted: boolean
  onSave: (label: "RELEVANT" | "NOT_RELEVANT", reasoning?: string) => void
  onToggleFlag: () => void
  onPrev: () => void
  onNext: () => void
}

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
  const [reasoning, setReasoning] = useState(existingLabel?.reasoning ?? "")
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    setLabel(existingLabel?.label ?? null)
    setReasoning(existingLabel?.reasoning ?? "")
  }, [pairId, existingLabel])

  const selectLabel = useCallback(
    (value: "RELEVANT" | "NOT_RELEVANT") => {
      setLabel(value)
      onSave(value, reasoning)
    },
    [onSave, reasoning],
  )

  const handleReasoningChange = useCallback(
    (text: string) => {
      setReasoning(text)
      if (!label) return
      if (debounceRef.current) clearTimeout(debounceRef.current)
      debounceRef.current = setTimeout(() => {
        onSave(label, text)
      }, 800)
    },
    [label, onSave],
  )

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [])

  useEffect(() => {
    if (submitted) return
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      switch (e.key.toLowerCase()) {
        case "r":
          selectLabel("RELEVANT")
          break
        case "n":
          selectLabel("NOT_RELEVANT")
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
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [submitted, selectLabel, onToggleFlag, onPrev, onNext])

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <button
          onClick={() => selectLabel("RELEVANT")}
          disabled={submitted}
          className={cn(
            "rounded-lg border-2 p-4 text-left transition-all",
            label === "RELEVANT"
              ? "border-navy-600 bg-navy-50 shadow-md"
              : "border-slate-200 bg-white hover:border-navy-300 hover:bg-navy-50/50",
            submitted && "opacity-60 cursor-not-allowed",
          )}
        >
          <div className="flex items-center gap-2">
            <span className={cn(
              "flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold",
              label === "RELEVANT"
                ? "bg-navy-600 text-white"
                : "border-2 border-slate-300 text-slate-400",
            )}>
              A
            </span>
            <span className={cn(
              "text-sm font-semibold",
              label === "RELEVANT" ? "text-navy-700" : "text-slate-600",
            )}>
              Relevant
            </span>
          </div>
          {label === "RELEVANT" && (
            <textarea
              value={reasoning}
              onChange={(e) => handleReasoningChange(e.target.value)}
              onClick={(e) => e.stopPropagation()}
              placeholder="Reasoning (optional)"
              disabled={submitted}
              rows={2}
              className="mt-3 w-full resize-none rounded-md border border-navy-200 bg-white px-3 py-2 text-sm text-navy-700 placeholder:text-navy-300 focus:border-navy-400 focus:outline-none"
            />
          )}
        </button>

        <button
          onClick={() => selectLabel("NOT_RELEVANT")}
          disabled={submitted}
          className={cn(
            "rounded-lg border-2 p-4 text-left transition-all",
            label === "NOT_RELEVANT"
              ? "border-slate-500 bg-slate-50 shadow-md"
              : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/50",
            submitted && "opacity-60 cursor-not-allowed",
          )}
        >
          <div className="flex items-center gap-2">
            <span className={cn(
              "flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold",
              label === "NOT_RELEVANT"
                ? "bg-slate-500 text-white"
                : "border-2 border-slate-300 text-slate-400",
            )}>
              B
            </span>
            <span className={cn(
              "text-sm font-semibold",
              label === "NOT_RELEVANT" ? "text-slate-700" : "text-slate-600",
            )}>
              Not Relevant
            </span>
          </div>
          {label === "NOT_RELEVANT" && (
            <textarea
              value={reasoning}
              onChange={(e) => handleReasoningChange(e.target.value)}
              onClick={(e) => e.stopPropagation()}
              placeholder="Reasoning (optional)"
              disabled={submitted}
              rows={2}
              className="mt-3 w-full resize-none rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 placeholder:text-slate-300 focus:border-slate-400 focus:outline-none"
            />
          )}
        </button>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
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
        <Button
          onClick={onNext}
          disabled={isLast}
          className="bg-navy-700 text-white hover:bg-navy-600"
        >
          Next
          <ChevronRight className="ml-1 h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
