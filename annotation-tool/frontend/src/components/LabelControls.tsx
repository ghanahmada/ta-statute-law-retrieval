import { useCallback, useEffect, useRef, useState } from "react"
import type { LabelData } from "@/types"
import { cn } from "@/lib/utils"

interface Props {
  pairId: string
  existingLabel: LabelData | null
  submitted: boolean
  onSave: (label: "RELEVANT" | "NOT_RELEVANT", reasoning?: string) => void
}

export function LabelControls({
  pairId,
  existingLabel,
  submitted,
  onSave,
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
      if (submitted) return
      setLabel(value)
      onSave(value, reasoning)
    },
    [onSave, reasoning, submitted],
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

  return (
    <div className="grid grid-cols-2 gap-3">
      <div
        className={cn(
          "rounded-lg border-2 p-4 text-left transition-all",
          label === "RELEVANT"
            ? "border-navy-600 bg-navy-50 shadow-md"
            : "border-slate-200 bg-white hover:border-navy-300 hover:bg-navy-50/50",
          submitted && "opacity-60",
        )}
      >
        <div
          onClick={() => selectLabel("RELEVANT")}
          className={cn("flex cursor-pointer items-center gap-2", submitted && "cursor-not-allowed")}
        >
          <span className={cn(
            "flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold",
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
            placeholder="Reasoning (optional)"
            disabled={submitted}
            rows={2}
            className="mt-3 w-full resize-none rounded-md border border-navy-200 bg-white px-3 py-2 text-sm text-navy-700 placeholder:text-navy-300 focus:border-navy-400 focus:outline-none"
          />
        )}
      </div>

      <div
        className={cn(
          "rounded-lg border-2 p-4 text-left transition-all",
          label === "NOT_RELEVANT"
            ? "border-slate-500 bg-slate-50 shadow-md"
            : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/50",
          submitted && "opacity-60",
        )}
      >
        <div
          onClick={() => selectLabel("NOT_RELEVANT")}
          className={cn("flex cursor-pointer items-center gap-2", submitted && "cursor-not-allowed")}
        >
          <span className={cn(
            "flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold",
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
            placeholder="Reasoning (optional)"
            disabled={submitted}
            rows={2}
            className="mt-3 w-full resize-none rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 placeholder:text-slate-300 focus:border-slate-400 focus:outline-none"
          />
        )}
      </div>
    </div>
  )
}
