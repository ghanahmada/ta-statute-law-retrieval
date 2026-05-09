import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"
import { ClipboardCheck } from "lucide-react"

type PairStatus = "unanswered" | "answered" | "flagged" | "flagged-answered"

interface Props {
  total: number
  currentIndex: number
  getStatus: (index: number) => PairStatus
  getPairId: (index: number) => string
  onGoTo: (index: number) => void
  onReview: () => void
}

const statusStyles: Record<PairStatus, string> = {
  unanswered: "border border-slate-300 bg-white text-slate-400 hover:border-navy-400",
  answered: "bg-navy-600 text-white hover:bg-navy-700",
  flagged: "border border-amber-400 bg-amber-100 text-amber-700 hover:bg-amber-200",
  "flagged-answered": "bg-amber-400 text-navy-900 ring-1 ring-navy-600 hover:bg-amber-500",
}

export function Sidebar({
  total,
  currentIndex,
  getStatus,
  getPairId,
  onGoTo,
  onReview,
}: Props) {
  return (
    <aside className="flex w-64 flex-col border-l border-navy-200 bg-white">
      <ScrollArea className="flex-1 p-3">
        <div className="grid grid-cols-8 gap-1.5">
          {Array.from({ length: total }, (_, i) => {
            const status = getStatus(i)
            const isCurrent = i === currentIndex
            return (
              <button
                key={i}
                onClick={() => onGoTo(i)}
                title={`${getPairId(i)} — ${status}`}
                className={cn(
                  "flex h-7 w-7 items-center justify-center rounded text-xs font-medium transition-all",
                  statusStyles[status],
                  isCurrent && "ring-2 ring-amber-400 scale-110",
                )}
              >
                {i + 1}
              </button>
            )
          })}
        </div>
      </ScrollArea>

      <Separator className="bg-navy-100" />

      <div className="p-3">
        <Button
          variant="outline"
          size="sm"
          onClick={onReview}
          className="w-full border-amber-400 text-amber-700 hover:bg-amber-50"
        >
          <ClipboardCheck className="mr-1.5 h-4 w-4" />
          Review & Submit
        </Button>
      </div>

      <Separator className="bg-navy-100" />

      <div className="space-y-1 p-3 text-xs text-navy-500">
        <div className="flex items-center gap-2">
          <span className="inline-block h-3 w-3 rounded border border-slate-300 bg-white" />
          Unanswered
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block h-3 w-3 rounded bg-navy-600" />
          Answered
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block h-3 w-3 rounded border border-amber-400 bg-amber-100" />
          Flagged
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block h-3 w-3 rounded bg-amber-400 ring-1 ring-navy-600" />
          Flagged + Answered
        </div>
      </div>
    </aside>
  )
}
