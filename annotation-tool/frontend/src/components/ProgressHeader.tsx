import { Progress } from "@/components/ui/progress"
import { GuidelinesDialog } from "./GuidelinesDialog"
import { User } from "lucide-react"

interface Props {
  annotatorName: string
  answered: number
  total: number
}

export function ProgressHeader({ annotatorName, answered, total }: Props) {
  const pct = total > 0 ? Math.round((answered / total) * 100) : 0

  return (
    <header className="flex items-center gap-4 border-b border-navy-200 bg-white px-4 py-3">
      <div className="flex items-center gap-2 text-sm font-medium text-navy-700">
        <User className="h-4 w-4 text-navy-500" />
        {annotatorName}
      </div>

      <div className="flex flex-1 items-center gap-3">
        <Progress
          value={pct}
          className="h-2.5 flex-1 bg-navy-100 [&>div]:bg-navy-600"
        />
        <span className="min-w-[60px] text-right text-sm font-semibold text-navy-700">
          {answered}/{total}
        </span>
      </div>

      <GuidelinesDialog />
    </header>
  )
}
