import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Flag } from "lucide-react"
import { cn } from "@/lib/utils"
import type { Pair } from "@/types"

interface Props {
  pair: Pair
  index: number
  total: number
  isFlagged: boolean
  submitted: boolean
  onToggleFlag: () => void
}

export function PairView({ pair, index, total, isFlagged, submitted, onToggleFlag }: Props) {
  const isHumanized = pair.variant === "humanized"

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Badge
          className={
            isHumanized
              ? "border-navy-300 bg-navy-100 text-navy-700 hover:bg-navy-100"
              : "border-amber-400 bg-amber-100 text-amber-700 hover:bg-amber-100"
          }
        >
          {pair.variant.toUpperCase()}
        </Badge>
        <span className="text-sm text-navy-500">
          Pair {index + 1} of {total} — {pair.pair_id}
        </span>
        <span className="text-sm text-navy-400">
          {pair.kuhperdata_book}
        </span>
        <Button
          variant="outline"
          size="sm"
          onClick={onToggleFlag}
          disabled={submitted}
          className={cn(
            "ml-auto h-7 px-2 transition-all",
            isFlagged
              ? "border-amber-400 bg-amber-100 text-amber-700 hover:bg-amber-200"
              : "border-navy-200 text-navy-400 hover:bg-navy-50 hover:text-amber-600",
          )}
        >
          <Flag className={cn("h-3.5 w-3.5", isFlagged && "fill-amber-500")} />
        </Button>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="border-navy-200">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-semibold uppercase tracking-wide text-navy-500">
              Query
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm leading-relaxed text-navy-800">{pair.query_text}</p>
          </CardContent>
        </Card>

        <Card className="border-navy-200">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-semibold uppercase tracking-wide text-navy-500">
              Article — {pair.article_title || pair.article_id}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="whitespace-pre-wrap text-sm leading-relaxed text-navy-800">
              {pair.article_text}
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
