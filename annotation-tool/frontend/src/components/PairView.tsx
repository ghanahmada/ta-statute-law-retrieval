import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { Pair } from "@/types"

interface Props {
  pair: Pair
  index: number
  total: number
}

export function PairView({ pair, index, total }: Props) {
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
            <ScrollArea className="max-h-[45vh]">
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-navy-800">
                {pair.article_text}
              </p>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
