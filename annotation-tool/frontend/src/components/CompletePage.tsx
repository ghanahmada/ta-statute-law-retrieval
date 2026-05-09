import { Card, CardContent } from "@/components/ui/card"
import { CheckCircle2, Scale } from "lucide-react"

export function CompletePage() {
  return (
    <div className="flex min-h-dvh items-center justify-center bg-navy-50 p-6">
      <Card className="w-full max-w-md border-navy-200 text-center shadow-lg">
        <CardContent className="space-y-4 pt-8 pb-8">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-navy-700">
            <CheckCircle2 className="h-9 w-9 text-amber-400" />
          </div>
          <h2 className="text-2xl font-semibold text-navy-800">
            All Done!
          </h2>
          <p className="text-sm text-navy-500">
            All 80 pairs have been submitted successfully.
            Thank you for your participation in this study.
          </p>
          <div className="mx-auto h-1 w-16 rounded-full bg-amber-400" />
          <div className="flex items-center justify-center gap-2 pt-2 text-xs text-navy-400">
            <Scale className="h-3.5 w-3.5" />
            KUHPerdata Annotation Study
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
