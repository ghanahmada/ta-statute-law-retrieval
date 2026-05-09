import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { ArrowRight, BookOpen } from "lucide-react"

interface Props {
  onStart: () => void
}

export function GuidelinesContent() {
  return (
    <div className="space-y-6 text-left text-navy-800">
      <section>
        <h3 className="mb-2 text-lg font-semibold text-navy-700">Task Overview</h3>
        <p className="text-sm leading-relaxed">
          You will review <strong>80 query-article pairs</strong> from the Indonesian Civil Code
          (Kitab Undang-Undang Hukum Perdata / KUHPerdata). Each pair consists of a legal query
          and a specific article from the statute corpus. Your task is to judge whether the
          article is <strong>relevant</strong> to answering the legal question posed in the query.
        </p>
      </section>

      <Separator className="bg-navy-100" />

      <section>
        <h3 className="mb-2 text-lg font-semibold text-navy-700">Label Definitions</h3>
        <div className="space-y-3">
          <div className="rounded-lg border border-navy-200 bg-navy-50 p-3">
            <span className="font-semibold text-navy-700">RELEVANT</span>
            <p className="mt-1 text-sm">
              The article directly addresses or materially relates to the legal issue in the query.
              This includes articles that establish the rule, define key terms, set conditions, or
              provide exceptions relevant to the query's legal question.
            </p>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <span className="font-semibold text-slate-700">NOT RELEVANT</span>
            <p className="mt-1 text-sm">
              The article does not meaningfully connect to the legal issue in the query.
              Even if the article belongs to the same legal domain, it is not relevant unless
              it directly bears on the specific question asked.
            </p>
          </div>
        </div>
      </section>

      <Separator className="bg-navy-100" />

      <section>
        <h3 className="mb-2 text-lg font-semibold text-navy-700">Tips</h3>
        <ul className="ml-4 list-disc space-y-1 text-sm">
          <li>Read the <strong>full article text</strong> before making your judgment.</li>
          <li>
            Consider whether the article would be cited in a legal argument answering the query.
          </li>
          <li>
            Use the <strong>flag</strong> feature to mark pairs you want to revisit later.
          </li>
          <li>You can navigate freely between pairs using the numbered sidebar.</li>
          <li>You can change your answer at any time before final submission.</li>
          <li>
            The variant label (HUMANIZED / SUMMARIZED) indicates how the query was formulated
            — this is for tracking purposes and should not affect your relevance judgment.
          </li>
        </ul>
      </section>

      <Separator className="bg-navy-100" />

      <section>
        <h3 className="mb-2 text-lg font-semibold text-navy-700">Keyboard Shortcuts</h3>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <kbd className="rounded border border-navy-200 bg-navy-50 px-1.5 py-0.5 font-mono text-xs">R</kbd>{" "}
            — Mark as RELEVANT
          </div>
          <div>
            <kbd className="rounded border border-navy-200 bg-navy-50 px-1.5 py-0.5 font-mono text-xs">N</kbd>{" "}
            — Mark as NOT RELEVANT
          </div>
          <div>
            <kbd className="rounded border border-navy-200 bg-navy-50 px-1.5 py-0.5 font-mono text-xs">F</kbd>{" "}
            — Toggle flag
          </div>
          <div>
            <kbd className="rounded border border-navy-200 bg-navy-50 px-1.5 py-0.5 font-mono text-xs">←</kbd>{" "}
            <kbd className="rounded border border-navy-200 bg-navy-50 px-1.5 py-0.5 font-mono text-xs">→</kbd>{" "}
            — Navigate pairs
          </div>
        </div>
      </section>
    </div>
  )
}

export function GuidelinesPage({ onStart }: Props) {
  return (
    <div className="flex min-h-dvh items-center justify-center bg-navy-50 p-6">
      <Card className="w-full max-w-2xl border-navy-200 shadow-lg">
        <div className="flex items-center gap-3 border-b border-navy-100 px-6 py-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-navy-700">
            <BookOpen className="h-5 w-5 text-amber-400" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-navy-800">Annotation Guidelines</h2>
            <p className="text-sm text-navy-500">Please read carefully before starting</p>
          </div>
        </div>
        <CardContent className="max-h-[65vh] overflow-y-auto p-6">
          <GuidelinesContent />
        </CardContent>
        <div className="border-t border-navy-100 px-6 py-4">
          <Button
            onClick={onStart}
            className="w-full bg-navy-700 text-white hover:bg-navy-600"
          >
            Start Annotating
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </Card>
    </div>
  )
}
