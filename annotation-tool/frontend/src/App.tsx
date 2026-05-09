import { useCallback, useEffect, useState } from "react"
import { TooltipProvider } from "@/components/ui/tooltip"
import { LoginPage } from "@/components/LoginPage"
import { GuidelinesPage } from "@/components/GuidelinesPage"
import { AnnotatePage } from "@/components/AnnotatePage"
import { ConfirmationPage } from "@/components/ConfirmationPage"
import { CompletePage } from "@/components/CompletePage"
import { useAnnotation } from "@/hooks/useAnnotation"
import { api } from "@/lib/api"

type View = "login" | "guidelines" | "annotate" | "confirmation" | "complete"

export default function App() {
  const [view, setView] = useState<View>("login")
  const [annotatorName, setAnnotatorName] = useState("")
  const annotation = useAnnotation()

  useEffect(() => {
    const token = localStorage.getItem("session_token")
    if (!token) return
    api.status().then(async (res) => {
      if (res.authenticated && res.name) {
        setAnnotatorName(res.name)
        if (res.submitted) {
          annotation.setSubmitted(true)
          setView("complete")
        } else {
          await annotation.init()
          setView("annotate")
        }
      }
    }).catch(() => {
      localStorage.removeItem("session_token")
    })
  }, [])

  const handleLogin = useCallback(async (name: string, submitted: boolean) => {
    setAnnotatorName(name)
    if (submitted) {
      annotation.setSubmitted(true)
      setView("complete")
    } else {
      await annotation.init()
      setView("guidelines")
    }
  }, [annotation])

  return (
    <TooltipProvider>
      {view === "login" && <LoginPage onLogin={handleLogin} />}
      {view === "guidelines" && (
        <GuidelinesPage onStart={() => setView("annotate")} />
      )}
      {view === "annotate" && (
        <AnnotatePage
          annotatorName={annotatorName}
          onReview={() => setView("confirmation")}
          annotation={annotation}
        />
      )}
      {view === "confirmation" && (
        <ConfirmationPage
          annotation={annotation}
          onBack={() => setView("annotate")}
          onSubmitted={() => setView("complete")}
        />
      )}
      {view === "complete" && <CompletePage />}
    </TooltipProvider>
  )
}
