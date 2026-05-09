import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { api } from "@/lib/api"
import { KeyRound, Scale } from "lucide-react"

interface Props {
  onLogin: (name: string, submitted: boolean) => void
}

export function LoginPage({ onLogin }: Props) {
  const [token, setToken] = useState("")
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!token.trim()) return
    setLoading(true)
    setError("")
    try {
      const res = await api.login(token.trim())
      localStorage.setItem("session_token", res.session_token)
      localStorage.setItem("annotator_name", res.annotator_name)
      onLogin(res.annotator_name, res.submitted)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-dvh items-center justify-center bg-navy-50 p-4">
      <Card className="w-full max-w-md border-navy-200 shadow-lg">
        <CardHeader className="text-center pb-2">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-navy-700">
            <Scale className="h-7 w-7 text-amber-400" />
          </div>
          <CardTitle className="text-2xl font-semibold text-navy-800">
            Legal Annotation Study
          </CardTitle>
          <p className="mt-1 text-sm text-navy-500">
            KUHPerdata Query-Article Relevance Assessment
          </p>
          <div className="mx-auto mt-3 h-1 w-16 rounded-full bg-amber-400" />
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label
                htmlFor="token"
                className="mb-1.5 block text-sm font-medium text-navy-700"
              >
                Access Token
              </label>
              <div className="relative">
                <KeyRound className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-navy-400" />
                <input
                  id="token"
                  type="password"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder="Paste your access token"
                  className="w-full rounded-md border border-navy-200 bg-white py-2.5 pl-10 pr-4 text-sm text-navy-800 placeholder:text-navy-300 focus:border-navy-500 focus:outline-none focus:ring-2 focus:ring-navy-500/20"
                  autoFocus
                />
              </div>
            </div>
            {error && (
              <p className="text-sm text-red-600">{error}</p>
            )}
            <Button
              type="submit"
              disabled={loading || !token.trim()}
              className="w-full bg-navy-700 text-white hover:bg-navy-600"
            >
              {loading ? "Verifying..." : "Continue"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
