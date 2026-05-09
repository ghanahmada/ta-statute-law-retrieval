import { useState } from "react"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { BookOpen } from "lucide-react"
import { GuidelinesContent } from "./GuidelinesPage"

export function GuidelinesDialog() {
  const [open, setOpen] = useState(false)

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger
        render={
          <Button variant="outline" size="sm" className="border-navy-300 text-navy-700 hover:bg-navy-100">
            <BookOpen className="mr-1.5 h-4 w-4" />
            Guidelines
          </Button>
        }
      />
      <DialogContent className="max-w-2xl border-navy-200 p-0">
        <DialogHeader className="border-b border-navy-100 px-6 py-4">
          <DialogTitle className="text-navy-800">Annotation Guidelines</DialogTitle>
        </DialogHeader>
        <ScrollArea className="max-h-[70vh] px-6 py-4">
          <GuidelinesContent />
        </ScrollArea>
      </DialogContent>
    </Dialog>
  )
}
