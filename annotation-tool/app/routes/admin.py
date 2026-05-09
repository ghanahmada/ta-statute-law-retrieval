import csv
import io
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Annotator, Label, Pair

router = APIRouter(prefix="/admin")


@router.get("/progress")
def progress(db: Session = Depends(get_db)):
    total = db.query(Pair).count()
    annotators = db.query(Annotator).all()
    rows = []
    for ann in annotators:
        done = db.query(Label).filter(Label.annotator == ann.name).count()
        rows.append({
            "annotator": ann.name,
            "done": done,
            "remaining": total - done,
            "pct": round(100 * done / total, 1) if total else 0,
            "submitted": ann.submitted_at is not None,
            "submitted_at": ann.submitted_at.isoformat() if ann.submitted_at else None,
        })
    overall = db.query(Label).count()
    return {"total_pairs": total, "annotators": rows, "total_labels": overall}


@router.get("/export")
def export(db: Session = Depends(get_db)):
    labels = db.query(Label).all()
    pairs = {p.pair_id: p for p in db.query(Pair).all()}

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["annotator", "pair_id", "case_id", "variant", "article_id",
                     "kuhperdata_book", "llm_label", "label", "confidence", "created_at"])
    for lbl in labels:
        pair = pairs.get(lbl.pair_id)
        writer.writerow([
            lbl.annotator,
            lbl.pair_id,
            pair.case_id if pair else "",
            pair.variant if pair else "",
            pair.article_id if pair else "",
            pair.kuhperdata_book if pair else "",
            pair.llm_label if pair else "",
            lbl.label,
            lbl.confidence,
            lbl.created_at.isoformat() if lbl.created_at else "",
        ])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=labels.csv"},
    )
