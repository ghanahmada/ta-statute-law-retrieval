import csv
import io
import os
from itertools import combinations
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from sklearn.metrics import cohen_kappa_score
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Annotator, Flag, Label, Pair

router = APIRouter(prefix="/admin")

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


def _check_admin(authorization: str = Header(default="")):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN not configured")
    token = authorization.replace("Bearer ", "").strip()
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")


@router.post("/reset/{annotator_name}")
def reset_annotator(annotator_name: str, _: None = Depends(_check_admin), db: Session = Depends(get_db)):
    ann = db.query(Annotator).filter(Annotator.name == annotator_name).first()
    if not ann:
        raise HTTPException(status_code=404, detail="Annotator not found")
    deleted_labels = db.query(Label).filter(Label.annotator == annotator_name).delete()
    deleted_flags = db.query(Flag).filter(Flag.annotator == annotator_name).delete()
    ann.submitted_at = None
    db.commit()
    return {
        "success": True,
        "annotator": annotator_name,
        "labels_deleted": deleted_labels,
        "flags_deleted": deleted_flags,
    }


@router.get("/progress")
def progress(_: None = Depends(_check_admin), db: Session = Depends(get_db)):
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


@router.get("/agreement")
def agreement(_: None = Depends(_check_admin), db: Session = Depends(get_db)):
    pairs = {p.pair_id: p.llm_label for p in db.query(Pair).order_by(Pair.pair_id).all()}
    pair_ids = sorted(pairs.keys())
    annotators = [a.name for a in db.query(Annotator).all()]

    def norm(lbl: str) -> str:
        lbl = lbl.upper().strip()
        if lbl in ("RELEVAN", "RELEVANT"):
            return "RELEVANT"
        if lbl in ("TIDAK_RELEVAN", "NOT_RELEVANT"):
            return "NOT_RELEVANT"
        return lbl

    labels_by_ann: dict[str, dict[str, str]] = {}
    distribution = []
    for ann in annotators:
        rows = db.query(Label).filter(Label.annotator == ann).all()
        ann_labels = {r.pair_id: r.label for r in rows}
        labels_by_ann[ann] = ann_labels
        total = len(ann_labels)
        rel = sum(1 for v in ann_labels.values() if norm(v) == "RELEVANT")
        distribution.append({
            "annotator": ann,
            "labeled": total,
            "total": len(pair_ids),
            "relevant": rel,
            "not_relevant": total - rel,
        })

    ann_with_labels = [a for a in annotators if len(labels_by_ann.get(a, {})) > 0]

    pairwise = []
    for a1, a2 in combinations(ann_with_labels, 2):
        common = [pid for pid in pair_ids
                  if pid in labels_by_ann[a1] and pid in labels_by_ann[a2]]
        if len(common) < 2:
            pairwise.append({
                "annotator1": a1, "annotator2": a2,
                "kappa": None, "overlap": len(common),
                "agreement_pct": None,
            })
            continue
        y1 = [norm(labels_by_ann[a1][pid]) for pid in common]
        y2 = [norm(labels_by_ann[a2][pid]) for pid in common]
        k = cohen_kappa_score(y1, y2)
        agree = sum(1 for a, b in zip(y1, y2) if a == b)
        pairwise.append({
            "annotator1": a1, "annotator2": a2,
            "kappa": round(k, 4), "overlap": len(common),
            "agreement_pct": round(agree / len(common) * 100, 1),
        })

    vs_llm = []
    for ann in ann_with_labels:
        common = [pid for pid in pair_ids
                  if pid in labels_by_ann[ann] and pairs[pid]]
        if len(common) < 2:
            vs_llm.append({"annotator": ann, "kappa": None, "overlap": len(common)})
            continue
        y_ann = [norm(labels_by_ann[ann][pid]) for pid in common]
        y_llm = [norm(pairs[pid]) for pid in common]
        k = cohen_kappa_score(y_ann, y_llm)
        agree = sum(1 for a, b in zip(y_ann, y_llm) if a == b)
        vs_llm.append({
            "annotator": ann, "kappa": round(k, 4),
            "overlap": len(common),
            "agreement_pct": round(agree / len(common) * 100, 1),
        })

    avg_kappa = None
    kappas = [p["kappa"] for p in pairwise if p["kappa"] is not None]
    if kappas:
        avg_kappa = round(sum(kappas) / len(kappas), 4)

    return {
        "distribution": distribution,
        "pairwise_kappa": pairwise,
        "average_kappa": avg_kappa,
        "vs_llm": vs_llm,
    }


@router.get("/export")
def export(_: None = Depends(_check_admin), db: Session = Depends(get_db)):
    labels = db.query(Label).all()
    pairs = {p.pair_id: p for p in db.query(Pair).all()}

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["annotator", "pair_id", "case_id", "variant", "article_id",
                     "kuhperdata_book", "llm_label", "label", "confidence",
                     "reasoning", "created_at"])
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
            lbl.reasoning or "",
            lbl.created_at.isoformat() if lbl.created_at else "",
        ])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=labels.csv"},
    )
