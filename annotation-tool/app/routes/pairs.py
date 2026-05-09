from datetime import datetime
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Annotator, Pair, Label, Flag
from app.routes.auth import get_current_annotator

router = APIRouter(prefix="/pairs")


class LabelRequest(BaseModel):
    pair_id: str
    label: str
    confidence: str


class FlagRequest(BaseModel):
    pair_id: str
    flagged: bool


def _check_not_submitted(annotator_name: str, db: Session):
    ann = db.query(Annotator).filter(Annotator.name == annotator_name).first()
    if ann and ann.submitted_at:
        raise HTTPException(status_code=403, detail="Already submitted — labels are locked")


@router.get("/all")
def all_pairs(
    authorization: str = Header(default=""),
    db: Session = Depends(get_db),
):
    get_current_annotator(authorization, db)
    pairs = db.query(Pair).order_by(Pair.pair_id).all()
    return [
        {
            "pair_id": p.pair_id,
            "case_id": p.case_id,
            "variant": p.variant,
            "query_text": p.query_text,
            "article_id": p.article_id,
            "article_title": p.article_title,
            "article_text": p.article_text,
            "kuhperdata_book": p.kuhperdata_book,
        }
        for p in pairs
    ]


@router.get("/labels")
def get_labels(
    authorization: str = Header(default=""),
    db: Session = Depends(get_db),
):
    annotator = get_current_annotator(authorization, db)
    labels = db.query(Label).filter(Label.annotator == annotator).all()
    flags = db.query(Flag).filter(Flag.annotator == annotator).all()
    return {
        "labels": {
            lbl.pair_id: {
                "label": lbl.label,
                "confidence": lbl.confidence,
            }
            for lbl in labels
        },
        "flagged": [f.pair_id for f in flags],
    }


@router.get("/next")
def next_pair(
    authorization: str = Header(default=""),
    db: Session = Depends(get_db),
):
    annotator = get_current_annotator(authorization, db)
    labeled_ids = {
        row.pair_id
        for row in db.query(Label.pair_id).filter(Label.annotator == annotator).all()
    }
    total = db.query(Pair).count()
    done = len(labeled_ids)

    pair = (
        db.query(Pair)
        .filter(Pair.pair_id.notin_(labeled_ids))
        .order_by(Pair.pair_id)
        .first()
    )
    if not pair:
        return {"done": True, "progress": {"done": done, "total": total}}

    return {
        "done": False,
        "pair_id": pair.pair_id,
        "case_id": pair.case_id,
        "variant": pair.variant,
        "query_text": pair.query_text,
        "article_id": pair.article_id,
        "article_title": pair.article_title,
        "article_text": pair.article_text,
        "kuhperdata_book": pair.kuhperdata_book,
        "progress": {"done": done, "total": total},
    }


@router.post("/label")
def submit_label(
    req: LabelRequest,
    authorization: str = Header(default=""),
    db: Session = Depends(get_db),
):
    annotator = get_current_annotator(authorization, db)
    _check_not_submitted(annotator, db)

    if req.label not in ("RELEVANT", "NOT_RELEVANT"):
        raise HTTPException(status_code=400, detail="label must be RELEVANT or NOT_RELEVANT")
    if req.confidence not in ("low", "medium", "high"):
        raise HTTPException(status_code=400, detail="confidence must be low/medium/high")

    existing = (
        db.query(Label)
        .filter(Label.annotator == annotator, Label.pair_id == req.pair_id)
        .first()
    )
    if existing:
        existing.label = req.label
        existing.confidence = req.confidence
        existing.created_at = datetime.utcnow()
    else:
        db.add(Label(
            annotator=annotator,
            pair_id=req.pair_id,
            label=req.label,
            confidence=req.confidence,
        ))
    db.commit()
    return {"success": True}


@router.post("/flag")
def toggle_flag(
    req: FlagRequest,
    authorization: str = Header(default=""),
    db: Session = Depends(get_db),
):
    annotator = get_current_annotator(authorization, db)
    _check_not_submitted(annotator, db)

    existing = (
        db.query(Flag)
        .filter(Flag.annotator == annotator, Flag.pair_id == req.pair_id)
        .first()
    )
    if req.flagged and not existing:
        db.add(Flag(annotator=annotator, pair_id=req.pair_id))
    elif not req.flagged and existing:
        db.delete(existing)
    db.commit()
    return {"success": True}


@router.post("/submit")
def final_submit(
    authorization: str = Header(default=""),
    db: Session = Depends(get_db),
):
    annotator_name = get_current_annotator(authorization, db)
    ann = db.query(Annotator).filter(Annotator.name == annotator_name).first()
    if ann.submitted_at:
        raise HTTPException(status_code=409, detail="Already submitted")

    total = db.query(Pair).count()
    done = db.query(Label).filter(Label.annotator == annotator_name).count()
    if done < total:
        raise HTTPException(
            status_code=400,
            detail=f"Only {done}/{total} pairs labeled. Complete all pairs before submitting.",
        )

    ann.submitted_at = datetime.utcnow()
    db.commit()
    return {"success": True, "submitted_at": ann.submitted_at.isoformat()}
