from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Pair, Label
from app.routes.auth import get_current_annotator

router = APIRouter(prefix="/pairs")


class LabelRequest(BaseModel):
    pair_id: str
    label: str
    confidence: str


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
        raise HTTPException(status_code=409, detail="Already labeled")

    db.add(Label(annotator=annotator, pair_id=req.pair_id, label=req.label, confidence=req.confidence))
    db.commit()

    labeled_ids = {
        row.pair_id
        for row in db.query(Label.pair_id).filter(Label.annotator == annotator).all()
    }
    next_pair = (
        db.query(Pair)
        .filter(Pair.pair_id.notin_(labeled_ids))
        .order_by(Pair.pair_id)
        .first()
    )

    return {"success": True, "next_pair_id": next_pair.pair_id if next_pair else None}
