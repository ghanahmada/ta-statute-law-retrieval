import uuid
import bcrypt
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Annotator

router = APIRouter(prefix="/auth")


class LoginRequest(BaseModel):
    token: str


@router.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    for ann in db.query(Annotator).all():
        if not ann.access_token_hash:
            continue
        if bcrypt.checkpw(req.token.encode("utf-8"), ann.access_token_hash.encode("utf-8")):
            session = str(uuid.uuid4())
            ann.session_token = session
            db.commit()
            return {
                "session_token": session,
                "annotator_name": ann.name,
                "submitted": ann.submitted_at is not None,
            }
    raise HTTPException(status_code=401, detail="Invalid access token")


@router.get("/status")
def auth_status(authorization: str = Header(default=""), db: Session = Depends(get_db)):
    try:
        name = get_current_annotator(authorization, db)
        ann = db.query(Annotator).filter(Annotator.name == name).first()
        return {
            "authenticated": True,
            "name": name,
            "submitted": ann.submitted_at is not None if ann else False,
        }
    except HTTPException:
        return {"authenticated": False}


def get_current_annotator(authorization: str, db: Session) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.removeprefix("Bearer ").strip()
    annotator = db.query(Annotator).filter(Annotator.session_token == token).first()
    if not annotator:
        raise HTTPException(status_code=401, detail="Invalid token")
    return annotator.name
