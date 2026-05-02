import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Annotator

router = APIRouter(prefix="/auth")


class LoginRequest(BaseModel):
    name: str


@router.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    annotator = db.query(Annotator).filter(Annotator.name == req.name).first()
    if not annotator:
        raise HTTPException(status_code=404, detail="Annotator not found")
    token = str(uuid.uuid4())
    annotator.session_token = token
    db.commit()
    return {"session_token": token, "annotator_name": annotator.name}


def get_current_annotator(authorization: str, db: Session) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.removeprefix("Bearer ").strip()
    annotator = db.query(Annotator).filter(Annotator.session_token == token).first()
    if not annotator:
        raise HTTPException(status_code=401, detail="Invalid token")
    return annotator.name
