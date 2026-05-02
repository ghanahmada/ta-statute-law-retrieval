import csv
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from app.database import Base, engine, SessionLocal
from app.models import Pair, Annotator
from app.routes import auth, pairs, admin

PAIRS_TSV = os.getenv("PAIRS_TSV", "/app/data/pairs.tsv")
ANNOTATORS_TXT = os.getenv("ANNOTATORS_TXT", "/app/data/annotators.txt")


def load_data():
    db: Session = SessionLocal()
    try:
        if db.query(Pair).count() == 0 and os.path.exists(PAIRS_TSV):
            with open(PAIRS_TSV, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    db.add(Pair(
                        pair_id=row["pair_id"],
                        case_id=row["case_id"],
                        variant=row["variant"],
                        query_text=row["query_text"],
                        article_id=row.get("article_id", ""),
                        article_title=row.get("article_title", ""),
                        article_text=row["article_text"],
                        llm_label=row["llm_label"],
                        kuhperdata_book=row.get("kuhperdata_book", ""),
                    ))
            db.commit()
            print(f"Loaded {db.query(Pair).count()} pairs")

        if db.query(Annotator).count() == 0 and os.path.exists(ANNOTATORS_TXT):
            with open(ANNOTATORS_TXT, encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    if name:
                        db.add(Annotator(name=name, session_token=None))
            db.commit()
            print(f"Loaded {db.query(Annotator).count()} annotators")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    load_data()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(auth.router)
app.include_router(pairs.router)
app.include_router(admin.router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def root():
    return FileResponse("app/static/index.html")


@app.get("/annotate")
def annotate():
    return FileResponse("app/static/annotate.html")


@app.get("/admin-view")
def admin_view():
    return FileResponse("app/static/admin.html")
