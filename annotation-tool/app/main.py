import csv
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session
from app.database import Base, engine, SessionLocal
from app.models import Pair, Annotator
from app.routes import auth, pairs, admin

PAIRS_TSV = os.getenv("PAIRS_TSV", "/app/data/pairs.tsv")
ANNOTATORS_TXT = os.getenv("ANNOTATORS_TXT", "/app/data/annotators.txt")


def _migrate_schema():
    inspector = inspect(engine)
    if "annotators" in inspector.get_table_names():
        cols = {c["name"] for c in inspector.get_columns("annotators")}
        with engine.begin() as conn:
            if "submitted_at" not in cols:
                conn.execute(text("ALTER TABLE annotators ADD COLUMN submitted_at DATETIME"))
            if "access_token_hash" not in cols:
                conn.execute(text("ALTER TABLE annotators ADD COLUMN access_token_hash TEXT"))


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
                        db.add(Annotator(name=name))
            db.commit()
            print(f"Loaded {db.query(Annotator).count()} annotators")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    _migrate_schema()
    load_data()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(auth.router)
app.include_router(pairs.router)
app.include_router(admin.router)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

if os.path.isdir(os.path.join(STATIC_DIR, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")


@app.get("/{full_path:path}")
def spa(full_path: str):
    if full_path:
        file_path = os.path.join(STATIC_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    return {"detail": "Frontend not built. Run: cd frontend && npm run build"}
