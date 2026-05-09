from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text
from app.database import Base


class Pair(Base):
    __tablename__ = "pairs"

    pair_id = Column(String, primary_key=True)
    case_id = Column(String)
    variant = Column(String)
    query_text = Column(Text)
    article_id = Column(String)
    article_title = Column(String)
    article_text = Column(Text)
    llm_label = Column(String)
    kuhperdata_book = Column(String)


class Annotator(Base):
    __tablename__ = "annotators"

    name = Column(String, primary_key=True)
    access_token_hash = Column(String, nullable=True)
    session_token = Column(String, nullable=True)
    submitted_at = Column(DateTime, nullable=True, default=None)


class Label(Base):
    __tablename__ = "labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    annotator = Column(String)
    pair_id = Column(String)
    label = Column(String)
    confidence = Column(String)
    reasoning = Column(Text, nullable=True, default=None)
    created_at = Column(DateTime, default=datetime.utcnow)


class Flag(Base):
    __tablename__ = "flags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    annotator = Column(String, nullable=False)
    pair_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
