import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any


def load_qrels(qrels_path: str) -> Dict[str, Set[str]]:
    """Load qrels.tsv and return dict mapping query_id -> set of relevant doc_ids (score > 0 only)."""
    qrels = {}
    with open(qrels_path, 'r', encoding='utf-8') as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id, doc_id, score = parts[0], parts[1], int(parts[2])
                if query_id not in qrels:
                    qrels[query_id] = set()
                if score > 0:
                    qrels[query_id].add(doc_id)
    return qrels


class DataLoader:
    """Load BEIR-format dataset (corpus.jsonl, queries.jsonl, qrels.tsv)."""

    def __init__(self, corpus_path: str, queries_path: str, qrels_path: str):
        self.corpus_path = Path(corpus_path)
        self.queries_path = Path(queries_path)
        self.qrels_path = Path(qrels_path)

        self.corpus: Dict[str, Dict[str, str]] = {}
        self.queries: Dict[str, Dict[str, Any]] = {}
        self.qrels: Dict[str, Dict[str, int]] = {}

    def load(self) -> "DataLoader":
        """Load all data files."""
        self._load_corpus()
        self._load_queries()
        self._load_qrels()
        return self

    def _load_corpus(self):
        """Load corpus.jsonl: {"_id": str, "title": str, "text": str}"""
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line.strip())
                self.corpus[doc["_id"]] = {
                    "title": doc.get("title", ""),
                    "text": doc["text"]
                }

    def _load_queries(self):
        """Load queries.jsonl: {"_id": str, "text": str, "metadata": dict}"""
        with open(self.queries_path, "r", encoding="utf-8") as f:
            for line in f:
                query = json.loads(line.strip())
                self.queries[query["_id"]] = {
                    "text": query["text"],
                    "metadata": query.get("metadata", {})
                }

    def _load_qrels(self):
        """Load qrels.tsv: query_id\tdoc_id\tscore (TREC format)"""
        with open(self.qrels_path, "r", encoding="utf-8") as f:
            header = True
            for line in f:
                if header:
                    header = False
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    query_id, doc_id, score = parts[0], parts[1], int(parts[2])
                    if query_id not in self.qrels:
                        self.qrels[query_id] = {}
                    self.qrels[query_id][doc_id] = score

    def get_corpus_texts(self) -> Tuple[List[str], List[str]]:
        """Return (doc_ids, texts) for encoding."""
        doc_ids = list(self.corpus.keys())
        texts = [self.corpus[did]["text"] for did in doc_ids]
        return doc_ids, texts

    def get_query_texts(self) -> Tuple[List[str], List[str]]:
        """Return (query_ids, texts) for encoding."""
        query_ids = list(self.queries.keys())
        texts = [self.queries[qid]["text"] for qid in query_ids]
        return query_ids, texts

    def get_relevance_pairs(self) -> List[Tuple[str, str, int]]:
        """Return all (query_id, doc_id, relevance) tuples."""
        pairs = []
        for qid, docs in self.qrels.items():
            for doc_id, score in docs.items():
                pairs.append((qid, doc_id, score))
        return pairs
