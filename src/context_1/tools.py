"""Tool implementations for Context-1 agent harness.

Four tools matching the Context-1 spec: search_corpus, grep_corpus,
read_document, prune_chunks.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from .hybrid_search import HybridSearcher


@dataclass
class ToolResult:
    content: str
    doc_ids_seen: list[str] = field(default_factory=list)
    doc_scores: dict[str, float] = field(default_factory=dict)
    tokens_added: int = 0
    tokens_removed: int = 0


class ToolExecutor:

    def __init__(
        self,
        corpus: dict[str, dict[str, str]],
        hybrid_searcher: HybridSearcher,
        token_counter: Callable[[str], int],
    ):
        self.corpus = corpus
        self.searcher = hybrid_searcher
        self.token_counter = token_counter

    def embed_query(self, query: str):
        if hasattr(self.searcher, "encode_query"):
            return self.searcher.encode_query(query)
        return None

    def search_corpus(
        self,
        query: str,
        exclude_ids: set[str],
        top_n: int = 10,
    ) -> ToolResult:
        results = self.searcher.search(
            query, top_n=top_n, exclude_ids=exclude_ids,
        )
        if not results:
            content = f"search_corpus({query!r}) returned 0 results."
            return ToolResult(
                content=content,
                tokens_added=self.token_counter(content),
            )

        lines = []
        doc_ids = []
        for doc_id, text, score in results:
            title = self.corpus[doc_id].get("title", doc_id)
            lines.append(f"[{doc_id}] {title}: {text[:500]}")
            doc_ids.append(doc_id)

        suggested = self._extract_key_terms(
            [text for _, text, _ in results], query,
        )
        suffix = ""
        if suggested:
            suffix = "\nSuggested terms from results: " + ", ".join(suggested)

        content = (
            f"search_corpus({query!r}) returned {len(results)} results:\n"
            + "\n".join(lines)
            + suffix
        )
        scores = {doc_id: score for doc_id, _, score in results}
        return ToolResult(
            content=content,
            doc_ids_seen=doc_ids,
            doc_scores=scores,
            tokens_added=self.token_counter(content),
        )

    @staticmethod
    def _extract_key_terms(
        doc_texts: list[str], query: str, top_n: int = 8,
    ) -> list[str]:
        query_tokens = set(query.lower().split())
        stop = {
            "yang", "dan", "atau", "di", "ke", "dari", "ini", "itu",
            "dengan", "untuk", "pada", "adalah", "dalam", "tidak",
            "akan", "telah", "oleh", "suatu", "jika", "bila", "maka",
            "dapat", "harus", "wajib", "atas", "bagi", "sebagai",
            "bahwa", "serta", "tersebut", "ia", "orang", "hal",
            "pasal", "ayat", "huruf", "angka",
        }
        counts: Counter = Counter()
        for text in doc_texts:
            seen_in_doc = set()
            words = text.lower().split()
            for i, w in enumerate(words):
                w = re.sub(r"[^a-z\-]", "", w)
                if len(w) < 3 or w in stop or w in query_tokens:
                    continue
                if w not in seen_in_doc:
                    counts[w] += 1
                    seen_in_doc.add(w)
                bigram_parts = []
                if i + 1 < len(words):
                    w2 = re.sub(r"[^a-z\-]", "", words[i + 1].lower())
                    if len(w2) >= 3 and w2 not in stop:
                        bg = f"{w} {w2}"
                        if bg not in seen_in_doc:
                            counts[bg] += 1
                            seen_in_doc.add(bg)
        return [term for term, _ in counts.most_common(top_n)]

    def grep_corpus(
        self, pattern: str, max_results: int = 5,
    ) -> ToolResult:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            content = f"Invalid regex pattern: {pattern} ({e})"
            return ToolResult(
                content=content,
                tokens_added=self.token_counter(content),
            )

        matches = []
        for doc_id, doc in self.corpus.items():
            full_text = f"{doc.get('title', '')} {doc['text']}"
            if regex.search(full_text):
                matches.append(
                    (doc_id, doc.get("title", doc_id), doc["text"][:300])
                )
                if len(matches) >= max_results:
                    break

        if not matches:
            content = f"grep_corpus({pattern!r}) found 0 matches."
            return ToolResult(
                content=content,
                tokens_added=self.token_counter(content),
            )

        lines = [f"[{did}] {title}: {snippet}" for did, title, snippet in matches]
        content = (
            f"grep_corpus({pattern!r}) found {len(matches)} matches:\n"
            + "\n".join(lines)
        )
        return ToolResult(
            content=content,
            doc_ids_seen=[did for did, _, _ in matches],
            tokens_added=self.token_counter(content),
        )

    def read_document(self, doc_id: str) -> ToolResult:
        if doc_id not in self.corpus:
            content = f"Document '{doc_id}' not found in corpus."
            return ToolResult(
                content=content,
                tokens_added=self.token_counter(content),
            )

        doc = self.corpus[doc_id]
        title = doc.get("title", doc_id)
        content = f"[{doc_id}] {title}:\n{doc['text']}"
        return ToolResult(
            content=content,
            doc_ids_seen=[doc_id],
            tokens_added=self.token_counter(content),
        )

    def prune_chunks(
        self, doc_ids: list[str], conversation_messages: list[dict],
    ) -> ToolResult:
        tokens_removed = 0
        for doc_id in doc_ids:
            escaped = re.escape(f"[{doc_id}]")
            pattern = re.compile(escaped + r"[^\[]*", re.DOTALL)
            for msg in conversation_messages:
                if msg.get("role") not in ("tool", "user"):
                    continue
                old_content = msg.get("content", "")
                if not isinstance(old_content, str):
                    continue
                new_content = pattern.sub(f"[{doc_id}] [PRUNED] ", old_content)
                if new_content != old_content:
                    tokens_removed += (
                        self.token_counter(old_content)
                        - self.token_counter(new_content)
                    )
                    msg["content"] = new_content

        content = (
            f"Pruned {len(doc_ids)} document(s) from context. "
            f"~{tokens_removed} tokens freed."
        )
        return ToolResult(
            content=content,
            tokens_removed=tokens_removed,
            tokens_added=self.token_counter(content),
        )
