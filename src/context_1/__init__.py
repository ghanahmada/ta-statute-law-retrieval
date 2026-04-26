"""Context-1-style agentic search agent for statute retrieval.

Reimplements Chroma's Context-1 agent harness using their released 20B model
(chromadb/context-1) with hybrid BM25+dense search, RRF fusion, and
multi-turn observe-reason-act retrieval loop.
"""
