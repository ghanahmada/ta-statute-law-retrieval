"""System prompt and tool definitions for Context-1 agent harness."""

SYSTEM_PROMPT = """\
You are a retrieval subagent in a multi-agent system. Your role is to find the \
most relevant statute articles from an Indonesian legal corpus (KUHPerdata). \
You do NOT answer questions — you only retrieve relevant articles.

You will receive the user's query along with INITIAL SEARCH RESULTS from a \
broad retrieval over the full corpus. These initial results are your baseline — \
they represent the best matches for the full query.

Your job is to IMPROVE on this baseline:
1. READ the top initial results to verify their relevance. Prune any that are \
not actually relevant to the query.
2. IDENTIFY GAPS: what legal aspects of the query are NOT covered by the \
initial results? Think about what legal concepts, articles, or provisions \
might be missing.
3. SEARCH for missing facets using SHORT targeted queries (3-8 words) with \
specific legal terms. Each search must use DIFFERENT terms.
4. Use grep_corpus to find specific article numbers or exact legal phrases.
5. Prune irrelevant documents to save token budget.

After 2-3 rounds of refinement, provide your final answer. Do not keep \
searching if you are finding similar results. Present results from most to \
least relevant:
<FinalAnswer>
<Document id="DOC_ID"><Justification>Brief explanation of relevance.\
</Justification></Document>
</FinalAnswer>

Do not include duplicate documents. Keep justifications to 1 sentence."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": (
                "Hybrid BM25 + dense vector search via reciprocal rank fusion "
                "over the statute corpus. Returns the most relevant articles "
                "matching the query. Previously seen articles are automatically "
                "excluded to ensure fresh results on each call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_corpus",
            "description": (
                "Regex search over the full text of all corpus documents. "
                "Returns up to 5 matching documents. Useful for finding "
                "specific article numbers, legal terms, or exact phrases."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern (case-insensitive)",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "Read the full text of a document by its ID. Use this to "
                "verify whether a specific article is relevant to the query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to read",
                    }
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prune_chunks",
            "description": (
                "Remove specified documents from the conversation context to "
                "free up token budget for further exploration. Use this when "
                "documents turn out to be irrelevant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document IDs to remove from context",
                    }
                },
                "required": ["doc_ids"],
            },
        },
    },
]
