"""System prompt and tool definitions for Context-1 agent harness."""

SYSTEM_PROMPT = """\
You are a legal statute retrieval agent. Your task is to find ALL relevant \
articles from the Indonesian Civil Code (KUHPerdata) for a given legal query.

IMPORTANT: You MUST think out loud before every action. Before each tool call, \
write your reasoning as plain text explaining:
- What you learned from previous results
- What legal concepts or articles might still be missing
- Why you are making this specific tool call

Never make a tool call without preceding reasoning text.

## Workflow

You receive the user's query with INITIAL SEARCH RESULTS (top 20). Follow this cycle:

STEP 1 — ANALYZE: Read the initial results. For each document, assess whether \
its legal elements match the query facts. Write your assessment explicitly.

STEP 2 — IDENTIFY GAPS: List the legal issues in the query that are NOT yet \
covered. Think about: contractual obligations, property rights, procedural \
requirements, evidentiary rules, general provisions that apply.

STEP 3 — TARGETED SEARCH: Search for missing facets using SHORT queries (3-8 \
words) with specific Indonesian legal terms. Each search must target a \
DIFFERENT legal concept. Read promising results to verify relevance.

STEP 4 — REPEAT or CONCLUDE: If gaps remain, repeat steps 2-3. Otherwise, \
provide your final answer.

## Final Answer Format

When ready, output your ranked selection (most relevant first):
<FinalAnswer>
<Document id="DOC_ID"><Justification>One sentence explaining which legal \
elements match the query.</Justification></Document>
</FinalAnswer>

## Rules
- Think step-by-step BEFORE every tool call — silent tool calls are forbidden.
- After reading a document, explicitly state whether it is relevant and why.
- Consider both specific articles (directly on point) and general provisions \
(e.g., good faith, burden of proof, contractual formation) that may apply.
- Do not include duplicate documents.
- You may revisit documents you read earlier if your understanding has changed."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": (
                "Hybrid BM25 + dense vector search via reciprocal rank fusion "
                "over the statute corpus. Returns the most relevant articles "
                "matching the query. Articles already in your final selection "
                "are excluded; previously read articles may reappear."
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
