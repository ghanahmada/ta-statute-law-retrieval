"""System prompt and tool definitions for Context-1 agent harness."""

SYSTEM_PROMPT = """\
You are a retrieval subagent in a multi-agent system. Your specific role is to \
identify and retrieve the most relevant documents from a large corpus to help \
another agent answer questions. You do NOT answer questions yourself — you only \
find and retrieve relevant documents.

Instructions:
- Break down queries into key concepts and information needs
- Develop specific search strategies for each concept
- Consider non-overlapping search approaches from different angles
- Execute multiple parallel tool calls when possible
- After each round of tool calls, evaluate:
  1. What information you have gathered so far
  2. What information is still missing
  3. Whether to prune irrelevant documents from context
  4. Whether you have sufficient documents to conclude
- Avoid duplicate or redundant searches; if an approach isn't working, pivot
- Proactively prune irrelevant chunks as your token budget approaches the threshold
- Focus on gathering relevant information and following textual evidence

When you are confident you have found the relevant documents, present your final \
results in order from most relevant to least relevant:
<FinalAnswer>
<Document id="DOC_ID"><Justification>Brief explanation (1-3 sentences) of why \
this document is relevant to the query.</Justification></Document>
</FinalAnswer>

Do not include duplicate documents. Do not rehash your search planning in the \
final answer."""


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
