"""Core agent loop implementing Context-1's observe-reason-act cycle.

The agent iteratively calls tools to search, read, and prune documents,
managing a token budget and deduplicating results across turns.
"""

import json
import re
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np
from openai import AsyncOpenAI

from .prompts import SYSTEM_PROMPT_HIERARCHY, SYSTEM_PROMPT_FLAT, TOOLS
from .tools import ToolExecutor, ToolResult
from .token_budget import TokenBudgetTracker


@dataclass
class AgentState:
    messages: list[dict] = field(default_factory=list)
    seen_doc_ids: set[str] = field(default_factory=set)
    read_doc_ids: set[str] = field(default_factory=set)
    selected_doc_ids: OrderedDict = field(default_factory=OrderedDict)
    doc_scores: dict[str, float] = field(default_factory=dict)
    frames: dict[str, list[str]] = field(default_factory=dict)
    frame_rejections: dict[str, str] = field(default_factory=dict)
    query_embeddings: list = field(default_factory=list)
    n_similarity_rejections: int = 0
    n_gate_triggers: int = 0
    is_done: bool = False
    turn_count: int = 0
    max_turns: int = 10
    budget: TokenBudgetTracker = field(default_factory=TokenBudgetTracker)
    error: str | None = None


class AgenticRetriever:

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        tool_executor: ToolExecutor,
        max_turns: int = 10,
        budget_size: int = 32_768,
        pad_to_k: int = 0,
        use_hierarchy: bool = True,
        use_coverage_gate: bool = True,
        use_similarity_guard: bool = True,
        similarity_threshold: float = 0.92,
    ):
        self.client = client
        self.model = model
        self.tool_executor = tool_executor
        self.max_turns = max_turns
        self.pad_to_k = pad_to_k
        self.budget_size = budget_size
        self.use_hierarchy = use_hierarchy
        self.use_coverage_gate = use_coverage_gate
        self.use_similarity_guard = use_similarity_guard
        self.similarity_threshold = similarity_threshold
        self._system_prompt = (
            SYSTEM_PROMPT_HIERARCHY if use_hierarchy else SYSTEM_PROMPT_FLAT
        )

    def _new_state(self) -> AgentState:
        state = AgentState(
            max_turns=self.max_turns,
            budget=TokenBudgetTracker(self.budget_size),
        )
        return state

    def _observe(self, state: AgentState, content: str, role: str = "user"):
        text = content + "\n" + state.budget.status_message()
        state.messages.append({"role": role, "content": text})
        state.budget.add(text)

    async def _infer(self, state: AgentState, force_conclude: bool = False):
        kwargs = dict(
            model=self.model,
            messages=state.messages,
            temperature=0,
            max_tokens=4096 if force_conclude else 2048,
        )

        if force_conclude:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "none"
        elif state.budget.at_hard_threshold:
            kwargs["tools"] = [
                t for t in TOOLS if t["function"]["name"] == "prune_chunks"
            ]
            kwargs["tool_choice"] = "auto"
        else:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        assistant_content = choice.message.content or ""
        if assistant_content:
            state.budget.add(assistant_content)

        reasoning = getattr(choice.message, "reasoning_content", None) or ""

        msg = {"role": "assistant", "content": assistant_content}
        if reasoning:
            msg["reasoning"] = reasoning
        if choice.message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]
        state.messages.append(msg)

        return choice

    def _parse_frame_declarations(self, state: AgentState, text: str):
        for m in re.finditer(r'^L2 FRAME:\s*(.+)$', text, re.MULTILINE):
            frame = m.group(1).strip().rstrip('.,:;')
            if frame and frame not in state.frames:
                state.frames[frame] = []

    def _parse_doc_references(self, state: AgentState, text: str):
        pattern = (
            r'<Document\s+id=["\']?([^"\'>\s]+)["\']?\s*>'
            r'<Justification>(.*?)</Justification>'
            r'</Document>'
        )
        for m in re.finditer(pattern, text, re.DOTALL):
            doc_id = m.group(1)
            justification = m.group(2).strip()
            if doc_id in self.tool_executor.corpus:
                state.selected_doc_ids[doc_id] = justification
                frame_match = re.search(r'L2\s*:\s*([^—\-–|]+)', justification)
                if frame_match:
                    frame_name = frame_match.group(1).strip().lower()
                    for declared_frame in state.frames:
                        if (frame_name in declared_frame.lower()
                                or declared_frame.lower() in frame_name):
                            if doc_id not in state.frames[declared_frame]:
                                state.frames[declared_frame].append(doc_id)
                            break

    def _validate_coverage(self, state: AgentState) -> tuple[bool, str]:
        if len(state.frames) < 2:
            return False, (
                "Declare at least 2 L2 frames before concluding. "
                f"Currently declared: {list(state.frames.keys()) or 'none'}."
            )
        covered = {f for f, docs in state.frames.items() if docs}
        rejected = set(state.frame_rejections.keys())
        unaccounted = set(state.frames.keys()) - covered - rejected
        if len(covered) >= 2 and not unaccounted:
            return True, ""
        return False, (
            f"Coverage gate: {len(covered)} frame(s) have supporting documents, "
            f"{len(unaccounted)} unaccounted. "
            f"Search remaining frames or explicitly reject them. "
            f"Unaccounted: {sorted(unaccounted)}."
        )

    def _act(
        self, state: AgentState, choice, force_conclude: bool = False,
    ) -> list[ToolResult]:
        results = []
        content = choice.message.content or ""

        if content:
            self._parse_frame_declarations(state, content)
            self._parse_doc_references(state, content)

        if force_conclude or choice.finish_reason == "stop" or not choice.message.tool_calls:
            state.is_done = True

            if self.use_coverage_gate:
                ok, reason = self._validate_coverage(state)
                if not ok:
                    state.n_gate_triggers += 1
                    state.error = f"gate_failure: {reason}"

            return results

        # Coverage gate on non-final turns
        if state.is_done and not force_conclude and self.use_coverage_gate:
            ok, reason = self._validate_coverage(state)
            if not ok:
                state.is_done = False
                state.n_gate_triggers += 1
                state.messages.append({
                    "role": "user",
                    "content": f"[SYSTEM] {reason}",
                })

        for tool_call in choice.message.tool_calls:
            fn_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                result = ToolResult(
                    content=f"Invalid JSON arguments: {tool_call.function.arguments}",
                )
                results.append(result)
                state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.content,
                })
                continue

            result = self._execute_tool(state, fn_name, args)
            results.append(result)

            state.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result.content,
            })
            state.budget.add(result.content)
            state.seen_doc_ids.update(result.doc_ids_seen)
            for did, score in result.doc_scores.items():
                if did not in state.doc_scores or score > state.doc_scores[did]:
                    state.doc_scores[did] = score

        return results

    def _execute_tool(
        self, state: AgentState, name: str, args: dict,
    ) -> ToolResult:
        # Early gate: require frame declarations before any search on early turns
        if name in ("search_corpus", "grep_corpus") and not state.frames and state.turn_count <= 2:
            return ToolResult(
                content=(
                    "Declare L2 frames before searching. Use 'L2 FRAME: <name>' "
                    "for each frame, then proceed with targeted searches."
                )
            )

        if name == "search_corpus":
            # Similarity guard
            if self.use_similarity_guard:
                query_text = args.get("query", "")
                new_emb = self.tool_executor.embed_query(query_text)
                if new_emb is not None and state.query_embeddings:
                    for prior_emb in state.query_embeddings:
                        sim = float(np.dot(new_emb, prior_emb))
                        if sim > self.similarity_threshold:
                            state.n_similarity_rejections += 1
                            return ToolResult(
                                content=(
                                    f"Query rejected: too similar to a prior query "
                                    f"(similarity {sim:.2f} > {self.similarity_threshold}). "
                                    f"Reformulate targeting a different L2 frame or L3 doctrine."
                                )
                            )
                if new_emb is not None:
                    state.query_embeddings.append(new_emb)

            exclude = set(state.selected_doc_ids.keys())
            return self.tool_executor.search_corpus(
                args.get("query", ""),
                exclude_ids=exclude,
            )
        elif name == "grep_corpus":
            return self.tool_executor.grep_corpus(
                args.get("pattern", ""),
            )
        elif name == "read_document":
            doc_id = args.get("doc_id", "")
            state.read_doc_ids.add(doc_id)
            return self.tool_executor.read_document(doc_id)
        elif name == "prune_chunks":
            result = self.tool_executor.prune_chunks(
                args.get("doc_ids", []),
                state.messages,
            )
            state.budget.remove(result.tokens_removed)
            return result
        elif name == "FinalAnswer":
            content = args.get("answer", "") or args.get("documents", "") or json.dumps(args)
            self._parse_frame_declarations(state, content)
            self._parse_doc_references(state, content)
            state.is_done = True
            return ToolResult(content="Final answer recorded.")
        else:
            return ToolResult(content=f"Unknown tool: {name}")

    def _bootstrap_search(self, state: AgentState, query: str):
        result = self.tool_executor.search_corpus(
            query, exclude_ids=set(), top_n=20,
        )
        state.seen_doc_ids.update(result.doc_ids_seen)
        for did, score in result.doc_scores.items():
            state.doc_scores[did] = score
        state.budget.add(result.content)
        return result

    async def run(self, query: str) -> AgentState:
        state = self._new_state()

        bootstrap = self._bootstrap_search(state, query)

        # System prompt in system role
        state.messages.append({"role": "system", "content": self._system_prompt})
        state.budget.add(self._system_prompt)

        user_content = (
            "--- USER QUERY ---\n"
            + query
            + "\n\n--- INITIAL SEARCH RESULTS (top 20 from full query) ---\n"
            + bootstrap.content
            + "\n\n"
            + state.budget.status_message()
        )
        state.messages.append({"role": "user", "content": user_content})
        state.budget.add(user_content)

        while not state.is_done and state.turn_count < state.max_turns:
            state.turn_count += 1
            is_last_turn = state.turn_count >= state.max_turns

            if (
                state.budget.at_soft_threshold
                and not state.budget.at_hard_threshold
            ):
                state.messages.append({
                    "role": "user",
                    "content": (
                        "[SYSTEM] WARNING: Token budget is running low. "
                        "Consider pruning irrelevant documents or providing "
                        "your final answer now. "
                        f"Original query: \"{query}\"\n"
                        + state.budget.status_message()
                    ),
                })

            if is_last_turn:
                state.messages.append({
                    "role": "user",
                    "content": (
                        "[SYSTEM] FINAL TURN: You MUST provide your final "
                        "answer NOW. Remember the original query:\n\n"
                        f"\"{query}\"\n\n"
                        "List the most relevant documents using "
                        "the <Document id=\"DOC_ID\"><Justification>..."
                        "</Justification></Document> format. No more tool calls."
                    ),
                })

            try:
                choice = await self._infer(state, force_conclude=is_last_turn)

                content = choice.message.content or ""
                has_tools = bool(choice.message.tool_calls)
                if not content and not has_tools and not is_last_turn:
                    state.messages.append({
                        "role": "user",
                        "content": (
                            "[SYSTEM] Your response was empty. You MUST either "
                            "call a tool or provide your <FinalAnswer>. "
                            "Think about the query and what to do next.\n"
                            f"Original query: \"{query}\""
                        ),
                    })
                    continue

                self._act(state, choice, force_conclude=is_last_turn)
            except Exception as e:
                state.error = str(e)
                state.is_done = True

        ranked_seen = sorted(
            state.seen_doc_ids,
            key=lambda d: state.doc_scores.get(d, 0),
            reverse=True,
        )

        if not state.selected_doc_ids and ranked_seen:
            for doc_id in ranked_seen[:10]:
                state.selected_doc_ids[doc_id] = "fallback: no final answer produced"

        if self.pad_to_k > 0 and len(state.selected_doc_ids) < self.pad_to_k:
            for doc_id in ranked_seen:
                if doc_id not in state.selected_doc_ids:
                    state.selected_doc_ids[doc_id] = "padded from seen"
                if len(state.selected_doc_ids) >= self.pad_to_k:
                    break

        return state
