"""Core agent loop implementing Context-1's observe-reason-act cycle.

The agent iteratively calls tools to search, read, and prune documents,
managing a token budget and deduplicating results across turns.
"""

import json
import re
from dataclasses import dataclass, field
from collections import OrderedDict

from openai import AsyncOpenAI

from .prompts import SYSTEM_PROMPT, TOOLS
from .tools import ToolExecutor, ToolResult
from .token_budget import TokenBudgetTracker


@dataclass
class AgentState:
    messages: list[dict] = field(default_factory=list)
    seen_doc_ids: set[str] = field(default_factory=set)
    selected_doc_ids: OrderedDict = field(default_factory=OrderedDict)
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
    ):
        self.client = client
        self.model = model
        self.tool_executor = tool_executor
        self.max_turns = max_turns
        self.budget_size = budget_size

    def _new_state(self) -> AgentState:
        state = AgentState(
            max_turns=self.max_turns,
            budget=TokenBudgetTracker(self.budget_size),
        )
        state.messages.append({"role": "system", "content": SYSTEM_PROMPT})
        state.budget.add(SYSTEM_PROMPT)
        return state

    def _observe(self, state: AgentState, content: str, role: str = "user"):
        text = content + "\n" + state.budget.status_message()
        state.messages.append({"role": role, "content": text})
        state.budget.add(text)

    async def _infer(self, state: AgentState):
        tools = TOOLS
        if state.budget.at_hard_threshold:
            tools = [t for t in TOOLS if t["function"]["name"] == "prune_chunks"]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=state.messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
            max_tokens=2048,
        )
        choice = response.choices[0]

        assistant_content = choice.message.content or ""
        if assistant_content:
            state.budget.add(assistant_content)

        msg = {"role": "assistant", "content": assistant_content}
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

    def _act(self, state: AgentState, choice) -> list[ToolResult]:
        results = []

        if choice.finish_reason == "stop" or not choice.message.tool_calls:
            content = choice.message.content or ""
            if "<FinalAnswer>" in content:
                self._parse_final_answer(state, content)
            state.is_done = True
            return results

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

        return results

    def _execute_tool(
        self, state: AgentState, name: str, args: dict,
    ) -> ToolResult:
        if name == "search_corpus":
            return self.tool_executor.search_corpus(
                args.get("query", ""),
                exclude_ids=state.seen_doc_ids,
            )
        elif name == "grep_corpus":
            return self.tool_executor.grep_corpus(
                args.get("pattern", ""),
            )
        elif name == "read_document":
            return self.tool_executor.read_document(
                args.get("doc_id", ""),
            )
        elif name == "prune_chunks":
            result = self.tool_executor.prune_chunks(
                args.get("doc_ids", []),
                state.messages,
            )
            state.budget.remove(result.tokens_removed)
            return result
        else:
            return ToolResult(content=f"Unknown tool: {name}")

    def _parse_final_answer(self, state: AgentState, text: str):
        pattern = (
            r'<Document\s+id=["\']?([^"\'>\s]+)["\']?\s*>'
            r'<Justification>(.*?)</Justification>'
            r'</Document>'
        )
        for m in re.finditer(pattern, text, re.DOTALL):
            doc_id = m.group(1)
            justification = m.group(2).strip()
            state.selected_doc_ids[doc_id] = justification

    async def run(self, query: str) -> AgentState:
        state = self._new_state()
        self._observe(state, query)

        while not state.is_done and state.turn_count < state.max_turns:
            state.turn_count += 1

            if (
                state.budget.at_soft_threshold
                and not state.budget.at_hard_threshold
            ):
                state.messages.append({
                    "role": "system",
                    "content": (
                        "WARNING: Token budget is running low. Consider "
                        "pruning irrelevant documents or providing your "
                        "final answer now. "
                        + state.budget.status_message()
                    ),
                })

            try:
                choice = await self._infer(state)
                self._act(state, choice)
            except Exception as e:
                state.error = str(e)
                state.is_done = True

        if not state.selected_doc_ids and state.seen_doc_ids:
            for doc_id in state.seen_doc_ids:
                state.selected_doc_ids[doc_id] = "fallback: no final answer produced"

        return state
