"""Token budget tracking for Context-1 agent harness.

Manages approximate token usage with soft/hard thresholds to guide
the agent's pruning and termination behavior.
"""

import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")


class TokenBudgetTracker:

    BUDGET = 32_768
    SOFT_THRESHOLD = 0.80
    HARD_THRESHOLD = 0.92

    def __init__(self, budget: int | None = None):
        if budget is not None:
            self.BUDGET = budget
        self.used = 0

    def count_tokens(self, text: str) -> int:
        return len(_encoder.encode(text))

    def add(self, text: str) -> int:
        n = self.count_tokens(text)
        self.used += n
        return n

    def remove(self, n_tokens: int):
        self.used = max(0, self.used - n_tokens)

    def status_message(self) -> str:
        return f"[Token usage: {self.used:,}/{self.BUDGET:,}]"

    @property
    def at_soft_threshold(self) -> bool:
        return self.used >= self.BUDGET * self.SOFT_THRESHOLD

    @property
    def at_hard_threshold(self) -> bool:
        return self.used >= self.BUDGET * self.HARD_THRESHOLD

    @property
    def remaining(self) -> int:
        return max(0, self.BUDGET - self.used)
