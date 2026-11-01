"""Token-budget enforcement without silently truncating current observations."""


class ContextBudgetError(ValueError):
    def __init__(self, message, stats=None):
        super().__init__(message)
        self.stats = stats or {}


def prepare_with_budget(messages, prepare, max_input_tokens=None, history_start=1):
    """prepare returns processor tensors; trim oldest *complete* history pairs.

    System instructions and the final current user message are immutable here.
    Count after multimodal processing, before any tensor is moved onto a GPU.
    """
    working = list(messages)
    dropped_pairs = 0
    original_count = None
    while True:
        prepared = prepare(working)
        count = int(prepared["input_ids"].shape[-1])
        if original_count is None:
            original_count = count
        if max_input_tokens is None or count <= max_input_tokens:
            return prepared, {"input_tokens": count, "dropped_history_pairs": dropped_pairs,
                              "input_tokens_before_trimming": original_count,
                              "max_input_tokens": max_input_tokens}
        if len(working) <= history_start + 1:
            raise ContextBudgetError(
                f"Current system+observation requires {count} input tokens, budget={max_input_tokens}; "
                "no current image, candidate or instruction was truncated and no action was executed.",
                {"input_tokens": count, "input_tokens_before_trimming": original_count,
                 "dropped_history_pairs": dropped_pairs, "max_input_tokens": max_input_tokens})
        roles = tuple(row.get("role") for row in working[history_start:history_start + 2])
        if len(working) < history_start + 3 or roles not in {("user", "assistant"), ("assistant", "tool")}:
            raise ContextBudgetError("Cannot trim malformed history without altering current instructions")
        # Current images live only in the final observation, not past history.
        del working[history_start:history_start + 2]
        dropped_pairs += 1
