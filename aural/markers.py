from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

THINK_OPEN_TEXT = "###"
THINK_CLOSE_TEXT = "###"
THINK_SPEED_PREFIX = "(Thinking speed: 4)###"
ANSWER_PREFIX = "Answer:"
EXPECTED_THINK_OPEN_IDS = (14374,)
EXPECTED_THINK_CLOSE_IDS = (14374,)


def encode_think_markers(
    tokenizer, *, verify_expected: bool = True
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    open_ids = tuple((int(x) for x in tokenizer.encode(THINK_OPEN_TEXT, add_special_tokens=False)))
    close_ids = tuple(
        (int(x) for x in tokenizer.encode(THINK_CLOSE_TEXT, add_special_tokens=False))
    )
    if not open_ids or not close_ids:
        raise ValueError("Qwen2.5 AURAL thinking delimiters must tokenize to non-empty sequences.")
    if verify_expected and (
        open_ids != EXPECTED_THINK_OPEN_IDS or close_ids != EXPECTED_THINK_CLOSE_IDS
    ):
        raise ValueError(
            f"Unexpected Qwen2.5-Omni thinking delimiter tokenization: {THINK_OPEN_TEXT!r}->{list(open_ids)}, {THINK_CLOSE_TEXT!r}->{list(close_ids)}; expected {list(EXPECTED_THINK_OPEN_IDS)} and {list(EXPECTED_THINK_CLOSE_IDS)}."
        )
    return (open_ids, close_ids)


def find_subsequence(
    sequence: Sequence[int],
    subsequence: Sequence[int],
    *,
    start: int = 0,
    end: Optional[int] = None,
) -> int:
    if not subsequence:
        raise ValueError("subsequence must not be empty")
    end = len(sequence) if end is None else min(int(end), len(sequence))
    start = max(0, int(start))
    last = end - len(subsequence)
    for idx in range(start, last + 1):
        if list(sequence[idx : idx + len(subsequence)]) == list(subsequence):
            return idx
    return -1


def find_all_subsequences(
    sequence: Sequence[int], subsequence: Sequence[int], *, end: Optional[int] = None
) -> List[int]:
    offsets: List[int] = []
    cursor = 0
    while True:
        idx = find_subsequence(sequence, subsequence, start=cursor, end=end)
        if idx < 0:
            return offsets
        offsets.append(idx)
        cursor = idx + len(subsequence)


def find_reason_span(
    sequence: Sequence[int],
    open_ids: Sequence[int],
    close_ids: Sequence[int],
    *,
    valid_len: Optional[int] = None,
    require_unique: bool = True,
) -> Tuple[int, int, int, int]:
    valid_len = len(sequence) if valid_len is None else min(int(valid_len), len(sequence))
    open_offsets = find_all_subsequences(sequence, open_ids, end=valid_len)
    close_offsets = find_all_subsequences(sequence, close_ids, end=valid_len)
    if tuple(open_ids) == tuple(close_ids):
        separator_offsets = open_offsets
        if len(separator_offsets) == 2:
            open_offsets = [separator_offsets[0]]
            close_offsets = [separator_offsets[1]]
        else:
            open_offsets, close_offsets = ([], [])
    if len(open_offsets) != 1 or len(close_offsets) != 1:
        if require_unique:
            raise ValueError(
                f"Expected exactly one Qwen2.5 AURAL thinking block, found open={len(open_offsets)} close={len(close_offsets)}."
            )
        return (-1, -1, -1, -1)
    open_start = open_offsets[0]
    reason_start = open_start + len(open_ids)
    close_start = close_offsets[0]
    close_end = close_start + len(close_ids)
    if close_start <= reason_start or close_end > valid_len:
        if require_unique:
            raise ValueError(
                f"Invalid Qwen2.5 AURAL thinking span: open_start={open_start}, reason_start={reason_start}, close_start={close_start}, close_end={close_end}, valid_len={valid_len}."
            )
        return (-1, -1, -1, -1)
    return (open_start, reason_start, close_start, close_end)
