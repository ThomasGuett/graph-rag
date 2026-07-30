"""Split raw document text into overlapping chunks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TextSpan:
    text: str
    ord: int
    char_start: int
    char_end: int

    @property
    def props(self) -> dict:
        return {
            "ord": self.ord,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "kind": "document_chunk",
        }


def chunk_text(text: str, *, chunk_size: int = 1200, chunk_overlap: int = 200) -> list[TextSpan]:
    """Paragraph-aware splitter with fixed-size windows and overlap.

    1. Split on blank lines into paragraphs.
    2. Pack paragraphs into windows up to ``chunk_size``.
    3. If a single paragraph exceeds ``chunk_size``, hard-split with overlap.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    if not text or not text.strip():
        return []

    paragraphs = _paragraphs_with_offsets(text)
    windows: list[tuple[int, int, str]] = []
    buf_parts: list[str] = []
    buf_start = 0
    buf_end = 0

    def flush() -> None:
        nonlocal buf_parts, buf_start, buf_end
        if not buf_parts:
            return
        piece = "\n\n".join(buf_parts)
        windows.append((buf_start, buf_end, piece))
        buf_parts = []

    for para, start, end in paragraphs:
        if len(para) > chunk_size:
            flush()
            windows.extend(_hard_split(para, start, chunk_size, chunk_overlap))
            continue

        candidate = "\n\n".join([*buf_parts, para]) if buf_parts else para
        if buf_parts and len(candidate) > chunk_size:
            flush()
            buf_parts = [para]
            buf_start = start
            buf_end = end
        else:
            if not buf_parts:
                buf_start = start
            buf_parts.append(para)
            buf_end = end

    flush()

    # Apply overlap between consecutive windows by prepending tail of previous.
    if chunk_overlap and len(windows) > 1:
        overlapped: list[tuple[int, int, str]] = [windows[0]]
        for i in range(1, len(windows)):
            prev_text = overlapped[-1][2]
            start, end, piece = windows[i]
            tail = prev_text[-chunk_overlap:]
            if tail and not piece.startswith(tail):
                piece = f"{tail}\n\n{piece}"
                start = max(0, overlapped[-1][1] - len(tail))
            overlapped.append((start, end, piece))
        windows = overlapped

    return [
        TextSpan(text=piece, ord=i, char_start=start, char_end=end)
        for i, (start, end, piece) in enumerate(windows)
    ]


def _paragraphs_with_offsets(text: str) -> list[tuple[str, int, int]]:
    parts: list[tuple[str, int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i] in "\r\n":
            i += 1
        if i >= n:
            break
        start = i
        while i < n:
            if text[i] == "\n" and i + 1 < n and text[i + 1] == "\n":
                break
            if text[i] == "\r" and i + 1 < n and text[i + 1] == "\n" and i + 2 < n and text[i + 2] == "\r":
                break
            i += 1
        raw = text[start:i]
        para = raw.strip()
        if para:
            leading = len(raw) - len(raw.lstrip())
            para_start = start + leading
            parts.append((para, para_start, para_start + len(para)))
        while i < n and text[i] in "\r\n":
            i += 1
    return parts


def _hard_split(
    text: str, base_start: int, chunk_size: int, chunk_overlap: int
) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    start = 0
    n = len(text)
    step = max(1, chunk_size - chunk_overlap)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end]
        out.append((base_start + start, base_start + end, piece))
        if end >= n:
            break
        start += step
    return out
