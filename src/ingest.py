from __future__ import annotations

from io import BytesIO


def extract_text(filename: str, content: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(content))
            return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        except Exception:
            return ""

    if lower.endswith(".txt") or lower.endswith(".md"):
        return content.decode("utf-8", errors="ignore").strip()

    return ""
