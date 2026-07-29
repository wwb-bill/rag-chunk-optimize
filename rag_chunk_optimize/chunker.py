import re
from rag_chunk_optimize.types import ChunkConfig

def chunk_text(text: str, config: ChunkConfig) -> list[str]:
    if config.strategy == "paragraph":
        segments = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    elif config.strategy == "fixed":
        segments = [text[i:i + config.size] for i in range(0, len(text), config.size)]
    else:
        segments = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]
        if not segments: segments = [text]
    chunks: list[str] = []
    current = ""
    for seg in segments:
        if current and len(current) + len(seg) > config.size:
            chunks.append(current.strip())
            current = seg if config.overlap == 0 else current[-config.overlap:] + " " + seg
        else:
            current = (current + " " + seg).strip() if current else seg
    if current.strip(): chunks.append(current.strip())
    return chunks or [text]
