"""Pair flat input folders without silently discarding unmatched files."""

from dataclasses import dataclass
from pathlib import Path


def normalize_text(text: str) -> str:
    return text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


@dataclass(frozen=True)
class InputPair:
    meeting_id: str
    transcript_path: Path | None
    summary_path: Path | None
    error: str | None = None


def discover_pairs(transcript_dir: str | Path, summary_dir: str | Path) -> list[InputPair]:
    def index(directory: str | Path) -> dict[str, list[Path]]:
        path = Path(directory)
        if not path.is_dir():
            raise ValueError(f"Input folder does not exist: {path}")
        found: dict[str, list[Path]] = {}
        for file in sorted(path.iterdir()):
            if file.is_file() and file.suffix.lower() in {".txt", ".md"}:
                found.setdefault(file.stem, []).append(file)
        return found

    transcripts, summaries = index(transcript_dir), index(summary_dir)
    pairs = []
    for stem in sorted(transcripts.keys() | summaries.keys()):
        left, right = transcripts.get(stem, []), summaries.get(stem, [])
        problems = []
        if not left:
            problems.append("Missing transcript")
        if not right:
            problems.append("Missing summary")
        if len(left) > 1:
            problems.append("Duplicate transcript stem")
        if len(right) > 1:
            problems.append("Duplicate summary stem")
        pairs.append(InputPair(meeting_id=stem,
                               transcript_path=left[0] if len(left) == 1 else None,
                               summary_path=right[0] if len(right) == 1 else None,
                               error="; ".join(problems) or None))
    return pairs
