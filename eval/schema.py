"""Golden-case schema for offline eval."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def _as_categories(value) -> tuple[str, ...]:
    """Normalize a JSON 'category' (str) or 'categories' (list) into a tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


@dataclass
class ExpectedFinding:
    line_range: tuple[int, int]
    categories: tuple[str, ...]
    min_impact: str
    note: str = ""


@dataclass
class GoldenCase:
    id: str
    description: str
    path: str
    diff_text: str
    expected: list[ExpectedFinding]
    false_positive_lines: list[int] = field(default_factory=list)

    @classmethod
    def load(cls, json_path: Path) -> GoldenCase:
        d = json.loads(json_path.read_text(encoding="utf-8"))
        diff_file = json_path.parent / d["diff_file"]
        expected = [
            ExpectedFinding(
                line_range=tuple(e["line_range"]),
                categories=_as_categories(e.get("category") or e.get("categories")),
                min_impact=e["min_impact"],
                note=e.get("note", ""),
            )
            for e in d["expected"]
        ]
        return cls(
            id=d["id"],
            description=d["description"],
            path=d["path"],
            diff_text=diff_file.read_text(encoding="utf-8"),
            expected=expected,
            false_positive_lines=d.get("false_positives", []),
        )
