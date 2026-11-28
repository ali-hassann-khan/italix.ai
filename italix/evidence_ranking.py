"""Rank changed lines while reserving evidence slots for distinct files."""

from collections import defaultdict
from itertools import zip_longest


def _line_priority(line: dict) -> tuple:
    text = line["text"].strip()
    informative = any(char.isalnum() for char in text) and not text.startswith(("#", "//"))
    return (not informative, line["operation"] != "+", line["id"])


def ranked_patch_lines(changes: list[dict], limit: int) -> list[dict]:
    """Take one informative representative per file before additional lines."""
    by_path = defaultdict(list)
    for line in changes:
        by_path[line["path"]].append(line)
    groups = [sorted(lines, key=_line_priority) for lines in by_path.values()]
    result = []
    for layer in zip_longest(*groups):
        for line in layer:
            if line is not None:
                result.append(line)
                if len(result) == limit:
                    return result
    return result


def cover_files(selected: list[dict], ranked: list[dict], budget: int) -> list[dict]:
    """Keep model choices when possible and fill uncovered file slots from the patch."""
    if not selected:
        return ranked[:budget]
    result = []
    seen = set()
    for line in (*selected, *ranked):
        path = line["path"]
        if path not in seen:
            result.append(line)
            seen.add(path)
            if len(result) == budget:
                return result
    used = {line["id"] for line in result}
    result.extend(line for line in selected if line["id"] not in used)
    return result[:budget]
