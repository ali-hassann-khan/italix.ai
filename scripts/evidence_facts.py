"""Extract source facts from pinned before/after files and unified Git hunks."""
import ast
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess

from scripts.acquire_calibration import git

OFFSET = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]+")
KINDS = {
    "test": re.compile(r"\b(?:test_|Test[A-Z]|pytest\.raises|assert(?:Equals|_eq)?\b|expect\()"),
    "condition": re.compile(r"\b(?:if|elif|else|switch|case|unless|while|return|raise|throw)\b|\?"),
    "dependency": re.compile(r"\b(?:import|require|include|dependency|dependencies)\b"),
    "declaration": re.compile(r"\b(?:def|fn|func|function|class|interface|struct|trait|type)\b"),
    "value": re.compile(r"(?<![\w.])\d+(?:\.\d+)*(?!\w)"),
}
COMMON = {"return", "default", "false", "true", "error", "assert", "test", "string", "import", "from", "func"}


def spans_for_python(source):
    try:
        tree = ast.parse(source)
    except (ValueError, SyntaxError):
        return None
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "test" if node.name.startswith("test_") else "function"
            text = ("async " if isinstance(node, ast.AsyncFunctionDef) else "") + f"def {node.name}({ast.unparse(node.args)})"
            if node.returns:
                text += f" -> {ast.unparse(node.returns)}"
            spans.extend(({"kind": kind, "name": node.name, "text": text, "start": node.lineno,
                           "end": max(node.lineno, node.body[0].lineno - 1)},
                          {"kind": "scope", "name": node.name, "text": "", "start": node.lineno, "end": node.end_lineno}))
        elif isinstance(node, (ast.If, ast.While, ast.IfExp)):
            spans.append({"kind": "condition", "name": "", "text": ast.unparse(node.test),
                          "start": node.test.lineno, "end": node.test.end_lineno})
        elif isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                kind, name = "import", ""
            else:
                target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                if not isinstance(target, ast.Name) or not target.id.isupper():
                    continue
                kind, name = "constant", target.id
            spans.append({"kind": kind, "name": name, "text": ast.unparse(node),
                          "start": node.lineno, "end": node.end_lineno})
    return spans


def hunk_groups(hunk):
    header, *body = hunk["lines"].split("\n")
    match = OFFSET.match(header)
    if not match:
        raise ValueError("invalid hunk header")
    old, new = map(int, match.groups())
    group = {"old": [], "new": []}
    previous = None
    for line in body:
        if line.startswith("-"):
            group["old"].append({"line": old, "text": line[1:]})
            old += 1
        elif line.startswith("+"):
            group["new"].append({"line": new, "text": line[1:]})
            new += 1
        elif line.startswith(" "):
            if group["old"] or group["new"]:
                yield group | {"preceding": previous, "following": line[1:]}
                group = {"old": [], "new": []}
            previous = line[1:]
            old += 1
            new += 1
    if group["old"] or group["new"]:
        yield group | {"preceding": previous, "following": None}


def before_after(paths, base, head, mirror, go_index):
    indexes = {}
    go_sources, go_slots = [], []
    for name in paths:
        suffix = Path(name).suffix
        if suffix not in {".py", ".go"}:
            continue
        for side, revision in (("old", base), ("new", head)):
            try:
                source = git(mirror, "show", f"{revision}:{name}")
            except subprocess.CalledProcessError:
                # A verified absent path is an empty version. Missing Git objects stay unknown.
                try:
                    if not git(mirror, "ls-tree", revision, "--", name):
                        indexes[name, side] = []
                except subprocess.CalledProcessError:
                    pass
                continue
            if suffix == ".py":
                parsed = spans_for_python(source)
                if parsed is not None:
                    indexes[name, side] = parsed
            elif go_index:
                go_slots.append((name, side))
                go_sources.append(source)
    if go_sources:
        output = subprocess.run([str(go_index)], input="".join(json.dumps(x) + "\n" for x in go_sources),
                                text=True, capture_output=True, check=True, timeout=25)
        parsed = [json.loads(line) for line in output.stdout.split("\n") if line]
        if len(parsed) != len(go_slots):
            raise ValueError("Go AST response is incomplete")
        indexes.update((slot, spans) for slot, spans in zip(go_slots, parsed) if spans is not None)
    return indexes


def overlap(spans, rows):
    lines = {row["line"] for row in rows}
    return [span for span in spans if any(span["start"] <= line <= span["end"] for line in lines)]


def case_facts(case, mirror, go_index):
    indexes = before_after(case["paths"], case["base"], case["head"], mirror, go_index)
    facts = []
    for hunk in case["evidence"]:
        path = hunk["path"]
        for edit in hunk_groups(hunk):
            group = {side: edit[side] for side in ("old", "new")}
            before = "\n".join(row["text"] for row in group["old"])
            after = "\n".join(row["text"] for row in group["new"])
            if not (before.strip() or after.strip()) or before == after:
                continue
            nodes = {side: overlap(indexes.get((path, side), []), rows) for side, rows in group.items()}
            annotations = sorted({key for key, pattern in KINDS.items() if pattern.search(before + "\n" + after)}
                                 | {item["kind"] for sides in nodes.values() for item in sides if item["kind"] != "scope"})
            symbols = []
            for kind in ("function", "test"):
                for side in ("old", "new"):
                    other = "new" if side == "old" else "old"
                    for item in nodes[side]:
                        if item["kind"] != kind:
                            continue
                        all_other = [s for s in indexes.get((path, other), []) if s["kind"] == kind and s["name"] == item["name"]]
                        if not all_other and (path, other) in indexes:
                            symbols.append({"kind": kind, "change": "removed" if side == "old" else "added",
                                            "name": item["name"], "signature": item["text"]})
                        elif side == "new" and all_other and all(s["text"] != item["text"] for s in all_other):
                            symbols.append({"kind": kind, "change": "signature_modified", "name": item["name"],
                                            "before": all_other[0]["text"], "after": item["text"]})
            if not symbols:
                scopes = {s["name"] for side in nodes.values() for s in side if s["kind"] == "scope"}
                symbols.extend({"kind": "scope", "change": "body_lines_edited", "name": name} for name in sorted(scopes))
            numeric = {side: sorted({token for row in rows for token in KINDS["value"].findall(row["text"])})
                       for side, rows in group.items()}
            api_candidates = [item for item in symbols if item["kind"] == "function" and (
                item["name"].split(".")[-1][0].isupper() if path.endswith(".go")
                else not item["name"].startswith("_"))]
            facts.append({"id": f"F{len(facts)+1:03}", "kind": annotations or ["source_edit"],
                          "path": path, "hunk": hunk["id"], "locations": group,
                          "context": {key: edit[key] for key in ("preceding", "following")},
                          "numeric_literals": numeric, "api_surface_candidates": api_candidates,
                          "symbols": symbols, "syntax": {side: [s for s in ss if s["kind"] in {"condition", "constant", "import"}]
                                                         for side, ss in nodes.items()},
                          "status": "direct_source_edit"})
    # A shared changed identifier is a link for review, rather than proof of a runtime relationship.
    relations = defaultdict(lambda: defaultdict(list))
    for fact in facts:
        names = {word for side in fact["locations"].values() for row in side for word in WORD.findall(row["text"])
                 if len(word) >= 5 and word.casefold() not in COMMON}
        for word in names:
            relations[word][fact["path"]].append(fact["id"])
    links = [{"identifier": word, "paths": sorted(paths), "facts_by_path": dict(paths),
              "status": "syntactic_cooccurrence"}
             for word, paths in sorted(relations.items()) if len(paths) > 1]
    return {"id": case["id"], "repo": case["repo"], "base": case["base"], "head": case["head"],
            "license": case["license"], "source_url": case["source_url"], "paths": case["paths"],
            "facts": facts, "cross_file_links": links}


def verify_lines(facts, mirror):
    """Check every changed line against its pinned Git file."""
    sources = {}
    checked = 0
    for fact in facts["facts"]:
        for side, revision in (("old", "base"), ("new", "head")):
            if not fact["locations"][side]:
                continue
            path = fact["path"]
            key = (path, revision)
            if key not in sources:
                sources[key] = git(mirror, "show", f"{facts[revision]}:{path}").split("\n")
            source = sources[key]
            for row in fact["locations"][side]:
                line = row["line"]
                if not 0 < line <= len(source) or source[line-1] != row["text"]:
                    raise ValueError(f"pinned source mismatch: {facts['id']} {fact['id']} {side}:{line}")
                checked += 1
    return checked
