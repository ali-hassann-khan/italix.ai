"""Extract a bounded Phase 14 review packet from shallow public Git mirrors."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess

from scripts.prepare_specialization import accepted_patch, messages_for, patch_shingles, read_rows, write_rows
from italix.privacy import contains_secret

SOURCES = {
    "go": ("golang/go", "BSD-3-Clause"),
    "kubectl": ("kubernetes/kubectl", "Apache-2.0"),
    "cli": ("cli/cli", "MIT"),
    "flask": ("pallets/flask", "BSD-3-Clause"),
    "httpx": ("encode/httpx", "BSD-3-Clause"),
}
SOURCE_SUFFIXES = {".go", ".py", ".rs", ".js", ".ts", ".java", ".kt", ".rb", ".php", ".cpp", ".cc", ".h", ".hpp", ".tsx", ".yaml", ".yml", ".toml", ".json"}
EXCLUDED = re.compile(r"(?:^|/)(?:vendor|testdata|fixtures|third_party|node_modules|generated|docs|\.github)(?:/|$)|(?:\.pb\.go|\.sum|lock\.json)$", re.I)
LICENSE_DOC = re.compile(r"^(?:licen[sc]es?|copying|copyright|notice|legal|unlicense|mit-license)(?:[-_.].*)?$", re.I)
LICENSE_MARKERS = {"BSD-3-Clause": "neither the name", "Apache-2.0": "apache license", "MIT": "permission is hereby granted"}
CREDENTIAL_PATH = re.compile(r"secret|credential|token|private.?key|\.pem$", re.I)


def git(directory, *args):
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], stderr=subprocess.DEVNULL,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"},
        text=True, errors="replace", timeout=30,
    )


def verify_tree(directory, revision, paths, spdx):
    tree = git(directory, "ls-tree", "-r", "--name-only", revision).splitlines()
    names = set(tree)
    root_files = {name.casefold(): name for name in names if "/" not in name}
    candidates = ("license", "license.txt", "license.md")
    if spdx == "MIT":
        candidates = ("license-mit", "mit-license", *candidates)
    root_name = next((root_files[name] for name in candidates if name in root_files), None)
    if not root_name:
        raise ValueError("missing_root_license")
    license_text = git(directory, "show", f"{revision}:{root_name}")
    lower = " ".join(license_text.lower().split())
    if LICENSE_MARKERS[spdx] not in lower or contains_secret(license_text):
        raise ValueError("unexpected_root_license")
    ancestors = {parent for path in paths for parent in PurePosixPath(path).parents
                 if parent != PurePosixPath(".")}
    for name in names:
        path = PurePosixPath(name)
        if (path.parent in ancestors and LICENSE_DOC.fullmatch(path.name)
                or path.name.lower() in {"license", "licenses", "copying"} and path in ancestors):
            raise ValueError("nested_license_or_notice")
    for name in paths:
        if name not in names:
            continue
        mode = git(directory, "ls-tree", revision, "--", name).split()[0]
        if mode not in {"100644", "100755"}:
            raise ValueError("non_regular_source")
        source = git(directory, "show", f"{revision}:{name}")
        if re.search(r"code generated|automatically generated|do not edit", "\n".join(source.splitlines()[:25]), re.I):
            raise ValueError("generated_file")
        if contains_secret(source):
            raise ValueError("sensitive_source")
        for identifier in re.findall(r"SPDX-License-Identifier:\s*([^\n]+)", source):
            if identifier.strip().rstrip("*/").strip() != spdx:
                raise ValueError("file_spdx_mismatch")
        header = " ".join(" ".join(source.splitlines()[:40]).lower().split())
        header_signals = {"apache-2.0": "Apache-2.0", "apache license": "Apache-2.0", "mit license": "MIT",
                          "gnu general public license": "GPL", "mozilla public license": "MPL",
                          "all rights reserved": "copyright"}
        for marker, declared in header_signals.items():
            if marker in header and declared not in {spdx, "copyright"}:
                raise ValueError("file_license_mismatch")
        if "all rights reserved" in header and "license" not in header:
            raise ValueError("ambiguous_file_header")
    notices = {name: git(directory, "show", f"{revision}:{name}") for name in names
               if "/" not in name and LICENSE_DOC.fullmatch(name)}
    if any(contains_secret(value) for value in notices.values()):
        raise ValueError("sensitive_notice")
    return root_name, notices


def acquire(root, output, *, sources=None, limit=100, scan=105, prior=None, split_targets=None):
    catalog = sources if sources is not None else {
        local: {"repo": repo, "license": spdx, "family": repo, "split": "unassigned"}
        for local, (repo, spdx) in SOURCES.items()
    }
    families = {}
    for item in catalog.values():
        split, family = item["split"], item["family"]
        if families.setdefault(family, split) != split:
            raise ValueError("a repository family cannot span splits")
    accepted, fingerprints, rejects = [], [], {}
    exact_keys, index = set(), defaultdict(list)

    def register(exact, shingles):
        exact_keys.add(exact)
        for shingle in shingles:
            index[shingle].append(len(fingerprints))
        fingerprints.append(shingles)

    if prior:
        for case in read_rows(prior):
            register(*patch_shingles(case["evidence"]))
    notices = {}
    for local, item in catalog.items():
        repo, spdx = item["repo"], item["license"]
        directory = root / (local + ".git")
        counts = Counter()
        rejects[repo] = counts
        if not directory.is_dir():
            counts["mirror_missing"] += 1
            continue
        for line in git(directory, "log", "--format=%H %P", "--no-merges").splitlines()[:scan]:
            revs = line.split()
            if len(revs) != 2:
                counts["boundary_or_root"] += 1
                continue
            head, base = revs
            try:
                subject = git(directory, "show", "-s", "--format=%s", head).strip()
                if re.search(r"\b(?:CVE-\d|GHSA-|exploit|vulnerability|gofmt|whitespace|cherry.pick|revert)\b", subject, re.I):
                    raise ValueError("excluded_subject")
                patch = git(directory, "diff", "--no-ext-diff", "--no-textconv", "--no-renames",
                            "--src-prefix=a/", "--dst-prefix=b/", "-U5", base, head, "--")
                evidence = accepted_patch(patch)
                paths = sorted({hunk["path"] for hunk in evidence})
                if any(EXCLUDED.search(path) or CREDENTIAL_PATH.search(path) for path in paths):
                    raise ValueError("excluded_path")
                if not any(PurePosixPath(path).suffix in SOURCE_SUFFIXES for path in paths):
                    raise ValueError("no_source_or_configuration")
                changed = [line[1:].strip() for h in evidence for line in h["lines"].splitlines()[1:]
                           if line.startswith(("+", "-")) and line[1:].strip()]
                if len(changed) < 5:
                    raise ValueError("too_small")
                if sources is not None:
                    removed, added = [], []
                    for hunk in evidence:
                        for line in hunk["lines"].split("\n")[1:]:
                            if line.startswith(("+", "-")):
                                compact = "".join(line[1:].split())
                                if compact:
                                    (added if line[0] == "+" else removed).append(compact)
                    if removed == added or re.match(r"^(?:release|prepare release|bump version|chore\(release\))",
                                                    subject, re.I):
                        raise ValueError("format_or_release_only")
                exact, shingles = patch_shingles(evidence)
                nearby = {idx for shingle in shingles for idx in index[shingle]}
                if exact in exact_keys or any(
                    len(shingles & fingerprints[idx]) >= .85 * len(shingles | fingerprints[idx])
                    for idx in nearby
                ):
                    raise ValueError("duplicate_change")
                pinned = {}
                for revision in (base, head):
                    root_license, revision_notices = verify_tree(directory, revision, paths, spdx)
                    for name, body in revision_notices.items():
                        checksum = sha256(body.encode()).hexdigest()
                        notices[checksum] = body
                        pinned.setdefault(revision, {})[name] = checksum
            except (ValueError, subprocess.CalledProcessError) as exc:
                counts[str(exc) if isinstance(exc, ValueError) else "missing_git_object"] += 1
                continue
            # Keep the commit subject out of model inputs and reference labels.
            register(exact, shingles)
            accepted.append({"id": f"{local}-{head[:10]}", "repo": repo, "base": base, "head": head,
                             "source_url": f"https://github.com/{repo}/commit/{head}", "license": spdx,
                             "license_url": f"https://github.com/{repo}/blob/{head}/{root_license}", "notices": pinned,
                             "diff": patch, "evidence": evidence, "paths": paths,
                             "messages": messages_for(evidence), "annotation_status": "awaiting_draft",
                             "family": item["family"], "split": item["split"],
                             "rights_clearance": "pending"})
            counts["eligible"] += 1
    by_repo = {item["repo"]: [] for item in catalog.values()}
    for row in accepted:
        by_repo[row["repo"]].append(row)
    for rows in by_repo.values():
        rows.sort(key=lambda row: len(row["paths"]) == 1)
    selected = []
    split_counts = Counter()
    while len(selected) < limit and any(by_repo.values()):
        for rows in by_repo.values():
            if rows and len(selected) < limit:
                row = rows.pop(0)
                split = row["split"]
                if split_targets and split_counts[split] >= split_targets.get(split, 0):
                    continue
                selected.append(row)
                split_counts[split] += 1
    output.mkdir(mode=stat.S_IRWXU)
    (output / "notices").mkdir()
    selected_notices = {digest for row in selected for files in row["notices"].values() for digest in files.values()}
    for checksum in selected_notices:
        (output / "notices" / checksum).write_text(notices[checksum])
    write_rows(output / "acquired.jsonl", selected)
    (output / "acquisition.json").write_text(json.dumps({
        "accepted": len(selected), "multi_file": sum(len(row["paths"]) > 1 for row in selected),
        "by_repository": dict(Counter(row["repo"] for row in selected)), "screening": rejects,
        "by_split": dict(split_counts), "source_catalog": catalog,
        "prior_sha256": sha256(prior.read_bytes()).hexdigest() if prior else None,
        "acquired_sha256": sha256((output / "acquired.jsonl").read_bytes()).hexdigest(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mirrors", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--scan", type=int, default=105)
    parser.add_argument("--split-targets", type=json.loads)
    args = parser.parse_args()
    acquire(args.mirrors, args.output, sources=json.loads(args.catalog.read_text()) if args.catalog else None,
            prior=args.prior, limit=args.limit, scan=args.scan, split_targets=args.split_targets)
