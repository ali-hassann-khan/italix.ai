"""Evaluate Phase 14 cases locally and build a static review packet."""
import argparse
from html import escape
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
from time import perf_counter

from italix.engines import LOCAL_MODELS, LocalMLXEngine
from italix.privacy import redact
from italix.semantic_review import semantic_diagnostics
from scripts.prepare_specialization import digest, read_rows, write_rows


def annotate(case):
    """Create conservative extractive references for a human to rewrite."""
    facts = []
    for path in case["paths"]:
        hunks = [hunk for hunk in case["evidence"] if hunk["path"] == path]
        choices = []
        for hunk in hunks:
            for line in hunk["lines"].splitlines()[1:]:
                if not line.startswith(("+", "-")):
                    continue
                code = line[1:].strip()
                words = re.findall(r"[A-Za-z_][A-Za-z_0-9]{3,}", code)
                if not code:
                    continue
                substantive = bool(words) and not code.startswith(("//", "#"))
                choices.append((substantive, line.startswith("+"), min(len(words), 5), hunk["id"], code, words))
        if not choices:
            continue
        _, added, _, hunk_id, code, words = max(choices, key=lambda row: row[:3])
        anchor = max(words, key=len, default=code)
        facts.append({"id": f"draft-{len(facts) + 1}", "text": f"In {path}, {'add' if added else 'remove'} `{code}`.",
                      "anchors": [anchor], "numbers": [], "evidence_ids": [hunk_id],
                      "basis": "exact_changed_line", "changed_line": ("+" if added else "-") + code})
    return {**case, "reference_summary": " ".join(fact["text"] for fact in facts), "facts": facts,
            "annotation_status": "automated_extractive_pending_human_review"}


def run(corpus, output, model_id, limit):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    spec = LOCAL_MODELS[model_id]
    spec.cached_path()
    engine = LocalMLXEngine(model_id=model_id)
    cases = read_rows(corpus)
    if limit is not None:
        cases = cases[:limit]
    with output.open("x") as stream:
        def write(row):
            print(json.dumps(row, ensure_ascii=False), file=stream, flush=True)

        write({"model": model_id, "revision": spec.revision, "corpus_sha256": digest(corpus),
               "temperature": 0, "max_tokens": 192, "mlx_lm": version("mlx-lm")})
        try:
            for case in cases:
                started = perf_counter()
                try:
                    result = engine.generate(case["messages"], max_tokens=192, temperature=0)
                except (OSError, RuntimeError, ValueError, ImportError) as exc:
                    write({"id": case["id"], "error": type(exc).__name__})
                    continue
                text = redact(result.text)
                elapsed = (perf_counter() - started) * 1000
                write({"id": case["id"], "prediction": text, "elapsed_ms": elapsed,
                       "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                       **(result.metadata or {}), **diagnose(case, text)})
        finally:
            engine.unload()


def diagnose(case, text):
    paths = {h["id"]: h["path"] for h in case["evidence"]}
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\s*```", text.strip(), re.I | re.S)
    return {"fact_count": len(case["facts"]),
            "coverage_file_count": len({paths[k] for fact in case["facts"] for k in fact["evidence_ids"]}),
            "diagnostics": semantic_diagnostics(text, case["facts"], paths),
            "fence_tolerant_diagnostics": semantic_diagnostics(fenced[1] if fenced else text, case["facts"], paths)}


def rescore(previous, corpus, results, output):
    header, *rows = read_rows(results)
    old = {case["id"]: case["messages"] for case in read_rows(previous)}
    new = {case["id"]: case for case in read_rows(corpus)}
    if header["corpus_sha256"] != digest(previous) or old != {key: case["messages"] for key, case in new.items()}:
        raise ValueError("prediction inputs or provenance differ")
    with output.open("x") as stream:
        print(json.dumps({**header, "corpus_sha256": digest(corpus),
                          "original_annotation_sha256": digest(previous)}), file=stream)
        for row in rows:
            if "prediction" in row:
                row.update(diagnose(new[row["id"]], row["prediction"]))
            print(json.dumps(row, ensure_ascii=False), file=stream)


def report(results):
    comparison = None
    for path in results:
        header, *rows = read_rows(path)
        key = (header["corpus_sha256"], header["max_tokens"], header["temperature"], tuple(row["id"] for row in rows))
        if comparison is not None and key != comparison:
            raise ValueError("results differ in corpus, decoding settings or case order")
        comparison = key
        responses = [row for row in rows if "diagnostics" in row]
        delays = sorted(row["elapsed_ms"] for row in responses[1:])
        facts = sum(r["fact_count"] for r in responses)
        files = sum(r["coverage_file_count"] for r in responses)
        tolerant = [r.get("fence_tolerant_diagnostics", r["diagnostics"]) for r in responses]
        def flagged(label):
            return sum(bool(r["diagnostics"][label]) for r in responses)
        print(json.dumps({"model": header["model"], "rows": len(rows), "errors": len(rows) - len(responses),
                          "valid_json": flagged("valid_json"),
                          "valid_after_unfencing": sum(r["valid_json"] for r in tolerant),
                          "tolerant_anchor_fact_recall": 1 - sum(len(r["missing_facts"]) for r in tolerant) / facts if facts else None,
                          "tolerant_anchor_file_recall": 1 - sum(len(r["missing_files"]) for r in tolerant) / files if files else None,
                          "tolerant_unmatched_claim_cases": sum(bool(r["unmatched_claim_indexes"]) for r in tolerant),
                          "tolerant_impact_flag_cases": sum(bool(r["security_or_performance_claims_for_review"]) for r in tolerant),
                          "at_output_cap": sum(r["completion_tokens"] >= header["max_tokens"] for r in responses),
                          "mean_prompt_tokens": statistics.mean(r["prompt_tokens"] for r in responses) if responses else None,
                          "anchor_fact_recall": 1 - sum(len(r["diagnostics"]["missing_facts"]) for r in responses) / facts if facts else None,
                          "anchor_file_recall": 1 - sum(len(r["diagnostics"]["missing_files"]) for r in responses) / files if files else None,
                          "unmatched_claim_cases": flagged("unmatched_claim_indexes"),
                          "impact_flag_cases": flagged("security_or_performance_claims_for_review"),
                          "duplicate_claim_cases": flagged("duplicate_claims"),
                          "mean_repetition_ratio": statistics.mean(r["diagnostics"]["repetition_ratio"] for r in responses) if responses else None,
                          "cold_ms": responses[0]["elapsed_ms"] if responses else None,
                          "warm_mean_ms": statistics.mean(delays) if delays else None,
                          "warm_median_ms": statistics.median(delays) if delays else None,
                          "warm_p95_ms": delays[math.ceil(.95 * len(delays)) - 1] if delays else None,
                          "peak_mlx_gb": max((r["peak_memory_gb"] or 0 for r in responses), default=None),
                          "mean_generation_tps": statistics.mean(r["generation_tps"] or 0 for r in responses) if responses else None}))


def packet(corpus, output, results):
    cases = read_rows(corpus)
    models = []
    for path in results:
        header, *rows = read_rows(path)
        if header["corpus_sha256"] != digest(corpus):
            raise ValueError("prediction file and review corpus differ")
        models.append((header["model"], {row["id"]: row for row in rows}))
    mapping = {}
    rng = random.Random(701)
    with output.open("x") as stream:
        def write(text):
            print(text, file=stream)
        write('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Phase 14 calibration review</title>
<style>body{font:16px system-ui;margin:2rem auto;max-width:78rem;padding:0 1rem}
pre{white-space:pre-wrap;background:#eee;padding:1rem}summary{cursor:pointer;font-weight:bold}
details{margin-bottom:1rem}textarea{width:100%;min-height:4rem}a{color:#14548b}</style>
<h1>Phase 14 calibration review</h1><p>Review the patch first. Draft claims are automated extracts.
Mark each claim supported, contradicted or unsupported. Add omitted facts with hunk IDs.
For candidates, also record duplicate claims and relevant files missed. Export before closing.</p>
<p><label>Reviewer pseudonym <input id="reviewer"></label> <button id="export">Export notes</button></p>''')
        for case in cases:
            ident = case["id"]
            write(f'<details><summary>{escape(ident)} · {len(case["paths"])} files</summary>')
            write(f'<p>{escape(case["repo"])} · {escape(case["license"])} · <a href="{escape(case["source_url"], quote=True)}">Pinned change</a></p>')
            for hunk in case["evidence"]:
                write(f'<details><summary>{escape(hunk["id"])} · {escape(hunk["path"])}</summary><pre>{escape(hunk["lines"])}</pre></details>')
            write('<h3>Automated reference draft</h3>')
            for fact in case["facts"]:
                write(f'<p>{escape(fact["text"])} [{escape(", ".join(fact["evidence_ids"]))}]</p>')
            write(f'<label>Reference corrections<textarea data-case="{escape(ident)}" data-part="reference"></textarea></label>')
            shuffled = models.copy()
            rng.shuffle(shuffled)
            mapping[ident] = {}
            for index, (model, rows) in enumerate(shuffled, 1):
                row = rows.get(ident)
                if row is None:
                    continue
                label = f"candidate-{index}"
                mapping[ident][label] = model
                write(f'<h3>Candidate {index}</h3><pre>{escape(row.get("prediction", "Runtime failure"))}</pre>')
                write(f'<label>Candidate {index} adjudication<textarea data-case="{escape(ident)}" data-part="{label}"></textarea></label>')
            write('</details>')
        write('''<script>document.querySelector("#export").onclick = () => {
const reviewer = document.querySelector("#reviewer").value.trim();
if (!reviewer) { alert("Enter a reviewer pseudonym."); return; }
const notes = Array.from(document.querySelectorAll("textarea")).filter(t => t.value.trim())
.map(t => ({case: t.dataset.case, part: t.dataset.part, notes: t.value.trim()}));
const a = document.createElement("a");
a.href = URL.createObjectURL(new Blob([JSON.stringify({reviewer, notes}, null, 2)], {type:"application/json"}));
a.download = "phase14-review.json"; a.click();
};</script></html>''')
    if models:
        with output.with_suffix(".key.json").open("x") as stream:
            json.dump(mapping, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("annotate", "run", "packet", "report"):
        sub = actions.add_parser(action)
        if action != "report":
            sub.add_argument("corpus", type=Path)
            sub.add_argument("output", type=Path)
        if action in {"packet", "report"}:
            sub.add_argument("results", nargs="*" if action == "packet" else "+", type=Path)
        if action == "run":
            sub.add_argument("--model", choices=LOCAL_MODELS, required=True)
            sub.add_argument("--limit", type=int)
    sub = actions.add_parser("rescore")
    for name in ("previous", "corpus", "result", "output"):
        sub.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.action == "annotate":
        if args.output.exists():
            parser.error("annotation output must be new")
        write_rows(args.output, [annotate(case) for case in read_rows(args.corpus)])
    elif args.action == "run":
        run(args.corpus, args.output, args.model, args.limit)
    elif args.action == "packet":
        packet(args.corpus, args.output, args.results)
    elif args.action == "rescore":
        rescore(args.previous, args.corpus, args.result, args.output)
    else:
        report(args.results)
