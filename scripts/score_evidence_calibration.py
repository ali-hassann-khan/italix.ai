"""Score the frozen evidence experiment from explicitly AI-adjudicated claim verdicts."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.evaluate_evidence_facts import cards, inspect_claims
from scripts.prepare_specialization import digest, read_rows


def shortlist(extracted, outputs):
    head, *cases = read_rows(extracted)
    files = {}
    all_rows = {}
    for file in outputs:
        meta, *rows = read_rows(file)
        if meta["fact_sha256"] != digest(extracted) or [r["id"] for r in rows] != [c["id"] for c in cases]:
            raise ValueError("incomplete or mismatched evidence calibration rows")
        if meta["model"] in files:
            raise ValueError("repeat model")
        files[meta["model"]] = digest(file)
        all_rows[meta["model"]] = {r["id"]: r for r in rows}
    selected = []
    for case in cases:
        given = {item["id"] for item in cards(case["facts"])}
        known = {f["id"]: f for f in case["facts"] if f["id"] in given}
        checked = {}
        for model, rows in all_rows.items():
            row = rows[case["id"]]
            claims, flags = inspect_claims(row.get("raw", ""), known)
            covered = {file for claim in claims for file in claim["files"]}
            missing = sorted(set(case["paths"]) - covered)
            checked[model] = row | {"claims": claims, "flags": sorted(set(flags + (
                ["incomplete_file_coverage"] if missing else []) + (
                ["output_budget"] if row.get("output_tokens", 0) >= 384 else []))),
                "missing_files": missing}
        eligible = {m: row for m, row in checked.items() if row["claims"]}
        chosen = min(eligible, key=lambda m: (
            len(set(eligible[m]["flags"]) - {"incomplete_file_coverage"}),
            len(eligible[m]["missing_files"]), m,
        )) if eligible else None
        selected.append({"id": case["id"], "paths": case["paths"], "selected_model": chosen,
                         "selected": eligible.get(chosen), "candidates": checked})
    return {"facts_sha256": digest(extracted), "model_result_sha256": files, "cases": selected}


def semantics(shortlist_path, old_path, references, verdicts, output):
    previous = json.loads(old_path.read_text())
    proposed = json.loads(shortlist_path.read_text())
    proposed_cases = {row["id"]: row for row in proposed["cases"]}
    reference_rows = {row["id"]: row for row in read_rows(references) if row["id"] in proposed_cases
                      and row["reference_claims"]}
    old_rows = {row["id"]: row for row in previous["records"]}
    checked = json.loads(verdicts.read_text())
    if set(checked) != set(reference_rows):
        raise ValueError("verdicts must cover every in-scope calibration case")
    metrics = {}
    for variant in ("old", "new"):
        aggregate = Counter()
        for key, adjudication in checked.items():
            old = old_rows[key]
            new = proposed_cases[key]["selected"] or {}
            claims = old["provisional_claims"] if variant == "old" else new.get("claims", [])
            judgments = adjudication[variant]
            labels = judgments["verdicts"]
            covered = judgments["required_facts"]
            required = reference_rows[key]["ai_review"]["claims"]
            passed = old["automatic_checks"] == "pass" if variant == "old" else bool(new and not new["flags"])
            if (len(labels) != len(claims) or set(labels) - {"S", "W", "C", "U"}
                    or len(set(covered)) != len(covered) or any(not 1 <= i <= len(required) for i in covered)):
                raise ValueError(f"verdict disagreement with claim shape: {variant} {key}")
            paths_by_hunk = {h["id"]: h["path"] for h in reference_rows[key]["evidence"]}
            true_files = {paths_by_hunk[h] for verdict, claim in zip(labels, claims) if verdict == "S"
                          for h in claim["evidence_ids"] if h in paths_by_hunk}
            explicit_files = set(judgments.get("supported_files", true_files))
            if not explicit_files <= true_files:
                raise ValueError(f"unsupported file credit: {variant} {key}")
            readiness = len(covered) == len(required) and bool(labels) and set(labels) == {"S"}
            if "ready" in judgments:  # Allows explicit AI materiality judgment alongside strict exact recall.
                readiness = judgments["ready"]
            aggregate.update({"cases": 1, "claims": len(claims), "facts": len(required),
                              "covered": len(covered), "supported_files": len(explicit_files),
                              "files": len(proposed_cases[key]["paths"]), "ready": int(readiness),
                              "autopasses": int(passed), "false_autopasses": int(passed and not readiness),
                              "edits": sum(mark != "S" for mark in labels) + len(required) - len(covered)})
            aggregate.update(labels)
        a = aggregate
        metrics[variant] = dict(a) | {
            "factual_precision": (a["S"] + a["W"]) / a["claims"] if a["claims"] else None,
            "claim_citation_support_rate": a["S"] / a["claims"] if a["claims"] else None,
            "required_fact_recall": a["covered"] / a["facts"] if a["facts"] else None,
            "cross_file_coverage": a["supported_files"] / a["files"] if a["files"] else None,
            "correction_units_per_case": a["edits"] / a["cases"],
            "automatic_false_acceptance_rate": a["false_autopasses"] / a["autopasses"] if a["autopasses"] else None,
        }
    with output.open("x") as stream:
        json.dump({"review_status": "AI-assisted_source_adjudication_not_human",
                   "old_results_sha256": digest(old_path), "shortlist_sha256": digest(shortlist_path),
                   "reference_sha256": digest(references), "verdicts_sha256": digest(verdicts),
                   "metrics": metrics}, stream, indent=2)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    select_cli = commands.add_parser("select")
    select_cli.add_argument("extracted", type=Path)
    select_cli.add_argument("output", type=Path)
    select_cli.add_argument("outputs", type=Path, nargs="+")
    score_cli = commands.add_parser("score")
    for name in ("shortlist_path", "old_path", "references", "verdicts", "output"):
        score_cli.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    if args.pop("command") == "select":
        with args.pop("output").open("x") as stream:
            json.dump(shortlist(**args), stream, indent=2)
    else:
        semantics(**args)
