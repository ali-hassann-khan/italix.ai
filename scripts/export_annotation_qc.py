"""Export evidence-linked AI adjudications without granting human or rights approval."""
import argparse
from collections import Counter
import json
from pathlib import Path

from italix.privacy import contains_secret
from scripts.prepare_specialization import digest, read_rows


def load_adjudications(path, cases):
    rows = {}
    current = None
    for line in path.read_text().splitlines():
        if line.startswith("@"):
            key, disposition = line[1:].split()
            if key in rows or key not in cases or disposition not in {"accepted", "corrected", "rejected"}:
                raise ValueError("invalid case or disposition")
            current = {"id": key, "disposition": disposition, "claims": [], "findings": []}
            rows[key] = current
        elif line.startswith("!") and current is not None:
            finding = line[1:].strip()
            if not finding or contains_secret(finding):
                raise ValueError("invalid review finding")
            current["findings"].append(finding)
        elif line:
            if current is None:
                raise ValueError("fact without case")
            ids, text = line.split("|", 1)
            ids = ids.split(",")
            available = {hunk["id"] for hunk in cases[current["id"]]["evidence"]}
            if not text.strip() or contains_secret(text) or not set(ids) <= available:
                raise ValueError(f"invalid fact or hunk in {current['id']}")
            current["claims"].append({"text": text, "evidence_ids": list(dict.fromkeys(ids))})
    if not rows or any(not row["claims"] for row in rows.values()):
        raise ValueError("adjudications must contain evidence-linked facts")
    return rows


def export(corpus, audit_path, results, adjudications, output, *, allow_partial=False):
    audit = json.loads(audit_path.read_text())
    candidates = json.loads(results.read_text())
    corpus_hash = digest(corpus)
    if corpus_hash != audit["corpus_sha256"] or corpus_hash != candidates["corpus_sha256"]:
        raise ValueError("audit, candidates and sources must share the same corpus")
    cases = {case["id"]: case for case in read_rows(corpus)}
    generated = {r["id"]: r for r in candidates["records"]}
    if not generated or not generated.keys() <= cases.keys() or (not allow_partial and set(generated) != set(cases)):
        raise ValueError("all acquired cases require matched model outputs")
    reviewed = load_adjudications(adjudications, cases)
    if not reviewed.keys() <= generated.keys() or not set(audit["initial_review_packet_ids"]) <= reviewed.keys():
        raise ValueError("all initial calibration cases require a review disposition")
    notice_ids = {value for case in cases.values() for notices in case["notices"].values()
                  for value in notices.values()}
    source_notices = corpus.parent / "notices"
    for key in notice_ids:
        if len(key) != 64 or any(ch not in "0123456789abcdef" for ch in key) or digest(source_notices / key) != key:
            raise ValueError("license notice unavailable or checksum mismatch")
    output.mkdir(mode=0o700)
    if notice_ids:
        notice_output = output / "notices"
        notice_output.mkdir(mode=0o700)
        for key in sorted(notice_ids):
            (notice_output / key).write_bytes((source_notices / key).read_bytes())
    counts = Counter({key: 0 for key in ("accepted", "corrected", "rejected", "unresolved")})
    with (output / "qc-corpus.jsonl").open("x") as stream:
        for key, case in cases.items():
            ai_review = reviewed.get(key)
            draft = generated.get(key)
            disposition = ai_review["disposition"] if ai_review else "unresolved"
            counts[disposition] += 1
            fields = {k: case[k] for k in ("id", "repo", "base", "head", "source_url",
                                           "license", "license_url", "notices", "paths", "diff", "evidence")}
            if disposition == "accepted":
                final = draft["provisional_claims"]
                available = {hunk["id"] for hunk in case["evidence"]}
                if not final or any(contains_secret(c["text"]) or not set(c["evidence_ids"])
                                    or not set(c["evidence_ids"]) <= available for c in final):
                    raise ValueError(f"accepted annotation has invalid citations: {key}")
            else:
                final = ai_review["claims"] if disposition == "corrected" else None
            record = {**fields, "draft": draft, "disposition": disposition,
                      "ai_review": {"status": "source_reviewed", "method": "AI-assisted review of pinned source hunks",
                                    "claims": ai_review["claims"], "findings": ai_review["findings"]}
                      if ai_review else {"status": "pending"},
                      "reference_claims": final,
                      "reference_summary": " ".join(claim["text"] for claim in final) if final else None,
                      "human_review": "pending", "rights_clearance": "pending",
                      "split": case.get("split", "unassigned"), "family": case.get("family", case["repo"]),
                      "training_eligible": False}
            print(json.dumps(record, ensure_ascii=False), file=stream)
    manifest = {"format": "italix-ai-annotation-qc-v1", "source_sha256": corpus_hash,
                "audit_sha256": digest(audit_path), "three_model_results_sha256": digest(results),
                "adjudication_sha256": digest(adjudications),
                "dataset_sha256": digest(output / "qc-corpus.jsonl"),
                "models": candidates["models"], "review_dispositions": counts,
                "license_notice_sha256": sorted(notice_ids),
                "model_annotated": len(generated), "ai_source_reviewed": len(reviewed), "independently_human_reviewed": 0, "human_rights_clearances": 0,
                "training_eligible": 0}
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "audit", "results", "adjudications", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    export(args.corpus, args.audit, args.results, args.adjudications, args.output, allow_partial=args.allow_partial)
