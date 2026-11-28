"""Evidence-linked diagnostics for research, independent of model qualification."""

import json
import re
from collections import Counter

from italix.evaluation import repetition_ratio

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:%|\s*(?:ms|seconds?|minutes?|hours?|days?|KiB|MiB|GiB)\b)?", re.I)
IMPACT = re.compile(r"\b(performance|speedup|faster|slower|throughput|latency|memory usage|"
                    r"vulnerabilit\w*|security|secure|exploits?|attack)\b", re.I)
RISK = re.compile(r"\brisk\s*[:=]\s*(low|medium|high)\b", re.I)


def numbers(text):
    units = {"seconds": "second", "minutes": "minute", "hours": "hour", "days": "day"}
    result = set()
    for match in NUMBER.finditer(text):
        parts = match.group().casefold().split()
        if len(parts) == 2:
            parts[-1] = units.get(parts[-1], parts[-1])
        result.add(" ".join(parts))
    return result


def semantic_diagnostics(text, facts, evidence_paths):
    """Report structural and lexical findings for subsequent human adjudication."""
    try:
        value = json.loads(text)
        claims = value["claims"]
        if not isinstance(value, dict) or set(value) != {"claims"} or not isinstance(claims, list):
            raise ValueError
        if not all(isinstance(claim, dict) and set(claim) == {"text", "evidence_ids"}
                   and isinstance(claim["text"], str) and claim["text"].strip()
                   and isinstance(claim["evidence_ids"], list)
                   and all(isinstance(key, str) for key in claim["evidence_ids"])
                   for claim in claims):
            raise ValueError
    except (ValueError, TypeError, KeyError):
        claims = []
        valid = False
    else:
        valid = True

    covered, covered_files, unknown_ids = set(), set(), set()
    unmatched, numeric, impact = [], [], []
    seen_values = {}
    for index, claim in enumerate(claims):
        wording = claim["text"]
        folded = wording.casefold()
        ids = set(claim["evidence_ids"])
        unknown_ids.update(ids - evidence_paths.keys())
        matched = [fact for fact in facts if any(
            re.search(r"(?<!\w)" + re.escape(alias.casefold()) + r"(?!\w)", folded)
            for alias in fact["anchors"]
        )]
        if not matched:
            unmatched.append(index)
        if IMPACT.search(wording) and not any(
            fact.get("impact_evidence", False) and set(fact["evidence_ids"]) <= ids
            for fact in matched
        ):
            impact.append(index)
        for fact in matched:
            required = set(fact["evidence_ids"])
            allowed = numbers(" ".join(fact["numbers"]))
            actual = numbers(wording)
            if allowed and actual - allowed:
                numeric.append({"claim": index, "fact": fact["id"], "unexpected": sorted(actual - allowed)})
            if allowed and actual:
                seen_values.setdefault(fact["id"], set()).add(tuple(sorted(actual)))
            if required <= ids and allowed <= actual and not (allowed and actual - allowed):
                covered.add(fact["id"])
                covered_files.update(evidence_paths[key] for key in required)
    required_files = {evidence_paths[key] for fact in facts for key in fact["evidence_ids"]}
    wording = [claim["text"] for claim in claims]
    return {
        "valid_json": valid,
        "missing_facts": sorted({fact["id"] for fact in facts} - covered),
        "missing_files": sorted(required_files - covered_files),
        "numeric_mismatches": numeric,
        "possible_contradictions": sorted(key for key, values in seen_values.items() if len(values) > 1),
        "conflicting_risk_labels": sorted(set(RISK.findall(text.casefold()))) if len(set(RISK.findall(text.casefold()))) > 1 else [],
        "unmatched_claim_indexes": unmatched,
        "unknown_evidence_ids": sorted(unknown_ids),
        "security_or_performance_claims_for_review": impact,
        "duplicate_claims": sum(count - 1 for count in Counter(word.casefold().strip() for word in wording).values()),
        "repetition_ratio": repetition_ratio(" ".join(wording) if valid else text),
    }
