"""Build a 20-case, offline blind-review packet from Phase 14/15 artifacts."""
import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import random
import re

from italix.context_compression import compress
from scripts.prepare_specialization import digest, messages_for, read_rows


def paired_lines(hunk):
    old, new = [], []
    for line in hunk["lines"].split("\n")[1:]:
        if line.startswith(" "):
            while len(old) < len(new):
                old.append("")
            while len(new) < len(old):
                new.append("")
            old.append(line)
            new.append(line)
        elif line.startswith("-"):
            old.append(line)
        elif line.startswith("+"):
            new.append(line)
        elif line.startswith("\\"):
            old.append(line)
            new.append(line)
    return "\n".join(old), "\n".join(new)


def make_packet(corpus, old_results, output, phase15):
    by_id = {row["id"]: row for row in read_rows(corpus)}
    comparisons = {}
    prior_ids = None
    for path in old_results:
        header, *rows = read_rows(path)
        ids = [row["id"] for row in rows]
        if header["corpus_sha256"] != digest(corpus) or len(ids) != 10 or (prior_ids and prior_ids != ids):
            raise ValueError("expected the matched 10-case Phase 14 artifacts")
        prior_ids = ids
        for row in rows:
            comparisons.setdefault(row["id"], []).append((header["model"] + "/phase14", row.get("prediction", "Runtime failure")))
    held = set(prior_ids)
    # Add ten unseen-to-this-calibration cases. Keep this packet separate from any final test.
    extra = sorted((row for key, row in by_id.items() if key not in held),
                   key=lambda row: (len(row["paths"]) == 1, row["repo"], row["id"]))
    balance, selected = {}, []
    for row in extra:
        if balance.get(row["repo"], 0) < 2:
            selected.append(row["id"])
            balance[row["repo"]] = balance.get(row["repo"], 0) + 1
    if len(selected) < 10:
        selected.extend(row["id"] for row in extra if row["id"] not in selected)
    ids = prior_ids + selected[:10]
    for path in phase15:
        header, *rows = read_rows(path)
        if header["cases_sha256"] != digest(corpus):
            raise ValueError("Phase 15 predictions belong to a different corpus")
        for row in rows:
            if row.get("budget") != 384 or row["layout"] != "compressed":
                continue
            tag = "/".join((header["model"], row["form"], row["layout"], str(row["budget"])))
            comparisons.setdefault(row["id"], []).append((tag, row["text"]))
    rng = random.Random(1501)
    mapping = {}
    with output.open("x") as stream:
        def write(html):
            print(html, file=stream)
        write("""<!doctype html><html lang="en"><meta charset="utf-8"><title>Phase 15 calibration</title>
<style>body{font:16px system-ui;margin:2rem auto;max-width:90rem;padding:0 1rem}summary{cursor:pointer;font-weight:bold}
details{margin:.8rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f3f3;padding:.6rem}
.diff{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:.5rem}.diff pre:first-child{background:#fff0f0}
.diff pre:nth-child(2){background:#eefbee}textarea{width:100%;min-height:3rem}fieldset{margin:.5rem 0}</style>
<h1>Phase 15 · 20 calibration cases</h1>
<p>Review the before/after code, correct each draft with cited hunks, and mark unreliable cases rejected. Model output is blinded.
Export before leaving. This page records a review submission for later adjudication.</p>
<p><label>Reviewer pseudonym <input id="reviewer"></label> <button id="export">Export review</button></p>
""")
        for key in ids:
            row = by_id[key]
            key_e = escape(key)
            write(f'<details data-id="{key_e}"><summary>{key_e} · {len(row["paths"])} files</summary>')
            write(f'<p>{escape(row["repo"])} · {escape(row["license"])} · <a href="{escape(row["source_url"], quote=True)}">Pinned change</a></p>')
            for hunk in row["evidence"]:
                old, new = paired_lines(hunk)
                write(f'<details><summary>{escape(hunk["id"])} · {escape(hunk["path"])}</summary>'
                      f'<pre>{escape(hunk["lines"].splitlines()[0])}</pre>'
                      f'<div class="diff"><pre>Before\n{escape(old)}</pre><pre>After\n{escape(new)}</pre></div></details>')
            compact = compress(row["evidence"])
            payload = messages_for(compact)[1]["content"]
            context_hash = sha256(payload.encode()).hexdigest()
            write("<details><summary>Compressed context to check for evidence loss</summary>")
            for hunk in compact:
                write(f'<pre>{escape(hunk["id"])} {escape(hunk["path"])}\n{escape(hunk["lines"])}</pre>')
            write(f'<label><input type="checkbox" data-part="reviewed_context_sha256" value="{context_hash}">'
                  'I checked that this exact compressed packet supports the corrected reference.</label></details>')
            write('<h3>Automated reference draft</h3><ul>')
            for fact in row["facts"]:
                write(f'<li>{escape(fact["text"])} [{escape(", ".join(fact["evidence_ids"]))}]</li>')
            write('''</ul><label>Reference decision <select data-part="decision"><option value="">Pending</option>
<option>Accept after checking</option><option>Correct</option><option>Reject</option></select></label>
<label>Corrections with hunk IDs or rejection reason<textarea data-part="reference"></textarea></label>
<details><summary>Blinded model outputs</summary>''')
            candidates = comparisons.get(key, []).copy()
            rng.shuffle(candidates)
            if not candidates:
                write('<p>No model output in the matched diagnostic set.</p>')
            mapping[key] = {}
            for i, (identity, text) in enumerate(candidates, 1):
                part = f"candidate-{i}"
                mapping[key][part] = identity
                write(f'<fieldset><legend>Candidate {i}</legend><pre>{escape(text)}</pre>'
                      '<label>Verdict <select data-part="' + part + '"><option value="">Pending</option>'
                      '<option>Supported</option><option>Incomplete</option><option>Contradicted</option>'
                      '<option>Unsupported</option><option>Unusable format</option></select></label>'
                      f'<label>Notes and evidence<textarea data-part="{part}-notes"></textarea></label></fieldset>')
            write('</details></details>')
        write("""<script>document.querySelector("#export").onclick = () => {
const reviewer = document.querySelector("#reviewer").value.trim();
if (!reviewer) { alert("Enter a reviewer pseudonym."); return; }
const cases = [...document.querySelectorAll("details[data-id]")].map(d => ({
  id: d.dataset.id, ...Object.fromEntries([...d.querySelectorAll("[data-part]")]
  .filter(x => x.value.trim() && (x.type !== "checkbox" || x.checked))
  .map(x => [x.dataset.part, x.value.trim()]))
})).filter(x => Object.keys(x).length > 1);
for (const row of cases) {
  if (["Correct", "Reject"].includes(row.decision) && !row.reference) {
    alert("Add correction evidence or a rejection reason for " + row.id); return;
  }
}
const a = document.createElement("a");
a.href = URL.createObjectURL(new Blob([JSON.stringify({reviewer, cases}, null, 2)], {type:"application/json"}));
a.download = "phase15-review.json"; a.click();
};</script></html>""")
    with output.with_suffix(".key.json").open("x") as stream:
        json.dump({"corpus_sha256": digest(corpus), "cases": ids, "blind_mapping": mapping}, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prior", type=Path, nargs="+", required=True)
    parser.add_argument("--matrix", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    make_packet(args.corpus, args.prior, args.output, args.matrix)
