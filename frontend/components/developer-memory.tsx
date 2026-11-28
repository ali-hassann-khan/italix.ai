"use client";

import { useEffect, useState } from "react";
import { OperatorUnlock } from "@/components/gateway-playground";
import { Button } from "@/components/ui/button";

interface Source {
  id: string;
  title: string;
  content: string;
  kind: string;
  timestamp: string;
  metadata: { citation: string };
}

export function DeveloperMemory() {
  const [session, setSession] = useState<"checking" | "locked" | "ready">("checking");
  const [question, setQuestion] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [answer, setAnswer] = useState("");
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [importPath, setImportPath] = useState("");
  const [importKind, setImportKind] = useState("import-git");
  const [approved, setApproved] = useState(false);
  const [retentionDays, setRetentionDays] = useState("90");

  useEffect(() => {
    fetch("/api/operator/session", { cache: "no-store" })
      .then(async (response) => setSession(response.ok && (await response.json()).authenticated ? "ready" : "locked"))
      .catch(() => setSession("locked"));
  }, []);

  async function run(payload: Record<string, unknown>) {
    setBusy(true);
    setStatus("");
    try {
      const response = await fetch("/api/operator/memory", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await response.json();
      if (response.status === 401) {
        setSession("locked");
        setQuestion("");
        setAnswer("");
        setSources([]);
        return;
      }
      if (!response.ok) {
        setStatus(body.detail);
        return;
      }
      if (payload.action === "import-git" || payload.action === "import-discussions") {
        setStatus(`Imported ${body.imported} source(s); ${body.duplicates} already present.`);
        setImportPath("");
        setApproved(false);
      } else if (payload.action === "redact") {
        setSources([]);
        setAnswer("");
        setStatus(`Redacted ${body.redacted} source(s).`);
      } else if (payload.action === "delete" || payload.action === "purge") {
        setSources([]);
        setAnswer("");
        setStatus(`Removed ${body.deleted} source(s).`);
      } else {
        setAnswer(body.answer ?? "");
        setSources(body.abstained ? (body.retrieved ?? []) : (body.sources ?? body.data ?? []));
        if (body.abstained) setStatus("Answer abstained. Review any available sources below.");
        else if (!body.answer && !body.data?.length) setStatus("No matching sources.");
      }
    } catch {
      setStatus("The local memory request failed.");
    } finally {
      setBusy(false);
    }
  }

  if (session === "checking") return <p role="status">Checking operator session…</p>;
  if (session === "locked") return <OperatorUnlock onUnlocked={() => setSession("ready")} />;

  return (
    <div className="space-y-5 text-sm text-zinc-300">
      <form className="space-y-3" onSubmit={(event) => { event.preventDefault(); void run({ action: "search", question }); }}>
        <label className="block" htmlFor="memory-question">Search imported engineering decisions</label>
        <input id="memory-question" className="w-full rounded-lg border border-zinc-700 bg-zinc-950 p-3"
          value={question} maxLength={2000} onChange={(event) => setQuestion(event.target.value)} />
        <div className="flex flex-wrap gap-2">
          <Button type="submit" disabled={busy || question.trim().length < 3}>Search sources</Button>
          <Button type="button" variant="outline" disabled={busy || question.trim().length < 3}
            onClick={() => void run({ action: "query", question })}>Ask local model</Button>
        </div>
      </form>
      <p role="status">{busy ? "Working…" : status}</p>
      {answer && <p className="whitespace-pre-wrap rounded-lg border border-zinc-700 p-4">{answer}</p>}
      <ul className="space-y-3">
        {sources.map((source) => (
          <li key={source.id} className="space-y-2 rounded-lg border border-zinc-700 p-4">
            <p className="font-semibold">{source.title}</p>
            <p className="text-xs text-zinc-400">{source.metadata.citation} · {source.kind} · {source.timestamp}</p>
            <p className="whitespace-pre-wrap break-words">{source.content}</p>
            <Button type="button" variant="outline" disabled={busy} onClick={() => {
              if (window.confirm("Delete this source from local Memory?")) void run({ action: "delete", sourceId: source.id });
            }}>Delete source</Button>
          </li>
        ))}
      </ul>
      <form className="space-y-2 border-t border-zinc-700 pt-4" onSubmit={(event) => {
        event.preventDefault();
        if (approved) void run({ action: importKind, filePath: importPath, approved });
      }}>
        <label className="block" htmlFor="memory-import-kind">Import source</label>
        <select className="rounded border border-zinc-700 bg-zinc-950 p-2" id="memory-import-kind"
          value={importKind} onChange={(event) => setImportKind(event.target.value)}>
          <option value="import-git">Local Git repository</option>
          <option value="import-discussions">Approved discussion export</option>
        </select>
        <label className="block" htmlFor="memory-import-path">Local path</label>
        <input className="w-full rounded border border-zinc-700 bg-zinc-950 p-2" id="memory-import-path" value={importPath}
          onChange={(event) => setImportPath(event.target.value)} maxLength={2048} required />
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={approved} onChange={(event) => setApproved(event.target.checked)} />
          I approve importing this source into local Memory
        </label>
        <Button type="submit" variant="outline" disabled={busy || !approved || !importPath.trim()}>Import source</Button>
      </form>
      <Button variant="outline" disabled={busy} onClick={() => {
        if (window.confirm("Apply credential redaction to existing local Memory sources?")) void run({ action: "redact" });
      }}>Redact existing sources</Button>
      <form className="space-y-2 border-t border-zinc-700 pt-4" onSubmit={(event) => {
        event.preventDefault();
        if (window.confirm("Delete all local Memory sources imported before this retention window?")) {
          void run({ action: "purge", retentionDays: Number(retentionDays) });
        }
      }}>
        <label htmlFor="memory-retention">Keep sources imported in the last</label>
        <div className="flex items-center gap-2">
          <input id="memory-retention" className="w-20 rounded-lg border border-zinc-700 bg-zinc-950 p-2"
            type="number" min={1} max={3650} required value={retentionDays} onChange={(event) => setRetentionDays(event.target.value)} />
          <span>days</span>
          <Button type="submit" variant="outline" disabled={busy}>Purge older sources</Button>
        </div>
      </form>
    </div>
  );
}
