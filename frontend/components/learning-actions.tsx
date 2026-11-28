"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import type { LearningStudioData } from "@/types/learning";

type Plan = {
  id: number;
  example_count: number;
  candidate_id: string;
  review_digest: string;
  model_revision: string;
  configuration: {
    optimizer_steps: number; rank: number; trainable_layers: number; learning_rate: number;
    batch_size: number; max_sequence_length: number; seed: number; lora_scale: number; lora_dropout: number;
  };
  dataset_checksums: Record<string, string>;
  queued_examples_sha256: string;
};

export function LearningActions({ data, onChange }: { data: LearningStudioData; onChange: () => void }) {
  const [workload, setWorkload] = useState("pr_summary");
  const [busy, setBusy] = useState(false);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [message, setMessage] = useState("");
  const [candidateId, setCandidateId] = useState("");
  const [content, setContent] = useState("");
  const [preview, setPreview] = useState("");
  const candidate = data.models.find((model) => model.id === candidateId);
  const activeJob = data.jobs.find((job) => job.status === "running");

  async function request(payload: Record<string, unknown>) {
    const response = await fetch("/api/operator/learning", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail ?? "Learning action failed.");
    return result;
  }

  async function run(action: string, payload: Record<string, unknown> = {}) {
    setBusy(true);
    setMessage("");
    try {
      const result = await request({ action, workload, ...payload });
      if (action === "plan") {
        setPlan(await request({ action: "review", jobId: result.id }));
      } else if (action === "review") {
        setPlan(result);
      } else if (action === "preview") {
        setPreview(result.output);
      } else {
        if (action === "execute") setPlan(null);
        setMessage(action === "recover" ? `Recovered ${result.recovered} interrupted job(s).`
          : action === "execute" ? "Local training started." : "Action completed.");
      }
      onChange();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Learning action failed.");
    } finally {
      setBusy(false);
    }
  }

  return <section className="space-y-5 rounded-xl border border-zinc-800 bg-zinc-950/60 p-5 text-sm text-zinc-300">
    <h2 className="font-semibold text-white">Training and candidate controls</h2>
    <div className="flex flex-wrap items-center gap-2">
      <label htmlFor="learning-workload">Workload</label>
      <select id="learning-workload" className="rounded border border-zinc-700 bg-zinc-950 p-2"
        value={workload} onChange={(event) => { setWorkload(event.target.value); setPlan(null); setCandidateId(""); setPreview(""); }}>
        <option value="pr_summary">PR summary</option><option value="change_triage">Change triage</option>
      </select>
      <Button disabled={busy} onClick={() => void run("plan")}>Create training plan</Button>
      <Button variant="outline" disabled={busy || !activeJob} onClick={() => {
        if (window.confirm("Recover interrupted jobs and return their examples to the queue?")) void run("recover");
      }}>Recover interrupted jobs</Button>
    </div>
    {data.jobs.filter((job) => job.status === "planned" && job.workload === workload).map((job) => (
      <Button key={job.id} variant="outline" disabled={busy} onClick={() => void run("review", { jobId: job.id })}>
        Review plan #{job.id}
      </Button>
    ))}
    {plan && <div className="space-y-2 rounded border border-zinc-800 p-3">
      <p>Plan #{plan.id} · {plan.example_count} approved examples · {plan.candidate_id}</p>
      <p>{plan.configuration.optimizer_steps} steps · rank {plan.configuration.rank} · {plan.configuration.trainable_layers} layers · LR {plan.configuration.learning_rate}</p>
      <p>Batch {plan.configuration.batch_size} · Sequence limit {plan.configuration.max_sequence_length} · Seed {plan.configuration.seed} · Scale {plan.configuration.lora_scale} · Dropout {plan.configuration.lora_dropout}</p>
      <p className="break-all font-mono text-xs">Model revision {plan.model_revision}</p>
      {Object.entries(plan.dataset_checksums).map(([key, value]) => <p key={key} className="break-all font-mono text-xs">{key} {value}</p>)}
      <p className="break-all font-mono text-xs">Approved examples SHA-256 {plan.queued_examples_sha256}</p>
      <Button disabled={busy || Boolean(activeJob)} onClick={() => {
        if (window.confirm(`Execute reviewed local LoRA plan #${plan.id}?`)) {
          void run("execute", { jobId: plan.id, digest: plan.review_digest });
        }
      }}>Execute reviewed plan</Button>
    </div>}
    {activeJob && <p role="status">Job #{activeJob.id} running · Step {activeJob.metrics.current_step ?? "pending"}</p>}
    {data.jobs.filter((job) => job.status === "failed" && job.workload === workload).slice(0, 3).map((job) => (
      <p key={job.id} className="text-amber-300">Job #{job.id} · {job.failure}</p>
    ))}
    <div className="space-y-2 border-t border-zinc-800 pt-4">
      <label htmlFor="learning-candidate">Candidate</label>
      <select id="learning-candidate" className="ml-2 rounded border border-zinc-700 bg-zinc-950 p-2"
        value={candidateId} onChange={(event) => { setCandidateId(event.target.value); setPreview(""); }}>
        <option value="">Select an adapter</option>
        {data.models.filter((model) => model.kind === "adapter" && model.workload === workload).map((model) =>
          <option key={model.id} value={model.id}>{model.id} · {model.status}</option>)}
      </select>
      <label className="block" htmlFor="candidate-preview">Preview diff</label>
      <textarea id="candidate-preview" className="w-full rounded border border-zinc-700 bg-zinc-950 p-2" rows={3}
        value={content} maxLength={20000} onChange={(event) => setContent(event.target.value)} />
      <div className="flex flex-wrap gap-2">
        <Button variant="outline" disabled={busy || !candidate || !content.trim()}
          onClick={() => { setPreview(""); void run("preview", { candidateId, content }); }}>Preview output</Button>
        <Button variant="outline" disabled={busy || !candidate} onClick={() => void run("evaluation", { candidateId })}>Run qualification</Button>
        <Button variant="outline" disabled={busy || !candidate?.qualified} onClick={() => {
          if (window.confirm(`Activate ${candidateId} for ${workload}?`)) void run("activate", { candidateId });
        }}>Activate qualified adapter</Button>
        <Button variant="outline" disabled={busy} onClick={() => {
          if (window.confirm(`Restore the base route for ${workload}?`)) void run("rollback");
        }}>Roll back to base</Button>
      </div>
      {preview && <pre className="whitespace-pre-wrap break-words">{preview}</pre>}
    </div>
    <p role="status">{busy ? "Working…" : message}</p>
  </section>;
}
