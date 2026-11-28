import { NextRequest, NextResponse } from "next/server";

import { getLearningStudioData, operatorGet, operatorMutation } from "@/lib/italix-operator";
import { isSameOrigin, readJsonBody } from "@/lib/operator-request";
import { OPERATOR_COOKIE, verifyOperatorSession } from "@/lib/operator-session";

export const runtime = "nodejs";

export async function GET(request: NextRequest) {
  try {
    if (!verifyOperatorSession(request.cookies.get(OPERATOR_COOKIE)?.value)) {
      return NextResponse.json(
        { detail: "Operator session required." },
        { status: 401, headers: { "Cache-Control": "no-store" } },
      );
    }
    const data = await getLearningStudioData();
    return NextResponse.json(data, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json(
      { detail: "Learning status is unavailable from the local ITALIX backend." },
      { status: 502, headers: { "Cache-Control": "no-store" } },
    );
  }
}

export async function POST(request: NextRequest) {
  const headers = { "Cache-Control": "no-store" };
  const reply = (body: unknown, status = 200) => NextResponse.json(body, { status, headers });
  if (!isSameOrigin(request)) return reply({ detail: "Same-origin request required." }, 403);

  try {
    if (!verifyOperatorSession(request.cookies.get(OPERATOR_COOKIE)?.value)) {
      return reply({ detail: "Operator session required." }, 401);
    }
    const input = await readJsonBody(request);
    if (!input.ok) return reply({ detail: input.message }, input.status);
    if (!input.value || typeof input.value !== "object" || Array.isArray(input.value)) {
      return reply({ detail: "Invalid learning request." }, 400);
    }
    const { action, jobId, digest, workload, candidateId, content } = input.value as Record<string, unknown>;
    const validWorkload = workload === "pr_summary" || workload === "change_triage";
    const validJob = typeof jobId === "number" && Number.isSafeInteger(jobId) && jobId > 0;
    const validCandidate = typeof candidateId === "string" && /^[a-z0-9][a-z0-9_-]{0,127}$/.test(candidateId);
    if (action === "review" && validJob) {
      return reply(await operatorGet(`/v1/training/jobs/${jobId}/review`));
    }
    let result;
    if (action === "plan" && validWorkload) {
      result = await operatorMutation("/v1/training/jobs", "POST", { workload });
    } else if (action === "execute" && validJob && typeof digest === "string" && /^[a-f0-9]{64}$/.test(digest)) {
      result = await operatorMutation(`/v1/training/jobs/${jobId}/execute`, "POST", { review_digest: digest });
    } else if (action === "recover") {
      result = await operatorMutation("/v1/training/recover", "POST");
    } else if (action === "rollback" && validWorkload) {
      result = await operatorMutation(`/v1/routes/${workload}/rollback`, "POST");
    } else if ((action === "evaluation" || action === "activate") && validCandidate && validWorkload) {
      result = await operatorMutation(`/v1/models/${candidateId}/${action}`, "POST", { workload });
    } else if (action === "preview" && validCandidate && typeof content === "string" && content.trim() && content.length <= 20000) {
      result = await operatorMutation(`/v1/models/${candidateId}/preview`, "POST", { content });
    } else {
      return reply({ detail: "Invalid learning request." }, 400);
    }
    if (!result.body || typeof result.body !== "object") return reply({ detail: "Invalid backend response." }, 502);
    if (result.status >= 400) {
      const message = "detail" in result.body && typeof result.body.detail === "string"
        ? result.body.detail : "The local backend rejected the request.";
      return reply({ detail: message }, result.status);
    }
    if (action === "preview") return reply(result.body, result.status);
    // The studio reads status from its existing aggregate endpoint.
    const visible = Object.fromEntries(Object.entries(result.body).filter(([key]) =>
      ["id", "status", "recovered", "candidate_id"].includes(key)));
    return reply(visible, result.status);
  } catch {
    return reply({ detail: "The local learning action is unavailable." }, 502);
  }
}
