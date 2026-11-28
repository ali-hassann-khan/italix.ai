import { NextRequest, NextResponse } from "next/server";

import { operatorMutation } from "@/lib/italix-operator";
import { isSameOrigin, readJsonBody } from "@/lib/operator-request";
import { OPERATOR_COOKIE, verifyOperatorSession } from "@/lib/operator-session";

function json(body: unknown, status: number) {
  return NextResponse.json(body, { status, headers: { "Cache-Control": "no-store" } });
}

export async function POST(request: NextRequest) {
  if (!isSameOrigin(request)) return json({ detail: "Same-origin request required." }, 403);
  try {
    if (!verifyOperatorSession(request.cookies.get(OPERATOR_COOKIE)?.value)) {
      return json({ detail: "Operator session required." }, 401);
    }
    const parsed = await readJsonBody(request, 8192);
    if (!parsed.ok) return json({ detail: parsed.message }, parsed.status);
    const body = parsed.value;
    if (!body || typeof body !== "object" || Array.isArray(body)) {
      return json({ detail: "Invalid memory request." }, 400);
    }
    const { action, question, sourceId, retentionDays, filePath, approved } = body as Record<string, unknown>;
    let result;
    if ((action === "search" || action === "query") && typeof question === "string"
        && question.trim().length >= 3 && question.length <= 2000) {
      result = await operatorMutation(`/v1/memory/${action}`, "POST", { question: question.trim() });
    } else if ((action === "import-git" || action === "import-discussions") && approved === true
               && typeof filePath === "string" && filePath.trim() && filePath.length <= 2048) {
      const path = filePath.trim();
      result = action === "import-git"
        ? await operatorMutation("/v1/memory/import/git", "POST", { repository_path: path })
        : await operatorMutation("/v1/memory/import/discussions", "POST", { file_path: path });
      if (result.status < 400 && typeof result.body === "object" && result.body) {
        return json(Object.fromEntries(Object.entries(result.body).filter(([key]) =>
          key === "imported" || key === "duplicates")), result.status);
      }
      return json({ detail: "The approved source could not be imported." }, result.status);
    } else if (action === "redact") {
      result = await operatorMutation("/v1/memory/redact", "POST");
    } else if (action === "delete" && typeof sourceId === "string" && /^mem_[a-f0-9]{20}$/.test(sourceId)) {
      result = await operatorMutation(`/v1/memory/sources/${sourceId}`, "DELETE");
    } else if (action === "purge" && typeof retentionDays === "number"
               && Number.isInteger(retentionDays) && retentionDays >= 1 && retentionDays <= 3650) {
      result = await operatorMutation("/v1/memory/retention/purge", "POST", { retention_days: retentionDays });
    } else {
      return json({ detail: "Invalid memory request." }, 400);
    }
    const detail = typeof result.body === "object" && result.body && "detail" in result.body
      && typeof result.body.detail === "string" ? result.body.detail : "Memory request failed.";
    return json(result.status >= 400 ? { detail } : result.body, result.status);
  } catch {
    return json({ detail: "The local ITALIX backend is unavailable." }, 502);
  }
}
