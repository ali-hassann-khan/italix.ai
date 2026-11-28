import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { chromium } from "playwright-core";

const origin = process.env.ITALIX_BROWSER_ORIGIN ?? "http://127.0.0.1:3000";
const unlockSecret = process.env.ITALIX_OPERATOR_UNLOCK_SECRET ?? "";
assert.ok(unlockSecret.length >= 32, "persistent operator unlock secret is required");

const browser = await chromium.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true,
});
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1100 }, reducedMotion: "reduce" });
  const page = await context.newPage();
  await page.goto(`${origin}/learning`, { waitUntil: "networkidle" });
  await page.getByPlaceholder("Operator unlock secret").fill(unlockSecret);
  await page.getByRole("button", { name: "Unlock playground" }).click();
  await page.getByText("No LoRA adapter is production-qualified").waitFor({ timeout: 15_000 });

  const apiResponse = await context.request.get(`${origin}/api/operator/learning`);
  assert.equal(apiResponse.status(), 200);
  const apiText = await apiResponse.text();
  assert.ok(apiText.length < 50_000, "sanitized learning response is unexpectedly large");
  for (const forbidden of ["prompt_json", "response_text", "training_losses", "validation_losses", "adapter_path", "log_path", "metrics_json", "examples"]) {
    assert.equal(apiText.includes(`\"${forbidden}\"`), false, `sensitive field ${forbidden} reached the browser`);
  }
  const data = JSON.parse(apiText);
  assert.equal(Array.isArray(data.queues), true);
  assert.equal(Array.isArray(data.jobs), true);
  assert.equal(Array.isArray(data.models), true);
  assert.equal(data.summary.qualified_adapters, 0);

  await page.getByRole("button", { name: "Create training plan" }).waitFor();
  const screenshots = process.env.ITALIX_GATEWAY_SCREENSHOTS ?? join(process.cwd(), "artifacts", "screenshots");
  await mkdir(screenshots, { recursive: true });
  await page.screenshot({ path: join(screenshots, "learning-desktop.png"), fullPage: true });
  await page.setViewportSize({ width: 500, height: 1000 });
  await page.screenshot({ path: join(screenshots, "learning-mobile.png"), fullPage: true });

  const memoryFixture = process.env.ITALIX_MEMORY_BROWSER_FIXTURE;
  if (memoryFixture) {
    const crossOrigin = await context.request.post(`${origin}/api/operator/memory`, {
      headers: { Origin: "https://untrusted.invalid" }, data: { action: "purge", retentionDays: 30 },
    });
    assert.equal(crossOrigin.status(), 403);
    await page.goto(`${origin}/memory`);
    await page.getByLabel("Import source", { exact: true }).selectOption("import-discussions");
    await page.getByLabel("Local path", { exact: true }).fill(memoryFixture);
    const importButton = page.getByRole("button", { name: "Import source", exact: true });
    assert.equal(await importButton.isDisabled(), true);
    await page.getByLabel("I approve importing this source into local Memory").check();
    await importButton.click();
    await page.getByText("Imported 1 source(s); 0 already present.").waitFor();
    await page.getByLabel("Search imported engineering decisions").fill("cache rollout");
    await page.getByRole("button", { name: "Search sources" }).click();
    await page.getByText(/Cache rollout.*REDACTED/).waitFor();
    page.once("dialog", (dialog) => dialog.accept());
    await page.getByRole("button", { name: "Delete source", exact: true }).click();
    await page.getByText("Removed 1 source(s).").waitFor();
    await page.getByRole("button", { name: "Search sources" }).click();
    await page.getByText("No matching sources.").waitFor();
    console.log("Developer Memory import, redaction, search, deletion and same-origin checks passed.");
  }

  console.log(JSON.stringify({
    pending_examples: data.summary.pending_examples,
    completed_jobs: data.summary.completed_jobs,
    experimental_adapters: data.summary.experimental_adapters,
    qualified_adapters: data.summary.qualified_adapters,
    response_bytes: apiText.length,
  }));
} finally {
  await browser.close();
}
