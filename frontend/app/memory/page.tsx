import { AppShell } from "@/components/app-shell";
import { DeveloperMemory } from "@/components/developer-memory";

import { getDashboardOverview } from "@/lib/italix-api";

export const dynamic = "force-dynamic";

export default async function MemoryPage() {
  const overview = await getDashboardOverview();
  return (
    <AppShell connected={overview.ok}>
      <main className="mx-auto max-w-4xl space-y-6">
        <h1 className="text-2xl font-semibold text-white">Developer Memory</h1>
        <DeveloperMemory />
      </main>
    </AppShell>
  );
}
