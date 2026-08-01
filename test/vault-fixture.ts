// Fixture vaults live inside the repo (tmp-test/, gitignored) — the test
// sandbox may block writes outside it.
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type { Embedder } from "../src/embed";

/** An embedder that returns exactly what a test wants it to. */
export const stubEmbedder = (
  model: string,
  dims: number,
  embed: Embedder["embed"] = async (texts) => texts.map(() => new Float32Array(dims).fill(1)),
): Embedder => ({ model, dims, embed });

const roots: string[] = [];

export function makeVault(files: Record<string, string>): string {
  const root = join(import.meta.dir, "..", "tmp-test", `v-${Bun.randomUUIDv7()}`);
  mkdirSync(root, { recursive: true });
  roots.push(root);
  for (const [rel, body] of Object.entries(files)) writeNote(root, rel, body);
  return root;
}

export function writeNote(root: string, rel: string, body: string): void {
  const abs = join(root, rel);
  mkdirSync(dirname(abs), { recursive: true });
  writeFileSync(abs, body);
}

export function cleanupVaults(): void {
  for (const r of roots.splice(0)) rmSync(r, { recursive: true, force: true });
}
