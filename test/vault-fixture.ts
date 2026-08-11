// Fixture vaults live inside the repo (tmp-test/, gitignored) — the test
// sandbox may block writes outside it.
import type { Database } from "bun:sqlite";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type { Embedder } from "../src/embed";

/** An embedder that returns exactly what a test wants it to. */
export const stubEmbedder = (
  model: string,
  dims: number,
  embed: Embedder["embed"] = async (texts) => texts.map(() => new Float32Array(dims).fill(1)),
): Embedder => ({ model, dims, embed });

/** set while a `withEnv` owns the process environment */
let held = false;

/**
 * Run `fn` with exactly the `VAULT_EMBED_*` / `VAULT_DECIDE_*` environment it asks for — the rest
 * cleared, so a developer's own key or endpoint cannot decide a test — then put
 * the real environment back. `fn` may be async: an async one reads the
 * environment *after* it returns its promise, so the restore waits for it.
 */
export function withEnv<T>(env: Record<string, string>, fn: () => T): T {
  // One process-wide environment, so two of these at once would restore each
  // other's state and leave a test reading someone else's endpoint. Refused
  // loudly rather than debugged later as flakiness.
  if (held) throw new Error("withEnv is not re-entrant: await the outer one first");
  const names = [
    "VAULT_EMBED_API_KEY",
    "VAULT_EMBED_ENDPOINT",
    "VAULT_EMBED_MODEL",
    "VAULT_EMBED_DIMS",
    "VAULT_DECIDE_API_KEY",
    "VAULT_DECIDE_ENDPOINT",
    "VAULT_DECIDE_MODEL",
  ];
  const prev = names.map((n) => [n, process.env[n]] as const);
  const restore = () => {
    for (const [n, v] of prev) {
      if (v === undefined) delete process.env[n];
      else process.env[n] = v;
    }
  };
  for (const n of names) delete process.env[n];
  for (const [n, v] of Object.entries(env)) process.env[n] = v;
  held = true;
  const done = () => {
    held = false;
    restore();
  };
  let out: T;
  try {
    out = fn();
  } catch (e) {
    done();
    throw e;
  }
  if (out instanceof Promise) return out.finally(done) as T;
  done();
  return out;
}

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

/** A note's stored vector, as floats — the blob comes back as bytes. */
export function vecOf(db: Database, id: number): Float32Array {
  const { emb } = db.query("select emb from vectors where note_id = ?").get(id) as {
    emb: Uint8Array;
  };
  return new Float32Array(emb.buffer.slice(emb.byteOffset, emb.byteOffset + emb.byteLength));
}

export function cleanupVaults(): void {
  for (const r of roots.splice(0)) rmSync(r, { recursive: true, force: true });
}
