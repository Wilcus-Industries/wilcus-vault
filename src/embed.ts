// Embedders are injected — the vault never hardcodes a provider. This slice
// ships only the deterministic one; the fetch-based API embedder lands with
// hybrid search.

export interface Embedder {
  /** identity of the vector space; a change forces a full re-embed */
  readonly model: string;
  readonly dims: number;
  embed(texts: string[]): Promise<Float32Array[]>;
}

/**
 * Scale to unit length, in place. Vectors are normalized on insert and on
 * query so cosine distance is well-defined whatever the provider returns.
 */
export function l2normalize(v: Float32Array): Float32Array {
  let sum = 0;
  for (const x of v) sum += x * x;
  const n = Math.sqrt(sum);
  if (n > 0) for (let i = 0; i < v.length; i++) v[i]! /= n;
  return v;
}

/**
 * Bag-of-tokens embedder: hashes each token into one of `dims` buckets. Purely
 * lexical — it exercises the plumbing, not semantics — and deterministic
 * across machines and Bun versions, which is what the evals need.
 */
export class TokenOverlapEmbedder implements Embedder {
  readonly model = "token-overlap-v1";
  constructor(readonly dims: number = 256) {}

  async embed(texts: string[]): Promise<Float32Array[]> {
    return texts.map((text) => {
      const v = new Float32Array(this.dims);
      for (const token of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
        v[fnv1a(token) % this.dims]! += 1;
      }
      return v;
    });
  }
}

/** FNV-1a, 32-bit — fixed by spec, so vectors don't shift under us. */
function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}
