// Embedders are injected — the vault never hardcodes a provider. Two ship:
// a deterministic one for tests and evals, and an OpenAI-compatible HTTP one.

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

export type FetchEmbedderOptions = {
  /** default `$VAULT_EMBED_MODEL` */
  model?: string;
  /** default `$VAULT_EMBED_DIMS` — must match what the model returns */
  dims?: number;
  /** OpenAI-compatible embeddings URL; default `$VAULT_EMBED_ENDPOINT` */
  endpoint?: string;
  /** default `$VAULT_EMBED_API_KEY`; never persisted and never logged */
  apiKey?: string;
  /** texts per request */
  batchSize?: number;
  /** characters per request; a single longer text goes in a request of its own */
  maxChars?: number;
  timeoutMs?: number;
  /** injected in tests — the suite never touches the network */
  fetch?: typeof fetch;
};

/**
 * OpenAI-compatible `POST /v1/embeddings`. Whole note bodies leave the machine
 * on every embed — that is the caller's provider choice to make, per DESIGN.md
 * § Embedding. The API key lives in a private field: it is never written to the
 * DB or frontmatter, never echoed in an error, and redacted out of any provider
 * response body we quote back.
 */
export class FetchEmbedder implements Embedder {
  readonly model: string;
  readonly dims: number;
  readonly #key: string;
  readonly #endpoint: string;
  readonly #batchSize: number;
  readonly #maxChars: number;
  readonly #timeoutMs: number;
  readonly #fetch: typeof fetch;

  constructor(o: FetchEmbedderOptions = {}) {
    const key = o.apiKey ?? process.env["VAULT_EMBED_API_KEY"] ?? "";
    if (key === "") {
      throw new Error("FetchEmbedder: no API key — pass apiKey or set VAULT_EMBED_API_KEY");
    }
    this.#key = key;
    this.model = o.model ?? process.env["VAULT_EMBED_MODEL"] ?? "text-embedding-3-small";
    this.dims = o.dims ?? Number(process.env["VAULT_EMBED_DIMS"] ?? 1536);
    this.#endpoint =
      o.endpoint ?? process.env["VAULT_EMBED_ENDPOINT"] ?? "https://api.openai.com/v1/embeddings";
    // ponytail: batch by count and by characters — characters approximate the
    // token budget a provider actually enforces. Count real tokens if a
    // provider starts rejecting batches.
    this.#batchSize = o.batchSize ?? 64;
    this.#maxChars = o.maxChars ?? 96_000;
    this.#timeoutMs = o.timeoutMs ?? 30_000;
    this.#fetch = o.fetch ?? fetch;
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    const out: Float32Array[] = [];
    let batch: string[] = [];
    let chars = 0;
    for (const text of texts) {
      if (batch.length > 0 && (batch.length >= this.#batchSize || chars + text.length > this.#maxChars)) {
        out.push(...(await this.#post(batch)));
        batch = [];
        chars = 0;
      }
      batch.push(text);
      chars += text.length;
    }
    if (batch.length > 0) out.push(...(await this.#post(batch)));
    return out;
  }

  async #post(input: string[]): Promise<Float32Array[]> {
    const res = await this.#fetch(this.#endpoint, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${this.#key}` },
      body: JSON.stringify({ model: this.model, input }),
      signal: AbortSignal.timeout(this.#timeoutMs), // a hung provider must not hang the indexer
    });
    if (!res.ok) {
      throw new Error(
        `embedding request failed: ${res.status} ${this.#redact(await res.text().catch(() => ""))}`,
      );
    }
    const { data } = (await res.json()) as { data?: { embedding?: number[]; index?: number }[] };
    if (!Array.isArray(data) || data.length !== input.length) {
      throw new Error(
        `embedder ${this.model} returned ${data?.length ?? 0} vectors for ${input.length} texts`,
      );
    }
    // The response order is not promised; `index` is what maps a vector back
    // to its text, and a mix-up here would mis-file every note's embedding.
    return [...data]
      .sort((a, b) => (a.index ?? 0) - (b.index ?? 0))
      .map((d) => {
        if (!Array.isArray(d.embedding) || d.embedding.length !== this.dims) {
          throw new Error(
            `embedder ${this.model} returned a vector of width ${d.embedding?.length ?? 0}, expected ${this.dims}`,
          );
        }
        return Float32Array.from(d.embedding);
      });
  }

  /** Providers echo request details into error bodies; the key never survives. */
  #redact(body: string): string {
    return body.slice(0, 200).replaceAll(this.#key, "[redacted]");
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
