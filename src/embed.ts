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
    // Caught here rather than as "every response is the wrong width" later.
    if (!Number.isInteger(this.dims) || this.dims < 1) {
      throw new Error(`FetchEmbedder: dims must be a positive integer, got ${this.dims}`);
    }
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
    // Parsed by hand so a 200 carrying an HTML error page is a clear message
    // rather than an unredacted parse error.
    const text = await res.text();
    let data: { embedding?: unknown; index?: unknown }[] | undefined;
    try {
      ({ data } = JSON.parse(text) as { data?: { embedding?: unknown; index?: unknown }[] });
    } catch {
      throw new Error(`embedder ${this.model} returned a non-JSON response: ${this.#redact(text)}`);
    }
    if (!Array.isArray(data) || data.length !== input.length) {
      throw new Error(
        `embedder ${this.model} returned ${data?.length ?? 0} vectors for ${input.length} texts`,
      );
    }
    // The response order is not promised; `index` is what maps a vector back to
    // its text, and a mix-up here would mis-file every note's embedding. Filing
    // by index rather than sorting also proves the indices are one per text:
    // each slot is claimed exactly once, or we refuse the whole response.
    const out = new Array<Float32Array>(input.length);
    for (const d of data) {
      const i = d.index;
      if (typeof i !== "number" || !Number.isInteger(i) || i < 0 || i >= out.length || out[i] !== undefined) {
        throw new Error(
          `embedder ${this.model} returned an unusable vector index ${String(i)} for ${input.length} texts`,
        );
      }
      out[i] = this.#vector(d.embedding);
    }
    return out;
  }

  /** A vector is only usable if every element is a finite number. */
  #vector(embedding: unknown): Float32Array {
    if (!Array.isArray(embedding) || embedding.length !== this.dims) {
      throw new Error(
        `embedder ${this.model} returned a vector of width ${Array.isArray(embedding) ? embedding.length : 0}, expected ${this.dims}`,
      );
    }
    // A string or a null (a provider's NaN, once JSON has been through it)
    // survives l2normalize as NaN and poisons the vec0 index until the next
    // full re-embed, so it never gets that far.
    if (!embedding.every((x) => typeof x === "number" && Number.isFinite(x))) {
      throw new Error(`embedder ${this.model} returned a vector that is not all finite numbers`);
    }
    return Float32Array.from(embedding as number[]);
  }

  /**
   * Providers echo request details into error bodies; the key never survives.
   * Redact *then* truncate — the other order leaks the head of a key that
   * happens to straddle the cut.
   */
  #redact(body: string): string {
    return body.replaceAll(this.#key, "[redacted]").slice(0, 200);
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
