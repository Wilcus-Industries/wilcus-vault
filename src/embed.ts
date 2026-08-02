// Embedders are injected — the vault never hardcodes a provider. Two ship: a
// deterministic one for tests and evals, and an OpenAI-compatible HTTP one
// whose unconfigured default is a small model on a local Ollama, never a cloud.

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

/** Zero-config: a small model on a local Ollama. Nothing leaves the machine. */
const LOCAL_ENDPOINT = "http://localhost:11434/v1/embeddings";
const LOCAL_MODEL = "all-minilm";
const LOCAL_DIMS = 384;
const NO_EMBEDDER =
  "no embedder configured: start Ollama (`ollama pull all-minilm`) or configure a remote provider";

/**
 * Vets a configured endpoint and answers whether it is on this machine (which
 * is what decides that no API key is required to reach it). `localhost:11434`
 * — the plausible typo — *parses*, as the scheme `localhost:` with no host at
 * all, so a hostname is checked for rather than left to `new URL` to reject.
 *
 * The message names the setting to go and fix but never quotes its value: a
 * URL may carry `user:password@`, and this error is printed, logged and pasted
 * into bug reports. The setting is where the reader can read it themselves.
 */
function isLocal(endpoint: string, setting: string): boolean {
  const url = URL.parse(endpoint);
  if (url === null || !["http:", "https:"].includes(url.protocol) || url.hostname === "") {
    throw new Error(`FetchEmbedder: invalid ${setting} — expected an http(s) URL with a host`);
  }
  return ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
}

export type FetchEmbedderOptions = {
  /** default `$VAULT_EMBED_MODEL`, else `all-minilm`; required for a remote endpoint */
  model?: string;
  /** default `$VAULT_EMBED_DIMS`, else 384 — must match what the model returns */
  dims?: number;
  /** OpenAI-compatible embeddings URL; default `$VAULT_EMBED_ENDPOINT`, else local Ollama */
  endpoint?: string;
  /**
   * default `$VAULT_EMBED_API_KEY` — but only once an endpoint has been chosen,
   * so the local default cannot collect a key meant for a remote provider.
   * Required for a remote endpoint; never persisted and never logged.
   */
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
 * OpenAI-compatible `POST /v1/embeddings`. Unconfigured it is a local Ollama
 * (`all-minilm`, 384 dims) and nothing leaves the machine; a remote provider is
 * an explicit choice — endpoint and key from the constructor or `VAULT_EMBED_*`
 * — because with one whole note bodies leave the machine on every embed, per
 * DESIGN.md § Embedding. The API key lives in a private field: it is never
 * written to the DB or frontmatter, never echoed in an error, and redacted out
 * of any provider response body we quote back.
 */
export class FetchEmbedder implements Embedder {
  readonly model: string;
  readonly dims: number;
  readonly #key: string;
  readonly #endpoint: string;
  /** nobody chose this endpoint — it is the built-in local default */
  readonly #defaulted: boolean;
  readonly #batchSize: number;
  readonly #maxChars: number;
  readonly #timeoutMs: number;
  readonly #fetch: typeof fetch;

  constructor(o: FetchEmbedderOptions = {}) {
    const setting = o.endpoint !== undefined ? "endpoint" : "VAULT_EMBED_ENDPOINT";
    const endpoint = o.endpoint ?? process.env["VAULT_EMBED_ENDPOINT"];
    this.#endpoint = endpoint ?? LOCAL_ENDPOINT;
    this.#defaulted = this.#endpoint === LOCAL_ENDPOINT;
    const local = endpoint === undefined || isLocal(endpoint, setting);
    // An ambient key belongs to whatever remote provider its owner configured.
    // The default endpoint is simply "whoever holds :11434", so it does not get
    // to collect that key by being the default — only a chosen endpoint, or an
    // explicit `apiKey` (a local gateway may want one), sends anything.
    const key = o.apiKey ?? (endpoint === undefined ? "" : (process.env["VAULT_EMBED_API_KEY"] ?? ""));
    // A daemon on this machine has no key to give; a remote provider always does.
    if (key === "" && !local) {
      throw new Error("FetchEmbedder: no API key — pass apiKey or set VAULT_EMBED_API_KEY");
    }
    this.#key = key;
    const envDims = process.env["VAULT_EMBED_DIMS"];
    const model = o.model ?? process.env["VAULT_EMBED_MODEL"];
    const dims = o.dims ?? (envDims === undefined ? undefined : Number(envDims));
    // The defaults describe the local model. Inheriting them for a remote
    // provider would post note bodies under a model name it never heard of —
    // and file the answer as if it were that vector space.
    const missing: string[] = [];
    if (model === undefined) missing.push("model");
    if (dims === undefined) missing.push("dims");
    if (!local && missing.length > 0) {
      throw new Error(
        `FetchEmbedder: a remote endpoint needs ${missing.join(" and ")} — pass ${missing.join("/")}` +
          ` or set ${missing.map((m) => `VAULT_EMBED_${m.toUpperCase()}`).join("/")}`,
      );
    }
    this.model = model ?? LOCAL_MODEL;
    this.dims = dims ?? LOCAL_DIMS;
    // Caught here rather than as "every response is the wrong width" later.
    if (!Number.isInteger(this.dims) || this.dims < 1) {
      throw new Error(`FetchEmbedder: dims must be a positive integer, got ${this.dims}`);
    }
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
    let res: Response;
    try {
      res = await this.#fetch(this.#endpoint, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          // A local daemon wants no key, and `Bearer ` alone is a malformed one.
          ...(this.#key === "" ? {} : { authorization: `Bearer ${this.#key}` }),
        },
        body: JSON.stringify({ model: this.model, input }),
        signal: AbortSignal.timeout(this.#timeoutMs), // a hung provider must not hang the indexer
      });
    } catch (e) {
      // Nothing is listening on the default endpoint: say what to do about it,
      // on the first attempt. There is no retry and above all no fall back to
      // somebody's cloud API — note bodies never leave the machine by accident.
      // Only about *this* endpoint, though: `ollama pull` is not the fix for a
      // vLLM the caller chose. A timeout is a different failure (something *is*
      // listening) and stays as it is; every other one survives as the `cause`.
      if (this.#defaulted && (e as { name?: string }).name !== "TimeoutError") {
        throw new Error(NO_EMBEDDER, { cause: e });
      }
      throw e;
    }
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
    // Replacing "" would splice [redacted] between every character of a
    // keyless (local) provider's perfectly quotable error.
    return (this.#key === "" ? body : body.replaceAll(this.#key, "[redacted]")).slice(0, 200);
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
