import { test, expect } from "bun:test";
import { FetchEmbedder, TokenOverlapEmbedder, l2normalize, type FetchEmbedderOptions } from "../src/embed";
import { withEnv } from "./vault-fixture";

const KEY = "sk-test-do-not-log-me";

/** The error a call rejected with — the message itself is what we assert on. */
const failure = (p: Promise<unknown>): Promise<Error> =>
  p.then(
    () => new Error("did not throw"),
    (e: Error) => e,
  );

/** A fetch stub that records its calls and answers with `dims`-wide vectors. */
function stubFetch(dims: number, reply?: (input: string[]) => Response) {
  const calls: { url: string; init: RequestInit; body: { model: string; input: string[] } }[] = [];
  const fn = (async (url: string | URL | Request, init: RequestInit = {}) => {
    const body = JSON.parse(String(init.body)) as { model: string; input: string[] };
    calls.push({ url: String(url), init, body });
    if (reply) return reply(body.input);
    return Response.json({
      // deliberately out of order: providers do not promise response order
      data: body.input
        .map((t, i) => ({ index: i, embedding: Array.from({ length: dims }, () => t.length) }))
        .reverse(),
    });
  }) as unknown as typeof fetch;
  return { calls, fn };
}

test("FetchEmbedder posts an OpenAI-shaped request and returns vectors in input order", async () => {
  const { calls, fn } = stubFetch(4);
  const e = new FetchEmbedder({ apiKey: KEY, dims: 4, model: "m-1", endpoint: "https://x/v1/embeddings", fetch: fn });
  expect(e.model).toBe("m-1");
  expect(e.dims).toBe(4);

  const out = await e.embed(["a", "bb", "ccc"]);
  expect(out.map((v) => Array.from(v))).toEqual([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
  ]);
  expect(calls).toHaveLength(1);
  expect(calls[0]!.url).toBe("https://x/v1/embeddings");
  expect(calls[0]!.body).toEqual({ model: "m-1", input: ["a", "bb", "ccc"] });
  const headers = calls[0]!.init.headers as Record<string, string>;
  expect(headers["authorization"]).toBe(`Bearer ${KEY}`);
  expect(calls[0]!.init.signal).toBeInstanceOf(AbortSignal); // timeout, not a hang
});

test("FetchEmbedder batches by count and by request size", async () => {
  const { calls, fn } = stubFetch(2);
  const e = new FetchEmbedder({ apiKey: KEY, dims: 2, fetch: fn, batchSize: 2 });
  expect(await e.embed(["a", "b", "c", "d", "e"])).toHaveLength(5);
  expect(calls.map((c) => c.body.input)).toEqual([["a", "b"], ["c", "d"], ["e"]]);

  calls.length = 0;
  const small = new FetchEmbedder({ apiKey: KEY, dims: 2, fetch: fn, batchSize: 100, maxChars: 6 });
  // whole notes embed whole (no chunking), so a text over the budget goes alone
  expect(await small.embed(["aaaa", "bb", "c".repeat(20), "d"])).toHaveLength(4);
  expect(calls.map((c) => c.body.input)).toEqual([["aaaa", "bb"], ["c".repeat(20)], ["d"]]);
});

test("FetchEmbedder never leaks the API key in errors or when inspected", async () => {
  const { fn } = stubFetch(4, () => new Response(`bad key ${KEY}`, { status: 401 }));
  const e = new FetchEmbedder({ apiKey: KEY, dims: 4, fetch: fn });
  const err = await failure(e.embed(["x"]));
  expect(err.message).toContain("401");
  expect(err.message).not.toContain(KEY);
  expect(err.message).toContain("[redacted]");
  expect(JSON.stringify(e)).not.toContain(KEY);
  expect(Bun.inspect(e)).not.toContain(KEY);
});

test("FetchEmbedder redacts the key before truncating a long error body", async () => {
  // the key straddles the 200-character cut: truncating first would leak its head
  const { fn } = stubFetch(4, () => new Response("x".repeat(190) + KEY + "y".repeat(50), { status: 500 }));
  const e = new FetchEmbedder({ apiKey: KEY, dims: 4, fetch: fn });
  const err = await failure(e.embed(["x"]));
  expect(err.message).toContain("500");
  expect(err.message).not.toContain(KEY.slice(0, 8));
});

test("FetchEmbedder rejects vectors that are not all finite numbers", async () => {
  // JSON has no NaN literal, so a provider's NaN arrives as null
  for (const bad of [`[1, "2", 3, 4]`, `[1, null, 3, 4]`, `[1, 2, 3, 1e999]`]) {
    const e = new FetchEmbedder({
      apiKey: KEY,
      dims: 4,
      model: "m-1",
      fetch: stubFetch(4, () => new Response(`{"data":[{"index":0,"embedding":${bad}}]}`)).fn,
    });
    // a non-finite element survives l2normalize and would poison the vec0 index
    await expect(e.embed(["x"])).rejects.toThrow(/m-1.*finite numbers/s);
  }
});

test("FetchEmbedder rejects a response whose indices are not one per text", async () => {
  const e = (body: string) =>
    new FetchEmbedder({
      apiKey: KEY,
      dims: 2,
      model: "m-1",
      fetch: stubFetch(2, () => new Response(body)).fn,
    });
  const vec = `"embedding":[1,2]`;
  // duplicated index: one text would silently get another text's vector
  await expect(e(`{"data":[{"index":0,${vec}},{"index":0,${vec}}]}`).embed(["a", "b"])).rejects.toThrow(
    /m-1.*index/s,
  );
  await expect(e(`{"data":[{"index":0,${vec}},{"index":7,${vec}}]}`).embed(["a", "b"])).rejects.toThrow(
    /m-1.*index/s,
  );
  await expect(e(`{"data":[{${vec}},{${vec}}]}`).embed(["a", "b"])).rejects.toThrow(/m-1.*index/s);
});

test("FetchEmbedder reports a non-JSON body instead of throwing a parse error", async () => {
  const { fn } = stubFetch(4, () => new Response(`<html>gateway timeout ${KEY}</html>`));
  const e = new FetchEmbedder({ apiKey: KEY, dims: 4, model: "m-1", fetch: fn });
  const err = await failure(e.embed(["x"]));
  expect(err.message).toMatch(/m-1.*non-JSON/s);
  expect(err.message).not.toContain(KEY);
});

test("FetchEmbedder refuses nonsense dims at construction", () => {
  expect(() => new FetchEmbedder({ apiKey: KEY, dims: 0 })).toThrow(/dims/);
  expect(() => new FetchEmbedder({ apiKey: KEY, dims: 1.5 })).toThrow(/dims/);
  withEnv({ VAULT_EMBED_DIMS: "wide" }, () => {
    expect(() => new FetchEmbedder({ apiKey: KEY })).toThrow(/dims/);
  });
});

test("FetchEmbedder unconfigured is a small local model, never a cloud provider", async () => {
  const { calls, fn } = withEnv({}, () => stubFetch(384));
  const e = withEnv({}, () => new FetchEmbedder({ fetch: fn }));
  expect(e.model).toBe("all-minilm");
  expect(e.dims).toBe(384);

  await e.embed(["a"]);
  expect(calls[0]!.url).toBe("http://localhost:11434/v1/embeddings");
  // no key exists to send, and a local daemon does not want one
  expect((calls[0]!.init.headers as Record<string, string>)["authorization"]).toBeUndefined();
});

test("FetchEmbedder fails fast and actionably when the local endpoint is unreachable", async () => {
  // what Bun throws on a refused connection — no network needed to reproduce it
  const refused = () => {
    throw new Error("Unable to connect. Is the computer able to access the url?");
  };
  const calls: string[] = [];
  const fn = (async (url: string | URL | Request) => {
    calls.push(String(url));
    refused();
  }) as unknown as typeof fetch;

  const e = withEnv({}, () => new FetchEmbedder({ fetch: fn }));
  const started = Bun.nanoseconds();
  const err = await failure(e.embed(["a"]));
  expect(err.message).toBe(
    "no embedder configured: start Ollama (`ollama pull all-minilm`) or configure a remote provider",
  );
  // one attempt, at the local endpoint: no retry storm, and above all no
  // silent second try against somebody's cloud API
  expect(calls).toEqual(["http://localhost:11434/v1/embeddings"]);
  expect((Bun.nanoseconds() - started) / 1e6).toBeLessThan(1_000);
});

test("FetchEmbedder zero-config never sends an ambient cloud key to localhost", async () => {
  // A key in the environment was put there for somebody's cloud provider. The
  // default endpoint is whatever process holds :11434 — it does not get to
  // harvest that key just by being the default.
  const { calls, fn } = stubFetch(384);
  const keyless = withEnv({ VAULT_EMBED_API_KEY: KEY }, () => new FetchEmbedder({ fetch: fn }));
  await keyless.embed(["a"]);
  expect(calls[0]!.url).toBe("http://localhost:11434/v1/embeddings");
  expect(JSON.stringify(calls[0]!.init.headers)).not.toContain(KEY);

  // An explicit key is a deliberate choice and still travels — a local gateway
  // (LiteLLM and friends) legitimately wants one.
  const gateway = withEnv({}, () => new FetchEmbedder({ apiKey: KEY, fetch: fn }));
  await gateway.embed(["a"]);
  expect((calls[1]!.init.headers as Record<string, string>)["authorization"]).toBe(`Bearer ${KEY}`);

  // ...and so is an endpoint: configure one and the environment's key is yours.
  const configured = withEnv({ VAULT_EMBED_API_KEY: KEY }, () =>
    new FetchEmbedder({ endpoint: "http://localhost:11434/v1/embeddings", fetch: fn }),
  );
  await configured.embed(["a"]);
  expect((calls[2]!.init.headers as Record<string, string>)["authorization"]).toBe(`Bearer ${KEY}`);
});

test("FetchEmbedder makes a remote endpoint name its model and dims", () => {
  const remote = (o: Partial<FetchEmbedderOptions> = {}) =>
    () => new FetchEmbedder({ apiKey: KEY, endpoint: "https://api.openai.com/v1/embeddings", ...o });
  // silently posting note bodies as all-minilm/384 is a wrong answer, not a default
  withEnv({}, () => {
    expect(remote()).toThrow(/model and dims/);
    expect(remote({ model: "text-embedding-3-small" })).toThrow(/dims/);
    expect(remote({ dims: 1536 })).toThrow(/model/);
    expect(remote({ model: "text-embedding-3-small", dims: 1536 })).not.toThrow();
  });
  withEnv({ VAULT_EMBED_MODEL: "text-embedding-3-small", VAULT_EMBED_DIMS: "1536" }, () => {
    expect(remote()).not.toThrow();
  });
});

test("FetchEmbedder says which setting holds a malformed endpoint", () => {
  withEnv({}, () => {
    expect(() => new FetchEmbedder({ endpoint: "localhost:11434" })).toThrow(/invalid endpoint.*localhost:11434/s);
  });
  withEnv({ VAULT_EMBED_ENDPOINT: "localhost:11434" }, () => {
    expect(() => new FetchEmbedder()).toThrow(/invalid VAULT_EMBED_ENDPOINT.*localhost:11434/s);
  });
});

test("FetchEmbedder keeps the underlying failure as the cause of 'start Ollama'", async () => {
  const refused = new Error("Unable to connect. Is the computer able to access the url?");
  const fn = (async () => {
    throw refused;
  }) as unknown as typeof fetch;
  const e = withEnv({}, () => new FetchEmbedder({ fetch: fn }));
  expect((await failure(e.embed(["a"]))).cause).toBe(refused);
});

test("FetchEmbedder only says 'start Ollama' about the endpoint it chose itself", async () => {
  // a local vLLM on :8000 is somebody's explicit choice; `ollama pull` is not
  // the fix for it, so its own error survives
  const refused = new Error("Unable to connect. Is the computer able to access the url?");
  const fn = (async () => {
    throw refused;
  }) as unknown as typeof fetch;
  const e = withEnv({}, () => new FetchEmbedder({ endpoint: "http://localhost:8000/v1/embeddings", fetch: fn }));
  expect(await failure(e.embed(["a"]))).toBe(refused);
});

test("FetchEmbedder reports a hung local endpoint as a timeout, not as 'start Ollama'", async () => {
  const fn = (async () => {
    throw new DOMException("The operation timed out.", "TimeoutError");
  }) as unknown as typeof fetch;
  const e = withEnv({}, () => new FetchEmbedder({ fetch: fn, timeoutMs: 5 }));
  const err = await failure(e.embed(["a"]));
  expect(err.message).not.toContain("no embedder configured");
  expect(err.name).toBe("TimeoutError");
});

test("FetchEmbedder demands a key for a remote endpoint but not for a local one", () => {
  withEnv({}, () => {
    expect(new FetchEmbedder({ dims: 4 })).toBeInstanceOf(FetchEmbedder);
    expect(() => new FetchEmbedder({ dims: 4, endpoint: "https://api.openai.com/v1/embeddings" })).toThrow(
      /VAULT_EMBED_API_KEY/,
    );
    for (const local of ["http://127.0.0.1:11434/v1/embeddings", "http://[::1]:11434/v1/embeddings"]) {
      expect(new FetchEmbedder({ dims: 4, endpoint: local })).toBeInstanceOf(FetchEmbedder);
    }
  });
  withEnv({ VAULT_EMBED_ENDPOINT: "https://api.openai.com/v1/embeddings" }, () => {
    expect(() => new FetchEmbedder({ dims: 4 })).toThrow(/VAULT_EMBED_API_KEY/);
  });
});

test("FetchEmbedder env config selects a remote provider over the local default", async () => {
  const { calls, fn } = stubFetch(1536);
  const e = withEnv(
    {
      VAULT_EMBED_API_KEY: KEY,
      VAULT_EMBED_ENDPOINT: "https://api.openai.com/v1/embeddings",
      VAULT_EMBED_MODEL: "text-embedding-3-small",
      VAULT_EMBED_DIMS: "1536",
    },
    () => new FetchEmbedder({ fetch: fn }),
  );
  expect([e.model, e.dims]).toEqual(["text-embedding-3-small", 1536]);
  await e.embed(["a"]);
  expect(calls[0]!.url).toBe("https://api.openai.com/v1/embeddings");
  expect((calls[0]!.init.headers as Record<string, string>)["authorization"]).toBe(`Bearer ${KEY}`);
});

test("FetchEmbedder quotes a keyless provider error body verbatim", async () => {
  // redacting the empty string would splice [redacted] between every character
  const { fn } = stubFetch(384, () => new Response("model 'all-minilm' not found", { status: 404 }));
  const e = withEnv({}, () => new FetchEmbedder({ fetch: fn }));
  const err = await failure(e.embed(["x"]));
  expect(err.message).toContain("404 model 'all-minilm' not found");
});

test("FetchEmbedder rejects a response that does not match the request", async () => {
  const short = new FetchEmbedder({
    apiKey: KEY,
    dims: 4,
    model: "m-1",
    fetch: stubFetch(4, () => Response.json({ data: [] })).fn,
  });
  await expect(short.embed(["a", "b"])).rejects.toThrow(/m-1.*0 vectors for 2/s);

  const narrow = new FetchEmbedder({
    apiKey: KEY,
    dims: 4,
    model: "m-1",
    fetch: stubFetch(2).fn, // 2-wide vectors for a 4-dim embedder
  });
  await expect(narrow.embed(["a"])).rejects.toThrow(/m-1.*width 2.*expected 4/s);
});

test("l2normalize leaves a zero vector alone (it has no direction)", () => {
  expect(Array.from(l2normalize(new Float32Array([0, 0, 0])))).toEqual([0, 0, 0]);
  expect(Math.hypot(...l2normalize(new Float32Array([3, 4])))).toBeCloseTo(1, 6);
});

test("TokenOverlapEmbedder stays deterministic across instances", async () => {
  const [a] = await new TokenOverlapEmbedder(16).embed(["Acme renewal"]);
  const [b] = await new TokenOverlapEmbedder(16).embed(["acme   RENEWAL!"]);
  expect(Array.from(a!)).toEqual(Array.from(b!));
});
