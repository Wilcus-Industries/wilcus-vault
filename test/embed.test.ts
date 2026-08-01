import { test, expect } from "bun:test";
import { FetchEmbedder, TokenOverlapEmbedder, l2normalize } from "../src/embed";

const KEY = "sk-test-do-not-log-me";

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
  const err = await e.embed(["x"]).then(
    () => null,
    (x: Error) => x,
  );
  expect(err?.message).toContain("401");
  expect(err?.message).not.toContain(KEY);
  expect(err?.message).toContain("[redacted]");
  expect(JSON.stringify(e)).not.toContain(KEY);
  expect(Bun.inspect(e)).not.toContain(KEY);
});

test("FetchEmbedder reads the key from the environment and refuses to run without one", () => {
  const prev = process.env["VAULT_EMBED_API_KEY"];
  try {
    process.env["VAULT_EMBED_API_KEY"] = "env-key";
    expect(new FetchEmbedder({ dims: 4 })).toBeInstanceOf(FetchEmbedder);
    delete process.env["VAULT_EMBED_API_KEY"];
    expect(() => new FetchEmbedder({ dims: 4 })).toThrow(/VAULT_EMBED_API_KEY/);
  } finally {
    if (prev === undefined) delete process.env["VAULT_EMBED_API_KEY"];
    else process.env["VAULT_EMBED_API_KEY"] = prev;
  }
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
