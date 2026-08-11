// FetchEmbedder's chat-side twin (#33): an OpenAI-compatible decider so
// `vault discards restore` can re-run the gate from the CLI, which otherwise
// wires no model. It speaks the library's own contract — `gatePrompt` in,
// `parseDecision` out — so the gate's rails hold whatever the provider says.
// Library callers keep injecting their own Decider; this is never a default.
import { gatePrompt, parseDecision, type Decider } from "./gate";
import { isLocal } from "./embed";

/** Zero-config endpoint: a local Ollama. The model is still yours to name. */
const LOCAL_CHAT = "http://localhost:11434/v1/chat/completions";

export type FetchDeciderOptions = {
  /**
   * default `$VAULT_DECIDE_MODEL`; always required. The local *embedding*
   * default can be pulled once and forgotten — a chat model that decides what
   * happens to your notes is a choice nobody else gets to make for you.
   */
  model?: string;
  /** OpenAI-compatible chat-completions URL; default `$VAULT_DECIDE_ENDPOINT`, else local Ollama */
  endpoint?: string;
  /**
   * default `$VAULT_DECIDE_API_KEY` — but only once an endpoint has been
   * chosen, same rule as FetchEmbedder: the local default does not get to
   * collect a key meant for a remote provider. Required remote; never logged.
   */
  apiKey?: string;
  timeoutMs?: number;
  /** injected in tests — the suite never touches the network */
  fetch?: typeof fetch;
};

/**
 * Build a `Decider` over an OpenAI-compatible `POST /v1/chat/completions`.
 * Configuration errors throw here, at construction — before a database is
 * opened or a candidate read. The reply is handed to `parseDecision`
 * unsoftened: a fenced or chatty answer is malformed, exactly as the library
 * treats it, because a guessed decision writes to the wrong file.
 */
export function fetchDecider(o: FetchDeciderOptions = {}): Decider {
  const setting = o.endpoint !== undefined ? "endpoint" : "VAULT_DECIDE_ENDPOINT";
  const endpoint = o.endpoint ?? process.env["VAULT_DECIDE_ENDPOINT"];
  const url = endpoint ?? LOCAL_CHAT;
  const local = endpoint === undefined || isLocal(endpoint, setting, "FetchDecider");
  const key = o.apiKey ?? (endpoint === undefined ? "" : (process.env["VAULT_DECIDE_API_KEY"] ?? ""));
  if (key === "" && !local) {
    throw new Error("FetchDecider: no API key — pass apiKey or set VAULT_DECIDE_API_KEY");
  }
  const model = o.model ?? process.env["VAULT_DECIDE_MODEL"];
  if (model === undefined || model === "") {
    throw new Error("FetchDecider: no model — pass model or set VAULT_DECIDE_MODEL");
  }
  const timeoutMs = o.timeoutMs ?? 60_000;
  const post = o.fetch ?? fetch;
  // Same discipline as FetchEmbedder: redact then truncate, so a key that
  // straddles the cut never leaks its head into an error message.
  const redact = (body: string): string =>
    (key === "" ? body : body.replaceAll(key, "[redacted]")).slice(0, 200);

  return async (input) => {
    const res = await post(url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(key === "" ? {} : { authorization: `Bearer ${key}` }),
      },
      body: JSON.stringify({ model, messages: [{ role: "user", content: gatePrompt(input) }] }),
      signal: AbortSignal.timeout(timeoutMs), // a hung provider must not hang the restore
    });
    if (!res.ok) {
      throw new Error(`decide request failed: ${res.status} ${redact(await res.text().catch(() => ""))}`);
    }
    // Parsed by hand so a 200 carrying an HTML error page is a clear message
    // rather than an unredacted parse error — FetchEmbedder's reasoning.
    const text = await res.text();
    let content: unknown;
    try {
      content = (JSON.parse(text) as { choices?: { message?: { content?: unknown } }[] })
        .choices?.[0]?.message?.content;
    } catch {
      throw new Error(`decider ${model} returned a non-JSON response: ${redact(text)}`);
    }
    if (typeof content !== "string") {
      throw new Error(`decider ${model} returned no message content: ${redact(text)}`);
    }
    return parseDecision(content.trim());
  };
}
