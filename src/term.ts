// Nothing this vault prints is text it wrote. A path, a title, a link target
// come from a note a human or an LLM authored; an error message now quotes up
// to 200 bytes of whatever answered the embedding endpoint. A bare `\r` or an
// ESC sequence in any of them rewrites the line the terminal has already drawn
// — so it goes through here first, on stdout and stderr alike.

/** Untrusted text, safe to echo: control characters print as `?`. */
export const safe = (text: string): string => text.replace(/\p{Cc}/gu, "?");

/**
 * An untrusted error as the message it was written as — never a stack trace,
 * never a redrawn terminal. Scrubbed line by line, so the newlines an error
 * deliberately carries (a usage message under a bad flag) still break lines.
 */
export const printable = (e: unknown): string =>
  (e instanceof Error ? e.message : String(e)).split("\n").map(safe).join("\n");
