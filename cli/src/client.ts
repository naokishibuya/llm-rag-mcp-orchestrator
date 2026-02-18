import type { SSEEvent, UserContext } from "./types.js";

function getUserContext(): UserContext {
  const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
  const city = tz.split("/").pop()!.replace(/_/g, " ");
  const local_time = new Intl.DateTimeFormat("en-US", {
    timeZone: tz,
    dateStyle: "full",
    timeStyle: "short",
  }).format(new Date());
  return { city, timezone: tz, local_time };
}

export async function fetchModels(server: string): Promise<string[]> {
  const res = await fetch(`${server}/models`);
  if (!res.ok) {
    throw new Error(`Failed to fetch models: HTTP ${res.status}`);
  }
  const data = (await res.json()) as { models: string[] };
  return data.models;
}

export async function* streamChat(
  messages: { role: string; content: string }[],
  model: string,
  server: string,
): AsyncGenerator<SSEEvent> {
  const res = await fetch(`${server}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      model,
      user_context: getUserContext(),
    }),
  });

  if (!res.ok || !res.body) {
    throw new Error(`HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6).trim();
      if (payload === "[DONE]") continue;

      try {
        const event = JSON.parse(payload) as SSEEvent;
        yield event;
      } catch {
        // skip malformed JSON
      }
    }
  }
}
