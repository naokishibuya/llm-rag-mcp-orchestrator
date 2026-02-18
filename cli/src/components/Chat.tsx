import React, { useState, useCallback } from "react";
import { Box, Text, useApp, useInput } from "ink";
import TextInput from "ink-text-input";
import { streamChat } from "../client.js";
import MessageView from "./MessageView.js";
import type {
  Message,
  AgentResult,
  ModerationInfo,
  CostInfo,
  ThinkingStep,
} from "../types.js";

type Props = {
  model: string;
  server: string;
  onSwitchModel: () => void;
};

export default function Chat({ model, server, onSwitchModel }: Props) {
  const { exit } = useApp();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // Handle Ctrl+C to exit
  useInput((_input, key) => {
    if (key.ctrl && _input === "c") {
      exit();
    }
  });

  const handleSubmit = useCallback(
    async (value: string) => {
      const trimmed = value.trim();
      if (!trimmed || loading) return;

      // Handle slash commands
      if (trimmed === "/quit" || trimmed === "/exit") {
        exit();
        return;
      }
      if (trimmed === "/clear") {
        setMessages([]);
        setInput("");
        return;
      }
      if (trimmed === "/model") {
        onSwitchModel();
        return;
      }
      if (trimmed === "/help") {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              "**Commands:**\n- `/model` — Switch model\n- `/clear` — Clear history\n- `/quit` — Exit\n- `/help` — Show this help",
          },
        ]);
        setInput("");
        return;
      }

      setInput("");
      const userMessage: Message = { role: "user", content: trimmed };
      const updatedMessages = [...messages, userMessage];
      setMessages(updatedMessages);
      setLoading(true);

      // Stream the response
      let thinking: ThinkingStep[] = [];
      let results: AgentResult[] = [];
      let moderation: ModerationInfo = { verdict: "allow" };
      let total: CostInfo | undefined;

      const assistantIdx = updatedMessages.length;
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "", thinking: [], isStreaming: true },
      ]);

      try {
        const payloadMessages = updatedMessages.map(({ role, content }) => ({
          role,
          content,
        }));

        for await (const event of streamChat(payloadMessages, model, server)) {
          if (event.type === "thinking") {
            thinking = [
              ...thinking,
              { step: event.step, detail: event.detail, tokens: event.tokens },
            ];
          } else if (event.type === "answer") {
            results = [...results, event.result];
          } else if (event.type === "done") {
            moderation = event.moderation;
            total = event.total;
          } else if (event.type === "error") {
            throw new Error(event.message);
          }

          const content = results.map((r) => r.text).join("\n\n");
          const meta =
            results.length > 0 ? { results, moderation, total } : undefined;

          setMessages((prev) => {
            const updated = [...prev];
            updated[assistantIdx] = {
              role: "assistant",
              content,
              meta,
              thinking: [...thinking],
              isStreaming: true,
            };
            return updated;
          });
        }

        // Finalize
        const finalContent = results.map((r) => r.text).join("\n\n");
        setMessages((prev) => {
          const updated = [...prev];
          updated[assistantIdx] = {
            role: "assistant",
            content: finalContent,
            meta: { results, moderation, total },
            thinking: [...thinking],
            isStreaming: false,
          };
          return updated;
        });
      } catch (err) {
        setMessages((prev) => {
          const updated = [...prev];
          updated[assistantIdx] = {
            role: "assistant",
            content: `Error: ${err instanceof Error ? err.message : "Unknown error"}`,
            isStreaming: false,
          };
          return updated;
        });
      } finally {
        setLoading(false);
      }
    },
    [messages, model, server, loading, exit, onSwitchModel],
  );

  return (
    <Box flexDirection="column" width="100%">
      <Box marginBottom={1}>
        <Text bold color="green">
          Chat
        </Text>
        <Text dimColor> — {model} — /help for commands</Text>
      </Box>

      {messages.map((msg, i) => (
        <MessageView key={i} message={msg} />
      ))}

      <Box>
        <Text bold color="blue">
          {"❯ "}
        </Text>
        <TextInput
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          placeholder={loading ? "Waiting for response..." : "Type a message..."}
        />
      </Box>
    </Box>
  );
}
