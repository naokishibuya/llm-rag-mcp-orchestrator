import React from "react";
import { Box, Text } from "ink";
import { renderMarkdown } from "../utils/markdown.js";
import ThinkingSteps from "./ThinkingSteps.js";
import TokenSummary from "./TokenSummary.js";
import type { Message } from "../types.js";

type Props = {
  message: Message;
};

export default function MessageView({ message }: Props) {
  if (message.role === "user") {
    return (
      <Box marginBottom={1}>
        <Text bold color="blue">
          {"❯ "}
        </Text>
        <Text>{message.content}</Text>
      </Box>
    );
  }

  // Assistant message
  const rendered = message.content ? renderMarkdown(message.content) : "";

  return (
    <Box flexDirection="column" marginBottom={1}>
      {message.thinking && message.thinking.length > 0 && (
        <ThinkingSteps
          steps={message.thinking}
          isStreaming={!!message.isStreaming}
        />
      )}

      {rendered && (
        <Box flexDirection="column" paddingLeft={1} borderStyle="round" borderColor="gray">
          <Text>{rendered}</Text>
        </Box>
      )}

      {message.meta && message.meta.results.length > 0 && (
        <>
          {message.meta.results.map((r, i) =>
            r.disclaimer ? (
              <Box key={i} marginTop={0} paddingLeft={1}>
                <Text dimColor italic>
                  Disclaimer: {r.disclaimer}
                </Text>
              </Box>
            ) : null,
          )}
          <Box paddingLeft={1}>
            <TokenSummary meta={message.meta} />
          </Box>
        </>
      )}

      {message.meta &&
        message.meta.moderation.verdict !== "allow" &&
        message.meta.moderation.reason && (
          <Box paddingLeft={1}>
            <Text color="yellow">
              Moderation: {message.meta.moderation.reason}
            </Text>
          </Box>
        )}
    </Box>
  );
}
