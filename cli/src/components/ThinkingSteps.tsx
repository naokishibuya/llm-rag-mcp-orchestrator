import React from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";
import type { ThinkingStep } from "../types.js";

type Props = {
  steps: ThinkingStep[];
  isStreaming: boolean;
};

export default function ThinkingSteps({ steps, isStreaming }: Props) {
  if (steps.length === 0 && !isStreaming) return null;

  // While streaming, show spinner with the latest step
  if (isStreaming) {
    const latest = steps[steps.length - 1];
    return (
      <Box flexDirection="column" marginBottom={1}>
        <Box>
          <Text color="cyan">
            <Spinner type="dots" />
          </Text>
          <Text color="gray">
            {" "}
            {latest ? latest.step : "Thinking..."}
          </Text>
        </Box>
      </Box>
    );
  }

  // After streaming, show step summary with details
  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text dimColor>Thinking ({steps.length} steps)</Text>
      {steps.map((s, i) => (
        <Box key={i} flexDirection="column" marginLeft={2}>
          <Text dimColor>
            {i + 1}. {s.step}
            {s.tokens ? ` [${s.tokens.input_tokens}/${s.tokens.output_tokens}]` : ""}
          </Text>
          {s.detail && (
            <Box marginLeft={3}>
              <Text dimColor italic wrap="truncate-end">
                {s.detail.length > 200 ? s.detail.slice(0, 200) + "..." : s.detail}
              </Text>
            </Box>
          )}
        </Box>
      ))}
    </Box>
  );
}
