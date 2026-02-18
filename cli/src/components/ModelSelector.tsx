import React, { useState, useEffect } from "react";
import { Box, Text, useInput } from "ink";
import Spinner from "ink-spinner";
import { fetchModels } from "../client.js";

type Props = {
  server: string;
  onSelect: (model: string) => void;
};

export default function ModelSelector({ server, onSelect }: Props) {
  const [models, setModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [cursor, setCursor] = useState(0);

  useEffect(() => {
    fetchModels(server)
      .then((m) => {
        setModels(m);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, [server]);

  useInput((input, key) => {
    if (loading || error || models.length === 0) return;

    if (key.upArrow) {
      setCursor((c) => (c > 0 ? c - 1 : models.length - 1));
    } else if (key.downArrow) {
      setCursor((c) => (c < models.length - 1 ? c + 1 : 0));
    } else if (key.return) {
      onSelect(models[cursor]);
    } else if (input === "q") {
      process.exit(0);
    }
  });

  if (loading) {
    return (
      <Box>
        <Text color="cyan">
          <Spinner type="dots" />
        </Text>
        <Text> Loading models from {server}...</Text>
      </Box>
    );
  }

  if (error) {
    return (
      <Box flexDirection="column">
        <Text color="red">Failed to load models: {error}</Text>
        <Text dimColor>Make sure the backend is running at {server}</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column">
      <Text bold>Select a model:</Text>
      <Text dimColor>(Use arrow keys, Enter to select, q to quit)</Text>
      <Box flexDirection="column" marginTop={1}>
        {models.map((model, i) => (
          <Box key={model}>
            <Text color={i === cursor ? "cyan" : undefined} bold={i === cursor}>
              {i === cursor ? "❯ " : "  "}
              {model}
            </Text>
          </Box>
        ))}
      </Box>
    </Box>
  );
}
