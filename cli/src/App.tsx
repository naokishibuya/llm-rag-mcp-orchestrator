import React, { useState } from "react";
import { Box } from "ink";
import ModelSelector from "./components/ModelSelector.js";
import Chat from "./components/Chat.js";

type AppState = "selecting-model" | "chatting";

type Props = {
  initialModel?: string;
  server: string;
};

export default function App({ initialModel, server }: Props) {
  const [state, setState] = useState<AppState>(
    initialModel ? "chatting" : "selecting-model",
  );
  const [model, setModel] = useState(initialModel || "");

  const handleModelSelect = (selectedModel: string) => {
    setModel(selectedModel);
    setState("chatting");
  };

  const handleSwitchModel = () => {
    setState("selecting-model");
  };

  return (
    <Box flexDirection="column" width="100%">
      {state === "selecting-model" && (
        <ModelSelector server={server} onSelect={handleModelSelect} />
      )}
      {state === "chatting" && (
        <Chat model={model} server={server} onSwitchModel={handleSwitchModel} />
      )}
    </Box>
  );
}
