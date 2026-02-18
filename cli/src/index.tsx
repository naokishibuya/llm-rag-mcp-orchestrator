#!/usr/bin/env node
import React from "react";
import { render } from "ink";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import App from "./App.js";
import { fetchModels } from "./client.js";

const DEFAULT_SERVER = "http://localhost:8000";

const argv = await yargs(hideBin(process.argv))
  .scriptName("chat-cli")
  .usage("$0 [command]")
  .option("model", {
    alias: "m",
    type: "string",
    description: "Model to use (skip model selection)",
  })
  .option("server", {
    alias: "s",
    type: "string",
    default: DEFAULT_SERVER,
    description: "Backend server URL",
  })
  .command("models", "List available models", {}, async (args) => {
    try {
      const models = await fetchModels(args.server as string);
      console.log("Available models:");
      for (const m of models) {
        console.log(`  ${m}`);
      }
    } catch (err) {
      console.error(
        `Failed to fetch models: ${err instanceof Error ? err.message : err}`,
      );
      process.exit(1);
    }
    process.exit(0);
  })
  .help()
  .parse();

render(<App initialModel={argv.model} server={argv.server} />);
