import React from "react";
import { Text } from "ink";
import type { ResponseMeta } from "../types.js";

type Props = {
  meta: ResponseMeta;
};

export default function TokenSummary({ meta }: Props) {
  const parts: string[] = [];

  const models = [...new Set(meta.results.map((r) => r.model))];
  parts.push(`Model: ${models.join(", ")}`);

  if (meta.total) {
    parts.push(`Tokens: ${meta.total.input_tokens}/${meta.total.output_tokens}`);
    const costStr =
      meta.total.cost === 0
        ? "$0.00 (free)"
        : `$${meta.total.cost.toFixed(6)}`;
    parts.push(`Cost: ${costStr}`);
  }

  const tools = meta.results.flatMap((r) => r.tools_used).filter(Boolean);
  if (tools.length > 0) {
    parts.push(`Tools: ${[...new Set(tools)].join(", ")}`);
  }

  return <Text dimColor>{parts.join(" | ")}</Text>;
}
