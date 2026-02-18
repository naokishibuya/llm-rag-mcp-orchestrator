export type TokenUsage = {
  input_tokens: number;
  output_tokens: number;
};

export type AgentResult = {
  intent: string;
  model: string;
  text: string;
  tools_used: string[];
  disclaimer?: string;
};

export type ModerationInfo = {
  verdict: string;
  reason?: string | null;
};

export type CostInfo = {
  input_tokens: number;
  output_tokens: number;
  cost: number;
};

export type ResponseMeta = {
  results: AgentResult[];
  moderation: ModerationInfo;
  total?: CostInfo;
};

export type ThinkingStep = {
  step: string;
  detail?: string;
  tokens?: TokenUsage;
};

export type Message = {
  role: "user" | "assistant";
  content: string;
  meta?: ResponseMeta;
  thinking?: ThinkingStep[];
  isStreaming?: boolean;
};

export type UserContext = {
  city: string;
  timezone: string;
  local_time: string;
};

// SSE event types from the backend
export type SSEThinkingEvent = {
  type: "thinking";
  step: string;
  detail?: string;
  tokens?: TokenUsage;
};

export type SSEAnswerEvent = {
  type: "answer";
  result: AgentResult;
};

export type SSEDoneEvent = {
  type: "done";
  moderation: ModerationInfo;
  total?: CostInfo;
};

export type SSEErrorEvent = {
  type: "error";
  message: string;
};

export type SSEEvent =
  | SSEThinkingEvent
  | SSEAnswerEvent
  | SSEDoneEvent
  | SSEErrorEvent;
