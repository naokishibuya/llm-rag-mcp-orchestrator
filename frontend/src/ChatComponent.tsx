import { useState, useRef, useEffect } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

const API_BASE = 'http://localhost:8000';

type TokenUsage = {
  input_tokens: number;
  output_tokens: number;
};

type AgentResult = {
  intent: string;
  model: string;
  text: string;
  tools_used: string[];
};

type ModerationInfo = {
  verdict: string;
  reason?: string | null;
};

type CostInfo = {
  input_tokens: number;
  output_tokens: number;
  cost: number;
};

type ResponseMeta = {
  results: AgentResult[];
  moderation: ModerationInfo;
  total?: CostInfo;
};

type ThinkingStep = {
  step: string;
  detail?: string;
  tokens?: TokenUsage;
};

type Message = {
  role: 'user' | 'assistant';
  content: string;
  meta?: ResponseMeta;
  thinking?: ThinkingStep[];
  isStreaming?: boolean;
};

type UserContext = {
  city: string;
  timezone: string;
};

type ChatProps = {
  model: string;
};

function ChainMetadata({ meta }: { meta: ResponseMeta }) {
  const total = meta.total;
  const parts: string[] = [];

  // Show model(s) used
  const models = [...new Set(meta.results.map(r => r.model))];
  parts.push(`Model: ${models.join(', ')}`);

  if (total) {
    parts.push(`Tokens: ${total.input_tokens}/${total.output_tokens}`);
    const costStr = total.cost === 0
      ? '$0.00 (free)'
      : `$${total.cost.toFixed(6)}`;
    parts.push(`Cost: ${costStr}`);
  }

  // Show tools used across all results
  const tools = meta.results.flatMap(r => r.tools_used).filter(Boolean);
  if (tools.length > 0) {
    parts.push(`Tools: ${[...new Set(tools)].join(', ')}`);
  }

  return (
    <div className="mt-2 pt-2 border-t border-gray-300 text-xs text-gray-500">
      {parts.join(' | ')}
    </div>
  );
}

function ThinkingSection({ steps, isStreaming }: { steps: ThinkingStep[]; isStreaming: boolean }) {
  const [collapsed, setCollapsed] = useState(false);

  // Auto-collapse ~1s after streaming finishes
  const wasStreaming = useRef(true);
  useEffect(() => {
    if (wasStreaming.current && !isStreaming) {
      const timer = setTimeout(() => setCollapsed(true), 1000);
      return () => clearTimeout(timer);
    }
    wasStreaming.current = isStreaming;
  }, [isStreaming]);

  if (steps.length === 0) return null;

  return (
    <div className="mb-2">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700 cursor-pointer"
      >
        <span className={`transition-transform ${collapsed ? '' : 'rotate-90'}`}>&#9654;</span>
        <span>Thinking{isStreaming ? '...' : ` (${steps.length} steps)`}</span>
        {isStreaming && (
          <span className="inline-block w-2 h-2 rounded-full bg-blue-400 animate-pulse ml-1" />
        )}
      </button>
      {!collapsed && (
        <div className="mt-1 ml-3 pl-2 border-l-2 border-gray-300 text-xs text-gray-500 space-y-0.5">
          {steps.map((s, i) => (
            <div key={i}>
              <span className="text-gray-400 mr-1">{i + 1}.</span>
              {s.step}
              {s.tokens && (
                <span className="text-gray-400 ml-1">
                  tokens: [{s.tokens.input_tokens}/{s.tokens.output_tokens}]
                </span>
              )}
              {s.detail && (
                <div className="ml-4 text-gray-400 italic break-words">
                  {s.detail}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function AssistantMessage({ content, meta, thinking, isStreaming }: {
  content: string;
  meta?: ResponseMeta;
  thinking?: ThinkingStep[];
  isStreaming?: boolean;
}) {
  return (
    <>
      {thinking && thinking.length > 0 && (
        <ThinkingSection steps={thinking} isStreaming={!!isStreaming} />
      )}
      {(() => {
        if (!meta || meta.results.length <= 1) {
          return (
            <>
              {content && (
                <div className="prose prose-sm max-w-none">
                  <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{content}</Markdown>
                </div>
              )}
              {meta && <ChainMetadata meta={meta} />}
              {meta && meta.moderation.verdict !== 'allow' && meta.moderation.reason && (
                <div className="mt-1 text-xs text-orange-600">
                  Moderation: {meta.moderation.reason}
                </div>
              )}
            </>
          );
        }

        return (
          <>
            {meta.results.map((result, i) => (
              <div key={i} className={i > 0 ? 'mt-3 pt-3 border-t border-gray-300' : ''}>
                <div className="prose prose-sm max-w-none">
                  <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{result.text}</Markdown>
                </div>
              </div>
            ))}
            <ChainMetadata meta={meta} />
            {meta.moderation.verdict !== 'allow' && meta.moderation.reason && (
              <div className="mt-2 text-xs text-orange-600">
                Moderation: {meta.moderation.reason}
              </div>
            )}
          </>
        );
      })()}
    </>
  );
}

export default function ChatComponent({ model }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [editingIdx, setEditingIdx] = useState<number | null>(null);
  const [editValue, setEditValue] = useState('');
  const [userContext] = useState<UserContext>(() => {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const city = tz.split('/').pop()!.replace(/_/g, ' ');
    return { city, timezone: tz };
  });

  const chatContainerRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessages = async (updatedMessages: Message[]) => {
    setLoading(true);

    const payloadMessages = updatedMessages.map(({ role, content }) => ({ role, content }));

    // Placeholder assistant message for streaming updates
    const assistantIdx = updatedMessages.length;
    const initialAssistant: Message = {
      role: 'assistant',
      content: '',
      thinking: [],
      isStreaming: true,
    };
    setMessages([...updatedMessages, initialAssistant]);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: payloadMessages,
          model,
          user_context: {
            ...userContext,
            local_time: new Intl.DateTimeFormat('en-US', {
              timeZone: userContext.timezone,
              dateStyle: 'full',
              timeStyle: 'short',
            }).format(new Date()),
          },
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let thinking: ThinkingStep[] = [];
      let results: AgentResult[] = [];
      let moderation: ModerationInfo = { verdict: 'allow' };
      let total: CostInfo | undefined;
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        // Keep the last potentially incomplete line in the buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6).trim();
          if (payload === '[DONE]') continue;

          let event;
          try {
            event = JSON.parse(payload);
          } catch {
            continue;
          }

          if (event.type === 'thinking') {
            thinking = [...thinking, { step: event.step, detail: event.detail, tokens: event.tokens }];
          } else if (event.type === 'answer') {
            results = [...results, event.result];
          } else if (event.type === 'done') {
            moderation = event.moderation;
            total = event.total;
          } else if (event.type === 'error') {
            throw new Error(event.message);
          }

          // Update the assistant message in-place
          const content = results.map(r => r.text).join('\n\n');
          const meta: ResponseMeta | undefined = results.length > 0
            ? { results, moderation, total }
            : undefined;

          setMessages(prev => {
            const updated = [...prev];
            updated[assistantIdx] = {
              role: 'assistant',
              content,
              meta,
              thinking: [...thinking],
              isStreaming: true,
            };
            return updated;
          });
        }
      }

      // Finalize: mark streaming as done
      const finalContent = results.map(r => r.text).join('\n\n');
      setMessages(prev => {
        const updated = [...prev];
        updated[assistantIdx] = {
          role: 'assistant',
          content: finalContent,
          meta: { results, moderation, total },
          thinking: [...thinking],
          isStreaming: false,
        };
        return updated;
      });
    } catch {
      setMessages(prev => {
        const updated = [...prev];
        updated[assistantIdx] = {
          role: 'assistant',
          content: 'Error contacting backend.',
          isStreaming: false,
        };
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !model) return;

    const userMessage: Message = { role: 'user', content: input };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput('');

    await sendMessages(updatedMessages);
  };

  const startEditing = (idx: number) => {
    setEditingIdx(idx);
    setEditValue(messages[idx].content);
  };

  const submitEdit = async (idx: number) => {
    const edited = editValue.trim();
    if (!edited || loading) return;
    setEditingIdx(null);

    const userMessage: Message = { role: 'user', content: edited };
    const updatedMessages = [...messages.slice(0, idx), userMessage];
    setMessages(updatedMessages);

    await sendMessages(updatedMessages);
  };

  const clearChat = () => setMessages([]);

  return (
    <div className="bg-white p-6 rounded-lg shadow border w-full max-w-4xl flex flex-col flex-1 min-h-0 overflow-hidden">
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm text-gray-500">
          {messages.length > 0 ? `${messages.length} messages` : ''}
        </span>
        {messages.length > 0 && (
          <button
            onClick={clearChat}
            className="text-xs text-gray-400 hover:text-gray-600 px-2 py-1"
          >
            Clear
          </button>
        )}
      </div>
      <div ref={chatContainerRef} className="flex-grow overflow-y-auto min-h-0 p-4 bg-gray-50 rounded-lg border mb-4">
        {messages.length === 0 && (
          <div className="text-gray-400 text-center">
            <p>Start the conversation!</p>
            <p className="text-xs mt-2">Try: "What is RAG?" or "AAPL stock price" or both!</p>
          </div>
        )}
        {messages.map((msg, idx) => (
          <div
            key={idx}

            className={`flex mb-2 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] p-3 rounded-lg shadow-sm ${
                msg.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'
              }`}
            >
              {msg.role === 'assistant' ? (
                <AssistantMessage
                  content={msg.content}
                  meta={msg.meta}
                  thinking={msg.thinking}
                  isStreaming={msg.isStreaming}
                />
              ) : editingIdx === idx ? (
                <div className="flex flex-col gap-2 min-w-[200px]">
                  <textarea
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    className="w-full p-2 rounded border border-blue-300 text-gray-800 bg-white focus:outline-none focus:ring-2 focus:ring-blue-300 resize-none"
                    rows={3}
                    autoFocus
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        submitEdit(idx);
                      } else if (e.key === 'Escape') {
                        setEditingIdx(null);
                      }
                    }}
                  />
                  <div className="flex gap-2 justify-end">
                    <button
                      onClick={() => setEditingIdx(null)}
                      className="text-xs text-blue-200 hover:text-white px-2 py-1"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={() => submitEdit(idx)}
                      disabled={!editValue.trim() || loading}
                      className="text-xs bg-white text-blue-500 px-3 py-1 rounded hover:bg-blue-50 disabled:opacity-50"
                    >
                      Send
                    </button>
                  </div>
                </div>
              ) : (
                <div className="group relative">
                  <div className="whitespace-pre-wrap">{msg.content}</div>
                  {!loading && (
                    <button
                      onClick={() => startEditing(idx)}
                      className="hidden group-hover:flex absolute -bottom-1 -left-1 items-center justify-center w-6 h-6 rounded-full bg-blue-600 hover:bg-blue-700 text-white text-xs shadow"
                      title="Edit message"
                    >
                      ✎
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          className="flex-grow border p-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-300"
          disabled={loading}
        />
        <button
          type="submit"
          className="bg-blue-500 text-white px-4 py-2 rounded-lg disabled:opacity-50 min-w-[70px]"
          disabled={loading || !input.trim()}
        >
          {loading ? (
            <span className="flex gap-1 justify-center">
              <span className="animate-bounce">.</span>
              <span className="animate-bounce" style={{ animationDelay: '0.1s' }}>.</span>
              <span className="animate-bounce" style={{ animationDelay: '0.2s' }}>.</span>
            </span>
          ) : 'Send'}
        </button>
      </form>
    </div>
  );
}
