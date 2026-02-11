import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const API_BASE = 'http://localhost:8000';

type ReflectionInfo = {
  action: string;
  score: number;
  feedback: string;
};

type Metrics = {
  input_tokens?: number;
  output_tokens?: number;
  cost?: number;
  tools_used?: string[];
  model?: string;
  latency_ms?: number;
  error?: string;
  reflection?: ReflectionInfo;
};

type ModerationInfo = {
  verdict: string;
  reason?: string | null;
};

type ChatResponse = {
  answer: string;
  intent: string;
  model: string;
  embedding_model: string;
  routing_rationale?: string | null;
  moderation: ModerationInfo;
  metrics: Metrics;
};

type ResponseMeta = Omit<ChatResponse, 'answer'>;

type Message = {
  role: 'user' | 'assistant';
  content: string;
  meta?: ResponseMeta;
};

type ChatProps = {
  model: string;
  embeddingModel: string;
};

function Metadata({ meta }: { meta: ResponseMeta }) {
  const parts: string[] = [];

  // Show intent (now inferred from tools used)
  parts.push(`Intent: ${meta.intent}`);
  parts.push(`Permission: ${meta.moderation.verdict}`);

  // Show tools used
  if (meta.metrics.tools_used && meta.metrics.tools_used.length > 0) {
    parts.push(`Tools: ${meta.metrics.tools_used.join(', ')}`);
  }

  // Show model
  if (meta.metrics.model) {
    parts.push(`Model: ${meta.metrics.model}`);
  }

  if (meta.metrics.input_tokens !== undefined) {
    parts.push(`Tokens: ${meta.metrics.input_tokens}/${meta.metrics.output_tokens}`);
  }

  if (meta.metrics.cost !== undefined) {
    const costStr = meta.metrics.cost === 0
      ? '$0.00 (free)'
      : `$${meta.metrics.cost.toFixed(6)}`;
    parts.push(`Cost: ${costStr}`);
  }

  if (meta.metrics.latency_ms !== undefined) {
    parts.push(`Latency: ${meta.metrics.latency_ms.toFixed(0)}ms`);
  }

  if (meta.metrics.reflection) {
    const r = meta.metrics.reflection;
    const score = r.score != null ? ` (${r.score.toFixed(1)})` : '';
    parts.push(`Reflected: ${r.action}${score}`);
  }

  return (
    <div className="mt-2 pt-2 border-t border-gray-300 text-xs text-gray-500">
      {parts.join(' | ')}
      {meta.moderation.verdict !== 'allow' && meta.moderation.reason && (
        <div className="mt-1 text-orange-600">
          Reason: {meta.moderation.reason}
        </div>
      )}
      {meta.metrics.error && (
        <div className="mt-1 text-orange-600">
          Error: {meta.metrics.error}
        </div>
      )}
    </div>
  );
}

export default function ChatComponent({ model, embeddingModel }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [useReflection, setUseReflection] = useState(true);

  const lastMsgRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (lastMsgRef.current) {
      lastMsgRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !model) return;

    const userMessage: Message = { role: 'user', content: input };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput('');
    setLoading(true);

    const payloadMessages = updatedMessages.map(({ role, content }) => ({ role, content }));

    try {
      const res = await axios.post<ChatResponse>(`${API_BASE}/chat`, {
        messages: payloadMessages,
        model,
        embedding_model: embeddingModel,
        use_reflection: useReflection
      });
      const { answer, ...meta } = res.data;
      setMessages([...updatedMessages, { role: 'assistant', content: answer, meta }]);
    } catch {
      setMessages([...updatedMessages, { role: 'assistant', content: 'Error contacting backend.' }]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => setMessages([]);

  return (
    <div className="bg-white p-6 rounded-lg shadow border w-full max-w-3xl flex flex-col flex-1 min-h-0 max-h-[calc(100vh-10rem)]">
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-500">
            {messages.length > 0 ? `${messages.length} messages` : ''}
          </span>
          <label className="flex items-center gap-1 text-xs text-gray-500 cursor-pointer">
            <input
              type="checkbox"
              checked={useReflection}
              onChange={(e) => setUseReflection(e.target.checked)}
              className="w-3 h-3"
            />
            <span title="Enable self-reflection (agent critiques and may revise its answer)">
              Reflect
            </span>
          </label>
        </div>
        {messages.length > 0 && (
          <button
            onClick={clearChat}
            className="text-xs text-gray-400 hover:text-gray-600 px-2 py-1"
          >
            Clear
          </button>
        )}
      </div>
      <div className="flex-grow overflow-y-auto min-h-0 p-4 bg-gray-50 rounded-lg border mb-4">
        {messages.length === 0 && (
          <div className="text-gray-400 text-center">
            <p>Start the conversation!</p>
            <p className="text-xs mt-2">Try: "What is RAG?" or "AAPL stock price" or both!</p>
          </div>
        )}
        {messages.map((msg, idx) => (
          <div
            key={idx}
            ref={idx === messages.length - 1 ? lastMsgRef : undefined}
            className={`flex mb-2 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] p-3 rounded-lg shadow-sm ${
                msg.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'
              }`}
            >
              {msg.role === 'assistant' ? (
                <div className="prose prose-sm max-w-none">
                  <Markdown remarkPlugins={[remarkGfm]}>{msg.content}</Markdown>
                </div>
              ) : (
                <div className="whitespace-pre-wrap">{msg.content}</div>
              )}
              {msg.role === 'assistant' && msg.meta && (
                <Metadata meta={msg.meta} />
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
