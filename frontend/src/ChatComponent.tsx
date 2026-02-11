import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const API_BASE = 'http://localhost:8000';

type ReflectionInfo = {
  action: string;
  score: number | null;
  feedback: string;
  input_tokens?: number;
  output_tokens?: number;
};

type IntentMetrics = {
  input_tokens: number;
  output_tokens: number;
  cost: number;
  tools_used: string[];
  reflection: ReflectionInfo | null;
};

type IntentResult = {
  answer: string;
  intent: string;
  agent: string;
  model: string;
  metrics: IntentMetrics;
};

type ModerationInfo = {
  verdict: string;
  reason?: string | null;
};

type RouterInfo = {
  input_tokens: number;
  output_tokens: number;
  cost: number;
};

type ChatResponse = {
  results: IntentResult[];
  moderation: ModerationInfo;
  router: RouterInfo;
};

type ResponseMeta = {
  results: IntentResult[];
  moderation: ModerationInfo;
  router: RouterInfo;
};

type Message = {
  role: 'user' | 'assistant';
  content: string;
  meta?: ResponseMeta;
};

type ChatProps = {
  model: string;
};

function IntentMetadata({ result }: { result: IntentResult }) {
  const parts: string[] = [];
  parts.push(`Intent: ${result.intent}`);
  parts.push(`Model: ${result.model}`);
  parts.push(`Tokens: ${result.metrics.input_tokens}/${result.metrics.output_tokens}`);

  const costStr = result.metrics.cost === 0
    ? '$0.00 (free)'
    : `$${result.metrics.cost.toFixed(6)}`;
  parts.push(`Cost: ${costStr}`);

  if (result.metrics.tools_used && result.metrics.tools_used.length > 0) {
    parts.push(`Tools: ${result.metrics.tools_used.join(', ')}`);
  }

  if (result.metrics.reflection) {
    const r = result.metrics.reflection;
    const score = r.score != null ? ` (${r.score.toFixed(1)})` : '';
    parts.push(`Reflected: ${r.action}${score}`);
  }

  return (
    <div className="mt-2 pt-2 border-t border-gray-300 text-xs text-gray-500">
      {parts.join(' | ')}
      {result.metrics.reflection && (
        <details className="mt-1">
          <summary className="cursor-pointer hover:text-gray-700">Reflection details</summary>
          <div className="mt-1 pl-2 border-l-2 border-gray-300">
            <div>Action: {result.metrics.reflection.action}</div>
            {result.metrics.reflection.score != null && (
              <div>Score: {result.metrics.reflection.score.toFixed(2)}</div>
            )}
            {result.metrics.reflection.feedback && (
              <div>Feedback: {result.metrics.reflection.feedback}</div>
            )}
            {(result.metrics.reflection.input_tokens || result.metrics.reflection.output_tokens) ? (
              <div>Reflection tokens: {result.metrics.reflection.input_tokens ?? 0}/{result.metrics.reflection.output_tokens ?? 0}</div>
            ) : null}
          </div>
        </details>
      )}
    </div>
  );
}

function AssistantMessage({ content, meta }: { content: string; meta?: ResponseMeta }) {
  if (!meta || meta.results.length <= 1) {
    // Single intent: one markdown block + one metadata line (same as before)
    return (
      <>
        <div className="prose prose-sm max-w-none">
          <Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown>
        </div>
        {meta && meta.results.length === 1 && (
          <IntentMetadata result={meta.results[0]} />
        )}
        {meta && meta.moderation.verdict !== 'allow' && meta.moderation.reason && (
          <div className="mt-1 text-xs text-orange-600">
            Moderation: {meta.moderation.reason}
          </div>
        )}
      </>
    );
  }

  // Multi-intent: each sub-result separated with its own metadata
  return (
    <>
      {meta.results.map((result, i) => (
        <div key={i} className={i > 0 ? 'mt-3 pt-3 border-t border-gray-300' : ''}>
          <div className="prose prose-sm max-w-none">
            <Markdown remarkPlugins={[remarkGfm]}>{result.answer}</Markdown>
          </div>
          <IntentMetadata result={result} />
        </div>
      ))}
      {meta.moderation.verdict !== 'allow' && meta.moderation.reason && (
        <div className="mt-2 text-xs text-orange-600">
          Moderation: {meta.moderation.reason}
        </div>
      )}
    </>
  );
}

export default function ChatComponent({ model }: ChatProps) {
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
        use_reflection: useReflection
      });
      const data = res.data;
      const content = data.results.map(r => r.answer).join('\n\n');
      const meta: ResponseMeta = {
        results: data.results,
        moderation: data.moderation,
        router: data.router,
      };
      setMessages([...updatedMessages, { role: 'assistant', content, meta }]);
    } catch {
      setMessages([...updatedMessages, { role: 'assistant', content: 'Error contacting backend.' }]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => setMessages([]);

  return (
    <div className="bg-white p-6 rounded-lg shadow border w-full max-w-4xl flex flex-col flex-1 min-h-0 max-h-[calc(100vh-10rem)]">
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
                <AssistantMessage content={msg.content} meta={msg.meta} />
              ) : (
                <div className="whitespace-pre-wrap">{msg.content}</div>
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
