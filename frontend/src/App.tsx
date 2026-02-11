import { useState, useEffect } from 'react';
import axios from 'axios';
import ChatComponent from './ChatComponent';

const API_BASE = 'http://localhost:8000';

export default function App() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');

  useEffect(() => {
    axios.get<{ models: string[] }>(`${API_BASE}/models`)
      .then((res) => {
        setModels(res.data.models);
        if (res.data.models.length > 0) {
          setSelectedModel(res.data.models[0]);
        }
      })
      .catch(() => {
        setModels([]);
      });
  }, []);

  return (
    <div className="h-screen flex flex-col items-center bg-gradient-to-br from-blue-50 to-white p-4 overflow-hidden">
      <h1 className="text-3xl font-bold mb-4 text-blue-600 shrink-0">LLM RAG MCP Orchestrator Chat Demo</h1>

      {models.length > 0 && (
        <div className="flex gap-4 mb-4 shrink-0">
          <label className="flex items-center gap-2 bg-white border rounded-lg shadow px-4 py-2">
            <span className="text-sm font-semibold text-gray-600">Chat Model</span>
            <select
              className="border rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-300"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {models.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}

      <ChatComponent model={selectedModel} />
    </div>
  );
}
