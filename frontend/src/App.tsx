import { useState, useEffect } from 'react';
import axios from 'axios';
import ChatComponent from './ChatComponent';

const API_BASE = 'http://localhost:8000';

export default function App() {
  const [models, setModels] = useState<string[]>([]);
  const [embeddings, setEmbeddings] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedEmbedding, setSelectedEmbedding] = useState<string>('');

  useEffect(() => {
    Promise.all([
      axios.get<{ models: string[] }>(`${API_BASE}/models`),
      axios.get<{ embeddings: string[] }>(`${API_BASE}/embeddings`),
    ])
      .then(([modelsRes, embeddingsRes]) => {
        setModels(modelsRes.data.models);
        setEmbeddings(embeddingsRes.data.embeddings);
        if (modelsRes.data.models.length > 0) {
          setSelectedModel(modelsRes.data.models[0]);
        }
        if (embeddingsRes.data.embeddings.length > 0) {
          setSelectedEmbedding(embeddingsRes.data.embeddings[0]);
        }
      })
      .catch(() => {
        setModels([]);
        setEmbeddings([]);
      });
  }, []);

  return (
    <div className="min-h-screen flex flex-col items-center bg-gradient-to-br from-blue-50 to-white p-6">
      <h1 className="text-4xl font-bold mb-6 text-blue-600">LLM RAG Chat Demo</h1>

      {(models.length > 0 || embeddings.length > 0) && (
        <div className="flex gap-4 mb-6">
          {models.length > 0 && (
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
          )}
          {embeddings.length > 0 && (
            <label className="flex items-center gap-2 bg-white border rounded-lg shadow px-4 py-2">
              <span className="text-sm font-semibold text-gray-600">RAG Embedding</span>
              <select
                className="border rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-300"
                value={selectedEmbedding}
                onChange={(e) => setSelectedEmbedding(e.target.value)}
              >
                {embeddings.map((emb) => (
                  <option key={emb} value={emb}>
                    {emb}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
      )}

      <ChatComponent model={selectedModel} embeddingModel={selectedEmbedding} />
    </div>
  );
}
