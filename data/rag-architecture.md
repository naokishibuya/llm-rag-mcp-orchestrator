# RAG Architecture Overview

Retrieval-Augmented Generation (RAG) combines information retrieval with text generation to produce accurate, grounded responses. The pipeline works in three stages:

## 1. Indexing

Documents are split into chunks, converted to vector embeddings, and stored in a vector database. This demo uses numpy-based cosine similarity search.

## 2. Retrieval

When a user asks a question, the query is embedded using the same model. The system finds the most similar document chunks using vector similarity search (top-k retrieval).

## 3. Generation

Retrieved chunks are injected into the LLM prompt as context. The model generates an answer grounded in this context, reducing hallucination.

## Key Benefits

- Answers are grounded in actual documents
- Knowledge can be updated without retraining the model
- Sources can be cited for transparency
- Works with any LLM (local or cloud)

## Supported LLM Providers

This demo supports two LLM providers:

- **Ollama**: Local models like Qwen, Mistral, LLaMA
- **Google Gemini**: Cloud-based with free tier available
