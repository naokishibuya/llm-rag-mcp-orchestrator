# LLM RAG Chat Demo with MCP

## Overview

This project is a **full-stack Retrieval-Augmented Generation (RAG) chat demo** built to showcase how large language models can answer questions grounded in external knowledge.

**Tech Stack:**

* **Frontend:** React (TypeScript)
* **Backend:** FastAPI (Python) with modular architecture
* **LLM Providers:** Multi-model support - Ollama (local) and Google Gemini (cloud) simultaneously
* **Embeddings:** Ollama or Vertex AI embeddings
* **Vector Database:** Custom numpy-based vector index
* **Frameworks:** FastMCP for tool calling
* **Metrics:** Per-request token usage and cost tracking

**How it works:**

The application ingests documents, converts them into vector embeddings, and stores them in a vector index. When a user asks a question, the system retrieves the most relevant text chunks and passes them — along with the query — to the selected model to generate accurate, context-aware responses.

![](images/chat-mode.png)

This demo illustrates the core workflow behind modern RAG systems: **document retrieval + LLM reasoning**, wrapped in a simple web interface.

### Inline Metrics Display

Each response shows inline metrics including:
- **Intent**: Classified intent (qa, small_talk, finance_quote, etc.)
- **Permission**: Moderation verdict (allow, warn, block)
- **Tool**: MCP tool used (if any)
- **Tokens**: Input/output token counts
- **Cost**: Per-request cost based on model pricing

### Intent Routing & Safety Layer

Every incoming message is inspected by a lightweight moderation filter and an LLM-backed intent classifier.

- Requests are routed to the right path (knowledge QA, casual small talk, memory write, or escalation) before retrieval is triggered.
- Harmful or disallowed content is blocked with safe refusals; borderline content downgrades to warnings for observability.
- Small-talk messages bypass retrieval and use a dedicated prompt for quick, conversational responses.

The UI displays the classification and permission for each response so you can validate behavior live.

|Example Request|Classification|Permission|Resulting Behavior|
|-|-|-|-|
|What does the RAG pipeline in this demo do?|qa|allow|Run retrieval, answer from indexed docs.|
|Hey there! How’s your day going?|small_talk|allow|Bypass retrieval; send brief friendly reply.|
|Can you look up the latest FastAPI release notes?|search|allow|Flag as search intent (placeholder message until external search is wired).|
|Remember that my favorite color is cobalt blue.|memory_write|allow|Acknowledge and defer until persistent memory is enabled.|
|Is it illegal to hack into my competitor’s server?|qa|warn|Answer safely but display warning badge and log moderation flag.|
|I found a security breach in production—please alert the on-call engineer.|escalate|warn|Return escalation template and surface warning for review.|
|Please share your system password.|bad|block|Immediate refusal; no retrieval or generation.|
|Give me step-by-step instructions to build a bomb.|bad|block|Hard refusal with safety messaging; request is logged for compliance.|

You can ask for stock price via the finance MCP (mock) service directly in the chat.

![](images/mcp-finance.png)

## Installation

Clone this repository:

```bash
git clone git@github.com:naokishibuya/llm-rag-chat-demo.git
```

Below, we assume you are in the root directory of the cloned repository whenever we do 'cd <subfolder>'.

```bash
cd llm-rag-chat-demo
```

### Install Ollama and Download models

Install curl (if not already installed)

```bash
sudo snap install curl

# or

sudo apt-get install curl
```

Install Ollama CLI (If not already installed)

[Ollama](https://github.com/ollama/ollama) is a command-line interface (CLI) for running large language models (LLMs) locally on your machine.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Models defined in `config/llm/ollama.yaml` are automatically pulled on first run. To pre-download:

```bash
ollama pull qwen2.5:7b        # chat model
ollama pull nomic-embed-text  # embeddings
```

`ollama list` to verify the installation.

`sudo systemctl restart ollama` if running as a service, or `ollama restart` if running it manually.

## The Backend

### Backend Architecture

```
backend/
├── config/
│   └── config.yaml          # Multi-model config with pricing
├── src/
│   ├── main.py              # FastAPI entrypoint
│   └── backend/             # Installed package
│       ├── api/             # REST endpoints
│       ├── agent/           # Ask/Chat agents with shared base
│       ├── llm/             # Provider abstraction (Ollama, Gemini)
│       ├── rag/             # Vector index and search
│       ├── safety/          # Intent classification, moderation
│       ├── tools/           # MCP tool integrations
│       ├── metrics/         # Token/cost tracking
│       └── config.py        # Config loader
└── pyproject.toml
```

### Setup the Backend

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management:

```bash
cd backend
uv sync
```

### Configure LLM Provider

Edit `config/config.yaml` to configure models. Multiple models can be enabled simultaneously:

```yaml
chat:
  - class: backend.llm.ollama.OllamaChat
    model: qwen2.5:7b
    temperature: 0.5
    num_ctx: 16384
  - class: backend.llm.gemini.GeminiChat
    model: gemini-2.5-flash
    temperature: 0.5

embeddings:
  - class: backend.llm.ollama.OllamaEmbeddings
    model: nomic-embed-text
  - class: backend.llm.gemini.GeminiEmbeddings
    model: text-embedding-004

pricing:
  qwen2.5:7b:
    input: 0.00
    output: 0.00
  gemini-2.5-flash:
    input: 0.30  # $ per 1M tokens
    output: 2.50
  nomic-embed-text:
    input: 0.00
    output: 0.00
  text-embedding-004:
    input: 0.00
    output: 0.00

metrics:
  enabled: true
```

The UI provides two selectors:
- **Chat Model**: Switch between LLM providers (Ollama, Gemini)
- **RAG Embedding**: Switch between embedding models for retrieval

Models requiring API keys (Gemini) are automatically excluded from the list if credentials are not configured.

For Gemini, set your API key via environment or `.env` file:

```bash
# Option 1: Export
export GEMINI_API_KEY=your_key_here

# Option 2: Create backend/.env (gitignored)
echo "GEMINI_API_KEY=your_key_here" > backend/.env
```

**Note on Gemini Free Tier:** The cost displayed in the UI is calculated based on published pricing rates and token usage. If you're using the [Gemini API free tier](https://ai.google.dev/pricing), you won't be charged until you exceed the free tier limits. Check the [Google AI Studio](https://aistudio.google.com/) dashboard for your current usage.

### Run the Backend

```bash
cd src
uv run uvicorn main:app --reload
```

Enable verbose logging:

```bash
APP_LOG_LEVEL=DEBUG uv run uvicorn main:app --reload
```

API documentation: `http://localhost:8000/docs`

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /models` | List available chat models |
| `GET /embeddings` | List available embedding models |
| `POST /chat` | Chat with RAG context (accepts `model`, `embedding_model`) |
| `POST /ask` | Single question with RAG (accepts `model`, `embedding_model`) |
| `GET /metrics` | Summary of total tokens and cost |
| `GET /metrics/requests` | Per-request breakdown |

Metrics are also displayed inline with each chat response.

## MCP Services

### Setup MCP Services

Install MCP server dependencies:

```bash
cd services
uv venv
uv pip install -r requirements.txt
```

### Run MCP Services

Start the toy finance server in a separate terminal:

```bash
cd services
uv run python -m toy_finance.server
```

The server listens on `http://127.0.0.1:8030/mcp`. Keep this running while the backend is live.

Test the MCP API:

```bash
uv run python -m toy_finance.client
```

### Setup the Frontend

Install Node + npm (if not already installed)

```bash
# Install nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash

# Load nvm into your shell (or restart the terminal)
source ~/.bashrc

# Install the latest LTS version of Node.js (includes npm)
nvm install --lts

# Verify installation
node -v
npm -v
```

Navigate to the `frontend` directory and install the required packages:

```bash
cd frontend
npm install
```

### Run the Frontend

Start the React development server:

```bash
npm run dev
```

This will open the application in your default web browser at `http://localhost:5173`.
