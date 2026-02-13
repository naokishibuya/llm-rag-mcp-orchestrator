# Multi-Agent Systems and Tool Calling

Modern LLM applications often use multiple specialized agents working together.

## Intent Routing

A classifier agent determines the type of request (Q&A, small talk, finance lookup, etc.) and routes it to the appropriate handler.

## Tool Calling

Agents can invoke external tools via protocols like MCP (Model Context Protocol). This demo includes a finance quote tool that returns mock stock prices.

## Safety Layer

Before processing, requests pass through moderation filters that detect harmful content using pattern matching and LLM-based classification.

## Agent Architecture

This demo implements:

- **BaseAgent**: Shared logic for intent classification, safety checks, and metrics tracking
- **AskAgent**: Handles single-turn Q&A with RAG retrieval
- **ChatAgent**: Manages multi-turn conversations with history

## Metrics Tracking

Every LLM call records token usage and estimated cost. This is critical for production systems to monitor spend and optimize prompts.

- Total requests processed
- Input and output token counts
- Estimated cost in USD (for cloud providers)
