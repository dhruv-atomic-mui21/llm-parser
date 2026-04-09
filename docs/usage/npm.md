# npm Usage Guide

This guide covers the Node.js/TypeScript (`npm`) usage of **LLM Context Forge**. 

## Installation

```bash
npm install llm-context-forge
# or
yarn add llm-context-forge
# or
pnpm add llm-context-forge
```

## 1. Token Counting

The `TokenCounter` accurately counts tokens using the same tokenization standard as the model you are targeting.

```typescript
import { TokenCounter } from 'llm-context-forge';

const counter = new TokenCounter("gpt-4o");
const tokens = counter.count("Hello, world!");
console.log(`Tokens: ${tokens}`); // Tokens: 4

// Check context window fit
const fits = counter.fitsInWindow("Your prompt...", { reserveOutput: 500 });

// Estimate cost before sending
const cost = counter.estimateCost("Your prompt...", { direction: "input" });
console.log(`Cost: $${cost.toFixed(6)}`);
```

## 2. Intelligent Chunking

Split large documents into semantically coherent pieces without cutting mid-sentence.

```typescript
import { DocumentChunker, ChunkStrategy } from 'llm-context-forge';

const chunker = new DocumentChunker("gpt-4o");

// Chunk respecting paragraph boundaries
const chunks = chunker.chunk(
    longDocument,
    {
        strategy: ChunkStrategy.PARAGRAPH,
        maxTokens: 500,
        overlapTokens: 50
    }
);

// Specialized chunkers
const codeChunks = chunker.chunkCode(sourceCode, { language: "typescript" });
const mdChunks = chunker.chunkMarkdown(readmeText);
```

## 3. Priority-Based Context Assembly

The core pattern for RAG applications — guarantee critical context fits while gracefully dropping lower-priority content.

```typescript
import { ContextWindow, Priority } from 'llm-context-forge';

const window = new ContextWindow("gpt-4o");

// System instructions — always included
window.addBlock("You are a legal assistant.", Priority.CRITICAL, "system");

// User query — high priority
window.addBlock("What is the statute of limitations?", Priority.HIGH, "query");

// RAG search results — included if space permits
window.addBlock(searchResult1, Priority.MEDIUM, "rag_1");
window.addBlock(searchResult2, Priority.LOW, "rag_2");

// Assemble: packs highest-priority blocks first
const prompt = window.assemble({ maxTokens: 4096 });

// See what was included/dropped
const usage = window.usage();
console.log(`Included: ${usage.numIncluded} blocks (${usage.includedTokens} tokens)`);
console.log(`Dropped:  ${usage.numExcluded} blocks`);
```

## 4. Cost Estimation

Pre-flight cost checks allow you to monitor LLM spending.

```typescript
import { CostCalculator } from 'llm-context-forge';

const calc = new CostCalculator("gpt-4o");

// Single prompt cost
const cost = calc.estimatePrompt("Your prompt text here");
console.log(`Input cost: $${cost.usd.toFixed(6)}`);

// Compare models
const comparison = calc.compareModels({
    texts: ["Document 1...", "Document 2..."],
    models: ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "gemini-flash"]
});

for (const [model, analysis] of Object.entries(comparison)) {
    console.log(`${model}: $${analysis.totalUsd.toFixed(6)} for ${analysis.totalTokens} tokens`);
}
```

## 5. Context Compression

If the prompt overflows, intelligently reduce the size rather than natively truncating everything.

```typescript
import { ContextCompressor, CompressionStrategy } from 'llm-context-forge';

const compressor = new ContextCompressor("gpt-4o");

// Extractive: keeps most important sentences via TF-IDF scoring
const result = compressor.compress(longText, { targetTokens: 200 });
console.log(`Compressed: ${result.originalTokens} → ${result.compressedTokens} tokens`);
console.log(`Savings: ${result.savingsPct.toFixed(1)}%`);

// Middle-out: preserves start and end, removes middle
const resultMiddleOut = compressor.compress(logText, { 
    targetTokens: 300, 
    strategy: CompressionStrategy.MIDDLE_OUT 
});
```

## 6. Conversation Management

Dynamically manage chat history windows.

```typescript
import { ConversationManager } from 'llm-context-forge';

const manager = new ConversationManager("gpt-4o");

manager.addMessage("system", "You are a helpful TypeScript tutor.");
manager.addMessage("user", "Explain decorators");
manager.addMessage("assistant", "Decorators are...");
// ... many more turns ...

// Auto-trim older messages to fit budget, preserving system prompt
const trimmed = manager.getContext({ maxTokens: 4096, preserveSystem: true });
```
