# pip Usage Guide

This guide covers the Python (`pip`) usage of **LLM Context Forge**. 

## Installation

```bash
pip install llm-context-forge
```

## 1. Token Counting

The `TokenCounter` accurately counts tokens using the same tokenization standard as the model you are targeting.

```python
from llm_context_forge import TokenCounter

counter = TokenCounter("gpt-4o")
tokens = counter.count("Hello, world!")
print(f"Tokens: {tokens}")  # Tokens: 4

# Check context window fit
fits = counter.fits_in_window("Your prompt...", reserve_output=500)

# Estimate cost before sending
cost = counter.estimate_cost("Your prompt...", direction="input")
print(f"Cost: ${cost:.6f}")
```

## 2. Intelligent Chunking

Split large documents into semantically coherent pieces without cutting mid-sentence.

```python
from llm_context_forge import DocumentChunker, ChunkStrategy

chunker = DocumentChunker("gpt-4o")

# Chunk respecting paragraph boundaries
chunks = chunker.chunk(
    long_document,
    strategy=ChunkStrategy.PARAGRAPH,
    max_tokens=500,
    overlap_tokens=50,
)

# Specialized chunkers
code_chunks = chunker.chunk_code(source_code, language="python")
md_chunks = chunker.chunk_markdown(readme_text)
```

## 3. Priority-Based Context Assembly

The core pattern for RAG applications — guarantee critical context fits while gracefully dropping lower-priority content.

```python
from llm_context_forge import ContextWindow, Priority

window = ContextWindow("gpt-4o")

# System instructions — always included
window.add_block("You are a legal assistant.", Priority.CRITICAL, "system")

# User query — high priority
window.add_block("What is the statute of limitations?", Priority.HIGH, "query")

# RAG search results — included if space permits
window.add_block(search_result_1, Priority.MEDIUM, "rag_1")
window.add_block(search_result_2, Priority.LOW, "rag_2")

# Assemble: packs highest-priority blocks first
prompt = window.assemble(max_tokens=4096)

# See what was included/dropped
usage = window.usage()
print(f"Included: {usage['num_included']} blocks ({usage['included_tokens']} tokens)")
print(f"Dropped:  {usage['num_excluded']} blocks")
```

## 4. Cost Estimation

Pre-flight cost checks allow you to monitor LLM spending.

```python
from llm_context_forge import CostCalculator

calc = CostCalculator("gpt-4o")

# Single prompt cost
cost = calc.estimate_prompt("Your prompt text here")
print(f"Input cost: ${cost.usd:.6f}")

# Compare models
comparison = calc.compare_models(
    texts=["Document 1...", "Document 2..."],
    models=["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "gemini-flash"],
)

for model, analysis in comparison.items():
    print(f"{model}: ${analysis.total_usd:.6f} for {analysis.total_tokens} tokens")
```

## 5. Context Compression

If the prompt overflows, intelligently reduce the size rather than natively truncating everything.

```python
from llm_context_forge import ContextCompressor, CompressionStrategy

compressor = ContextCompressor("gpt-4o")

# Extractive: keeps most important sentences via TF-IDF scoring
result = compressor.compress(long_text, target_tokens=200)
print(f"Compressed: {result.original_tokens} → {result.compressed_tokens} tokens")
print(f"Savings: {result.savings_pct:.1f}%")

# Middle-out: preserves start and end, removes middle
result = compressor.compress(log_text, target_tokens=300, strategy=CompressionStrategy.MIDDLE_OUT)
```

## 6. Conversation Management

Dynamically manage chat history windows.

```python
from llm_context_forge import ConversationManager

manager = ConversationManager("gpt-4o")

manager.add_message("system", "You are a helpful Python tutor.")
manager.add_message("user", "Explain decorators")
manager.add_message("assistant", "Decorators are...")
# ... many more turns ...

# Auto-trim older messages to fit budget, preserving system prompt
trimmed = manager.get_context(max_tokens=4096, preserve_system=True)
```
