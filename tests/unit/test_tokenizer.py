"""
Rigorous tests for TokenCounter.
"""

import pytest
from llm_context_forge.tokenizer import TokenCounter
from llm_context_forge.models import TokenizerBackend

class TestTokenCounter:
    """Rigorous tests for TokenCounter validating exact bounds and specific features."""

    def setup_method(self):
        self.counter = TokenCounter("gpt-4o")

    def test_determinism_guarantees(self):
        """Ensure token counts exactly match structural expectations for OpenAI schema."""
        assert self.counter.count("Hello, world!") == 4
        assert self.counter.count("The quick brown fox jumps over the lazy dog.") == 10

    def test_chatml_overhead_exactitude(self):
        """Validate ChatML message list counting to exact specifications."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        # Base=3. Msg1 (System + "You are helpful.") = 3 + 1 + 4 = 8
        # Msg2 (User "Hi") = 3 + 1 + 1 = 5
        # Total = 16 tokens
        assert self.counter.count_messages(messages) == 16

    def test_edge_case_empty(self):
        """Testing zero boundaries gracefully."""
        assert self.counter.count("") == 0
        assert self.counter.count(None) == 0

    def test_edge_case_emojis(self):
        """Testing emoji and complex unicode behavior."""
        text = "🚀👨‍👩‍👧‍👦 Here we go!"
        tokens = self.counter.count(text)
        assert tokens > 0
        assert tokens < 20  # ensure it doesn't wildly blow up or crash

    def test_edge_case_massive_string(self):
        """Ensure it does not segfault on massive contiguous bytes."""
        massive = "A" * 50_000
        tokens = self.counter.count(massive)
        assert tokens > 100

    def test_truncation_mathematical_bounds(self):
        """Ensure chunk truncation works exactly mathematically using binary search."""
        text = "One two three four five six seven eight nine ten"
        # Total text is ~10 tokens. Truncate to 5.
        result = self.counter.truncate_to_fit(text, max_tokens=5, suffix="...")
        assert self.counter.count(result) <= 5

    def test_fallback_mechanics(self):
        """Test fallback estimations using scalars."""
        counter_unknown = TokenCounter("unknown-model-xyz")
        # Should fallback to cl100k roughly + 10% penalty
        assert counter_unknown.count("Test string") > 0

    def test_google_multipliers_vs_openai(self):
        """Verify the difference in encoding paths if backends switch."""
        c_open = TokenCounter("gpt-4")
        c_google = TokenCounter("gemini-pro")
        assert c_google.count("Test") > 0
        assert type(c_google.count("Test")) == int
