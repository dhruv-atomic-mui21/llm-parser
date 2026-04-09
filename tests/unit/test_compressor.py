"""
Rigorous tests for ContextCompressor.
"""

import pytest
from llm_context_forge.compressor import ContextCompressor, CompressionStrategy

class TestContextCompressor:
    """Rigorous semantic and boundary validation."""

    def setup_method(self):
        self.compressor = ContextCompressor("gpt-4o")

    def test_semantic_fidelity_extractive(self):
        """Ensure TF-IDF retains highest-priority sentence mathematically."""
        # Sentence 1 is pure noise. Sentence 2 has complex uniquely countable nouns.
        text = "This is a simple sentence. However, this subsequent quantum entanglement configuration matrix requires deep analysis."
        
        result = self.compressor.compress(text, target_tokens=15, strategy=CompressionStrategy.EXTRACTIVE)
        
        # Validate that the quantum sentence survives
        assert "quantum entanglement" in result.text
        assert result.compressed_tokens <= 15
        assert result.savings_pct >= 0

    def test_middle_out_truncation(self):
        """Ensure boundary bytes are strictly enforced in Middle-Out dropping logic."""
        long_text = "START OF FILE. " + "Middle noise padding data " * 100 + "END OF FILE."
        
        result = self.compressor.compress(long_text, target_tokens=30, strategy=CompressionStrategy.MIDDLE_OUT)
        
        # It must natively extract bounds and inject marker
        assert "START OF FILE" in result.text
        assert "END OF FILE" in result.text
        assert "middle content removed" in result.text
        assert result.compressed_tokens <= 30

    def test_map_reduce(self):
        """Map reduce execution does not violate tokens bounds."""
        text = "A. B. C. D. E. F. G. H. I. J. K. L. M. N. O. P."
        res = self.compressor.compress(text, target_tokens=10, strategy=CompressionStrategy.MAP_REDUCE)
        assert res.compressed_tokens <= 10
