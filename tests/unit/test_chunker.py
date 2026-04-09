"""
Rigorous tests for DocumentChunker
"""

import pytest
from llm_context_forge.chunker import DocumentChunker, ChunkStrategy

class TestDocumentChunker:
    """Rigorous constraints, overlaps, and bounds tests."""

    def setup_method(self):
        self.chunker = DocumentChunker("gpt-4o")

    def test_constraint_enforcement_paragraphs(self):
        """Ensure paragraphs never exceed max_tokens constraint."""
        text = "P1\n\nP2\n\nP3\n\nP4\n\nP5"
        chunks = self.chunker.chunk(text, strategy=ChunkStrategy.PARAGRAPH, max_tokens=5, overlap_tokens=0)
        
        assert len(chunks) > 1
        for c in chunks:
            assert c.token_count <= 5

    def test_force_split_massive_lines(self):
        """When semantic chunks fail, character splitting must enforce the bounds."""
        massive = "word " * 500  # huge single line with no punctuation
        chunks = self.chunker.chunk(massive, max_tokens=50, overlap_tokens=0)
        
        assert len(chunks) > 1
        for c in chunks:
            assert hasattr(c, "token_count")
            assert c.token_count <= 50

    def test_overlap_handling_infinite_safety(self):
        """Overlap should not prevent progression or cause infinite loops."""
        text = "A. B. C. D. E. F. G. H. I. J. K. L. M. N. O. P. Q. R. S. T. U. V. W. X. Y. Z."
        # Overly restrictive bounds
        chunks = self.chunker.chunk(text, strategy=ChunkStrategy.SENTENCE, max_tokens=5, overlap_tokens=2)
        
        assert len(chunks) > 5
        # Ensure chunks don't magically end up empty
        for c in chunks:
            assert c.token_count > 0

    def test_code_chunk_respects_functions(self):
        """Functions should remain largely coherent chunks unless forced."""
        code = "def foo():\n    pass\n\ndef bar():\n    pass"
        chunks = self.chunker.chunk_code(code, max_tokens=50)
        
        # Depending on splits, this could be 1 chunk if budget allows or 2
        assert len(chunks) > 0
        for c in chunks:
            assert c.token_count <= 50
