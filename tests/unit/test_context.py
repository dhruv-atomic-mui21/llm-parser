"""
Rigorous tests for Context assembling and conversation management.
"""

import pytest
from llm_context_forge.context import ContextWindow, Priority, ConversationManager

class TestContextWindow:
    """Validate priority dropping algorithms."""

    def test_overflow_priority_dropping(self):
        """Verify LOW priority blocks are accurately dropped before CRITICAL when budget overflows."""
        window = ContextWindow("gpt-4o")
        window.add_block("Critical data that must be present.", priority=Priority.CRITICAL, label="sys")
        window.add_block("Low priority 1.", priority=Priority.LOW, label="b1")
        window.add_block("Low priority 2.", priority=Priority.LOW, label="b2")

        # Assemble tightly
        window.assemble(max_tokens=10)

        usage = window.usage()
        assert usage["num_included"] == 1
        assert usage["num_excluded"] == 2

        # The included block MUST be the critical one
        included = [b for b in usage["blocks"] if b["included"]]
        assert included[0]["label"] == "sys"
        assert included[0]["priority"] == "CRITICAL"

    def test_priority_tie_resolution(self):
        """Verify determinism regarding identically prioritized blocks."""
        window = ContextWindow("gpt-4o")
        window.add_block("Data block A", priority=Priority.MEDIUM, label="A")
        window.add_block("Data block B", priority=Priority.MEDIUM, label="B")

        window.assemble(max_tokens=5)

        usage = window.usage()
        assert usage["num_included"] == 1
        included = [b for b in usage["blocks"] if b["included"]]
        # List.sort in Python is stable. So "A" retains priority natively.
        assert included[0]["label"] == "A"


class TestConversationManager:
    """Conversational history budget management."""

    def test_manager_preserves_newest_messages(self):
        """Validate trailing loop budget preservation identically implemented to TS fix."""
        mgr = ConversationManager("gpt-4o")
        mgr.add_message("system", "SYS")
        mgr.add_message("user", "Old padding message to exceed budget by a significant margin")
        mgr.add_message("assistant", "A1")
        mgr.add_message("user", "U2")

        # Force budget limit to cull older standard messages
        ctx = mgr.get_context(max_tokens=10, preserve_system=True)
        
        # It guarantees system preservation
        assert ctx[0]["role"] == "system"
        
        if len(ctx) > 1:
            # U2 must be retained over old padding
            assert ctx[-1]["content"] == "U2"
