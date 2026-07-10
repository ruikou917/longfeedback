"""Source dataset adapters producing canonical trajectories.

The generic conversation core is dependency-light; source-specific readers
(for example LMSYS) require the ``research`` extra for pyarrow.
"""

from longfeedback.data.conversations import (
    PII_FILTER_VERSION,
    ConversationRecord,
    ConversationTurn,
    conversation_exclusion_reason,
    conversation_to_trajectory,
    sanitize_text,
    split_by_conversation_hash,
)

__all__ = [
    "PII_FILTER_VERSION",
    "ConversationRecord",
    "ConversationTurn",
    "conversation_exclusion_reason",
    "conversation_to_trajectory",
    "sanitize_text",
    "split_by_conversation_hash",
]
