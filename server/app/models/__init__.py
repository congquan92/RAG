"""
ORM Models package — re-export Base và tất cả models.

Import models ở đây để đảm bảo SQLAlchemy metadata biết tất cả tables
TRƯỚC khi gọi Base.metadata.create_all() trong lifespan.
"""

from app.core.database import Base
from app.models.chat import ChatMessage, ChatSession
from app.models.document import Document, IngestionTask

__all__ = [
    "Base",
    "ChatSession",
    "ChatMessage",
    "Document",
    "IngestionTask",
]
