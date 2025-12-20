
import uuid
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Any, List, Sequence
from agent_framework import ChatMessage, ChatMessageStoreProtocol
from agent_framework._serialization import SerializationMixin

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select

Base = declarative_base()


class ChatMessageORM(Base):
    """SQLAlchemy model for storing chat messages in PostgreSQL."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    thread_id = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, user_id={self.user_id}, thread_id={self.thread_id})>"


class PostgresStoreState(SerializationMixin):
    """State model for serializing and deserializing Postgres chat message store data."""

    thread_id: str
    postgres_url: str | None = None
    table_name: str = "chat_messages"
    max_messages: int | None = None

    def __init__(
        self,
        thread_id: str,
        postgres_url: str | None = None,
        table_name: str = "chat_messages",
        max_messages: int | None = None,
    ) -> None:
        self.thread_id = thread_id
        self.postgres_url = postgres_url
        self.table_name = table_name
        self.max_messages = max_messages


class PostgresChatMessageStore(ChatMessageStoreProtocol):
    """PostgreSQL-backed implementation of ChatMessageStore using SQLAlchemy."""

    def __init__(
        self,
        postgres_url: str,
        thread_id: str | None = None,
        table_name: str = "chat_messages",
        max_messages: int | None = None,
        user_id: str | None = None,
    ) -> None:
        if not postgres_url:
            raise ValueError("postgres_url is required for PostgreSQL connection")

        self.postgres_url = postgres_url
        self.thread_id = thread_id or f"thread_{uuid.uuid4()}"
        self.user_id = user_id or self.thread_id
        self._table_name = table_name
        self.max_messages = max_messages

        self.engine = create_async_engine(self.postgres_url, echo=True, future=True)
        self.Session = sessionmaker(
            bind=self.engine, class_=AsyncSession, expire_on_commit=False
        )
        self._session = None

    async def _create_table_if_needed(self) -> None:
        """Automatically create the table if it does not exist."""
        async with self.engine.begin() as conn:
            # Create all tables asynchronously
            await conn.run_sync(Base.metadata.create_all)

    async def _ensure_session(self) -> None:
        """Ensure that the session is initialized."""
        if self._session is None:
            self._session = self.Session()
        await self._create_table_if_needed()

    def get_table_name(self) -> str:
        return self._table_name

    async def list_messages(self) -> List[ChatMessage]:
        """Get all messages from the store in chronological order."""
        await self._ensure_session()

        result = await self._session.execute(
            select(ChatMessageORM)
            .filter_by(thread_id=self.thread_id)
            .order_by(ChatMessageORM.created_at.asc())
        )
        messages = result.scalars().all()
        return [self._deserialize_message(msg.message) for msg in messages]

    async def add_messages(self, messages: Sequence[ChatMessage]) -> None:
        """Add messages to the PostgreSQL store."""
        if not messages:
            return

        await self._ensure_session()

        for message in messages:
            user_id = self.user_id
            serialized_message = self._serialize_message(message)
            new_message = ChatMessageORM(
                user_id=user_id, thread_id=self.thread_id, message=serialized_message
            )
            self._session.add(new_message)

        await self._session.commit()

        # Apply message limit if configured
        if self.max_messages is not None:
            await self._trim_messages()

    async def serialize(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize the current store state for persistence."""
        state = PostgresStoreState(
            thread_id=self.thread_id,
            postgres_url=self.postgres_url,
            table_name=self._table_name,
            max_messages=self.max_messages,
        )
        return state.to_dict(**kwargs)

    @classmethod
    async def deserialize(
        cls, serialized_store_state: Any, **kwargs: Any
    ) -> "PostgresChatMessageStore":
        if not serialized_store_state:
            raise ValueError("serialized_store_state is required for deserialization")

        state = PostgresChatMessageStore.from_dict(serialized_store_state, **kwargs)

        return cls(
            postgres_url=state.postgres_url,
            thread_id=state.thread_id,
            table_name=state.table_name,
            max_messages=state.max_messages,
        )

    async def update_from_state(
        self, serialized_store_state: Any, **kwargs: Any
    ) -> None:
        if not serialized_store_state:
            return

        state = PostgresChatMessageStore.from_dict(serialized_store_state, **kwargs)

        self.thread_id = state.thread_id
        if state.postgres_url is not None:
            self.postgres_url = state.postgres_url
        self.max_messages = state.max_messages

        if state.postgres_url and state.postgres_url != getattr(
            self, "_last_postgres_url", None
        ):
            self._last_postgres_url = state.postgres_url
            await self._ensure_session()

    async def _trim_messages(self) -> None:
        """Trim the messages table to the maximum number of messages."""
        await self._ensure_session()

        result = await self._session.execute(
            select(ChatMessageORM).filter_by(thread_id=self.thread_id)
        )
        messages = result.scalars().all()
        count = len(messages)

        if count > self.max_messages:
            # Delete the oldest messages
            excess_count = count - self.max_messages
            await self._session.execute(
                select(ChatMessageORM)
                .filter_by(thread_id=self.thread_id)
                .order_by(ChatMessageORM.created_at.asc())
                .limit(excess_count)
            )
            await self._session.commit()

    def _serialize_message(self, message: ChatMessage) -> str:
        """Serialize a ChatMessage to JSON string."""
        return message.to_json(separators=(",", ":"))

    def _deserialize_message(self, serialized_message: str) -> ChatMessage:
        """Deserialize a JSON string to ChatMessage."""
        return ChatMessage.from_json(serialized_message)

    async def clear(self) -> None:
        """Remove all messages from the store."""
        # await self._ensure_session()
        # await self._session.execute(
        #     select(ChatMessageORM).filter_by(thread_id=self.thread_id).delete()
        # )
        # await self._session.commit()
        ...

    async def aclose(self) -> None:
        """Close the PostgreSQL session."""
        if self._session:
            await self._session.close()
