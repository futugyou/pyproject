import json
import uuid
from collections.abc import Sequence
from typing import Any
from pydantic import BaseModel
import asyncpg
from agent_framework import ChatMessage


class PostgresStoreState(BaseModel):
    """State model for serializing and deserializing Postgres chat message store data."""

    thread_id: str
    postgres_url: str
    table_name: str = "chat_messages"
    max_messages: int | None = None


class PostgresChatMessageStore:
    """PostgreSQL-backed implementation of ChatMessageStore using asyncpg."""

    def __init__(
        self,
        postgres_url: str,
        thread_id: str | None = None,
        table_name: str = "chat_messages",
        max_messages: int | None = None,
    ) -> None:
        """Initialize the PostgreSQL chat message store.

        Args:
            postgres_url: PostgreSQL connection URL (for example, "postgresql://localhost:5432/mydb").
            thread_id: Unique identifier for this conversation thread.
                      If not provided, a UUID will be auto-generated.
            table_name: Name of the table to store messages in.
            max_messages: Maximum number of messages to retain in the database.
                         When exceeded, oldest messages are automatically trimmed.
        """
        if not postgres_url:
            raise ValueError("postgres_url is required for PostgreSQL connection")

        self.postgres_url = postgres_url
        self.thread_id = thread_id or f"thread_{uuid.uuid4()}"
        self._table_name = table_name
        self.max_messages = max_messages

        self._postgres_client = None  # Initialize as None initially

    async def _ensure_client(self):
        """Ensure the PostgreSQL client is initialized and connected."""
        if self._postgres_client is None:
            self._postgres_client = await asyncpg.connect(self.postgres_url)

    def get_table_name(self) -> str:
        return f"{self._table_name}_{self.thread_id.replace('-', '_')}"

    async def add_messages(self, messages: Sequence[ChatMessage]) -> None:
        """Add messages to the PostgreSQL store."""
        if not messages:
            return

        # Ensure PostgreSQL client is connected before proceeding
        await self._ensure_client()

        # Insert messages into the PostgreSQL database
        async with self._postgres_client.transaction():
            for message in messages:
                serialized_message = self._serialize_message(message)
                await self._postgres_client.execute(
                    f"INSERT INTO {self.get_table_name()} (message) VALUES ($1)",
                    serialized_message,
                )

        # Apply message limit if configured
        if self.max_messages is not None:
            await self._trim_messages()

    async def list_messages(self) -> list[ChatMessage]:
        """Get all messages from the store in chronological order."""
        # Ensure PostgreSQL client is connected before proceeding
        await self._ensure_client()

        messages = await self._postgres_client.fetch(
            f"SELECT message FROM {self.get_table_name()} ORDER BY created_at ASC"
        )

        return [self._deserialize_message(msg["message"]) for msg in messages]

    async def serialize_state(self, **kwargs: Any) -> Any:
        """Serialize the current store state for persistence."""
        state = PostgresStoreState(
            thread_id=self.thread_id,
            postgres_url=self.postgres_url,
            table_name=self._table_name,
            max_messages=self.max_messages,
        )
        return state.model_dump(**kwargs)

    async def deserialize_state(
        self, serialized_store_state: Any, **kwargs: Any
    ) -> None:
        """Deserialize state data into this store instance."""
        if serialized_store_state:
            state = PostgresStoreState.model_validate(serialized_store_state, **kwargs)
            self.thread_id = state.thread_id
            self._table_name = state.table_name
            self.max_messages = state.max_messages

            # Reconnect to PostgreSQL if the URL changed
            if state.postgres_url and state.postgres_url != self.postgres_url:
                self.postgres_url = state.postgres_url
                await self._ensure_client()

    async def _trim_messages(self) -> None:
        """Trim the messages table to the maximum number of messages."""
        # Ensure PostgreSQL client is connected before proceeding
        await self._ensure_client()

        count = await self._postgres_client.fetchval(
            f"SELECT COUNT(*) FROM {self.get_table_name()}"
        )
        if count > self.max_messages:
            # Delete the oldest messages
            await self._postgres_client.execute(
                f"DELETE FROM {self.get_table_name()} WHERE created_at IN (SELECT created_at FROM {self.get_table_name()} ORDER BY created_at ASC LIMIT {count - self.max_messages})"
            )

    def _serialize_message(self, message: ChatMessage) -> str:
        """Serialize a ChatMessage to JSON string."""
        return message.to_json(separators=(",", ":"))

    def _deserialize_message(self, serialized_message: str) -> ChatMessage:
        """Deserialize a JSON string to ChatMessage."""
        return ChatMessage.from_json(serialized_message)

    async def clear(self) -> None:
        """Remove all messages from the store."""
        # Ensure PostgreSQL client is connected before proceeding
        await self._ensure_client()

        await self._postgres_client.execute(f"DELETE FROM {self.get_table_name()}")

    async def aclose(self) -> None:
        """Close the PostgreSQL connection."""
        if self._postgres_client:
            await self._postgres_client.close()
