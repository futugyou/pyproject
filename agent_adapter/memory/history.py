import json
from typing import Any, MutableSequence, Sequence, cast

from sqlalchemy import create_engine, Column, String, Integer, Text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.future import select

from agent_framework import ChatMessage, Context, ContextProvider, Role
from agent_framework.exceptions import (
    AgentException,
    ServiceInitializationError,
    ServiceInvalidRequestError,
)

Base = declarative_base()


class Message(Base):
    """ORM model for storing messages in PostgreSQL."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    conversation_id = Column(String, nullable=False)
    message_id = Column(String, nullable=False)
    author_name = Column(String, nullable=True)
    application_id = Column(String, nullable=True)
    agent_id = Column(String, nullable=True)
    user_id = Column(String, nullable=True)
    thread_id = Column(String, nullable=True)


class PostgresProvider(ContextProvider):
    """PostgreSQL context provider with dynamic, filterable schema."""

    def __init__(
        self,
        db_url: str = "postgresql+asyncpg://user:password@localhost/mydb",
        application_id: str | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
        context_prompt: str = ContextProvider.DEFAULT_CONTEXT_PROMPT,
        overwrite_index: bool = False,
    ):
        """Create a PostgreSQL Context Provider."""
        self.db_url = db_url
        self.application_id = application_id
        self.agent_id = agent_id
        self.user_id = user_id
        self.thread_id = thread_id
        self.context_prompt = context_prompt
        self._index_initialized = False

        # Initialize asynchronous database engine and session
        self.engine = create_async_engine(self.db_url, echo=True)
        self.Session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        # Create tables asynchronously
        self._initialize()

    async def _initialize(self) -> None:
        """Initialize database tables asynchronously."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def _add(
        self,
        *,
        data: dict[str, Any] | list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Inserts one or many documents with partition fields populated."""
        self._validate_filters()

        docs = data if isinstance(data, list) else [data]

        async with self.Session() as session:
            async with session.begin():
                for doc in docs:
                    message = Message(
                        role=doc["role"],
                        content=doc["content"],
                        conversation_id=doc["conversation_id"],
                        message_id=doc["message_id"],
                        author_name=doc.get("author_name"),
                        application_id=doc.get("application_id", self.application_id),
                        agent_id=doc.get("agent_id", self.agent_id),
                        user_id=doc.get("user_id", self.user_id),
                        thread_id=doc.get("thread_id", self.thread_id),
                    )
                    session.add(message)
            await session.commit()

    async def _postgres_search(
        self, text: str, *, num_results: int = 10
    ) -> list[dict[str, Any]]:
        """Runs a simple text search on messages."""
        self._validate_filters()

        async with self.Session() as session:
            stmt = (
                select(Message)
                .filter(Message.content.ilike(f"%{text}%"))
                .limit(num_results)
            )
            result = await session.execute(stmt)
            messages = result.scalars().all()

        return [
            {"role": m.role, "content": m.content, "message_id": m.message_id}
            for m in messages
        ]

    async def search_all(self, page_size: int = 200) -> list[dict[str, Any]]:
        """Returns all documents in the database."""
        async with self.Session() as session:
            stmt = select(Message).limit(page_size)
            result = await session.execute(stmt)
            messages = result.scalars().all()

        return [
            {"role": m.role, "content": m.content, "message_id": m.message_id}
            for m in messages
        ]

    @property
    def _effective_thread_id(self) -> str | None:
        """Resolves the active thread id."""
        return self.thread_id

    @override
    async def thread_created(self, thread_id: str | None) -> None:
        """Called when a new thread is created."""
        self.thread_id = thread_id

    @override
    async def invoked(
        self,
        request_messages: ChatMessage | Sequence[ChatMessage],
        response_messages: ChatMessage | Sequence[ChatMessage] | None = None,
        invoke_exception: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an invocation occurs."""
        self._validate_filters()

        request_messages_list = (
            [request_messages]
            if isinstance(request_messages, ChatMessage)
            else list(request_messages)
        )
        response_messages_list = (
            [response_messages]
            if isinstance(response_messages, ChatMessage)
            else list(response_messages)
            if response_messages
            else []
        )
        messages_list = [*request_messages_list, *response_messages_list]

        messages: list[dict[str, Any]] = []
        for message in messages_list:
            if (
                message.role.value
                in {Role.USER.value, Role.ASSISTANT.value, Role.SYSTEM.value}
                and message.text
                and message.text.strip()
            ):
                shaped: dict[str, Any] = {
                    "role": message.role.value,
                    "content": message.text,
                    "conversation_id": self.thread_id,
                    "message_id": message.message_id,
                    "author_name": message.author_name,
                }
                messages.append(shaped)
        if messages:
            await self._add(data=messages)

    @override
    async def invoking(
        self, messages: ChatMessage | MutableSequence[ChatMessage], **kwargs: Any
    ) -> Context:
        """Called before invoking the model to provide scoped context."""
        self._validate_filters()

        messages_list = (
            [messages] if isinstance(messages, ChatMessage) else list(messages)
        )
        input_text = "\n".join(
            msg.text for msg in messages_list if msg and msg.text and msg.text.strip()
        )

        memories = await self._postgres_search(text=input_text)
        line_separated_memories = "\n".join(
            str(memory.get("content", ""))
            for memory in memories
            if memory.get("content")
        )

        return Context(
            messages=[
                ChatMessage(
                    role="user",
                    text=f"{self.context_prompt}\n{line_separated_memories}",
                )
            ]
            if line_separated_memories
            else None
        )

    def _validate_filters(self) -> None:
        """Validates that at least one filter is provided."""
        if (
            not self.agent_id
            and not self.user_id
            and not self.application_id
            and not self.thread_id
        ):
            raise ServiceInitializationError(
                "At least one of the filters: agent_id, user_id, application_id, or thread_id is required."
            )
