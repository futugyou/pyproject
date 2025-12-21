import uuid
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from agent_framework import WorkflowCheckpoint

from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import NoResultFound


Base = declarative_base()


class WorkflowCheckpointORM(Base):
    __tablename__ = "python_workflow_checkpoints"

    checkpoint_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String, nullable=False)
    timestamp = Column(String)
    messages = Column(JSON, default=dict)
    shared_state = Column(JSON, default=dict)
    pending_request_info_events = Column(JSON, default=dict)
    iteration_count = Column(Integer, default=0)
    metadata_data = Column(JSON, default=dict)
    version = Column(String, default="1.0")

    def to_dict(self):
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "timestamp": self.timestamp,
            "messages": self.messages,
            "shared_state": self.shared_state,
            "pending_request_info_events": self.pending_request_info_events,
            "iteration_count": self.iteration_count,
            "metadata": self.metadata_data,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


# Convert WorkflowCheckpoint to WorkflowCheckpointORM
def workflow_checkpoint_to_orm(checkpoint: WorkflowCheckpoint) -> WorkflowCheckpointORM:
    return WorkflowCheckpointORM(
        checkpoint_id=checkpoint.checkpoint_id,
        workflow_id=checkpoint.workflow_id,
        timestamp=checkpoint.timestamp,
        messages=checkpoint.messages,
        shared_state=checkpoint.shared_state,
        pending_request_info_events=checkpoint.pending_request_info_events,
        iteration_count=checkpoint.iteration_count,
        metadata_data=checkpoint.metadata,
        version=checkpoint.version,
    )


# Convert WorkflowCheckpointORM to WorkflowCheckpoint
def workflow_checkpoint_from_orm(orm: WorkflowCheckpointORM) -> WorkflowCheckpoint:
    return WorkflowCheckpoint(
        checkpoint_id=orm.checkpoint_id,
        workflow_id=orm.workflow_id,
        timestamp=orm.timestamp,
        messages=orm.messages,
        shared_state=orm.shared_state,
        pending_request_info_events=orm.pending_request_info_events,
        iteration_count=orm.iteration_count,
        metadata=orm.metadata_data,
        version=orm.version,
    )


class PostgresCheckpointStorage:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = create_async_engine(self.db_url, echo=True, future=True)
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _create_table_if_needed(self) -> None:
        """Automatically create the table if it does not exist."""
        async with self._init_lock:
            if self._initialized:
                return

            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                self._initialized = True

    @asynccontextmanager
    async def _get_session(self) -> AsyncSession:
        await self._create_table_if_needed()
        async with self.SessionLocal() as session:
            yield session

    async def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> str:
        async with self._get_session() as session:
            orm_checkpoint = workflow_checkpoint_to_orm(checkpoint)
            session.add(orm_checkpoint)
            await session.commit()
            return orm_checkpoint.checkpoint_id

    async def load_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        async with self._get_session() as session:
            stmt = select(WorkflowCheckpointORM).filter(
                WorkflowCheckpointORM.checkpoint_id == checkpoint_id
            )
            result = await session.execute(stmt)
            orm_checkpoint = result.scalar_one_or_none()
            return (
                workflow_checkpoint_from_orm(orm_checkpoint) if orm_checkpoint else None
            )

    async def list_checkpoint_ids(self, workflow_id: str | None = None) -> list[str]:
        async with self._get_session() as session:
            stmt = select(WorkflowCheckpointORM.checkpoint_id)
            if workflow_id:
                stmt = stmt.filter(WorkflowCheckpointORM.workflow_id == workflow_id)

            result = await session.execute(stmt)
            return [row[0] for row in result.fetchall()]

    async def list_checkpoints(
        self, workflow_id: str | None = None
    ) -> list[WorkflowCheckpoint]:
        async with self._get_session() as session:
            stmt = select(WorkflowCheckpointORM)
            if workflow_id:
                stmt = stmt.filter(WorkflowCheckpointORM.workflow_id == workflow_id)

            result = await session.execute(stmt)
            orm_checkpoints = result.scalars().all()
            return [workflow_checkpoint_from_orm(orm) for orm in orm_checkpoints]

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        async with self._get_session() as session:
            stmt = select(WorkflowCheckpointORM).filter(
                WorkflowCheckpointORM.checkpoint_id == checkpoint_id
            )
            result = await session.execute(stmt)
            orm_checkpoint = result.scalar_one_or_none()

            if orm_checkpoint:
                await session.delete(orm_checkpoint)
                await session.commit()
                return True
            return False
