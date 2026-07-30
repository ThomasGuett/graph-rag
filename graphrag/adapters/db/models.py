from datetime import datetime
from typing import Any
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

_EMPTY_JSONB = sql_text("'{}'::jsonb")


class Base(DeclarativeBase):
    pass


class Node(Base):
    __tablename__ = "nodes"

    id: Mapped[Any] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    props: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=_EMPTY_JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="node", cascade="all, delete-orphan")


class Edge(Base):
    __tablename__ = "edges"
    __table_args__ = (UniqueConstraint("src_id", "dst_id", "type", name="edges_unique_triple"),)

    id: Mapped[Any] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    src_id: Mapped[Any] = mapped_column(UUID(as_uuid=True), ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False)
    dst_id: Mapped[Any] = mapped_column(UUID(as_uuid=True), ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    props: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=_EMPTY_JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[Any] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    node_id: Mapped[Any] = mapped_column(UUID(as_uuid=True), ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(2000), nullable=True)
    props: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=_EMPTY_JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    node: Mapped[Node] = relationship(back_populates="chunks")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[Any] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    node_id: Mapped[Any] = mapped_column(
        UUID(as_uuid=True), ForeignKey("nodes.id", ondelete="SET NULL"), nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    props: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=_EMPTY_JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    jobs: Mapped[list["IngestJob"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class IngestJob(Base):
    __tablename__ = "ingest_jobs"

    id: Mapped[Any] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[Any] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    progress: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=_EMPTY_JSONB)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="jobs")


class Community(Base):
    __tablename__ = "communities"

    id: Mapped[Any] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    node_id: Mapped[Any | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("nodes.id", ondelete="SET NULL"), nullable=True
    )
    member_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    props: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=_EMPTY_JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    members: Mapped[list["CommunityMember"]] = relationship(
        back_populates="community", cascade="all, delete-orphan"
    )


class CommunityMember(Base):
    __tablename__ = "community_members"

    community_id: Mapped[Any] = mapped_column(
        UUID(as_uuid=True), ForeignKey("communities.id", ondelete="CASCADE"), primary_key=True
    )
    node_id: Mapped[Any] = mapped_column(
        UUID(as_uuid=True), ForeignKey("nodes.id", ondelete="CASCADE"), primary_key=True
    )

    community: Mapped[Community] = relationship(back_populates="members")
