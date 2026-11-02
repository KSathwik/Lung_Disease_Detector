"""
Database Connection & SQLAlchemy Models
Uses SQLite for development, easily switchable to PostgreSQL for production.
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Float, Integer, DateTime, Text, ForeignKey, Boolean, JSON
from datetime import datetime, timezone
from typing import Optional, List
import os

# ─── Database URL ──────────────────────────────────────────────────────────────
# SQLite for development. For production PostgreSQL, use:
# DATABASE_URL = "postgresql+asyncpg://user:password@localhost/lung_db"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./lung_disease.db")
SQL_ECHO = os.getenv("SQL_ECHO", "false").lower() in ("true", "1", "yes")

engine = create_async_engine(
    DATABASE_URL,
    echo=SQL_ECHO,
    future=True
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


# ─── Base Model ────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ─── ORM Models ───────────────────────────────────────────────────────────────

class Patient(Base):
    """Patient record table."""
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(10), nullable=False)
    contact: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    medical_history: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    scans: Mapped[List["LungScan"]] = relationship("LungScan", back_populates="patient")


class LungScan(Base):
    """Lung scan upload and metadata table."""
    __tablename__ = "lung_scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    patient_id: Mapped[int] = mapped_column(Integer, ForeignKey("patients.id"), nullable=False)
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    image_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    image_size_bytes: Mapped[int] = mapped_column(Integer, nullable=True)
    scan_type: Mapped[str] = mapped_column(String(50), default="X-Ray")  # X-Ray, CT, MRI
    scan_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    preprocessed: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    patient: Mapped["Patient"] = relationship("Patient", back_populates="scans")
    predictions: Mapped[List["Prediction"]] = relationship("Prediction", back_populates="scan")


class Prediction(Base):
    """ML Model prediction results table."""
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    scan_id: Mapped[int] = mapped_column(Integer, ForeignKey("lung_scans.id"), nullable=False)

    # Model identity: model_used = canonical model_id, selected_model = display name, model_version = config version
    model_used: Mapped[str] = mapped_column(String(50), nullable=False)
    selected_model: Mapped[str] = mapped_column(String(100), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Six-class probabilities exactly as returned to the client, and the checkpoint that produced them
    probabilities: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    model_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    inconclusive: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True, default=False)

    # Final/Selected Result
    final_condition: Mapped[str] = mapped_column(String(100), nullable=False)
    final_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    urgency_level: Mapped[str] = mapped_column(String(20), default="routine")
    alternative_conditions: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    key_findings: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    precautions: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    scan: Mapped["LungScan"] = relationship("LungScan", back_populates="predictions")
    report: Mapped[Optional["Report"]] = relationship("Report", back_populates="prediction", uselist=False)


class Report(Base):
    """Medical report generated after prediction."""
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    prediction_id: Mapped[int] = mapped_column(Integer, ForeignKey("predictions.id"), nullable=False)
    report_content: Mapped[str] = mapped_column(Text, nullable=False)
    generated_by: Mapped[str] = mapped_column(String(100), default="AI System")
    reviewed_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    review_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="report")


# ─── DB Init & Session ─────────────────────────────────────────────────────────

# Columns added after the first release; create_all() does not alter existing tables.
_ADDED_COLUMNS = {
    "predictions": {
        "probabilities": "JSON",
        "model_sha256": "VARCHAR(64)",
        "inconclusive": "BOOLEAN",
        "model_version": "VARCHAR(20)",
    },
}


def _add_missing_columns(sync_conn):
    from sqlalchemy import inspect, text
    insp = inspect(sync_conn)
    for table, cols in _ADDED_COLUMNS.items():
        if not insp.has_table(table):
            continue
        existing = {c["name"] for c in insp.get_columns(table)}
        for name, ddl in cols.items():
            if name not in existing:
                sync_conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


async def init_db():
    """Initialize database — create all tables and add columns introduced since the table was created."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)


async def get_db():
    """Dependency: yield an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
