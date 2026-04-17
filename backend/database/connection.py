"""
Database Connection & SQLAlchemy Models
Uses SQLite for development, easily switchable to PostgreSQL for production.
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Float, Integer, DateTime, Text, ForeignKey, Boolean, JSON
from datetime import datetime
from typing import Optional, List
import os

# ─── Database URL ──────────────────────────────────────────────────────────────
# SQLite for development. For production PostgreSQL, use:
# DATABASE_URL = "postgresql+asyncpg://user:password@localhost/lung_db"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./lung_disease.db")

engine = create_async_engine(
    DATABASE_URL,
    echo=True,          # Set False in production
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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
    scan_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
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

    # Model results
    model_used: Mapped[str] = mapped_column(String(50), nullable=False)   # "CNN" or "ResNet"
    selected_model: Mapped[str] = mapped_column(String(50), nullable=True) # best performing model

    # CNN Results
    cnn_primary_condition: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    cnn_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cnn_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cnn_all_probabilities: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # ResNet Results
    resnet_primary_condition: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    resnet_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    resnet_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    resnet_all_probabilities: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Final/Selected Result
    final_condition: Mapped[str] = mapped_column(String(100), nullable=False)
    final_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    urgency_level: Mapped[str] = mapped_column(String(20), default="routine")
    alternative_conditions: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    key_findings: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    precautions: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    scan: Mapped["LungScan"] = relationship("LungScan", back_populates="predictions")
    report: Mapped[Optional["Report"]] = relationship("Report", back_populates="prediction", uselist=False)


class ModelMetrics(Base):
    """Store training metrics for each model."""
    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    f1_score: Mapped[float] = mapped_column(Float, nullable=False)
    auc_roc: Mapped[float] = mapped_column(Float, nullable=False)
    training_loss: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    validation_loss: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    confusion_matrix: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    class_names: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    training_samples: Mapped[int] = mapped_column(Integer, nullable=True)
    test_samples: Mapped[int] = mapped_column(Integer, nullable=True)
    epochs: Mapped[int] = mapped_column(Integer, nullable=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="report")


# ─── DB Init & Session ─────────────────────────────────────────────────────────

async def init_db():
    """Initialize database — create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependency: yield an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
