"""Shared test fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import Base, get_db
from main import app
from ml.inference import InferenceEngine


TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(TEST_DB_URL, echo=False)
TestSession = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def override_get_db():
    async with TestSession() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


@pytest_asyncio.fixture
async def db_session():
    async with TestSession() as session:
        yield session


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def remote_client():
    """Client whose requests appear to come from a non-loopback address."""
    transport = ASGITransport(app=app, client=("203.0.113.7", 50000))
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(scope="session", autouse=True)
def _initialize_engine():
    # httpx's ASGITransport does not run the FastAPI lifespan, so do what startup does once per session
    InferenceEngine().initialize()


@pytest.fixture(autouse=True)
def uploads_dir(tmp_path, monkeypatch):
    """Keep uploaded test images out of the working tree."""
    from api.routes import predictions
    d = tmp_path / "uploads"
    d.mkdir()
    monkeypatch.setattr(predictions, "UPLOADS_DIR", d)
    return d


class _StubModel:
    """Stands in for the classifier: returns fixed probabilities for any input."""
    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float32)
        self.calls = 0

    def predict(self, x, verbose=0):
        self.calls += 1
        assert x.shape == (1, 224, 224, 3)
        return self.probs[None, :]


@pytest.fixture
def stub_model():
    """Install a stub classifier on the engine singleton for one test; returns an installer taking 6 probabilities."""
    engine = InferenceEngine()
    saved = (engine.densenet_model, engine.base_grad_model, engine.top_layers, engine.model_sha256, engine.checkpoint_verified)

    def install(probs, sha256="ab" * 32):
        engine.densenet_model = _StubModel(probs)
        engine.base_grad_model = None
        engine.top_layers = None
        engine.model_sha256 = sha256
        engine.checkpoint_verified = False  # a stub is never the canonical checkpoint
        return engine.densenet_model

    yield install
    (engine.densenet_model, engine.base_grad_model, engine.top_layers, engine.model_sha256,
     engine.checkpoint_verified) = saved

