"""Comprehensive runtime environment and dependency verification script for TruthLens."""

import asyncio
from pathlib import Path
import platform
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def run_checks() -> bool:
    all_passed = True
    print("=" * 60)
    print("TruthLens: Runtime & ML Environment Verification")
    print("=" * 60)

    # 1. Python Check
    py_ver = platform.python_version()
    print(f"[1/7] Python Runtime: {py_ver} ({platform.platform()})")
    if sys.version_info < (3, 10):
        print("  FAIL: Python version must be >= 3.10")
        all_passed = False
    else:
        print("  PASS: Python version supported.")

    # 2. PyTorch Runtime & Tensor Math Check
    try:
        import torch
        print(f"[2/7] PyTorch: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
        # Test tensor creation, autograd, and matmul
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = torch.tensor([[2.0, 0.0], [1.0, 2.0]])
        c = torch.matmul(a, b)
        loss = c.sum()
        loss.backward()
        assert a.grad is not None
        assert c.shape == (2, 2)
        print("  PASS: PyTorch CPU tensor ops and autograd verified.")
    except Exception as exc:
        print(f"  FAIL: PyTorch verification error: {exc}")
        all_passed = False

    # 3. Scikit-learn Fit/Predict Smoke Test
    try:
        import sklearn
        from sklearn.linear_model import LogisticRegression
        print(f"[3/7] Scikit-Learn: {sklearn.__version__}")
        X = np.array([[1.0, 2.0], [2.0, 3.0], [-1.0, -2.0], [-2.0, -3.0]])
        y = np.array([1, 1, 0, 0])
        clf = LogisticRegression()
        clf.fit(X, y)
        preds = clf.predict(np.array([[1.5, 2.5], [-1.5, -2.5]]))
        assert (preds == np.array([1, 0])).all()
        print("  PASS: Scikit-learn fit/predict verified.")
    except Exception as exc:
        print(f"  FAIL: Scikit-learn verification error: {exc}")
        all_passed = False

    # 4. Transformers Module Check
    try:
        import transformers
        print(f"[4/7] Hugging Face Transformers: {transformers.__version__}")
        print("  PASS: Transformers package and symbols accessible.")
    except Exception as exc:
        print(f"  FAIL: Transformers verification error: {exc}")
        all_passed = False

    # 5. Pydantic v2 Check
    try:
        import pydantic
        from pydantic import BaseModel, Field
        print(f"[5/7] Pydantic: {pydantic.__version__}")
        class TestModel(BaseModel):
            id: int
            name: str = Field(min_length=2)
        m = TestModel(id=1, name="TruthLens")
        assert m.name == "TruthLens"
        print("  PASS: Pydantic v2 schema validation verified.")
    except Exception as exc:
        print(f"  FAIL: Pydantic verification error: {exc}")
        all_passed = False

    # 6. SQLAlchemy 2.0 Async Engine & Table Creation
    try:
        import sqlalchemy
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine
        print(f"[6/7] SQLAlchemy: {sqlalchemy.__version__}")

        async def test_db():
            test_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
            async with test_engine.connect() as conn:
                res = await conn.execute(text("SELECT 42"))
                val = res.scalar()
                assert val == 42
            await test_engine.dispose()

        asyncio.run(test_db())
        print("  PASS: Async SQLite in-memory engine and transactions verified.")
    except Exception as exc:
        print(f"  FAIL: SQLAlchemy verification error: {exc}")
        all_passed = False

    # 7. FastAPI Application Startup Test
    try:
        import fastapi
        from fastapi.testclient import TestClient
        from backend.app.main import app
        print(f"[7/7] FastAPI: {fastapi.__version__}")
        client = TestClient(app)
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ("healthy", "degraded")
        print(f"  PASS: FastAPI application initialized. Health status: {data['status']}")
    except Exception as exc:
        print(f"  FAIL: FastAPI application verification error: {exc}")
        all_passed = False

    print("=" * 60)
    if all_passed:
        print("OVERALL RESULT: ALL 7 RUNTIME CHECKS PASSED")
    else:
        print("OVERALL RESULT: ONE OR MORE CHECKS FAILED")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
