"""Unit tests for ML runtime stability and operations."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from backend.app.services.health_service import verify_ml_runtime


def test_pytorch_tensor_ops():
    """Verify PyTorch tensor math and backward pass."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = torch.tensor([[0.5, -1.0], [2.0, 1.5]])
    z = torch.matmul(x, y)
    assert z.shape == (2, 2)
    assert not torch.isnan(z).any()

    # Backpropagation check
    loss = z.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 2)


def test_sklearn_training_inference():
    """Verify Scikit-learn estimator training and prediction."""
    X = np.array([[0.0, 1.0], [1.0, 1.0], [5.0, 6.0], [6.0, 7.0]])
    y = np.array([0, 0, 1, 1])
    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)
    preds = clf.predict(np.array([[0.5, 0.5], [5.5, 6.5]]))
    assert list(preds) == [0, 1]


def test_verify_ml_runtime_service():
    """Verify health service returns available runtime status."""
    ml_health = verify_ml_runtime()
    assert ml_health.status == "available"
    assert ml_health.tensor_ops == "pass"
    assert ml_health.torch_version is not None
