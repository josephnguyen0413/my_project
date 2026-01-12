import torch
import os.path
import pytest
from messi.data import corrupt_mnist, normalize

@pytest.mark.skipif(
    not os.path.exists("data/processed/train_images.pt"), 
    reason="Data files not found"
)
def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000, "Dataset did not have the correct number of samples"
    assert len(test) == 5000, "Dataset did not have the correct number of samples"
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()

def test_normalize():
    """Test the normalize function."""
    images = torch.randn(10, 1, 28, 28)
    normalized = normalize(images)
    # After normalization, mean should be ~0 and std should be ~1
    assert torch.isclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(normalized.std(), torch.tensor(1.0), atol=1e-6)
