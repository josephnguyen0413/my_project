import os
import pytest
from messi.train import train

def test_train_runs_one_epoch():
    if not os.path.exists("data/processed/train_images.pt"):
        pytest.skip("Data files not found")
    train(lr=1e-3, batch_size=8, epochs=1)
