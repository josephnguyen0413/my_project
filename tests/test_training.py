from messi.train import train


def test_train_runs_one_epoch():
    """Basic smoke test: training completes without errors."""
    train(lr=1e-3, batch_size=8, epochs=1)
