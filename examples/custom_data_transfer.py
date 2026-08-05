from __future__ import annotations

import numpy as np

from twostage_transdnn import TrainingConfig, TwoStageTransferRegressor


def main() -> None:
    rng = np.random.default_rng(42)
    n, p, num_groups = 600, 10, 3
    groups = rng.integers(1, num_groups + 1, size=n)
    x = rng.normal(size=(n, p))
    group_shift = np.array([0.0, -0.5, 0.75])
    y = np.sin(x[:, 0]) + x[:, 1] ** 2 + group_shift[groups - 1] + rng.normal(0, 0.1, size=n)

    train_end, val_end = 420, 510
    x_train, y_train, g_train = x[:train_end], y[:train_end], groups[:train_end]
    x_val, y_val, g_val = x[train_end:val_end], y[train_end:val_end], groups[train_end:val_end]
    x_test, y_test, g_test = x[val_end:], y[val_end:], groups[val_end:]

    model = TwoStageTransferRegressor(
        input_dim=p,
        hidden_sizes=(32, 32),
        training=TrainingConfig(max_epochs=100, patience=8, batch_size=64),
    )
    model.fit(x_train, y_train, g_train, x_val, y_val, g_val)
    pred = model.predict(x_test, g_test)
    mse = np.mean((y_test - pred) ** 2)
    print(f"mse={mse:.4f}")


if __name__ == "__main__":
    main()
