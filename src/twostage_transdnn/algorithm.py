from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


ArrayLike = np.ndarray | torch.Tensor
ModelFactory = Callable[[int], nn.Module]


@dataclass(frozen=True)
class TrainingConfig:
    """Training options for neural-network regressors."""

    batch_size: int = 256
    max_epochs: int = 500
    patience: int = 10
    learning_rate: float = 1e-3
    device: str | torch.device | None = None


class MLPRegressor(nn.Module):
    """Simple fully connected regressor used by the simulation experiments."""

    def __init__(self, input_dim: int, hidden_sizes: Iterable[int] = (64, 64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.extend([nn.Linear(last_dim, int(hidden_dim)), nn.ReLU()])
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_tensor(x: ArrayLike, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.tensor(x, dtype=torch.float32, device=device)


def train_regressor(
    model: nn.Module,
    x_train: ArrayLike,
    y_train: ArrayLike,
    x_val: ArrayLike,
    y_val: ArrayLike,
    config: TrainingConfig,
) -> nn.Module:
    """Train a regressor with early stopping on validation MSE."""

    device = resolve_device(config.device)
    model = model.to(device)
    x_train_t = to_tensor(x_train, device)
    y_train_t = to_tensor(y_train, device).view(-1, 1)
    x_val_t = to_tensor(x_val, device)
    y_val_t = to_tensor(y_val, device).view(-1, 1)

    loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=config.batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    best_loss = np.inf
    patience_counter = 0
    best_state: dict[str, torch.Tensor] | None = None

    for _ in range(config.max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val_t), y_val_t).item()

        if val_loss < best_loss:
            best_loss = val_loss
            # Preserve legacy behavior: keep the state object exactly as PyTorch
            # exposes it instead of deep-copying tensors.
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


class TwoStageTransferRegressor:
    """Pooled model plus group-specific residual models.

    The estimator implements the transfer-learning algorithm used in the
    simulations:

    1. Fit one pooled model on all groups.
    2. Compute residuals from the pooled model.
    3. Fit one residual model per group.
    4. Predict with pooled prediction + group residual prediction.

    It only requires feature matrix `X`, target vector `y`, and group labels.
    Data generation, scenario definitions, and experiment bookkeeping live
    outside this class.
    """

    def __init__(
        self,
        input_dim: int,
        model_factory: ModelFactory | None = None,
        training: TrainingConfig | None = None,
        hidden_sizes: Iterable[int] = (64, 64, 64),
    ):
        self.input_dim = int(input_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.model_factory = model_factory or (lambda dim: MLPRegressor(dim, self.hidden_sizes))
        self.training = training or TrainingConfig()
        self.device = resolve_device(self.training.device)
        self.pooled_model_: nn.Module | None = None
        self.group_models_: dict[int, nn.Module] = {}
        self.groups_: np.ndarray | None = None

    def fit(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        groups_train: ArrayLike,
        x_val: ArrayLike | None = None,
        y_val: ArrayLike | None = None,
        groups_val: ArrayLike | None = None,
    ) -> TwoStageTransferRegressor:
        x_train_np = to_numpy(x_train)
        y_train_np = to_numpy(y_train).reshape(-1)
        groups_train_np = to_numpy(groups_train).astype(int).reshape(-1)
        x_val_np = x_train_np if x_val is None else to_numpy(x_val)
        y_val_np = y_train_np if y_val is None else to_numpy(y_val).reshape(-1)
        groups_val_np = groups_train_np if groups_val is None else to_numpy(groups_val).astype(int).reshape(-1)

        self.groups_ = np.array(sorted(np.unique(groups_train_np).tolist()), dtype=int)
        self.pooled_model_ = train_regressor(
            self.model_factory(self.input_dim),
            x_train_np,
            y_train_np,
            x_val_np,
            y_val_np,
            self.training,
        )

        pooled_train_pred = self.predict_pooled(x_train_np)
        pooled_val_pred = self.predict_pooled(x_val_np)
        train_residuals = y_train_np - pooled_train_pred
        val_residuals = y_val_np - pooled_val_pred

        self.group_models_ = {}
        for group_id in self.groups_:
            train_idx = groups_train_np == group_id
            val_idx = groups_val_np == group_id
            if not np.any(train_idx):
                continue
            if not np.any(val_idx):
                val_idx = train_idx
                x_val_group = x_train_np[val_idx]
                y_val_group = train_residuals[val_idx]
            else:
                x_val_group = x_val_np[val_idx]
                y_val_group = val_residuals[val_idx]
            self.group_models_[int(group_id)] = train_regressor(
                self.model_factory(self.input_dim),
                x_train_np[train_idx],
                train_residuals[train_idx],
                x_val_group,
                y_val_group,
                self.training,
            )
        return self

    def predict_pooled(self, x: ArrayLike) -> np.ndarray:
        if self.pooled_model_ is None:
            raise RuntimeError("Call fit before predict.")
        x_t = to_tensor(x, self.device)
        self.pooled_model_.eval()
        with torch.no_grad():
            return self.pooled_model_(x_t).view(-1).cpu().numpy()

    def predict_residual(self, x: ArrayLike, groups: ArrayLike) -> np.ndarray:
        if not self.group_models_:
            raise RuntimeError("Call fit before predict.")
        x_t = to_tensor(x, self.device)
        groups_np = to_numpy(groups).astype(int).reshape(-1)
        residual = np.zeros(len(groups_np), dtype=float)
        for group_id, model in self.group_models_.items():
            idx = groups_np == group_id
            if not np.any(idx):
                continue
            idx_pos = np.where(idx)[0]
            model.eval()
            with torch.no_grad():
                residual[idx] = model(x_t[idx_pos]).view(-1).cpu().numpy()
        unseen = sorted(set(groups_np.tolist()) - set(self.group_models_))
        if unseen:
            raise ValueError(f"Cannot predict residuals for unseen groups: {unseen}")
        return residual

    def predict(self, x: ArrayLike, groups: ArrayLike) -> np.ndarray:
        return self.predict_pooled(x) + self.predict_residual(x, groups)
