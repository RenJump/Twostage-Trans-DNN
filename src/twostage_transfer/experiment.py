from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .scenarios import LOW_DIM_F0, ScenarioConfig


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def set_seed(seed: int, device: torch.device, include_python_random: bool = False) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if include_python_random:
        random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_group_features(z: np.ndarray, num_groups: int) -> np.ndarray:
    group_features = np.zeros((len(z), num_groups))
    for i, group_id in enumerate(z):
        group_features[i, group_id - 1] = 1.0
    return group_features


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.ReLU()])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_dataloader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: MLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    config: ScenarioConfig,
    device: torch.device,
) -> MLP:
    model = model.to(device)
    loader = create_dataloader(x_train, y_train, batch_size=config.batch_size)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    best_loss = np.inf
    patience_counter = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(config.max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), y_val).item()

        if val_loss < best_loss:
            best_loss = val_loss
            # Preserve legacy early-stopping behavior for reproducibility.
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def freeze_layers(model: MLP, layer_indices_to_freeze: tuple[int, ...]) -> None:
    layers = list(model.net.children())
    for idx in layer_indices_to_freeze:
        if idx < len(layers):
            for param in layers[idx].parameters():
                param.requires_grad = False


def copy_and_freeze_model(source_model: MLP, config: ScenarioConfig, device: torch.device) -> MLP:
    new_model = MLP(source_model.net[0].in_features, config.hidden_sizes)
    new_model.load_state_dict(source_model.state_dict())
    new_model = new_model.to(device)
    freeze_layers(new_model, config.freeze_layer_indices)
    return new_model


def generate_low_dim_data(config: ScenarioConfig, n: int) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    p, num_groups = config.input_dim, config.num_groups
    group_probs = np.array([group_id for group_id in range(1, num_groups + 1)])
    group_probs = group_probs / group_probs.sum()
    group_counts = (group_probs * n).astype(int)
    group_counts[np.argmax(group_probs)] += n - group_counts.sum()

    x_grouped: dict[int, np.ndarray] = {}
    y_grouped: dict[int, np.ndarray] = {}
    z_grouped: dict[int, np.ndarray] = {}
    y_clean_grouped: dict[int, np.ndarray] = {}
    f0 = LOW_DIM_F0[config.f0_variant or ""]

    for group_id in range(1, num_groups + 1):
        n_l = group_counts[group_id - 1]
        x_l = np.random.uniform(0, 1, size=(n_l, p))
        y_clean_l = f0(x_l, group_id)
        y_clean_grouped[group_id] = y_clean_l
        x_grouped[group_id] = x_l
        y_grouped[group_id] = y_clean_l.copy()
        z_grouped[group_id] = np.full(n_l, group_id)

    var_y = np.var(np.concatenate(list(y_clean_grouped.values())))
    for group_id in range(1, num_groups + 1):
        y_grouped[group_id] = y_grouped[group_id] + np.random.randn(len(y_grouped[group_id])) * np.sqrt(var_y / float(config.snr))

    return x_grouped, y_grouped, y_clean_grouped, z_grouped


def _group_params(num_groups: int) -> dict[int, dict[str, float]]:
    weight_choices = [1 / k for k in range(2, 11)]
    freq_choices = [1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4, 5]
    shift_choices = [2 * np.pi / k for k in range(1, 7)]
    return {
        group_id: {
            "weight": float(np.random.choice(weight_choices)),
            "freq": float(np.random.choice(freq_choices)),
            "shift": float(np.random.choice(shift_choices)),
        }
        for group_id in range(1, num_groups + 1)
    }


def _latent_transform(z: np.ndarray, group_id: int, w: np.ndarray, b: np.ndarray, v: np.ndarray, group_params: dict[int, dict[str, float]]) -> np.ndarray:
    params = group_params[group_id]
    z_transformed = z.copy()
    for col in range(5, 10):
        z_transformed[:, col] = z[:, col] + params["weight"] * np.sin(params["shift"] + params["freq"] * z[:, col])
    h = np.tanh(z_transformed @ w.T + b)
    return h @ v.T


def generate_latent_100_data(config: ScenarioConfig, n: int) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    num_groups = config.num_groups
    group_counts = np.full(num_groups, n // num_groups)
    group_counts[0] += n - group_counts.sum()
    group_params = _group_params(num_groups)
    w = np.random.normal(0, 1 / np.sqrt(10), (50, 10))
    b = np.random.normal(0, 0.1, 50)
    v = np.random.normal(0, 1 / np.sqrt(50), (config.input_dim, 50))

    x_grouped: dict[int, np.ndarray] = {}
    y_grouped: dict[int, np.ndarray] = {}
    y_clean_grouped: dict[int, np.ndarray] = {}
    z_grouped: dict[int, np.ndarray] = {}
    for group_id in range(1, num_groups + 1):
        n_l = group_counts[group_id - 1]
        z_l = np.random.normal(0, 1, size=(n_l, 10))
        x_l = _latent_transform(z_l, group_id, w, b, v, group_params)
        y_l = np.sum(z_l**2, axis=1)
        x_grouped[group_id] = x_l
        y_clean_grouped[group_id] = y_l.copy()
        y_grouped[group_id] = y_l + np.random.randn(n_l) * float(config.noise_std)
        z_grouped[group_id] = np.full(n_l, group_id)
    return x_grouped, y_grouped, y_clean_grouped, z_grouped


def generate_latent_tail_data(config: ScenarioConfig, n: int) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    num_groups = config.num_groups
    group_counts = np.full(num_groups, n // num_groups)
    group_counts[0] += n - group_counts.sum()
    group_params = _group_params(num_groups)
    freq_choices = [1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4, 5]
    group_func_freqs = {
        group_id: [float(np.random.choice(freq_choices)) for _ in range(3)]
        for group_id in range(1, num_groups + 1)
    }

    w = np.random.normal(0, 1 / np.sqrt(10), (50, 10))
    b = np.random.normal(0, 0.1, 50)
    v_base = np.random.normal(0, 1 / np.sqrt(50), (config.input_dim - 90, 50))
    _v_tail = np.random.normal(0, 1 / np.sqrt(50), (90, 50))

    x_grouped: dict[int, np.ndarray] = {}
    y_grouped: dict[int, np.ndarray] = {}
    y_clean_grouped: dict[int, np.ndarray] = {}
    z_grouped: dict[int, np.ndarray] = {}
    for group_id in range(1, num_groups + 1):
        n_l = group_counts[group_id - 1]
        z_l = np.random.normal(0, 1, size=(n_l, 10))
        z_transformed = z_l.copy()
        h = np.tanh(z_transformed @ w.T + b)
        x_base = h @ v_base.T
        x_tail = np.random.normal(0, 1, (n_l, 90))
        x_l = np.concatenate([x_base, x_tail], axis=1)

        start = config.input_dim - 90 + (group_id - 1) * 3
        f1, f2, f3 = group_func_freqs[group_id]
        y_l = (
            np.sum(z_l**2, axis=1)
            + np.sin(f1 * x_l[:, start])
            + 1 / (1 + np.exp(f2 * x_l[:, start + 1]))
            + np.tanh(f3 * x_l[:, start + 2])
        )
        x_grouped[group_id] = x_l
        y_clean_grouped[group_id] = y_l.copy()
        y_grouped[group_id] = y_l + np.random.normal(0, float(config.noise_std), len(y_l))
        z_grouped[group_id] = np.full(n_l, group_id)
    return x_grouped, y_grouped, y_clean_grouped, z_grouped


def generate_data(config: ScenarioConfig, n: int):
    if config.kind == "low_dim":
        return generate_low_dim_data(config, n)
    if config.kind == "latent_100":
        return generate_latent_100_data(config, n)
    if config.kind == "latent_tail":
        return generate_latent_tail_data(config, n)
    raise ValueError(f"Unsupported scenario kind: {config.kind}")


def split_grouped_data(x_grouped, y_grouped, y_clean_grouped, z_grouped, num_groups: int) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {
        "x_train": [],
        "x_val": [],
        "x_test": [],
        "y_train": [],
        "y_val": [],
        "y_test": [],
        "y_clean_train": [],
        "y_clean_val": [],
        "y_clean_test": [],
        "z_train": [],
        "z_val": [],
        "z_test": [],
    }
    for group_id in range(1, num_groups + 1):
        x_l, y_l, y_clean_l, z_l = x_grouped[group_id], y_grouped[group_id], y_clean_grouped[group_id], z_grouped[group_id]
        n_l = len(x_l)
        n_train, n_val = int(n_l * 0.7), int(n_l * 0.15)
        train_end, val_end = n_train, n_train + n_val
        parts["x_train"].append(x_l[:train_end])
        parts["x_val"].append(x_l[train_end:val_end])
        parts["x_test"].append(x_l[val_end:])
        parts["y_train"].append(y_l[:train_end])
        parts["y_val"].append(y_l[train_end:val_end])
        parts["y_test"].append(y_l[val_end:])
        parts["y_clean_train"].append(y_clean_l[:train_end])
        parts["y_clean_val"].append(y_clean_l[train_end:val_end])
        parts["y_clean_test"].append(y_clean_l[val_end:])
        parts["z_train"].append(z_l[:train_end])
        parts["z_val"].append(z_l[train_end:val_end])
        parts["z_test"].append(z_l[val_end:])
    return {
        key: np.vstack(value) if key.startswith("x_") else np.hstack(value)
        for key, value in parts.items()
    }


def run_one_experiment(config: ScenarioConfig, n: int, seed: int, device: torch.device | None = None) -> dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed, device, include_python_random=config.kind == "latent_tail")

    data = split_grouped_data(*generate_data(config, n), num_groups=config.num_groups)
    x_train, x_val, x_test = data["x_train"], data["x_val"], data["x_test"]
    y_train, y_val = data["y_train"], data["y_val"]
    y_clean_test = data["y_clean_test"]
    z_train, z_val, z_test = data["z_train"], data["z_val"], data["z_test"]

    group_train = create_group_features(z_train, config.num_groups)
    group_val = create_group_features(z_val, config.num_groups)
    group_test = create_group_features(z_test, config.num_groups)

    x_train_with_group = np.hstack([x_train, group_train])
    x_val_with_group = np.hstack([x_val, group_val])
    x_test_with_group = np.hstack([x_test, group_test])

    def to_tensor(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32).to(device)

    x_train_t = to_tensor(x_train)
    y_train_t = to_tensor(y_train).view(-1, 1)
    x_val_t = to_tensor(x_val)
    y_val_t = to_tensor(y_val).view(-1, 1)
    x_test_t = to_tensor(x_test)
    x_train_group_t = to_tensor(x_train_with_group)
    x_val_group_t = to_tensor(x_val_with_group)
    x_test_group_t = to_tensor(x_test_with_group)

    model_f0 = train_model(MLP(config.input_dim, config.hidden_sizes), x_train_t, y_train_t, x_val_t, y_val_t, config, device)
    model_f0_with_group = train_model(
        MLP(config.input_dim + config.num_groups, config.hidden_sizes),
        x_train_group_t,
        y_train_t,
        x_val_group_t,
        y_val_t,
        config,
        device,
    )

    with torch.no_grad():
        f0_train_pred = model_f0(x_train_t).view(-1).cpu().numpy()
        val_f0_pred = model_f0(x_val_t).view(-1).detach().cpu().numpy()
    residuals = y_train - f0_train_pred
    val_residuals = y_val - val_f0_pred

    group_models: dict[int, MLP] = {}
    for group_id in range(1, config.num_groups + 1):
        train_idx = np.where(z_train == group_id)[0]
        val_idx = z_val == group_id
        group_models[group_id] = train_model(
            MLP(config.input_dim, config.hidden_sizes),
            to_tensor(x_train[train_idx]),
            to_tensor(residuals[train_idx]).view(-1, 1),
            to_tensor(x_val[val_idx]),
            to_tensor(val_residuals[val_idx]).view(-1, 1),
            config,
            device,
        )

    group_only_models: dict[int, MLP] = {}
    for group_id in range(1, config.num_groups + 1):
        train_idx = np.where(z_train == group_id)[0]
        val_idx = z_val == group_id
        group_only_models[group_id] = train_model(
            MLP(config.input_dim, config.hidden_sizes),
            to_tensor(x_train[train_idx]),
            to_tensor(y_train[train_idx]).view(-1, 1),
            to_tensor(x_val[val_idx]),
            to_tensor(y_val[val_idx]).view(-1, 1),
            config,
            device,
        )

    frozen_finetune_models: dict[int, MLP] = {}
    for group_id in range(1, config.num_groups + 1):
        train_idx = np.where(z_train == group_id)[0]
        val_idx = z_val == group_id
        frozen_model = copy_and_freeze_model(model_f0, config, device)
        frozen_finetune_models[group_id] = train_model(
            frozen_model,
            to_tensor(x_train[train_idx]),
            to_tensor(y_train[train_idx]).view(-1, 1),
            to_tensor(x_val[val_idx]),
            to_tensor(y_val[val_idx]).view(-1, 1),
            config,
            device,
        )

    with torch.no_grad():
        pred_f0 = model_f0(x_test_t).view(-1).cpu().numpy()
        pred_f0_with_group = model_f0_with_group(x_test_group_t).view(-1).cpu().numpy()
        pred_f0_plus = np.array(
            [
                model_f0(x_test_t[i : i + 1]).cpu().item()
                + group_models[z_test[i]](x_test_t[i : i + 1]).cpu().item()
                for i in range(len(x_test))
            ]
        )
        pred_group_only = np.array(
            [group_only_models[z_test[i]](x_test_t[i : i + 1]).cpu().item() for i in range(len(x_test))]
        )
        pred_frozen = np.array(
            [frozen_finetune_models[z_test[i]](x_test_t[i : i + 1]).cpu().item() for i in range(len(x_test))]
        )

    mse_f0 = [mean_squared_error(y_clean_test, pred_f0)]
    mse_f0_with_group = [mean_squared_error(y_clean_test, pred_f0_with_group)]
    mse_f0_plus_group = []
    mse_group_only = [mean_squared_error(y_clean_test, pred_group_only)]
    mse_frozen = [mean_squared_error(y_clean_test, pred_frozen)]
    for group_id in range(1, config.num_groups + 1):
        idx = z_test == group_id
        mse_f0.append(mean_squared_error(y_clean_test[idx], pred_f0[idx]))
        mse_f0_with_group.append(mean_squared_error(y_clean_test[idx], pred_f0_with_group[idx]))
        mse_f0_plus_group.append(mean_squared_error(y_clean_test[idx], pred_f0_plus[idx]))
        mse_group_only.append(mean_squared_error(y_clean_test[idx], pred_group_only[idx]))
        mse_frozen.append(mean_squared_error(y_clean_test[idx], pred_frozen[idx]))

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "f0": mse_f0,
        "f0_with_group": mse_f0_with_group,
        "f0_plus_collective": mean_squared_error(y_clean_test, pred_f0_plus),
        "f0_plus_group": mse_f0_plus_group,
        "group_only": mse_group_only,
        "froz_finetune": mse_frozen,
    }


def run_many(config: ScenarioConfig, dataset_sizes: list[int], seeds: list[int], device: torch.device | None = None) -> dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[int, dict[str, Any]] = {}
    all_seeds: dict[int, dict[int, dict[str, Any]]] = {}

    for n in dataset_sizes:
        print(f"\n==== Running {config.name} for dataset size: {n} ====")
        seed_results: dict[int, dict[str, Any]] = {}
        for seed in seeds:
            print(f"  Seed: {seed}", flush=True)
            seed_results[seed] = run_one_experiment(config, n, seed, device=device)

        all_seeds[n] = seed_results
        results[n] = {
            "f0": np.mean([r["f0"] for r in seed_results.values()], axis=0),
            "f0_with_group": np.mean([r["f0_with_group"] for r in seed_results.values()], axis=0),
            "f0_plus_collective": float(np.mean([r["f0_plus_collective"] for r in seed_results.values()])),
            "f0_plus_group": np.mean([r["f0_plus_group"] for r in seed_results.values()], axis=0),
            "group_only": np.mean([r["group_only"] for r in seed_results.values()], axis=0),
            "froz_finetune": np.mean([r["froz_finetune"] for r in seed_results.values()], axis=0),
        }
        print(f"  f0 only overall      : {results[n]['f0'][0]:.4f}")
        print(f"  2-stage overall      : {results[n]['f0_plus_collective']:.4f}")
        print(f"  group-only overall   : {results[n]['group_only'][0]:.4f}")
        print(f"  frozen-ft overall    : {results[n]['froz_finetune'][0]:.4f}")
        print(f"  f0+group overall     : {results[n]['f0_with_group'][0]:.4f}")

    return {"averaged_results": results, "all_seeds_results": all_seeds, "device": str(device)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def save_results(config: ScenarioConfig, run_output: dict[str, Any], dataset_sizes: list[int], seeds: list[int], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_data = []
    averaged = run_output["averaged_results"]
    for n in dataset_sizes:
        row = {
            "dataset_size": n,
            "f0_overall": averaged[n]["f0"][0],
            "f0_with_group_overall": averaged[n]["f0_with_group"][0],
            "2stage_overall": averaged[n]["f0_plus_collective"],
            "group_only_overall": averaged[n]["group_only"][0],
            "froz_finetune_overall": averaged[n]["froz_finetune"][0],
        }
        for idx in range(config.summary_groups):
            row[f"f0_group{idx + 1}"] = averaged[n]["f0"][idx + 1]
            row[f"f0_with_group_group{idx + 1}"] = averaged[n]["f0_with_group"][idx + 1]
            row[f"2stage_group{idx + 1}"] = averaged[n]["f0_plus_group"][idx]
            row[f"group_only_group{idx + 1}"] = averaged[n]["group_only"][idx + 1]
            row[f"froz_finetune_group{idx + 1}"] = averaged[n]["froz_finetune"][idx + 1]
        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)
    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "device": run_output["device"],
            "scenario": config.name,
            "dataset_sizes": dataset_sizes,
            "seeds": seeds,
            "num_groups": config.num_groups,
            "learning_rate": config.learning_rate,
        },
        "averaged_results": averaged,
        "all_seeds_results": run_output["all_seeds_results"],
    }

    json_path = output_dir / f"{config.output_prefix}_training_results_full_repro_{timestamp}.json"
    csv_path = output_dir / f"{config.output_prefix}_training_results_summary_repro_{timestamp}.csv"
    json_path.write_text(json.dumps(_jsonable(output_data), indent=2), encoding="utf-8")
    df_summary.to_csv(csv_path, index=False)
    return json_path, csv_path
