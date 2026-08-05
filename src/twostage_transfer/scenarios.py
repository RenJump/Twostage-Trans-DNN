from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

ScenarioKind = Literal["low_dim", "latent_100", "latent_tail"]


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    kind: ScenarioKind
    output_prefix: str
    dataset_sizes: tuple[int, ...]
    num_groups: int
    input_dim: int
    hidden_sizes: tuple[int, ...]
    max_epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    freeze_layer_indices: tuple[int, ...]
    summary_groups: int
    snr: float | None = None
    noise_std: float | None = None
    f0_variant: str | None = None


def f0_sc1(x: np.ndarray, group_id: int) -> np.ndarray:
    q = x
    q_k = x[:, group_id - 1]
    phi1 = np.sum(q**2, axis=1)
    phi2 = np.sum(np.abs(q), axis=1)
    phi3 = np.sum(q[:, :5] ** 2, axis=1) - q_k**2
    phi4 = np.sum(np.abs(q[:, :5]), axis=1) - np.abs(q_k)
    return np.log1p(phi1 * phi2) + np.sqrt(1 + np.abs(phi3 + phi4))


def f0_sc2(x: np.ndarray, group_id: int) -> np.ndarray:
    q = x
    q_k = x[:, group_id - 1]
    phi1 = np.sum(q**2, axis=1)
    phi2 = np.sum(np.abs(q), axis=1)
    phi3 = np.sum(q[:, :5] ** 2, axis=1) - q_k**2
    phi4 = np.sum(np.abs(q[:, :5]), axis=1) - np.abs(q_k)
    return np.sqrt((phi1 * phi2) + (phi3 * phi4)) * ((phi1 + phi3) ** 2 / (phi2 + phi4) ** 2)


def _split_10(x: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(np.split(x, 10, axis=1))


def _group_weight(group_id: int, num_groups: int = 5) -> float:
    return (num_groups - group_id + 1) / (num_groups * (num_groups + 1) / 2)


def _selective_sine_tail(x: np.ndarray, group_id: int, multiply_by_q: bool = False) -> tuple[np.ndarray, ...]:
    q = list(_split_10(x))
    weight = _group_weight(group_id)
    active_tail_indices = {
        1: (7, 8, 9),
        2: (5, 8, 9),
        3: (5, 6, 9),
        4: (5, 6, 7),
        5: (6, 7, 8),
    }[group_id]
    for idx in active_tail_indices:
        base = q[idx]
        perturbation = base * np.sin(group_id * base) if multiply_by_q else np.sin(group_id * base)
        q[idx] = base + weight * perturbation
    return tuple(q)


def f0_extrasn1(x: np.ndarray, group_id: int) -> np.ndarray:
    q1, q2, q3, q4, q5, q6, q7, q8, q9, q10 = _selective_sine_tail(x, group_id)
    terms = (
        np.sin(np.tanh(q1**2))
        + np.log1p(np.abs(q2))
        + np.exp(-np.sqrt(1 + q3**2))
        + np.cos(np.log1p(q4**2))
        - np.sin(np.tanh(np.abs(q5)))
        + np.cos(np.sqrt(1 + q6**2))
        - np.log1p(np.tanh(q7**2))
        - np.sin(np.log1p(q8**2))
        + np.cos(np.tanh(q9**2))
        - np.exp(-np.sqrt(1 + q10**2))
    )
    return terms.squeeze()


def f0_extrasn2(x: np.ndarray, group_id: int) -> np.ndarray:
    q1, q2, q3, q4, q5, q6, q7, q8, q9, q10 = _selective_sine_tail(x, group_id, multiply_by_q=True)
    weight = _group_weight(group_id)
    shared_terms = (
        np.sin(np.tanh(q1**2))
        + np.log1p(np.abs(q2))
        + np.exp(-np.sqrt(1 + q3**2))
        + np.cos(np.log1p(q4**2))
        - np.sin(np.tanh(np.abs(q5)))
    )
    tail_terms = (
        np.cos(np.sqrt(1 + q6**2))
        - np.log1p(np.tanh(q7**2))
        - np.sin(np.log1p(q8**2))
        + np.cos(np.tanh(q9**2))
        - np.exp(-np.sqrt(1 + q10**2))
    )
    return (shared_terms + (1 + weight) * tail_terms).squeeze()


def _hierarchical_features(q: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q1, q2, q3, q4, q5, q6, q7, q8, q9, q10 = q
    h1_1 = np.sin(q1) * q2**2 + np.exp(q3) - q4 * q5
    h1_2 = np.cos(q2) + q3 * np.tanh(q4) + q5**3
    h1_3 = np.log(1 + q6 * q7 - np.sin(q8)) - np.tanh(q9 * q10)
    h1_4 = np.exp(-np.abs(q8)) + q9 / (1 + q10**2)
    return h1_1, h1_2, h1_3, h1_4


def f0_extrasn3(x: np.ndarray, group_id: int) -> np.ndarray:
    h1_1, h1_2, h1_3, h1_4 = _hierarchical_features(_selective_sine_tail(x, group_id))
    h1 = np.concatenate([h1_1, h1_2, h1_3, h1_4], axis=1)
    return np.sqrt(1 + h1[:, 0] ** 2) + np.sqrt(1 + h1[:, 1] ** 2) + np.abs(h1[:, 2] - h1[:, 3])


def f0_extrasn4(x: np.ndarray, group_id: int) -> np.ndarray:
    q = list(_split_10(x))
    weight = _group_weight(group_id)
    for idx in (1, 3, 5, 7, 9):
        q[idx] = q[idx] + weight * np.sin(group_id * q[idx])
    h1_1, h1_2, h1_3, h1_4 = _hierarchical_features(tuple(q))
    h1 = np.concatenate([h1_1, h1_2, h1_3, h1_4], axis=1)
    return np.sqrt(np.abs(h1[:, 0] * h1[:, 2]) + np.abs(h1[:, 1] + h1[:, 3]) ** 2)


LOW_DIM_F0: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "sc1": f0_sc1,
    "sc2": f0_sc2,
    "extrasn1": f0_extrasn1,
    "extrasn2": f0_extrasn2,
    "extrasn3": f0_extrasn3,
    "extrasn4": f0_extrasn4,
}


def _low_dim(name: str, snr: float, prefix: str, f0_variant: str) -> ScenarioConfig:
    return ScenarioConfig(
        name=name,
        kind="low_dim",
        output_prefix=prefix,
        dataset_sizes=(1000, 3000, 5000, 10000, 30000, 50000),
        num_groups=5,
        input_dim=10,
        hidden_sizes=(64, 64, 64),
        max_epochs=500,
        patience=10,
        batch_size=256,
        learning_rate=1e-3,
        freeze_layer_indices=(0, 2, 4),
        summary_groups=5,
        snr=snr,
        f0_variant=f0_variant,
    )


SCENARIOS: dict[str, ScenarioConfig] = {
    "nn_sc1_snr10": _low_dim("nn_sc1_snr10", 10, "sn1_snr10", "sc1"),
    "nn_sc1_snr5": _low_dim("nn_sc1_snr5", 5, "sn1_snr5", "sc1"),
    "nn_sc1_snr2": _low_dim("nn_sc1_snr2", 2, "sn1_snr2", "sc1"),
    "nn_sc2_snr10": _low_dim("nn_sc2_snr10", 10, "sn2_snr10", "sc2"),
    "nn_sc2_snr5": _low_dim("nn_sc2_snr5", 5, "sn2_snr5", "sc2"),
    "nn_sc2_snr2": _low_dim("nn_sc2_snr2", 2, "sn2_snr2", "sc2"),
    "nn_extrasn1_snr10": _low_dim("nn_extrasn1_snr10", 10, "extrasn1_snr10", "extrasn1"),
    "nn_extrasn1_snr5": _low_dim("nn_extrasn1_snr5", 5, "extrasn1_snr5", "extrasn1"),
    "nn_extrasn1_snr2": _low_dim("nn_extrasn1_snr2", 2, "extrasn1_snr2", "extrasn1"),
    "nn_extrasn2_snr10": _low_dim("nn_extrasn2_snr10", 10, "extrasn2_snr10", "extrasn2"),
    "nn_extrasn2_snr5": _low_dim("nn_extrasn2_snr5", 5, "extrasn2_snr5", "extrasn2"),
    "nn_extrasn2_snr2": _low_dim("nn_extrasn2_snr2", 2, "extrasn2_snr2", "extrasn2"),
    "nn_extrasn3_snr10": _low_dim("nn_extrasn3_snr10", 10, "extrasn3_snr10", "extrasn3"),
    "nn_extrasn3_snr5": _low_dim("nn_extrasn3_snr5", 5, "extrasn3_snr5", "extrasn3"),
    "nn_extrasn3_snr2": _low_dim("nn_extrasn3_snr2", 2, "extrasn3_snr2", "extrasn3"),
    "nn_extrasn4_snr10": _low_dim("nn_extrasn4_snr10", 10, "extrasn4_snr10", "extrasn4"),
    "nn_extrasn4_snr5": _low_dim("nn_extrasn4_snr5", 5, "extrasn4_snr5", "extrasn4"),
    "nn_extrasn4_snr2": _low_dim("nn_extrasn4_snr2", 2, "extrasn4_snr2", "extrasn4"),
    "nn_sc3_std01": ScenarioConfig(
        name="nn_sc3_std01",
        kind="latent_100",
        output_prefix="sn3_std01",
        dataset_sizes=(5000, 10000, 30000, 50000),
        num_groups=30,
        input_dim=100,
        hidden_sizes=(128, 128, 128, 128),
        max_epochs=2000,
        patience=10,
        batch_size=256,
        learning_rate=1e-3,
        freeze_layer_indices=(0, 2, 4, 6),
        summary_groups=5,
        noise_std=0.1,
    ),
    "nn_sc3_std1": ScenarioConfig(
        name="nn_sc3_std1",
        kind="latent_100",
        output_prefix="sn3_std1",
        dataset_sizes=(5000, 10000, 30000, 50000),
        num_groups=30,
        input_dim=100,
        hidden_sizes=(128, 128, 128, 128),
        max_epochs=2000,
        patience=10,
        batch_size=256,
        learning_rate=1e-3,
        freeze_layer_indices=(0, 2, 4, 6),
        summary_groups=5,
        noise_std=1.0,
    ),
    "nn_sc4_std01": ScenarioConfig(
        name="nn_sc4_std01",
        kind="latent_tail",
        output_prefix="sn4_std01",
        dataset_sizes=(5000, 10000, 30000, 50000),
        num_groups=30,
        input_dim=500,
        hidden_sizes=(256, 256, 256, 256, 256, 256, 256),
        max_epochs=2000,
        patience=10,
        batch_size=256,
        learning_rate=1e-3,
        freeze_layer_indices=(0, 2, 4, 6, 8, 10, 12),
        summary_groups=5,
        noise_std=0.1,
    ),
    "nn_sc4_std1": ScenarioConfig(
        name="nn_sc4_std1",
        kind="latent_tail",
        output_prefix="sn4_std1",
        dataset_sizes=(5000, 10000, 30000, 50000),
        num_groups=30,
        input_dim=500,
        hidden_sizes=(256, 256, 256, 256, 256, 256, 256),
        max_epochs=2000,
        patience=10,
        batch_size=256,
        learning_rate=1e-3,
        freeze_layer_indices=(0, 2, 4, 6, 8, 10, 12),
        summary_groups=5,
        noise_std=1.0,
    ),
}


def get_scenario(name: str) -> ScenarioConfig:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"Unknown scenario {name!r}. Valid scenarios: {valid}") from exc
