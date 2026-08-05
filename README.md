# Transfer Learning in Nonparametric Regression with Deep ReLU Networks

This repository contains code for the ICML 2026 paper **Transfer Learning in Nonparametric Regression with Deep ReLU Networks**.

Authors: Junpeng Ren, Carlos Misael Madrid Padilla, Yanzhen Chen, and Oscar Hernan Madrid Padilla.

The code provides a PyTorch implementation of the two-stage transfer learning estimator used in the paper, together with simulation settings for reproducing the neural-network experiments.

The algorithm is:

1. train one pooled model using all groups;
2. compute residuals from the pooled model;
3. train one residual model for each group;
4. predict with `pooled_prediction + group_residual_prediction`.

## Use On Your Own Data

```python
from twostage_transdnn import TrainingConfig, TwoStageTransferRegressor

model = TwoStageTransferRegressor(
    input_dim=X_train.shape[1],
    hidden_sizes=(64, 64, 64),
    training=TrainingConfig(
        max_epochs=500,
        patience=10,
        batch_size=256,
    ),
)

model.fit(X_train, y_train, group_train, X_val, y_val, group_val)
y_pred = model.predict(X_test, group_test)
```

Inputs:

```text
X:      shape (n_samples, n_features)
y:      shape (n_samples,)
group:  shape (n_samples,), integer group labels
```

See `examples/custom_data_transfer.py` for a minimal runnable example.

## Repository Layout

```text
.
|-- src/twostage_transdnn/
|   |-- algorithm.py      # reusable two-stage estimator
|   |-- scenarios.py      # simulation settings
|   `-- experiment.py     # simulation runner
|-- scripts/
|   `-- run_experiments.py
|-- examples/
|   `-- custom_data_transfer.py
|-- README.md
|-- pyproject.toml
`-- requirements.txt
```

## Install

The package name is `twostage-transdnn`; import it as `twostage_transdnn`.

```bash
pip install -e .
```

Install a PyTorch build that matches your CPU/GPU environment.

## Run A Simulation

```bash
python scripts/run_experiments.py \
  --scenario nn_sc1_snr5 \
  --dataset-sizes 1000 \
  --seeds 42,43,44 \
  --output-dir outputs
```

Available scenarios:

```text
nn_sc1_snr10, nn_sc1_snr5, nn_sc1_snr2
nn_sc2_snr10, nn_sc2_snr5, nn_sc2_snr2
nn_sc3_std01, nn_sc3_std1
nn_sc4_std01, nn_sc4_std1
nn_extrasn1_snr10, nn_extrasn1_snr5, nn_extrasn1_snr2
nn_extrasn2_snr10, nn_extrasn2_snr5, nn_extrasn2_snr2
nn_extrasn3_snr10, nn_extrasn3_snr5, nn_extrasn3_snr2
nn_extrasn4_snr10, nn_extrasn4_snr5, nn_extrasn4_snr2
```
