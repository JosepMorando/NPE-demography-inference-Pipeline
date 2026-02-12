from __future__ import annotations

from typing import Any, Dict

import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform, posterior_nn


def _build_density_estimator_builder(flow_config: Dict[str, Any]):
    hidden_sizes = list(flow_config.get("hidden_sizes", [128, 128]))
    hidden_features = int(hidden_sizes[0]) if hidden_sizes else 128
    return posterior_nn(
        model="nsf",
        hidden_features=hidden_features,
        num_transforms=int(flow_config.get("num_layers", 4)),
        num_bins=int(flow_config.get("num_bins", 8)),
        dropout_probability=float(flow_config.get("dropout", 0.0)),
        z_score_theta="none",
        z_score_x="none",
    )


def build_sbi_prior(theta_dim: int, bounds: tuple[float, float] = (-10.0, 10.0)) -> BoxUniform:
    low = torch.full((theta_dim,), float(bounds[0]))
    high = torch.full((theta_dim,), float(bounds[1]))
    return BoxUniform(low=low, high=high)


def train_npe_sbi(
    theta: torch.Tensor,
    x: torch.Tensor,
    flow_config: Dict[str, Any],
    train_config: Dict[str, Any],
    device: str,
):
    prior = build_sbi_prior(theta.shape[1])
    density_estimator_builder = _build_density_estimator_builder(flow_config)
    inference = NPE(prior=prior, density_estimator=density_estimator_builder, device=device)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train(
        training_batch_size=int(train_config.get("batch_size", 64)),
        learning_rate=float(train_config.get("lr", 5e-4)),
        max_num_epochs=int(train_config.get("max_epochs", 200)),
        validation_fraction=float(train_config.get("val_frac", 0.1)),
        stop_after_epochs=int(train_config.get("early_stop_patience", 25)),
        clip_max_norm=5.0,
        show_train_summary=True,
    )
    posterior = inference.build_posterior(density_estimator)
    return posterior, density_estimator


def load_sbi_posterior(ckpt: Dict[str, Any], device: torch.device):
    theta_dim = int(ckpt["theta_dim"])
    x_dim = int(ckpt["x_dim"])
    flow_config = ckpt.get("flow_config", {})
    prior_bounds = tuple(ckpt.get("prior_bounds", (-10.0, 10.0)))

    prior = build_sbi_prior(theta_dim, bounds=(float(prior_bounds[0]), float(prior_bounds[1])))
    density_estimator_builder = _build_density_estimator_builder(flow_config)
    density_estimator = density_estimator_builder(
        torch.zeros(2, theta_dim, dtype=torch.float32),
        torch.zeros(2, x_dim, dtype=torch.float32),
    ).to(device)
    density_estimator.load_state_dict(ckpt["state_dict"])
    density_estimator.eval()

    inference = NPE(prior=prior, density_estimator=density_estimator_builder, device=str(device))
    posterior = inference.build_posterior(density_estimator)
    return posterior

