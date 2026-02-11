from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributions import Normal


def _searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def rational_quadratic_spline(
    inputs: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivatives: torch.Tensor,
    inverse: bool = False,
    left: float = -3.0,
    right: float = 3.0,
    bottom: float = -3.0,
    top: float = 3.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inside_mask = (inputs >= left) & (inputs <= right)
    inputs = inputs.clamp(left, right)

    num_bins = widths.shape[-1]
    widths = nn.functional.softmax(widths, dim=-1)
    heights = nn.functional.softmax(heights, dim=-1)

    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    derivatives = min_derivative + nn.functional.softplus(derivatives)

    cumwidths = torch.cumsum(widths, dim=-1)
    cumheights = torch.cumsum(heights, dim=-1)
    cumwidths = nn.functional.pad(cumwidths, (1, 0), value=0.0)
    cumheights = nn.functional.pad(cumheights, (1, 0), value=0.0)

    cumwidths = (right - left) * cumwidths + left
    cumheights = (top - bottom) * cumheights + bottom

    widths = (right - left) * widths
    heights = (top - bottom) * heights

    if inverse:
        bin_idx = _searchsorted(cumheights, inputs)
    else:
        bin_idx = _searchsorted(cumwidths, inputs)

    bin_idx = bin_idx.clamp(min=0, max=num_bins - 1)
    gather_shape = bin_idx.unsqueeze(-1)

    input_cumwidths = torch.gather(cumwidths, -1, gather_shape).squeeze(-1)
    input_bin_widths = torch.gather(widths, -1, gather_shape).squeeze(-1)
    input_cumheights = torch.gather(cumheights, -1, gather_shape).squeeze(-1)
    input_bin_heights = torch.gather(heights, -1, gather_shape).squeeze(-1)

    input_delta = input_bin_heights / input_bin_widths

    input_derivatives = torch.gather(derivatives, -1, gather_shape).squeeze(-1)
    input_derivatives_plus_one = torch.gather(
        derivatives, -1, (gather_shape + 1).clamp(max=derivatives.shape[-1] - 1)
    ).squeeze(-1)

    if inverse:
        y = inputs
        a = (y - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        b = input_bin_heights * input_derivatives - (y - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (y - input_cumheights)

        discriminant = b ** 2 - 4 * a * c
        discriminant = torch.clamp(discriminant, min=0.0)
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        theta = root

        x = theta * input_bin_widths + input_cumwidths

        numerator = input_delta ** 2 * (
            input_derivatives_plus_one * theta ** 2
            + 2 * input_delta * theta * (1 - theta)
            + input_derivatives * (1 - theta) ** 2
        )
        denominator = (
            input_delta
            + (input_derivatives_plus_one + input_derivatives - 2 * input_delta) * theta * (1 - theta)
        ) ** 2
        logabsdet = torch.log(numerator) - torch.log(denominator)
        x = torch.where(inside_mask, x, inputs)
        logabsdet = torch.where(inside_mask, logabsdet, torch.zeros_like(logabsdet))
        return x, -logabsdet

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus = theta * (1 - theta)
    numerator = input_bin_heights * (
        input_delta * theta ** 2 + input_derivatives * theta_one_minus
    )
    denominator = input_delta + (
        input_derivatives_plus_one + input_derivatives - 2 * input_delta
    ) * theta_one_minus
    outputs = input_cumheights + numerator / denominator

    derivative_numerator = input_delta ** 2 * (
        input_derivatives_plus_one * theta ** 2
        + 2 * input_delta * theta_one_minus
        + input_derivatives * (1 - theta) ** 2
    )
    derivative_denominator = denominator ** 2
    logabsdet = torch.log(derivative_numerator) - torch.log(derivative_denominator)
    outputs = torch.where(inside_mask, outputs, inputs)
    logabsdet = torch.where(inside_mask, logabsdet, torch.zeros_like(logabsdet))
    return outputs, logabsdet


@dataclass
class FlowConfig:
    hidden_sizes: Iterable[int]
    num_layers: int
    num_bins: int
    tail_bound: float
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_derivative: float = 1e-3
    dropout: float = 0.0


class SplineCouplingLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        mask: torch.Tensor,
        hidden_sizes: Iterable[int],
        num_bins: int,
        tail_bound: float,
        min_bin_width: float,
        min_bin_height: float,
        min_derivative: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.register_buffer("mask", mask)
        self.register_buffer("mask_bool", mask.bool())
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        transform_dim = int((1 - mask).sum().item())
        in_dim = int(mask.sum().item()) + context_dim
        out_dim = transform_dim * (3 * num_bins + 1)

        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def _split_inputs(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        masked = inputs[:, self.mask_bool]
        transform = inputs[:, ~self.mask_bool]
        return masked, transform

    def forward(self, inputs: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        masked, transform = self._split_inputs(inputs)
        net_in = torch.cat([masked, context], dim=-1)
        params = self.net(net_in)

        transform_dim = transform.shape[-1]
        params = params.view(inputs.shape[0], transform_dim, 3 * self.num_bins + 1)

        widths = params[..., : self.num_bins]
        heights = params[..., self.num_bins : 2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins :]

        outputs, logabsdet = rational_quadratic_spline(
            transform,
            widths,
            heights,
            derivatives,
            inverse=False,
            left=-self.tail_bound,
            right=self.tail_bound,
            bottom=-self.tail_bound,
            top=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        out = inputs.clone()
        out[:, self.mask_bool] = masked
        out[:, ~self.mask_bool] = outputs
        return out, logabsdet.sum(dim=-1)

    def inverse(self, inputs: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        masked, transform = self._split_inputs(inputs)
        net_in = torch.cat([masked, context], dim=-1)
        params = self.net(net_in)

        transform_dim = transform.shape[-1]
        params = params.view(inputs.shape[0], transform_dim, 3 * self.num_bins + 1)

        widths = params[..., : self.num_bins]
        heights = params[..., self.num_bins : 2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins :]

        outputs, logabsdet = rational_quadratic_spline(
            transform,
            widths,
            heights,
            derivatives,
            inverse=True,
            left=-self.tail_bound,
            right=self.tail_bound,
            bottom=-self.tail_bound,
            top=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        out = inputs.clone()
        out[:, self.mask_bool] = masked
        out[:, ~self.mask_bool] = outputs
        return out, logabsdet.sum(dim=-1)


class NeuralSplineFlow(nn.Module):
    def __init__(self, theta_dim: int, context_dim: int, config: FlowConfig) -> None:
        super().__init__()
        self.theta_dim = theta_dim
        self.context_dim = context_dim
        self.config = config

        masks = []
        for i in range(config.num_layers):
            mask = torch.zeros(theta_dim)
            mask[i % 2 :: 2] = 1.0
            masks.append(mask)

        self.layers = nn.ModuleList(
            [
                SplineCouplingLayer(
                    input_dim=theta_dim,
                    context_dim=context_dim,
                    mask=mask,
                    hidden_sizes=config.hidden_sizes,
                    num_bins=config.num_bins,
                    tail_bound=config.tail_bound,
                    min_bin_width=config.min_bin_width,
                    min_bin_height=config.min_bin_height,
                    min_derivative=config.min_derivative,
                    dropout=config.dropout,
                )
                for mask in masks
            ]
        )
        self.register_buffer("base_loc", torch.zeros(theta_dim))
        self.register_buffer("base_scale", torch.ones(theta_dim))

    def log_prob(self, theta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        z = theta
        logabsdet_total = torch.zeros(theta.shape[0], device=theta.device)
        for layer in self.layers:
            z, logabsdet = layer.forward(z, context)
            logabsdet_total += logabsdet
        base_dist = Normal(self.base_loc, self.base_scale)
        log_prob = base_dist.log_prob(z).sum(dim=-1) + logabsdet_total
        return log_prob

    def sample(self, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        context = context.expand(num_samples, -1)
        z = torch.randn(num_samples, self.theta_dim, device=context.device)
        x = z
        for layer in reversed(self.layers):
            x, _ = layer.inverse(x, context)
        return x
