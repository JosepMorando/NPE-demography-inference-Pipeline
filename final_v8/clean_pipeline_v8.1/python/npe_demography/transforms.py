"""Parameter transformations for constrained parameters.

This module handles transformations between biological (constrained) and
unconstrained spaces for neural density estimation. By working in unconstrained
space, the NSF can learn flexible posteriors without violating
constraints, and then we transform back to biological parameters.

Transformations:
  - Positive-only (Ne, gap-encoded T): log transform
  - Fractions [0,1]: logit transform
  - Durations: log transform (if positive-only)
  - Times: cumulative gap parameterization to enforce ordering constraints
"""
from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np

TIME_GAP_DEPENDENCIES = {
    "T_BG01": None,
    "T_CORE": "T_BG01",
    "T_SOUTH_LOW": "T_CORE",
    "T_EAST": "T_SOUTH_LOW",
    "T_SOUTH_MID": "T_CORE",
    "T_INT": "T_CORE",
    "T_CENTRAL": "T_INT",
    "T_PYRENEES": "T_INT",
}

TIME_GAP_ORDER = [
    "T_BG01",
    "T_CORE",
    "T_SOUTH_LOW",
    "T_EAST",
    "T_SOUTH_MID",
    "T_INT",
    "T_CENTRAL",
    "T_PYRENEES",
]


def _get_time_gap_parent(key: str, available: set[str]) -> str | None:
    parent = TIME_GAP_DEPENDENCIES.get(key)
    if parent in available:
        return parent
    return None


# Small epsilon to avoid log(0) and division by zero
EPS = 1e-8


def transform_to_unconstrained(params: Dict[str, Any], theta_keys: Tuple[str, ...]) -> Dict[str, float]:
    """Transform biological parameters to unconstrained space.
    
    This transformation ensures the NSF learns in a space where:
    - All real values are valid (no positivity constraints)
    - Gaussian distributions make sense
    - Standardization doesn't break constraints
    
    Parameters
    ----------
    params : dict
        Biological parameters (constrained)
    theta_keys : tuple
        Parameter names to transform
        
    Returns
    -------
    dict
        Transformed parameters (unconstrained)
    """
    unconstrained = {}
    
    for key in theta_keys:
        if key not in params:
            raise KeyError(f"Parameter '{key}' not found in params dict")
        
        value = float(params[key])
        
        # Population sizes: log transform (must be positive)
        if key.startswith('N_'):
            if value <= 0:
                raise ValueError(f"{key}={value} must be positive for log transform")
            unconstrained[key] = np.log(value)
        
        # Times: cumulative gap parameterization (must respect ordering)
        elif key.startswith('T_'):
            available = set(theta_keys)
            parent = _get_time_gap_parent(key, available)
            base = float(params[parent]) if parent is not None else 0.0
            gap = float(value) - base
            if gap <= 0:
                raise ValueError(f"{key}={value} must be > parent time {base} for gap transform")
            unconstrained[key] = np.log(gap)
        
        # Fractions [0,1]: logit transform
        # CRITICAL: Match ANY key containing '_FRAC' to catch per-population params
        # (e.g., BN_TIME_FRAC, BN_SIZE_FRAC, BN_TIME_FRAC_BG01, etc.)
        elif '_FRAC' in key:
            # Clip to [eps, 1-eps] to avoid log(0)
            p = np.clip(value, EPS, 1.0 - EPS)
            # logit(p) = log(p / (1-p))
            unconstrained[key] = np.log(p / (1.0 - p))
        
        # Durations: log transform if positive
        # CRITICAL: Match BN_DUR OR BN_DUR_* to catch per-population params
        elif key == 'BN_DUR' or key.startswith('BN_DUR_'):
            if value <= 0:
                raise ValueError(f"{key}={value} must be positive for log transform")
            unconstrained[key] = np.log(value)
        
        # Other parameters: pass through unchanged
        else:
            unconstrained[key] = value
    
    return unconstrained


def inverse_transform_from_unconstrained(unconstrained: Dict[str, float],
                                        theta_keys: Tuple[str, ...]) -> Dict[str, float]:
    """Transform unconstrained parameters back to biological space.
    
    Parameters
    ----------
    unconstrained : dict
        Transformed parameters (unconstrained)
    theta_keys : tuple
        Parameter names to inverse transform
        
    Returns
    -------
    dict
        Biological parameters (constrained)
    """
    biological = {}
    
    for key in theta_keys:
        if key not in unconstrained:
            raise KeyError(f"Parameter '{key}' not found in unconstrained dict")
        
        value = float(unconstrained[key])
        
        # Population sizes: exp transform
        if key.startswith('N_'):
            biological[key] = np.exp(value)
        
        # Times: exp transform (gap space)
        elif key.startswith('T_'):
            biological[key] = np.exp(value)
        
        # Fractions: inverse logit (sigmoid)
        # CRITICAL: Match ANY key containing '_FRAC'
        elif '_FRAC' in key:
            # sigmoid(x) = 1 / (1 + exp(-x))
            biological[key] = 1.0 / (1.0 + np.exp(-value))
        
        # Durations: exp transform
        # CRITICAL: Match BN_DUR OR BN_DUR_*
        elif key == 'BN_DUR' or key.startswith('BN_DUR_'):
            biological[key] = np.exp(value)
        
        # Other parameters: pass through unchanged
        else:
            biological[key] = value
    
    available = set(theta_keys)
    for key in TIME_GAP_ORDER:
        if key not in available:
            continue
        parent = _get_time_gap_parent(key, available)
        if parent is None:
            continue
        biological[key] = biological[parent] + biological[key]

    return biological


def transform_theta_vector(theta: np.ndarray, theta_keys: Tuple[str, ...]) -> np.ndarray:
    """Transform a theta vector (or batch of vectors) to unconstrained space.
    
    Parameters
    ----------
    theta : np.ndarray
        Theta vector(s) in biological space. Shape (n_params,) or (n_samples, n_params)
    theta_keys : tuple
        Parameter names corresponding to theta dimensions
        
    Returns
    -------
    np.ndarray
        Theta vector(s) in unconstrained space
    """
    if theta.ndim == 1:
        # Single vector
        params_dict = {k: theta[i] for i, k in enumerate(theta_keys)}
        unconstrained = transform_to_unconstrained(params_dict, theta_keys)
        return np.array([unconstrained[k] for k in theta_keys], dtype=np.float32)
    
    elif theta.ndim == 2:
        # Batch of vectors
        n_samples = theta.shape[0]
        result = np.zeros_like(theta)
        for i in range(n_samples):
            params_dict = {k: theta[i, j] for j, k in enumerate(theta_keys)}
            unconstrained = transform_to_unconstrained(params_dict, theta_keys)
            result[i] = np.array([unconstrained[k] for k in theta_keys], dtype=np.float32)
        return result
    
    else:
        raise ValueError(f"theta must be 1D or 2D, got shape {theta.shape}")


def inverse_transform_theta_vector(theta_unconstrained: np.ndarray, 
                                   theta_keys: Tuple[str, ...]) -> np.ndarray:
    """Transform theta vector(s) from unconstrained back to biological space.
    
    Parameters
    ----------
    theta_unconstrained : np.ndarray
        Theta vector(s) in unconstrained space. Shape (n_params,) or (n_samples, n_params)
    theta_keys : tuple
        Parameter names corresponding to theta dimensions
        
    Returns
    -------
    np.ndarray
        Theta vector(s) in biological space
    """
    if theta_unconstrained.ndim == 1:
        # Single vector
        unconstrained_dict = {k: theta_unconstrained[i] for i, k in enumerate(theta_keys)}
        biological = inverse_transform_from_unconstrained(unconstrained_dict, theta_keys)
        return np.array([biological[k] for k in theta_keys], dtype=np.float32)
    
    elif theta_unconstrained.ndim == 2:
        # Batch of vectors
        n_samples = theta_unconstrained.shape[0]
        result = np.zeros_like(theta_unconstrained)
        for i in range(n_samples):
            unconstrained_dict = {k: theta_unconstrained[i, j] for j, k in enumerate(theta_keys)}
            biological = inverse_transform_from_unconstrained(unconstrained_dict, theta_keys)
            result[i] = np.array([biological[k] for k in theta_keys], dtype=np.float32)
        return result
    
    else:
        raise ValueError(f"theta_unconstrained must be 1D or 2D, got shape {theta_unconstrained.shape}")


def validate_biological_params(params: Dict[str, Any], theta_keys: Tuple[str, ...]) -> None:
    """Validate that biological parameters satisfy hard constraints.
    
    Raises ValueError if any constraint is violated.
    
    Parameters
    ----------
    params : dict
        Biological parameters to validate
    theta_keys : tuple
        Parameter names to check
    """
    for key in theta_keys:
        if key not in params:
            continue
        
        value = float(params[key])
        
        # Population sizes must be positive
        if key.startswith('N_'):
            if value <= 0:
                raise ValueError(f"Population size {key}={value} must be positive")
        
        # Times must be positive
        elif key.startswith('T_'):
            if value <= 0:
                raise ValueError(f"Time {key}={value} must be positive")
        
        # Fractions must be in [0, 1]
        # CRITICAL: Match ANY key containing '_FRAC'
        elif '_FRAC' in key:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"Fraction {key}={value} must be in [0, 1]")
        
        # Durations must be positive
        # CRITICAL: Match BN_DUR OR BN_DUR_*
        elif key == 'BN_DUR' or key.startswith('BN_DUR_'):
            if value <= 0:
                raise ValueError(f"Duration {key}={value} must be positive")

    available = set(theta_keys)
    for key in TIME_GAP_ORDER:
        if key not in available:
            continue
        parent = _get_time_gap_parent(key, available)
        if parent is None or parent not in params:
            continue
        if params[key] <= params[parent]:
            raise ValueError(f"Time ordering violated: {key}={params[key]} <= {parent}={params[parent]}")
