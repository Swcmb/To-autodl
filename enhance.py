# -*- coding: utf-8 -*-
"""
Graph feature augmentation utilities for EM project.

Provided augmentations:
- random_permute_features: randomly permute rows (nodes)
- add_noise: additive Gaussian noise on features
- attribute_mask: global column-wise mask (set selected feature columns to 0)
- noise_then_mask: add noise first, then apply attribute mask

Design goals:
- Preserve input type: numpy.ndarray in -> ndarray out; torch.Tensor in -> Tensor out
- Determinism via seed
- Minimal assumptions: operate on dense feature matrices of shape [num_nodes, num_features]
"""

from typing import Optional, Dict, Callable, Any, Tuple
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False


def _is_torch_tensor(x: Any) -> bool:
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def _to_numpy(x: Any) -> Tuple[np.ndarray, Optional[Any]]:
    """
    Convert input to numpy array and return (np_array, like).
    'like' is the original object for restoring type later.
    """
    if _is_torch_tensor(x):
        # Ensure CPU and detach; assume dense tensor
        return x.detach().cpu().numpy(), x
    elif isinstance(x, np.ndarray):
        return x, x
    else:
        raise TypeError("Unsupported feature type. Expect numpy.ndarray or torch.Tensor.")


def _from_numpy_like(x_np: np.ndarray, like: Any) -> Any:
    """
    Restore numpy array back to the same type as 'like'.
    """
    if _is_torch_tensor(like):
        # Keep dtype consistent with original tensor dtype if possible
        dtype = like.dtype if like is not None else None
        t = torch.from_numpy(x_np)
        if dtype is not None and t.dtype != dtype:
            try:
                t = t.to(dtype)
            except Exception:
                pass
        return t
    elif isinstance(like, np.ndarray):
        return x_np
    else:
        # Fallback to numpy
        return x_np


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()


def random_permute_features(X: Any, seed: Optional[int] = None) -> Any:
    """
    Randomly permute rows (node order) of feature matrix.
    """
    if _is_torch_tensor(X):
        if X.dim() != 2:
            raise ValueError(f"Expect 2D feature matrix, got shape {tuple(X.shape)}")
        N = X.size(0)
        # 使用 torch.randperm，保持设备一致
        g = np.random.default_rng(seed)
        # 通过 numpy 生成种子，再转 torch 手动设置生成器以确定性；简化：使用 numpy 索引再搬回同设备
        idx_np = g.permutation(N)
        idx = torch.as_tensor(idx_np, device=X.device)
        return X.index_select(0, idx)
    # numpy 路径
    x_np, like = _to_numpy(X)
    if x_np.ndim != 2:
        raise ValueError(f"Expect 2D feature matrix, got shape {x_np.shape}")
    N = x_np.shape[0]
    g = _rng(seed)
    idx = g.permutation(N)
    out = x_np[idx]
    return _from_numpy_like(out, like)


def add_noise(X: Any, noise_std: float = 0.01, seed: Optional[int] = None) -> Any:
    """
    Additive Gaussian noise to features.
    """
    if noise_std <= 0:
        return X
    if _is_torch_tensor(X):
        if X.dim() != 2:
            raise ValueError(f"Expect 2D feature matrix, got shape {tuple(X.shape)}")
        # 在 tensor 设备上生成噪声；采用 numpy 种子派生后用 torch.randn 近似
        # 为了确定性，可以使用 numpy 生成噪声后转 tensor 到同设备
        g = _rng(seed)
        noise_np = g.normal(loc=0.0, scale=noise_std, size=tuple(X.shape)).astype(str(X.detach().cpu().numpy().dtype), copy=False)
        noise = torch.from_numpy(noise_np).to(X.device, dtype=X.dtype)
        return X + noise
    x_np, like = _to_numpy(X)
    if x_np.ndim != 2:
        raise ValueError(f"Expect 2D feature matrix, got shape {x_np.shape}")
    g = _rng(seed)
    noise = g.normal(loc=0.0, scale=noise_std, size=x_np.shape).astype(x_np.dtype, copy=False)
    out = x_np + noise
    return _from_numpy_like(out, like)


def attribute_mask(X: Any, mask_rate: float = 0.1, seed: Optional[int] = None) -> Any:
    """
    Global column-wise feature mask. Randomly select a subset of columns and set them to zero.
    """
    if mask_rate <= 0:
        return X
    if _is_torch_tensor(X):
        if X.dim() != 2:
            raise ValueError(f"Expect 2D feature matrix, got shape {tuple(X.shape)}")
        N, D = X.size(0), X.size(1)
        g = _rng(seed)
        k = int(np.floor(mask_rate * D))
        if k <= 0:
            return X
        k = min(k, D)
        cols = torch.as_tensor(g.choice(D, size=k, replace=False), device=X.device)
        out = X.clone()
        out[:, cols] = 0
        return out
    x_np, like = _to_numpy(X)
    if x_np.ndim != 2:
        raise ValueError(f"Expect 2D feature matrix, got shape {x_np.shape}")
    N, D = x_np.shape
    g = _rng(seed)
    k = int(np.floor(mask_rate * D))
    if k <= 0:
        return _from_numpy_like(x_np, like)
    k = min(k, D)
    cols = g.choice(D, size=k, replace=False)
    out = x_np.copy()
    out[:, cols] = 0
    return _from_numpy_like(out, like)


def noise_then_mask(
    X: Any,
    noise_std: float = 0.01,
    mask_rate: float = 0.1,
    seed: Optional[int] = None,
) -> Any:
    """
    Composite augmentation: add noise first, then apply attribute mask.
    """
    base = int(seed) if seed is not None else None
    seed_noise = None if base is None else base
    seed_mask = None if base is None else base + 1
    # 直接复用上面两个函数（已具备 tensor-native 路径）
    x1 = add_noise(X, noise_std=noise_std, seed=seed_noise)
    x2 = attribute_mask(x1, mask_rate=mask_rate, seed=seed_mask)
    return x2


# Registry for name-based selection
AUGMENT_FUNCS: Dict[str, Callable[..., Any]] = {
    "random_permute_features": random_permute_features,
    "add_noise": add_noise,
    "attribute_mask": attribute_mask,
    "noise_then_mask": noise_then_mask,
    "none": lambda X, **kwargs: X,
    "null": lambda X, **kwargs: X,
    "": lambda X, **kwargs: X,
}


def apply_augmentation(
    name: str,
    X: Any,
    *,
    noise_std: float = 0.01,
    mask_rate: float = 0.1,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """
    Unified entry to apply augmentation by name.

    Supported names:
      - 'random_permute_features'
      - 'add_noise'
      - 'attribute_mask'
      - 'noise_then_mask'
      - 'none'/'null'/'' (no-op)

    Extra kwargs are forwarded to the underlying function.

    Returns:
        Augmented features with the same type as input.
    """
    key = (name or "").strip().lower()
    # keep original exact keys too
    func = AUGMENT_FUNCS.get(key)
    if func is None:
        # try original name without lowering
        func = AUGMENT_FUNCS.get(name)
    if func is None:
        raise ValueError(f"Unknown augmentation name: {name}")
    if func is random_permute_features:
        return func(X, seed=seed, **kwargs)
    if func is add_noise:
        return func(X, noise_std=noise_std, seed=seed, **kwargs)
    if func is attribute_mask:
        return func(X, mask_rate=mask_rate, seed=seed, **kwargs)
    if func is noise_then_mask:
        return func(X, noise_std=noise_std, mask_rate=mask_rate, seed=seed, **kwargs)
    # no-op or custom
    return func(X, **kwargs)