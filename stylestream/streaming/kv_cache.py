"""KV cache management for StyleStream streaming inference.

Provides per-layer and multi-layer key/value caching for incremental
attention computation during chunk-by-chunk streaming inference.

StyleStream spec:
    - Max buffer: 250 frames (5 seconds @ 50Hz)
    - FIFO eviction: oldest frames discarded when buffer full
    - Separate caches for Conformer (6 layers), DiT (16 layers),
      HuBERT (24 layers)
"""

from __future__ import annotations

import torch
from torch import Tensor


class LayerKVCache:
    """KV cache for a single attention layer.

    Stores past key and value tensors and provides methods to
    append new chunks and evict old frames.

    Parameters
    ----------
    max_frames : int
        Maximum number of frames to cache (default 250 = 5s @ 50Hz).
        When exceeded, oldest frames are evicted (FIFO).
    """

    def __init__(self, max_frames: int = 250) -> None:
        self.max_frames = max_frames
        self._k: Tensor | None = None  # (B, H, T_cached, D)
        self._v: Tensor | None = None  # (B, H, T_cached, D)

    @property
    def length(self) -> int:
        """Number of cached frames."""
        if self._k is None:
            return 0
        return self._k.shape[2]

    @property
    def is_empty(self) -> bool:
        """Whether cache has any stored frames."""
        return self._k is None

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Append new key/value and return full (past + new) K, V.

        Parameters
        ----------
        k, v : Tensor
            New key/value of shape (B, H, T_new, D).

        Returns
        -------
        full_k, full_v : Tensor
            Concatenated (past + new) of shape (B, H, T_total, D)
            where T_total = min(T_cached + T_new, max_frames).
        """
        if self._k is None:
            self._k = k
            self._v = v
        else:
            self._k = torch.cat([self._k, k], dim=2)
            self._v = torch.cat([self._v, v], dim=2)

        self.trim()

        return self._k, self._v

    def get(self) -> tuple[Tensor | None, Tensor | None]:
        """Get current cached K, V (or None if empty)."""
        return self._k, self._v

    def reset(self) -> None:
        """Clear the cache."""
        self._k = None
        self._v = None

    def trim(self) -> None:
        """Evict oldest frames if cache exceeds max_frames."""
        if self._k is not None and self._k.shape[2] > self.max_frames:
            self._k = self._k[:, :, -self.max_frames:, :]
            self._v = self._v[:, :, -self.max_frames:, :]


class MultiLayerKVCache:
    """KV cache manager for multiple attention layers.

    Manages per-layer KV caches for a model with N attention layers.
    Provides convenient indexing and bulk operations.

    Parameters
    ----------
    num_layers : int
        Number of attention layers to cache.
    max_frames : int
        Maximum frames per layer cache (default 250).
    """

    def __init__(self, num_layers: int, max_frames: int = 250) -> None:
        self.caches = [LayerKVCache(max_frames) for _ in range(num_layers)]

    def __getitem__(self, layer_idx: int) -> LayerKVCache:
        """Get cache for a specific layer."""
        return self.caches[layer_idx]

    def __len__(self) -> int:
        """Number of layers."""
        return len(self.caches)

    @property
    def length(self) -> int:
        """Number of cached frames (from layer 0)."""
        if not self.caches:
            return 0
        return self.caches[0].length

    def reset(self) -> None:
        """Reset all layer caches."""
        for cache in self.caches:
            cache.reset()

    def trim_all(self) -> None:
        """Trim all layers to max_frames."""
        for cache in self.caches:
            cache.trim()
