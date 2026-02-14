"""
RoPE (Rotary Position Embedding) utilities for KV cache position re-encoding.
When stitching caches from different contexts, keys were computed with positions
from their original sequences. We strip the old RoPE and re-apply RoPE for the
global (stitched) positions so attention scores are correct.
"""

from __future__ import annotations

import torch
from typing import Any, Optional


def _get_rope_inv_freq(config: Any, device: torch.device) -> torch.Tensor:
    """Compute inverse frequencies for default RoPE from model config."""
    rope_params = getattr(config, "rope_parameters", None) or {}
    base = float(rope_params.get("rope_theta", 10000.0))
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    dim = head_dim
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
    )
    return inv_freq


def compute_rope_cos_sin(
    config: Any,
    position_ids: torch.Tensor,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    attention_scaling: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cos and sin for RoPE at the given positions.
    position_ids: (batch, seq_len) or (seq_len,) — positions (e.g. 0, 1, 2, ...).
    Returns cos, sin each of shape (batch, seq_len, head_dim) for broadcasting to (B, H, L, D).
    """
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    batch, seq_len = position_ids.shape
    inv_freq = _get_rope_inv_freq(config, device)
    # Use float32 for freq computation to avoid BFloat16/float mix in matmul
    inv_freq_expanded = inv_freq[None, :, None].expand(batch, -1, 1).to(torch.float32).to(device)
    position_ids_expanded = position_ids.float().to(device)[:, None, :]
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * attention_scaling).to(dtype)
    sin = (emb.sin() * attention_scaling).to(dtype)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (same as Qwen2/LLaMA)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply RoPE: x_embed = x * cos + rotate_half(x) * sin."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def apply_rotary_inverse(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Inverse RoPE (rotate by -theta): x_raw = x * cos - rotate_half(x) * sin."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) - (rotate_half(x) * sin)


def reencode_past_kvs_for_stitching(
    past_kvs: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    original_position_ids: torch.Tensor,
    config: Any,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Re-encode keys in the past KV cache so they use RoPE for global positions
    instead of their original positions. Values are unchanged (RoPE is not
    applied to V in standard models).

    past_kvs: tuple of (K, V) per layer; K, V shape (batch, num_heads, seq_len, head_dim).
    original_position_ids: (seq_len,) — for each position in the cache, the
        position it had in its source sequence (before stitching).
    config: model config with rope_parameters (e.g. rope_theta).
    """
    if past_kvs is None or len(past_kvs) == 0:
        return past_kvs
    k0 = past_kvs[0][0]
    batch, num_heads, seq_len, head_dim = k0.shape
    dtype = k0.dtype
    if original_position_ids.device != device:
        original_position_ids = original_position_ids.to(device)
    global_positions = torch.arange(seq_len, device=device, dtype=original_position_ids.dtype)

    rope_params = getattr(config, "rope_parameters", None) or {}
    attention_scaling = float(rope_params.get("attention_scaling", 1.0))
    if "attention_scaling" not in (rope_params or {}):
        attention_scaling = 1.0

    cos_old, sin_old = compute_rope_cos_sin(
        config, original_position_ids.unsqueeze(0), head_dim, device, dtype, attention_scaling
    )
    cos_new, sin_new = compute_rope_cos_sin(
        config, global_positions.unsqueeze(0), head_dim, device, dtype, attention_scaling
    )

    result = []
    for (k, v) in past_kvs:
        k_inv = apply_rotary_inverse(k, cos_old, sin_old, unsqueeze_dim=1)
        k_reenc = apply_rotary(k_inv, cos_new, sin_new, unsqueeze_dim=1)
        result.append((k_reenc, v))
    return tuple(result)
