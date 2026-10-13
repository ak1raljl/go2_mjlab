"""Global-context, single-query cross attention for the PIE estimator."""

from typing import Any

import torch
from torch import nn

from .pie_model import PIEActorModel


class MTEActorModel(PIEActorModel):
  """Replace only PIE fusion; inherit recurrence, losses and export contract."""

  policy_architecture = "mte-global-local-cross-attention"

  def _build_fusion(self, cfg: dict[str, Any]) -> int:
    heads = int(cfg.get("attention_heads", 8))
    if heads <= 0 or self.token_dim % heads:
      raise ValueError("token_dim must be divisible by positive attention_heads.")
    obsolete = {key for key in cfg if key.startswith("transformer_")}
    if obsolete:
      raise ValueError(f"MTE does not use Transformer config keys: {sorted(obsolete)}")
    global_dim = int(cfg.get("global_dim", 32))
    # CNN flatten traverses columns first, then rows. Derive its grid once;
    # these are feature-image coordinates, not metric terrain coordinates.
    with torch.no_grad():
      feature = self.depth_encoder(torch.zeros(1, *self.depth_shape))
    rows, columns = feature.shape[-2:]
    yy, xx = torch.meshgrid(
      torch.linspace(-1, 1, rows), torch.linspace(-1, 1, columns), indexing="ij"
    )
    self.register_buffer(
      "grid_xy", torch.stack((xx, yy), dim=-1).reshape(-1, 2), persistent=False
    )
    self.position_proj = nn.Linear(2, self.token_dim)
    self.global_mlp = nn.Sequential(
      nn.Linear(self.token_dim, global_dim), nn.ELU(),
      nn.Linear(global_dim, global_dim), nn.ELU(),
    )
    self.query_mlp = nn.Sequential(
      nn.Linear(self.token_dim + global_dim, self.token_dim), nn.ELU(),
      nn.Linear(self.token_dim, self.token_dim), nn.ELU(),
    )
    self.attention = nn.MultiheadAttention(
      self.token_dim, heads, dropout=0.0, batch_first=True
    )
    self.fusion_proj = nn.Sequential(
      nn.Linear(2 * self.token_dim + global_dim, self.memory_hidden_dim), nn.ELU()
    )
    return self.memory_hidden_dim

  def _trunk(self, history: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    p = self.history_encoder(self.history_normalizer(history))
    local = self._encode_depth(depth) + self.position_proj(self.grid_xy)
    global_context = self.global_mlp(local).amax(dim=-2)
    query = self.query_mlp(torch.cat((p, global_context), dim=-1))
    leading_shape = p.shape[:-1]
    local = local.reshape(-1, self.grid_xy.shape[0], self.token_dim)
    attended, _ = self.attention(
      query.reshape(-1, 1, self.token_dim), local, local, need_weights=False
    )
    attended = attended.reshape(*leading_shape, self.token_dim)
    return self.fusion_proj(torch.cat((p, global_context, attended), dim=-1))
