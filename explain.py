"""
explain.py

Turns the raw attention tensors produced by the models into human-readable
importance scores:

  - node_importance_from_attention(): GATv2Conv edge attention -> a single
    importance score per ROI ("which brain regions did the model focus on")
  - run_transformer_with_attention(): manually replays a nn.TransformerEncoder
    forward pass while also returning each layer's self-attention weights,
    since nn.TransformerEncoder does not expose them by default -> used to
    get a single importance score per pseudo-dynamic window ("which time
    windows did the model focus on")

Tested against: Python 3.9.0
"""

from typing import List, Tuple

import torch
import torch.nn as nn


def node_importance_from_attention(
    edge_index: torch.Tensor, alpha: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """
    Aggregate GATv2Conv attention weights into one importance score per node.

    `alpha` has shape [E, heads] (one weight per edge per attention head).
    For each node, importance = the average attention *received* from its
    neighbors across all incoming edges and all heads - i.e. how strongly
    other ROIs attended to this ROI when the model aggregated information.

    Returns
    -------
    torch.Tensor of shape [num_nodes], importance score per node.
    """
    alpha_mean_heads = alpha.mean(dim=-1)  # [E]
    target_nodes = edge_index[1]

    importance = torch.zeros(num_nodes, device=alpha.device)
    counts = torch.zeros(num_nodes, device=alpha.device)
    importance.index_add_(0, target_nodes, alpha_mean_heads)
    counts.index_add_(0, target_nodes, torch.ones_like(alpha_mean_heads))
    counts = counts.clamp(min=1.0)

    return importance / counts


def run_transformer_with_attention(
    encoder: nn.TransformerEncoder, x: torch.Tensor
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Manually replays `encoder`'s forward pass (assumes batch_first=True,
    norm_first=False, which is how DynamicAttentionGNN builds it) while
    capturing each layer's self-attention weights, since
    nn.TransformerEncoder.forward() does not return them.

    Parameters
    ----------
    x : torch.Tensor [batch, seq_len, embed_dim]
        The window-embedding sequence for one or more subjects.

    Returns
    -------
    output : torch.Tensor [batch, seq_len, embed_dim]
        Same as encoder(x) would return.
    attn_weights_per_layer : list of torch.Tensor [batch, seq_len, seq_len]
        Self-attention weight matrices, one per transformer layer,
        averaged over attention heads.
    """
    out = x
    attn_weights_per_layer = []

    for layer in encoder.layers:
        attn_output, attn_weights = layer.self_attn(
            out, out, out, need_weights=True, average_attn_weights=True
        )
        out2 = out + layer.dropout1(attn_output)
        out2 = layer.norm1(out2)
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(out2))))
        out3 = out2 + layer.dropout2(ff)
        out3 = layer.norm2(out3)
        out = out3
        attn_weights_per_layer.append(attn_weights)  # [batch, seq_len, seq_len]

    if encoder.norm is not None:
        out = encoder.norm(out)

    return out, attn_weights_per_layer


def window_importance_from_attention(attn_weights_per_layer: List[torch.Tensor]) -> torch.Tensor:
    """
    Collapse the per-layer [batch, seq_len, seq_len] self-attention matrices
    into a single importance score per window (key position), averaged over
    layers, heads (already averaged in run_transformer_with_attention), and
    query positions. Assumes batch size 1.

    Returns
    -------
    torch.Tensor of shape [seq_len], importance score per pseudo-window.
    """
    stacked = torch.stack(attn_weights_per_layer, dim=0)  # [layers, batch, seq, seq]
    stacked = stacked[:, 0, :, :]  # -> [layers, seq, seq], batch size 1
    importance = stacked.mean(dim=(0, 1))  # average over layers and query positions -> [seq]
    return importance


def top_k_indices(scores: torch.Tensor, k: int) -> List[Tuple[int, float]]:
    """Return the top-k (index, score) pairs, sorted descending by score."""
    k = min(k, scores.numel())
    top_scores, top_idx = torch.topk(scores, k)
    return list(zip(top_idx.tolist(), top_scores.tolist()))
