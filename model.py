"""
model.py

Two models are defined:

1. DynamicAttentionGNN
   Spatial attention (GATv2Conv) is applied to each pseudo-dynamic window
   graph independently, producing one embedding per window. A Transformer
   encoder ("sequence attention") then attends across the N_WINDOWS
   embeddings to produce a single subject-level embedding, which feeds a
   classification head. This stands in for temporal attention across real
   DFC windows, but here operates over statistically resampled windows
   (see pseudo_dynamic.py).

2. StaticGNN
   The same spatial-attention block applied to the single original static
   matrix graph per subject - no window sequence, no sequence attention.
   This is the honest static-FC baseline used for comparison.

Tested against: Python 3.9.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch

import config


class SpatialGATEncoder(nn.Module):
    """Stacked GATv2Conv layers + mean/max pooling -> one embedding per graph."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = config.HIDDEN_DIM,
        heads: int = config.NUM_GAT_HEADS,
        num_layers: int = config.GAT_LAYERS,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_dim, heads=heads, edge_dim=1, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=1, dropout=dropout)
            )

        # mean pool + max pool concatenated -> project back to hidden_dim
        self.pool_proj = nn.Linear(hidden_dim * heads * 2, hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch, return_attention: bool = False):
        """
        Parameters
        ----------
        return_attention : bool
            If True, also return the last GAT layer's attention weights
            (edge_index_out, alpha) for brain-region-importance analysis.
            Default False keeps this fully backward compatible with
            train.py / test.py, which never pass this flag.
        """
        h = x
        attn_edge_index, attn_weights = None, None
        last_idx = len(self.convs) - 1

        for idx, conv in enumerate(self.convs):
            if return_attention and idx == last_idx:
                h, (attn_edge_index, attn_weights) = conv(
                    h, edge_index, edge_attr=edge_attr, return_attention_weights=True
                )
            else:
                h = conv(h, edge_index, edge_attr=edge_attr)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        mean_pooled = global_mean_pool(h, batch)
        max_pooled = global_max_pool(h, batch)
        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)
        embedding = F.relu(self.pool_proj(pooled))

        if return_attention:
            return embedding, attn_edge_index, attn_weights
        return embedding  # [num_graphs_in_batch, hidden_dim]


class DynamicAttentionGNN(nn.Module):
    """
    Spatial GAT per pseudo-dynamic window, followed by a Transformer
    encoder across the window sequence ("sequence attention"), followed by
    a classification head.
    """

    def __init__(
        self,
        in_channels: int = config.NUM_ROIS,
        hidden_dim: int = config.HIDDEN_DIM,
        gat_heads: int = config.NUM_GAT_HEADS,
        gat_layers: int = config.GAT_LAYERS,
        num_attention_layers: int = config.NUM_ATTENTION_LAYERS,
        transformer_heads: int = config.TRANSFORMER_HEADS,
        dropout: float = config.DROPOUT,
        num_classes: int = config.NUM_CLASSES,
    ):
        super().__init__()
        self.spatial_encoder = SpatialGATEncoder(
            in_channels, hidden_dim, gat_heads, gat_layers, dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.sequence_attention = nn.TransformerEncoder(encoder_layer, num_layers=num_attention_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, window_graph_lists):
        """
        Parameters
        ----------
        window_graph_lists : list[list[torch_geometric.data.Data]]
            One list of N_WINDOWS graphs per subject, batch_size subjects.

        Returns
        -------
        logits : torch.Tensor [batch_size, num_classes]
        """
        batch_size = len(window_graph_lists)
        n_windows = len(window_graph_lists[0])

        # Flatten every window graph across every subject into one big PyG
        # batch so the spatial GAT runs in a single vectorized forward pass.
        flat_graphs = [g for subject_graphs in window_graph_lists for g in subject_graphs]
        big_batch = Batch.from_data_list(flat_graphs)
        device = next(self.parameters()).device
        big_batch = big_batch.to(device)

        window_embeddings = self.spatial_encoder(
            big_batch.x, big_batch.edge_index, big_batch.edge_attr, big_batch.batch
        )  # [batch_size * n_windows, hidden_dim]

        window_embeddings = window_embeddings.view(batch_size, n_windows, -1)  # [B, N, H]
        attended = self.sequence_attention(window_embeddings)  # [B, N, H]
        subject_embedding = attended.mean(dim=1)  # [B, H]

        logits = self.classifier(subject_embedding)
        return logits


class StaticGNN(nn.Module):
    """Static-FC baseline: spatial GAT on the single original matrix graph
    per subject, no sequence attention."""

    def __init__(
        self,
        in_channels: int = config.NUM_ROIS,
        hidden_dim: int = config.HIDDEN_DIM,
        gat_heads: int = config.NUM_GAT_HEADS,
        gat_layers: int = config.GAT_LAYERS,
        dropout: float = config.DROPOUT,
        num_classes: int = config.NUM_CLASSES,
    ):
        super().__init__()
        self.spatial_encoder = SpatialGATEncoder(in_channels, hidden_dim, gat_heads, gat_layers, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, batch):
        """
        Parameters
        ----------
        batch : torch_geometric.data.Batch
            A standard PyG batch of single graphs (one per subject).
        """
        embedding = self.spatial_encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        logits = self.classifier(embedding)
        return logits
