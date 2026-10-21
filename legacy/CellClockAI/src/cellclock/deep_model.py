"""Optional reference architecture for the proposed large-scale study.

Not trained on the public pilot. Requires the ``deep`` extra (PyTorch).
RNA, methylation, accessibility and protein inputs must be harmonized to
predeclared regulatory units. This reference uses Gaussian losses on normalized
features; read-count likelihoods and sequence encoders are future extensions.
"""
import torch
from torch import nn


class GraphBlock(nn.Module):
    def __init__(self, width, n_edges):
        super().__init__()
        self.self_map = nn.Linear(width, width)
        self.message = nn.Linear(width, width, bias=False)
        self.edge_logits = nn.Parameter(torch.zeros(n_edges))
        self.norm = nn.LayerNorm(width)

    def forward(self, h, edges):
        source, dest = edges
        out = torch.zeros_like(h)
        weights = self.edge_logits.sigmoid()
        messages = self.message(h[:, source]) * weights[None, :, None]
        out.index_add_(1, dest, messages)
        degree = torch.zeros(h.shape[1], device=h.device, dtype=h.dtype)
        degree.index_add_(0, dest, weights)
        out = out / degree.clamp_min(1)[None, :, None]
        return self.norm(h + torch.nn.functional.gelu(self.self_map(h) + out))


class DynamicRegulatoryModel(nn.Module):
    """Sparse graph encoder with action-conditioned dynamics and separate heads.

    x and observed: [batch, regulatory_units, modalities].
    control: [batch, regulatory_units, 2] for signed action and exposure.
    edges: [2, edges], directed source to destination, supplied before training.
    dt: [batch], dimensionless normalized interval.
    """
    def __init__(self, n_units, edges, n_modalities=4, width=128, layers=6,
                 n_functional=4, n_identities=8):
        super().__init__()
        edges = torch.as_tensor(edges, dtype=torch.long)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edges must have shape [2, E]")
        if edges.numel() and (edges.min() < 0 or edges.max() >= n_units):
            raise ValueError("edge index outside the regulatory-unit vocabulary")
        self.register_buffer("edges", edges)
        self.unit_embedding = nn.Parameter(torch.randn(n_units, width) * .02)
        self.modality_encoders = nn.ModuleList(nn.Linear(2, width) for _ in range(n_modalities))
        self.control_encoder = nn.Linear(2, width)
        self.blocks = nn.ModuleList(GraphBlock(width, edges.shape[1]) for _ in range(layers))
        self.drift = nn.Sequential(nn.Linear(width * 2, width), nn.Tanh(), nn.Linear(width, width))
        self.reconstruction = nn.Linear(width, n_modalities)
        self.functional = nn.Linear(width, n_functional)
        self.identity = nn.Linear(width, n_identities)
        self.viability = nn.Linear(width, 1)

    def forward(self, x, observed, control, dt):
        if x.shape != observed.shape or control.shape != (*x.shape[:2], 2):
            raise ValueError("x, observed and control dimensions disagree")
        # Missing measurements are never substituted as observed biological zero.
        present = observed.bool()
        safe_x = torch.where(present, x, torch.zeros_like(x))
        views = []
        for m, encoder in enumerate(self.modality_encoders):
            values = torch.stack([safe_x[..., m], present[..., m].to(x.dtype)], dim=-1)
            views.append(encoder(values) * present[..., m, None])
        h = sum(views) / present.sum(-1, keepdim=True).clamp_min(1)
        h = h + self.unit_embedding[None]
        for block in self.blocks:
            h = block(h, self.edges)
        action = self.control_encoder(control)
        next_h = h + dt[:, None, None] * self.drift(torch.cat([h, action], dim=-1))
        pooled = next_h.mean(1)
        return {"next_features": self.reconstruction(next_h), "latent": next_h,
                "functional": self.functional(pooled), "identity_logits": self.identity(pooled),
                "viability_logit": self.viability(pooled).squeeze(-1)}


def masked_feature_loss(predicted, target, observed):
    valid = observed.bool()
    if not valid.any():
        return predicted.sum() * 0
    # Boolean selection occurs before subtraction to avoid NaN propagation.
    return (predicted[valid] - target[valid]).square().mean()
