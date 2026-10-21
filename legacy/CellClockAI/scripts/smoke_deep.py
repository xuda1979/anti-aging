"""Exercise the optional architecture on synthetic tensors, never biological data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from cellclock.deep_model import DynamicRegulatoryModel, masked_feature_loss

torch.manual_seed(17)
model = DynamicRegulatoryModel(20, [list(range(19)), list(range(1, 20))], width=16, layers=2)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
x = torch.randn(8, 20, 4)
observed = torch.rand_like(x) > .2
control = torch.zeros(8, 20, 2)
for step in range(3):
    opt.zero_grad()
    out = model(x, observed, control, torch.ones(8))
    loss = masked_feature_loss(out["next_features"], x, observed)
    loss.backward()
    opt.step()
    print(f"synthetic tensor smoke step={step} loss={loss.item():.6f}")
