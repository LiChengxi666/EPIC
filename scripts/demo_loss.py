"""CPU-only demonstration. Does not load a model or contact a cluster."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from epic import epic_loss

old = torch.full((2, 5), -2.0)
current = (old + torch.tensor([[.01, .02, .03, .04, .05],
                              [-.01, -.02, -.03, 0., 0.]])).requires_grad_()
mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
advantages = torch.tensor([[1.] * 5, [-1.] * 5])
loss = epic_loss(old, current, advantages, mask)
loss.backward()
print(f"EPIC loss: {loss.item():.6f}")
print("Current-token gradients:", current.grad)
assert torch.isfinite(loss) and torch.isfinite(current.grad).all()
assert torch.equal(current.grad[1, 3:], torch.zeros(2))
