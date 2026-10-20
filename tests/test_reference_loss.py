"""Verify the compact API against the experiment implementation, not vice versa."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epic import epic_loss
from test_loss_kernels import NS, ActorConfig, MODES


class ReferenceLossTests(unittest.TestCase):
    def test_forward_backward_parity(self):
        rng = torch.Generator().manual_seed(71)
        for dtype in (torch.float32, torch.float64):
            for scale in (.001, .1, 1., 12.):
                for low, high in ((5e-4, 2e-3), (.01, .05)):
                    old = -torch.rand(4, 17, generator=rng, dtype=dtype) * 8 - .1
                    current = (old + scale * torch.randn(4, 17, generator=rng, dtype=dtype)).requires_grad_()
                    other = current.detach().clone().requires_grad_()
                    mask = torch.ones_like(old)
                    mask[0, 10:] = 0
                    mask[1, 3:] = 0
                    mask[2, :] = 0
                    adv = torch.randn(4, 1, generator=rng, dtype=dtype).expand_as(old)
                    cfg = ActorConfig({'policy_loss': {
                        'prefix_exact_kl_delta_low': low,
                        'prefix_exact_kl_delta_high': high}})
                    expected, _ = NS['compute_policy_loss_' + MODES[0]](old, other, adv, mask, config=cfg)
                    actual = epic_loss(old, current, adv, mask, delta_low=low, delta_high=high)
                    expected.backward()
                    actual.backward()
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
                    torch.testing.assert_close(current.grad, other.grad, rtol=2e-5, atol=1e-7)

    def test_invalid_inputs(self):
        old = torch.full((2, 3), -2.)
        adv = torch.ones_like(old)
        with self.assertRaises(ValueError):
            epic_loss(old, old, adv, torch.zeros_like(old))
        with self.assertRaises(ValueError):
            epic_loss(old, old, adv, adv, delta_low=0)

    def test_current_token_only_gradient(self):
        old = torch.full((1, 3), -2.)
        current = (old + .001).requires_grad_()
        epic_loss(old, current, torch.tensor([[0., 0., 1.]]), torch.ones_like(old)).backward()
        self.assertEqual(current.grad[0, 0].item(), 0)
        self.assertEqual(current.grad[0, 1].item(), 0)


if __name__ == '__main__':
    unittest.main()
