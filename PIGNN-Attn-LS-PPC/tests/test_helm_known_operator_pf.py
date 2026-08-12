import unittest

import torch

from helm_known_operator_pf import (
    DifferentiableHELMPath,
    HELMKOLConfig,
    _equation_value,
    _rectangular_jacobian,
)


class HELMKnownOperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.Y = torch.tensor(
            [[5.0 - 15.0j, -5.0 + 15.0j], [-5.0 + 15.0j, 5.0 - 15.0j]],
            dtype=torch.complex128,
        )
        self.bus_type = torch.tensor([[1, 3]], dtype=torch.long)
        self.Vfixed = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float64)
        target_voltage = torch.tensor(
            [[1.0 + 0.0j, 0.97 * torch.exp(torch.tensor(-0.07j))]],
            dtype=torch.complex128,
        )
        self.Sset = target_voltage * (target_voltage @ self.Y.T).conj()

    def test_rectangular_jacobian_matches_autograd(self):
        vr = torch.tensor([1.0, 0.99], dtype=torch.float64, requires_grad=True)
        vi = torch.tensor([0.0, -0.03], dtype=torch.float64, requires_grad=True)

        def equation(z):
            return _equation_value(
                self.Y, z[:2].unsqueeze(0), z[2:].unsqueeze(0), self.bus_type
            ).squeeze(0)

        exact = _rectangular_jacobian(self.Y, vr.detach(), vi.detach(), self.bus_type[0])
        automatic = torch.autograd.functional.jacobian(equation, torch.cat([vr, vi]))
        torch.testing.assert_close(exact, automatic, atol=1e-10, rtol=1e-10)

    def test_series_pade_reduces_residual_and_backpropagates(self):
        operator = DifferentiableHELMPath(
            HELMKOLConfig(series_order=8, path_order=3, linear_regularization=1e-10)
        )
        logits = torch.zeros((1, 4, 3), dtype=torch.float64, requires_grad=True)
        Vc, diagnostics = operator(
            self.Y, self.Sset, self.Vfixed, self.bus_type, logits
        )
        initial = operator._max_residual(
            self.Y,
            torch.complex(self.Vfixed[..., 0], torch.zeros_like(self.Vfixed[..., 0])),
            self.Sset,
            self.bus_type,
        )
        self.assertLess(diagnostics["max_mismatch"].item(), initial.item())
        loss = (Vc.real.square() + Vc.imag.square()).sum()
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().max().item(), 0.0)

    def test_endpoint_constraints_are_hard_projected(self):
        operator = DifferentiableHELMPath(
            HELMKOLConfig(series_order=6, path_order=2, linear_regularization=1e-10)
        )
        logits = torch.randn((1, 4, 2), dtype=torch.float64)
        Vc, _ = operator(self.Y, self.Sset, self.Vfixed, self.bus_type, logits)
        torch.testing.assert_close(Vc[0, 0], torch.tensor(1.0 + 0.0j, dtype=torch.complex128))


if __name__ == "__main__":
    unittest.main()
