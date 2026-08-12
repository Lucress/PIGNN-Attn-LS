import unittest

import torch
import torch.nn.functional as F

from known_operator_pf import (
    DifferentiablePFConfig,
    DifferentiablePowerFlow,
    power_flow_loss_and_gradient,
    power_flow_max_mismatch,
)


def two_bus_case(dtype=torch.complex128):
    y = torch.tensor(4.0 - 12.0j, dtype=dtype)
    Y = torch.stack(
        [torch.stack((y, -y)), torch.stack((-y, y))],
    )
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    V_true = torch.tensor([[[1.0, 0.0], [0.98, -0.05]]], dtype=real_dtype)
    Vc = V_true[..., 0].to(dtype) * torch.exp(1j * V_true[..., 1].to(dtype))
    Sset = Vc * torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1).conj()
    bus_type = torch.tensor([[1, 3]], dtype=torch.long)
    V_flat = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=real_dtype)
    return Y, Sset, bus_type, V_true, V_flat


class KnownOperatorPFTests(unittest.TestCase):
    def test_reference_dpf_mse_objective_and_gradient_match(self):
        """Match the author's ``S_calc`` packing and ``torch.nn.MSELoss``."""
        Y, Sset, bus_type, _, V0 = two_bus_case()
        V = V0.clone().requires_grad_(True)
        loss, grad_vm, grad_va = power_flow_loss_and_gradient(Y, V, Sset, bus_type)

        Vc = V[..., 0].to(Y.dtype) * torch.exp(1j * V[..., 1].to(Y.dtype))
        Scalc = Vc * torch.matmul(Y.conj(), Vc.conj().unsqueeze(-1)).squeeze(-1)
        p_mask = bus_type != 1
        q_mask = (bus_type != 1) & (bus_type != 2)
        out = torch.cat((Scalc.real[p_mask], Scalc.imag[q_mask]))
        target = torch.cat((Sset.real[p_mask], Sset.imag[q_mask]))
        reference_loss = F.mse_loss(out, target)
        expected, = torch.autograd.grad(reference_loss, V)

        self.assertTrue(torch.allclose(loss, reference_loss, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(grad_vm, expected[..., 0], atol=1e-10, rtol=1e-9))
        self.assertTrue(torch.allclose(grad_va, expected[..., 1], atol=1e-10, rtol=1e-9))

    def test_one_step_matches_reference_torch_adam(self):
        Y, Sset, bus_type, _, V0 = two_bus_case()
        lr = 0.003377
        betas = (0.979681, 0.963442)
        eps = 1e-8

        vm_pq = V0[:, 1, 0].clone().requires_grad_(True)
        va_pq = V0[:, 1, 1].clone().requires_grad_(True)
        optimizer = torch.optim.Adam(
            (vm_pq, va_pq), lr=lr, betas=betas, eps=eps
        )
        optimizer.zero_grad()
        vm = torch.stack((V0[:, 0, 0], vm_pq), dim=1)
        va = torch.stack((V0[:, 0, 1], va_pq), dim=1)
        Vc = vm.to(Y.dtype) * torch.exp(1j * va.to(Y.dtype))
        Scalc = Vc * torch.matmul(Y.conj(), Vc.conj().unsqueeze(-1)).squeeze(-1)
        reference_loss = F.mse_loss(
            torch.cat((Scalc.real[:, 1], Scalc.imag[:, 1])),
            torch.cat((Sset.real[:, 1], Sset.imag[:, 1])),
        )
        reference_loss.backward()
        optimizer.step()
        expected = V0.clone()
        expected[:, 1, 0] = vm_pq.detach()
        expected[:, 1, 1] = va_pq.detach()

        layer = DifferentiablePowerFlow(
            DifferentiablePFConfig(
                train_steps=1,
                eval_steps=1,
                lr=lr,
                optimizer="adam",
                beta1=betas[0],
                beta2=betas[1],
                eps=eps,
            )
        )
        actual, _ = layer(
            V0,
            Y,
            Sset,
            bus_type,
            V_fixed=V0,
            differentiable=True,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-11, rtol=1e-10))

    def test_sparse_adjoint_gradient_matches_autograd(self):
        Y, Sset, bus_type, _, V0 = two_bus_case()
        V = V0.clone().requires_grad_(True)
        loss, grad_vm, grad_va = power_flow_loss_and_gradient(
            Y.to_sparse_coo(), V, Sset, bus_type
        )
        # Compare against PyTorch AD on the packed polar state.
        expected, = torch.autograd.grad(loss, V)
        self.assertTrue(torch.allclose(grad_vm, expected[..., 0], atol=1e-10, rtol=1e-9))
        self.assertTrue(torch.allclose(grad_va, expected[..., 1], atol=1e-10, rtol=1e-9))

    def test_eval_reduces_residual_and_preserves_specified_voltage(self):
        Y, Sset, bus_type, _, V0 = two_bus_case()
        layer = DifferentiablePowerFlow(
            DifferentiablePFConfig(
                train_steps=3,
                eval_steps=3000,
                lr=1e-2,
                optimizer="adam",
                tol=1e-6,
            )
        )
        out, diagnostics = layer(
            V0,
            Y.to_sparse_coo(),
            Sset,
            bus_type,
            V_fixed=V0,
            differentiable=False,
        )
        self.assertLess(
            diagnostics["final_max_mismatch"].item(),
            diagnostics["initial_max_mismatch"].item(),
        )
        self.assertTrue(torch.equal(out[:, 0], V0[:, 0]))
        self.assertTrue(diagnostics["converged"].item())

    def test_training_unroll_backpropagates_to_surrogate_output(self):
        Y, Sset, bus_type, _, V0 = two_bus_case(torch.complex64)
        V0 = V0.clone().requires_grad_(True)
        layer = DifferentiablePowerFlow(
            DifferentiablePFConfig(
                train_steps=2,
                eval_steps=10,
                lr=1e-3,
                optimizer="adam",
            )
        )
        out, _ = layer(
            V0,
            Y,
            Sset,
            bus_type,
            V_fixed=V0.detach(),
            differentiable=True,
        )
        loss = out[:, 1].square().sum()
        loss.backward()
        self.assertIsNotNone(V0.grad)
        self.assertTrue(torch.isfinite(V0.grad).all())

    def test_exact_state_is_certified_without_iterations(self):
        Y, Sset, bus_type, V_true, _ = two_bus_case()
        layer = DifferentiablePowerFlow(
            DifferentiablePFConfig(eval_steps=10, tol=1e-10)
        )
        out, diagnostics = layer(
            V_true,
            Y,
            Sset,
            bus_type,
            V_fixed=V_true,
            differentiable=False,
        )
        residual = power_flow_max_mismatch(Y, out, Sset, bus_type)
        self.assertLessEqual(residual.item(), 1e-10)
        self.assertEqual(diagnostics["iterations"].item(), 0)
        self.assertTrue(diagnostics["converged"].item())

    def test_block_diagonal_graphs_stop_independently(self):
        Y, Sset, bus_type, V_true, V_flat = two_bus_case()
        Y_block = torch.block_diag(Y, Y).to_sparse_coo()
        S_block = torch.cat((Sset, Sset), dim=1)
        bus_block = torch.cat((bus_type, bus_type), dim=1)
        V0 = torch.cat((V_true, V_flat), dim=1)
        Vfixed = torch.cat((V_true, V_flat), dim=1)
        layer = DifferentiablePowerFlow(
            DifferentiablePFConfig(eval_steps=3000, lr=1e-2, tol=1e-6)
        )
        _, diagnostics = layer(
            V0,
            Y_block,
            S_block,
            bus_block,
            V_fixed=Vfixed,
            sizes=torch.tensor([2, 2]),
            differentiable=False,
        )
        self.assertEqual(diagnostics["iterations"][0].item(), 0)
        self.assertGreater(diagnostics["iterations"][1].item(), 0)
        self.assertTrue(diagnostics["converged"].all())


if __name__ == "__main__":
    unittest.main()
