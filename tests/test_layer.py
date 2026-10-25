"""End-to-end layer gradients against independent hard bit flips."""
from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from conv_rosa_transformer.rosa_layer import (
    NBitQKVRosa, NBitQKVRosaCpp, NBitQKVRosaForward, NBitQKVRosaForwardCpp,
)
from conv_rosa_transformer.rosa_reference import backward_bruteforce, forward_naive
from conv_rosa_transformer.rosa_transformer import RosaTransformer


def _pack_independently(bits):
    return [sum(int(bit) * (1 << j) for j, bit in enumerate(row))
            for row in bits.tolist()]


class LayerTest(unittest.TestCase):
    def audit_layer(self, mode, D=3, T=6, groups=2, batch=2, strided=False):
        generator = torch.Generator().manual_seed(893)
        C = groups * D
        layer = NBitQKVRosa(C, D, tau=0.7, credit_to_logit=mode).double()
        with torch.no_grad():
            layer.emb0.copy_(torch.randn(1, 1, C, generator=generator, dtype=torch.float64))
            layer.emb1.copy_(torch.randn(1, 1, C, generator=generator, dtype=torch.float64))
        shape = (batch, C, T) if strided else (batch, T, C)
        tensors = []
        for _ in range(3):
            z = torch.randn(shape, generator=generator, dtype=torch.float64)
            if strided:
                z = z.transpose(1, 2)
            if T:
                z[:, 0, 0] = 0  # Forward uses strict z>0, including at zero.
            tensors.append(z.requires_grad_())
        q, k, v = tensors
        # A repeated two-symbol stream explicitly exercises the unsigned high bit.
        if D == 64:
            with torch.no_grad():
                q.fill_(-1)
                k.fill_(-1)
                q[:, ::2, 63] = 1
                k[:, ::2, 63] = 1
        gy = torch.randn(batch, T, C, generator=generator, dtype=torch.float64)
        actual = layer._hard_rosa_from_qkv(q, k, v)
        (actual * gy).sum().backward()
        expected_y = torch.empty_like(actual)
        expected_credits = [torch.zeros_like(z) for z in tensors]
        # Independent differentiable embedding expression for its true gradients.
        e0 = layer.emb0.detach().clone().requires_grad_()
        e1 = layer.emb1.detach().clone().requires_grad_()
        embedding_loss = (e0.sum() + e1.sum()) * 0
        for b in range(batch):
            for g in range(groups):
                sl = slice(g * D, (g + 1) * D)
                qs = _pack_independently(q[b, :, sl] > 0)
                ks = _pack_independently(k[b, :, sl] > 0)
                vb = (v[b, :, sl] > 0).to(torch.uint8)
                fwd, _, routes = forward_naive(qs, ks, vb, e0[0, 0, sl], e1[0, 0, sl], D)
                expected_y[b, :, sl] = fwd
                credits = backward_bruteforce(
                    qs, ks, vb, gy[b, :, sl], e0[0, 0, sl], e1[0, 0, sl], D,
                )
                for dst, src in zip(expected_credits, credits):
                    dst[b, :, sl] = src
                for t, r in enumerate(routes):
                    payload = e0[0, 0, sl] if r < 0 else (
                        vb[r + 1].to(torch.float64) * 2 - 1
                    ) * e1[0, 0, sl]
                    embedding_loss = embedding_loss + (gy[b, t, sl] * payload).sum()
        embedding_loss.backward()
        torch.testing.assert_close(actual, expected_y, rtol=0, atol=0)
        for z, credit in zip(tensors, expected_credits):
            expected = (1 - 2 * (z.detach() > 0).to(z.dtype)) * credit
            if mode == "bernoulli":
                p = torch.sigmoid(z.detach() / 0.7)
                expected *= p * (1 - p) / 0.7
            torch.testing.assert_close(z.grad, expected, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(layer.emb0.grad, e0.grad, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(layer.emb1.grad, e1.grad, atol=1e-12, rtol=1e-12)

    def test_grouped_batched_bernoulli(self):
        self.audit_layer("bernoulli")

    def test_grouped_batched_identity_noncontiguous(self):
        self.audit_layer("identity", strided=True)

    def test_unsigned_bit_63(self):
        self.audit_layer("identity", D=64, T=5, groups=1, batch=1)

    def test_empty_sequence(self):
        self.audit_layer("bernoulli", T=0)

    def test_empty_batch(self):
        self.audit_layer("identity", batch=0)

    def test_compatibility_aliases_and_state_dict(self):
        import hard_qkv_rosa_explained as explained
        from conv_rosa_transformer import rosa_reference
        for alias in (NBitQKVRosaCpp, NBitQKVRosaForward, NBitQKVRosaForwardCpp,
                      explained.NBitQKVRosa, rosa_reference.NBitQKVRosa):
            self.assertIs(alias, NBitQKVRosa)
        a, b = NBitQKVRosa(6, 3), NBitQKVRosaCpp(6, 3)
        expected_keys = {"emb0", "emb1"}
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            expected_keys.update((f"{name}.weight", f"{name}.bias"))
        self.assertEqual(set(a.state_dict()), expected_keys)
        b.load_state_dict(a.state_dict(), strict=True)
        x = torch.randn(2, 4, 6, generator=torch.Generator().manual_seed(8))
        torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
        a(x).sum().backward()
        for parameter in a.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_bfloat16_layer(self):
        layer = NBitQKVRosa(4, 2).to(dtype=torch.bfloat16)
        x = torch.randn(1, 4, 4, dtype=torch.bfloat16, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertEqual(x.grad.dtype, x.dtype)
        self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_transformer_training_smoke(self):
        model = RosaTransformer(vocab_size=8, channels=8, n_layers=1,
                                n_bits=2, max_seq_len=5)
        tokens = torch.tensor([[1, 2, 3, 4, 5], [2, 1, 2, 1, 2]])
        logits = model(tokens)
        self.assertEqual(logits.shape, (2, 5, 8))
        torch.nn.functional.cross_entropy(logits.reshape(-1, 8), tokens.reshape(-1)).backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        self.assertIs(model.blocks[0].rosa.q_proj, model.blocks[0].rosa.k_proj)

    def test_demo_checkpoint_reconstructs_configured_kernel(self):
        # Exercise the export path without training or evaluating a model.
        # A configuration/weight mismatch fails strict state-dict loading.
        from conv_rosa_transformer import multiply_demo
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "checkpoint.pt"
            argv = ["multiply_demo.py", "--steps", "0", "--channels", "8",
                    "--layers", "1", "--n-bits", "2", "--conv-kernel", "7",
                    "--out", str(output)]
            with patch("sys.argv", argv), patch.object(
                multiply_demo, "evaluate", return_value=(0.0, 0.0, [], [0.0] * 20),
            ), redirect_stdout(StringIO()):
                multiply_demo.main()
            checkpoint = torch.load(output, weights_only=True)
            restored = RosaTransformer(**checkpoint["config"])
            restored.load_state_dict(checkpoint["state_dict"], strict=True)
            self.assertEqual(restored.blocks[0].conv.kernel, 7)

    def test_invalid_configuration(self):
        for C, D in ((0, 1), (8, 0), (65, 65), (8, 3)):
            with self.assertRaises(ValueError):
                NBitQKVRosa(C, D)
        for tau in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                NBitQKVRosa(8, 4, tau=tau)
        with self.assertRaises(ValueError):
            NBitQKVRosa(8, 4, credit_to_logit="unknown")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_cuda_transfer_and_gradient(self):
        cpu = NBitQKVRosa(6, 3).double()
        gpu = NBitQKVRosa(6, 3).double().cuda()
        gpu.load_state_dict(cpu.state_dict())
        x = torch.randn(1, 5, 6, dtype=torch.float64, requires_grad=True)
        x_gpu = x.detach().cuda().requires_grad_()
        y, y_gpu = cpu(x), gpu(x_gpu)
        y.sum().backward()
        y_gpu.sum().backward()
        torch.testing.assert_close(y, y_gpu.cpu(), rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(x.grad, x_gpu.grad.cpu(), rtol=1e-11, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
