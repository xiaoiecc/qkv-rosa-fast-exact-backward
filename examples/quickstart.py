"""Minimal PyTorch example showing NBitQKVRosa in a standard training loop."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qkv_rosa import NBitQKVRosa


def main():
    torch.manual_seed(0)
    layer = NBitQKVRosa(channels=64, n_bits=8)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)

    x = torch.randn(2, 32, 64)
    target = torch.randn_like(x)

    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        out = layer(x)
        loss = (out - target).square().mean()
        loss.backward()
        optimizer.step()
        print(f"step {step}: loss = {loss.item():.6f}")


if __name__ == "__main__":
    main()
