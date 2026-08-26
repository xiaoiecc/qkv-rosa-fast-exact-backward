"""Hard n-bit QKV-ROSA Transformer: (short conv ∥ QKV-ROSA) mixer + SwiGLU FFN.

The SDPA/attention sublayer of a standard pre-norm Transformer block is replaced by
two parallel context mixers:
  - NBitQKVRosaCpp: hard suffix-match retrieval (exact one-bit counterfactual
    backward, C++ extension; see csrc/)
  - ShortConv: depthwise causal conv1d (a learnable longer token-shift) covering
    local smooth statistics that hard longest-match retrieval cannot express.
The FFN is SwiGLU.

Design reference: toy experiments showing conv (local) and ROSA (long-range
retrieval) are complementary mixers; single parallel block is the sweet spot per
layer. Q/K projections of each ROSA layer are weight-tied by default: hard
threshold codes require identical Q/K encodings for matches to fire.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from rosa_layer import NBitQKVRosaCpp


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class ShortConv(nn.Module):
    """Depthwise causal conv1d: a learnable K-tap token-shift."""

    def __init__(self, channels, kernel=4):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, bias=True)

    def forward(self, x):  # [B,T,C]
        h = F.pad(x.transpose(1, 2), (self.kernel - 1, 0))
        return self.conv(h).transpose(1, 2)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ConvRosaBlock(nn.Module):
    """Pre-norm block: x += ROSA(norm x) + Conv(norm x);  x += SwiGLU(norm x)."""

    def __init__(self, channels, n_bits, conv_kernel=4, ffn_hidden=None, tau=1.0,
                 credit_to_logit="bernoulli", tie_qk=True):
        super().__init__()
        self.norm1 = RMSNorm(channels)
        self.rosa = NBitQKVRosaCpp(channels=channels, n_bits=n_bits, tau=tau,
                                        credit_to_logit=credit_to_logit)
        if tie_qk:
            self.rosa.k_proj = self.rosa.q_proj
        self.conv = ShortConv(channels, conv_kernel)
        self.norm2 = RMSNorm(channels)
        hidden = ffn_hidden or ((int(8 * channels / 3) + 31) // 32) * 32
        self.ffn = SwiGLU(channels, hidden)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.rosa(h) + self.conv(h)
        x = x + self.ffn(self.norm2(x))
        return x


class RosaTransformer(nn.Module):
    def __init__(self, vocab_size, channels=128, n_layers=4, n_bits=8, conv_kernel=4,
                 max_seq_len=64, tau=1.0, credit_to_logit="bernoulli", tie_qk=True,
                 tie_embeddings=True):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.channels = int(channels)
        self.max_seq_len = int(max_seq_len)
        self.tok_emb = nn.Embedding(vocab_size, channels)
        self.pos_emb = nn.Embedding(max_seq_len, channels)
        self.blocks = nn.ModuleList([
            ConvRosaBlock(channels, n_bits, conv_kernel, tau=tau,
                          credit_to_logit=credit_to_logit, tie_qk=tie_qk)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(channels)
        self.lm_head = nn.Linear(channels, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        if T > self.max_seq_len:
            raise ValueError("sequence longer than max_seq_len")
        h = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for b in self.blocks:
            h = b(h)
        return self.lm_head(self.norm_f(h))

    @torch.no_grad()
    def generate(self, idx, max_new, stop_token=None):
        for _ in range(max_new):
            logits = self(idx)[:, -1, :]
            nxt = logits.argmax(-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_token is not None and bool((nxt == stop_token).all()):
                break
        return idx
