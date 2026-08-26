import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rosa_transformer import RosaTransformer
MUL, EQ, EOS, PAD = 10, 11, 12, 13
EXPR_LEN = 22; MAX_ANS = 20; T_MAX = 43
a, b = 1234567891, 9876543211
p = a * b
expr = [int(c) for c in str(a)] + [MUL] + [int(c) for c in str(b)] + [EQ]
ans_rev = [int(c) for c in str(p)][::-1] + [0] * (MAX_ANS - len(str(p)))
seq = expr + ans_rev + [EOS]
seq += [PAD] * (T_MAX - len(seq))
labels = list(seq)
for i in range(EXPR_LEN): labels[i] = -100
labels = [(-100 if t == PAD else t) for t in labels]
x = torch.tensor([seq] * 16); y = torch.tensor([labels] * 16)
torch.manual_seed(0)
model = RosaTransformer(vocab_size=14, channels=128, n_layers=3, n_bits=32, conv_kernel=4, max_seq_len=T_MAX)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
lf = nn.CrossEntropyLoss(ignore_index=-100)
for step in range(300):
    loss = lf(model(x)[:, :-1].reshape(-1, 14), y[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0: print(step, round(loss.item(), 4), flush=True)
idx = torch.tensor([expr])
out = model.generate(idx, 21)[0].tolist()
digits = [t for t in out[EXPR_LEN:EXPR_LEN + 20] if 0 <= t <= 9]
pred = "".join(map(str, digits))[::-1].lstrip("0") or "0"
print("true:", p)
print("pred:", pred)
