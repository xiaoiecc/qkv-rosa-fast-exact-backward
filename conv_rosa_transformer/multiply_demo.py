"""Demo: small Conv∥ROSA Transformer learns 10-digit x 10-digit multiplication.

Pipeline: train -> export weights (multiply_ckpt.pt) -> evaluate with two ACCs:
  * digit_acc: per-position accuracy over the answer digits
  * exact_acc: whole product string must match

Note on output format: the model emits the product LEAST-significant digit first
(reversed), which makes carry propagation causal with generation and the task far
more learnable; evaluation reverses the string back to natural order.
"""
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rosa_transformer import RosaTransformer

MUL, EQ, EOS, PAD = 10, 11, 12, 13
VOCAB = 14
EXPR_LEN = 10 + 1 + 10 + 1          # dddddddddd x dddddddddd =
MAX_ANS = 20                        # 10-digit x 10-digit < 1e20
T_MAX = EXPR_LEN + MAX_ANS + 1      # + EOS


def make_example(rng):
    a = rng.randint(10**9, 10**10 - 1)
    b = rng.randint(10**9, 10**10 - 1)
    p = a * b
    expr = [int(c) for c in str(a)] + [MUL] + [int(c) for c in str(b)] + [EQ]
    ans_rev = [int(c) for c in str(p)][::-1]      # least-significant first
    ans_rev = ans_rev + [0] * (MAX_ANS - len(ans_rev))  # right-pad zeros (they become leading zeros)
    seq = expr + ans_rev + [EOS]
    seq = seq + [PAD] * (T_MAX - len(seq))
    # loss mask: predict answer digits + EOS (positions after EQ)
    labels = list(seq)
    for i in range(EXPR_LEN):
        labels[i] = -100
    # do not train on padding
    labels = [(-100 if t == PAD else t) for t in labels]
    return seq, labels, p


def batch(rng, bsz):
    xs, ys = [], []
    for _ in range(bsz):
        s, l, _ = make_example(rng)
        xs.append(s); ys.append(l)
    return torch.tensor(xs), torch.tensor(ys)


@torch.no_grad()
def evaluate(model, pairs, device="cpu"):
    model.eval()
    digit_ok = digit_n = exact_ok = 0
    pos_ok = [0] * MAX_ANS   # per reversed-position (low-order first) accuracy
    pos_n = [0] * MAX_ANS
    examples = []
    for a, b in pairs:
        p = a * b
        expr = [int(c) for c in str(a)] + [MUL] + [int(c) for c in str(b)] + [EQ]
        idx = torch.tensor([expr], device=device)
        out = model.generate(idx, MAX_ANS + 1, stop_token=None)[0].tolist()
        gen = out[EXPR_LEN:]
        digits = []
        for t in gen[:MAX_ANS]:
            if t in (EOS, PAD):
                break
            digits.append(t if 0 <= t <= 9 else -1)
        pred_str = "".join(str(d) if d >= 0 else "#" for d in digits)[::-1].lstrip("0") or "0"
        true_str = str(p)
        if pred_str == true_str:
            exact_ok += 1
        pred_pad = pred_str.ljust(len(true_str), "#")
        for i, c in enumerate(true_str):
            digit_n += 1
            ok = i < len(pred_pad) and pred_pad[i] == c
            if ok:
                digit_ok += 1
            rev_pos = len(true_str) - 1 - i
            pos_n[rev_pos] += 1
            pos_ok[rev_pos] += bool(ok)
        if len(examples) < 5:
            examples.append((a, b, true_str, pred_str))
    model.train()
    pos_acc = [round(pos_ok[i] / max(1, pos_n[i]), 3) for i in range(MAX_ANS)]
    return digit_ok / digit_n, exact_ok / len(pairs), examples, pos_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--n-bits", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--conv-kernel", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-pairs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "multiply_ckpt.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = RosaTransformer(vocab_size=VOCAB, channels=args.channels, n_layers=args.layers,
                            n_bits=args.n_bits, conv_kernel=args.conv_kernel, max_seq_len=T_MAX)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e3:.0f}K  T_MAX={T_MAX}", flush=True)

    test_rng = random.Random(999)
    test_pairs = [(test_rng.randint(10**9, 10**10 - 1), test_rng.randint(10**9, 10**10 - 1))
                  for _ in range(200)]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    t0 = time.time()
    losses = []
    for step in range(args.steps):
        x, y = batch(rng, args.batch)
        logits = model(x)
        loss = lossf(logits[:, :-1].reshape(-1, VOCAB), y[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % args.eval_every == 0 or step == args.steps - 1:
            da, ea, ex, pa = evaluate(model, test_pairs[:args.eval_pairs])
            print(f"step {step:5d} loss {sum(losses[-50:])/min(50,len(losses)):.4f} "
                  f"digit_acc {da:.4f} exact_acc {ea:.4f} ({time.time()-t0:.0f}s)", flush=True)
            print(f"   per-pos acc (units->high): {pa[:10]}", flush=True)

    # export weights
    ckpt = dict(config=dict(vocab_size=VOCAB, channels=args.channels, n_layers=args.layers,
                            n_bits=args.n_bits, conv_kernel=4, max_seq_len=T_MAX),
                state_dict=model.state_dict(),
                token_map=dict(MUL=MUL, EQ=EQ, EOS=EOS, PAD=PAD))
    torch.save(ckpt, args.out)
    print(f"weights exported -> {args.out}", flush=True)

    # final evaluation
    da, ea, ex, pa = evaluate(model, test_pairs)
    print(f"per-pos acc (units->high): {pa}", flush=True)
    print(f"\nFINAL digit_acc={da:.4f} exact_acc={ea:.4f} on {len(test_pairs)} held-out pairs", flush=True)
    for a, b, ts, ps in ex:
        mark = "OK " if ps == ts else "BAD"
        print(f"  [{mark}] {a} x {b} = {ts} | pred {ps}", flush=True)
    metrics = dict(digit_acc=da, exact_acc=ea, steps=args.steps, params=n_params,
                   walltime_s=time.time() - t0)
    with open(os.path.join(os.path.dirname(args.out), "multiply_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
