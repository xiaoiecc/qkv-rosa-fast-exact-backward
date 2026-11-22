# Benchmarks

Scripts for measuring CPU backward speed, full PyTorch layer updates, coordinate contraction, and paired comparisons against another Git checkout.

Summary results are reported in [`docs/benchmarks.md`](../docs/benchmarks.md) ([中文版](../docs/benchmarks.zh-CN.md)).

## Quick measurements

```bash
# Single-engine timing across random and structured inputs
python benchmarks/subquadratic_engine.py --quick --output build/bench_quick.json
python benchmarks/subquadratic_engine.py --expanded --repeats 5 --output build/bench_full.json

# Coordinate contraction ratio across sequence lengths and bit widths
c++ -O3 -std=c++17 benchmarks/interval_support.cpp -o build/interval-support
./build/interval-support
```

## Paired comparison against a reference checkout

To benchmark the current tree against a separate Git checkout on the same pinned CPU:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Random Gaussian / Bernoulli / biased QKV
python benchmarks/random_qkv.py --reference-checkout ../reference --cpu 0 --repeats 5 --output build/random.json

# Full PyTorch NBitQKVRosa update steps
python benchmarks/compare_layer_step.py --reference-checkout ../reference --cpu 0 --repeats 5 --output build/layer.json

# Structured stress inputs (unary, periodic, lagged_copy, near_fanout)
python benchmarks/frontier_structured.py --reference-checkout ../reference --cpu 0 --repeats 5 --output build/structured.json
```
