#ifndef ROSA_TRAINING_H
#define ROSA_TRAINING_H
/* Standalone C ABI for the one batched hard QKV-ROSA implementation.
 * Build unified_core.cpp as C++17; this header needs no Python or Torch.
 * Floating arrays are IEEE binary64. Logits/outputs have layout [B,G,T,D],
 * embeddings [G,D], packed Q/K and routes [B,G,T]. Row-major contiguous.
 * The caller allocates outputs; inputs must remain alive during the call.
 */
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROSA_TRAINING_ABI_VERSION 1u
#define ROSA_TRAINING_COUNTERS 35
#define ROSA_FORWARD_COUNTERS 4

typedef struct RosaTrainingOptions {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t credit_order;       /* 1: original single-bit; 2: trust + joint. */
    int32_t randomized;         /* Exact root acceleration; never thins credit. */
    int32_t logit_mode;         /* 0: Bernoulli sigmoid derivative; 1: identity. */
    int32_t joint_pairs;        /* Selected disjoint blocks per stream. */
    int32_t joint_rows;         /* Maximum query rows inspected for proposals. */
    int32_t active_rows;        /* 1: nonzero-g rows, 0: evenly spaced rows. */
    int32_t repair_budget;      /* 0: exact; >0: explicit legacy research MC. */
    int32_t reserved;
    double tau;
    double trust_diagonal;      /* 0.5 means ||g|| / output-set diameter. */
    uint64_t proposal_seed;
    uint64_t repair_seed;       /* Independent from proposal randomness. */
} RosaTrainingOptions;

/* Fill the complete struct with second-order exact defaults. */
void rosa_training_defaults(RosaTrainingOptions* options);

/* Return codes: 0 success; 1 invalid shape/options; 2 allocation failure;
 * 3 numerical/core failure; 4 nonfinite input. GradScaler handling belongs
 * to the framework adapter, which propagates nonfinite gradients to it.
 * Forward counters are [B*G,4]: passes, transition probes, hash probes,
 * endpoint queries. Every pointer is required unless its size is zero.
 */
int rosa_training_forward(
    const RosaTrainingOptions* options, int b, int groups, int n, int d,
    const double* q_logits, const double* k_logits, const double* v_logits,
    const double* emb0, const double* emb1,
    uint64_t* q, uint64_t* k, uint8_t* v_bits,
    int32_t* lengths, int32_t* routes, double* output, uint64_t* counters);

/* Backward uses packed codes from this same forward and the same logits and
 * embeddings. The finite trust model is frozen for the local block update;
 * it is not the true Hessian of an arbitrary downstream model.
 * Counters are [B*G,35]; indices are documented in the Python adapter.
 * pair_codes and pair_costs may be NULL. If present their layouts are
 * [B*G,joint_pairs,2] and [B*G,joint_pairs,4]. Codes < T*D are K bits;
 * codes >= T*D are Q bits. Costs: C_i,C_j,C_ij,kappa.
 */
int rosa_training_backward(
    const RosaTrainingOptions* options, int b, int groups, int n, int d,
    const uint64_t* q, const uint64_t* k, const uint8_t* v_bits,
    const double* q_logits, const double* k_logits, const double* v_logits,
    const double* grad_y, const double* emb0, const double* emb1,
    double* grad_q, double* grad_k, double* grad_v,
    double* grad_emb0, double* grad_emb1, uint64_t* counters,
    uint64_t* pair_codes, double* pair_costs);

#ifdef __cplusplus
}
#endif
#endif
