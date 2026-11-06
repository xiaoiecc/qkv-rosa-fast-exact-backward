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
#define ROSA_APPROXIMATION_ABI_VERSION 1u
#define ROSA_APPROXIMATION_COUNTERS 40
#define ROSA_SUBQUADRATIC_COUNTERS 22

typedef struct RosaTrainingOptions {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t credit_order;       /* 1: original single-bit; 2: trust + joint. */
    int32_t randomized;         /* Full-key cache acceleration; never thins credit. */
    int32_t logit_mode;         /* 0: Bernoulli sigmoid derivative; 1: identity. */
    int32_t joint_pairs;        /* Selected disjoint blocks per stream. */
    int32_t joint_rows;         /* Maximum query rows inspected for proposals. */
    int32_t active_rows;        /* 1: nonzero-g rows, 0: evenly spaced rows. */
    int32_t repair_budget;      /* Reserved for ABI layout; must be zero. */
    int32_t reserved;
    double tau;
    double trust_diagonal;      /* 0.5 means ||g|| / output-set diameter. */
    uint64_t proposal_seed;
    uint64_t repair_seed;       /* Independent from proposal randomness. */
} RosaTrainingOptions;

typedef struct RosaApproximationOptions {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t sample_rows;        /* Positive correction-row budget per stream. */
    int32_t importance;         /* 0 uniform strata; 1 norm + uniform floor. */
    int32_t sample_channels;    /* 0: row estimator; 1..64: centered-feature PPS. */
    int32_t reserved;
    uint64_t seed;              /* Salt independent of joint proposal RNG. */
} RosaApproximationOptions;

/* Fill the complete struct with second-order exact defaults. */
void rosa_training_defaults(RosaTrainingOptions* options);
void rosa_approximation_defaults(RosaApproximationOptions* options);

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

/* Optional aggressive approximation of the same structural engine. The hard
 * forward and V/embedding gradients retain complete evaluation. Use either
 * full deletion + a one-symbol control with stratified residual-row sampling,
 * or centered-feature PPS. Complete and approximate modes can both use FFT.
 * Selected joint corners retain complete internal matcher evaluations.
 * saved_lengths/routes must be from the same forward. Approximation counters
 * have layout [B*G,40].
 */
int rosa_training_backward_approx(
    const RosaTrainingOptions* options, const RosaApproximationOptions* approximation,
    int b, int groups, int n, int d,
    const uint64_t* q, const uint64_t* k, const uint8_t* v_bits,
    const double* q_logits, const double* k_logits, const double* v_logits,
    const double* grad_y, const double* emb0, const double* emb1,
    double* grad_q, double* grad_k, double* grad_v,
    double* grad_emb0, double* grad_emb1, uint64_t* counters,
    uint64_t* pair_codes, double* pair_costs,
    const int32_t* saved_lengths, const int32_t* saved_routes,
    uint64_t* approximation_counters);


/* Diagnostic alias of rosa_training_backward[_approx], not another engine.
 * For fixed packed width, numeric dimension and joint budget,
 * time is O~(T^(7/4)); workspace is O~(T^(4/3)). Route decisions are exact;
 * batched floating reductions and FFT are numerically approximate, even with
 * approximation == NULL. randomized only changes fully verified query caching
 * and is bitwise answer-equivalent to this entry point with randomized == 0.
 * Non-NULL approximation selects explicit residual-row or channel sampling.
 * repair_budget is reserved and must be zero.
 * subquadratic_counters: [B*G,22], names in native_training.py.
 */
int rosa_training_backward_subquadratic(
    const RosaTrainingOptions* options, const RosaApproximationOptions* approximation,
    int b, int groups, int n, int d,
    const uint64_t* q, const uint64_t* k, const uint8_t* v_bits,
    const double* q_logits, const double* k_logits, const double* v_logits,
    const double* grad_y, const double* emb0, const double* emb1,
    double* grad_q, double* grad_k, double* grad_v,
    double* grad_emb0, double* grad_emb1, uint64_t* counters,
    uint64_t* pair_codes, double* pair_costs,
    const int32_t* saved_lengths, const int32_t* saved_routes,
    uint64_t* approximation_counters, uint64_t* subquadratic_counters);

#ifdef __cplusplus
}
#endif
#endif
