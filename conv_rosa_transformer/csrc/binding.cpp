#include <torch/extension.h>
#include "index.h"
#include "pipeline.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // bit packing: rows of D bits -> python ints (bit order matches the reference)
    m.def("pack_bits", [](torch::Tensor bits_t_d) {
        auto vb = bits_t_d.cpu().contiguous().to(torch::kLong);
        const int64_t* p = vb.data_ptr<int64_t>();
        int64_t T = vb.size(0), D = vb.size(1);
        std::vector<int64_t> out(T);
        for (int64_t t = 0; t < T; ++t) {
            int64_t x = 0;
            for (int64_t j = 0; j < D; ++j)
                if (p[t * D + j]) x |= int64_t(1) << j;
            out[t] = x;
        }
        return out;
    });

    // forward certificates: per-position match length and route endpoint
    m.def("matching_stats", [](const std::vector<int64_t>& q, const std::vector<int64_t>& k) {
        rosa::CausalCutSuffixIndex idx(q, k);
        return idx.matching_stats();
    });

    // chosen adaptive repair backend (diagnostic)
    m.def("repair_backend", &rosa::repair_backend);

    // exact one-bit counterfactual VJP credits (backward)
    m.def("exact_stream_bit_credits", &rosa::exact_stream_bit_credits);
}
