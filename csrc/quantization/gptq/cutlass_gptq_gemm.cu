#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include "cutlass_extensions/torch_utils.hpp"

#include "core/registration.h"

#include "cutlass/cutlass.h"
#include <limits>

#include "cute/tensor.hpp"


namespace cutlass_gptq {

template<typename scalar_t>
void run_cutlass_gptq_gemm(const scalar_t* a,
                           const uint32_t* b_q_weight,
                           const uint32_t* b_gptq_qzeros,
                           const scalar_t* b_gptq_scales, const int* b_g_idx,
                           scalar_t* c, scalar_t* temp_dq, int size_m, int size_n,
                           int size_k, int groups, bool use_exllama, int bit) {



    }


template void run_cutlass_gptq_gemm<half>(
    const half*, const uint32_t*, const uint32_t*,
    const half*, const int*, half*, half*,
    int, int, int, int, bool, int
);

template void run_cutlass_gptq_gemm<__nv_bfloat16>(
    const __nv_bfloat16*, const uint32_t*, const uint32_t*,
    const __nv_bfloat16*, const int*, __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, bool, int
);

}