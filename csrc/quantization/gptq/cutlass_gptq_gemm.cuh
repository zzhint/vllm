#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include "cutlass_extensions/torch_utils.hpp"

#include "core/registration.h"

#include "cutlass/cutlass.h"
#include <limits>

#include "cute/tensor.hpp"

#ifndef cutlass_gptq_gemm_cuh
#define cutlass_gptq_gemm_cuh


namespace cutlass_gptq {
using namespace cute;

template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];

  __forceinline__ __device__ scalar_t& operator[](int idx) {
        return val[idx];
    }
};

static constexpr int M_BLOCK = 64;
static constexpr int N_BLOCK = 128;

template<typename scalar_t>
struct MMA_Operator {
    using mma_op = UniversalFMA<scalar_t,scalar_t,scalar_t>;
    using thr_layout = Layout<Shape<_16,_8,_1>>;
};

template<>
struct MMA_Operator<half> {
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using thr_layout = Layout<Shape<_1,_4,_1>>;
};

template<>
struct MMA_Operator<__nv_bfloat16> {
    using mma_op = SM80_16x8x16_F32BF16BF16F32_TN;
    using thr_layout = Layout<Shape<_1,_4,_1>>;
};

template<typename _scalar_t, 
    int _bM, int _bN, int _bK, int _kStage,
    int _bit, int _group_size>
struct GPTQ_GemmConfig {
    
    using scalar_t = _scalar_t;
    static constexpr int bM = _bM;
    static constexpr int bN = _bN;
    static constexpr int bK = _bK;
    static constexpr int kStage = _kStage;
    static constexpr int bit = _bit;
    static constexpr int group_size = _group_size;

    
    //using mma_op = cute::conditional_t<cute::is_same_v<scalar_t, half>, SM80_16x8x16_F32F16F16F32_TN, SM80_16x8x16_F32BF16BF16F32_TN>;
    using mma_op = typename MMA_Operator<scalar_t>::mma_op;
    using thr_layout = typename MMA_Operator<scalar_t>::thr_layout;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int aM = 16;
    static constexpr int aK = 16;

    using TiledMMA = decltype(make_tiled_mma(mma_atom{}, thr_layout{}, Tile<Int<bM>, Int<bN>, Int<aK>>{}));
    static constexpr int threads = size(TiledMMA{});   

    using accscalar_t = typename mma_traits::ValTypeD;
    static constexpr bool is_acc_type_match = cute::is_same_v<scalar_t, accscalar_t>;

    //how many data in one uint32_t
    static constexpr int q_div = 32 / bit;
    static constexpr int bit_mask = (1 << bit) - 1;

    static constexpr int bK_q = bK / q_div;
    static constexpr int bN_q = bN / q_div;
    static constexpr int n_bK_in_one_group = group_size / bK;
    
    //smem size
    static constexpr int smem_size_A = bM * bK * kStage * sizeof(scalar_t);
    static constexpr int smem_size_B = bN * bK * kStage * sizeof(scalar_t);
    static constexpr int smem_size_Bq = bN * bK_q * kStage * sizeof(uint32_t);
    static constexpr int smem_size_B_zeros = bN * sizeof(uint32_t);
    static constexpr int smem_size_B_scales = bN * sizeof(scalar_t);
    static constexpr int smem_size_C = bM * bN * sizeof(scalar_t);

    static constexpr int smem_size_AB = smem_size_A 
        + smem_size_B + smem_size_Bq
        + smem_size_B_zeros + smem_size_B_scales;
    static constexpr int smem_size = smem_size_C > smem_size_AB 
            ? smem_size_C : smem_size_AB;


    //using TiledMMA = decltype(make_tiled_mma(UniversalFMA<T,T,T>{},
    //                             Layout<Shape<_16,_8,_1>>{}, Tile<Int<bM>, Int<bN>, Int<bK>>{}));

    //using CTATiler = Shape<Int<bM>, Int<bN>, Int<bK>>;

/*
    using SmemLayoutA = Layout< Shape<Int<bM>, Int<bK>>, Stride<Int<bK>, Int<1>> >;
    using SmemLayoutB = Layout< Shape<Int<bN>, Int<bK>>, Stride<Int<bK>, Int<1>> >;
    using SmemLayoutBq = Layout< Shape<Int<bN>,  Int<bK_q>>, Stride<Int<1>, Int<bN>> >;
*/
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, scalar_t>;

    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    using SmemLayoutAtomAB = decltype(composition(
        Swizzle<3,3,3>{}, Layout<Shape<Int<8>, Int<bK>>, Stride<Int<bK>, Int<1>>>{}));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomAB{}, 
                                        Shape<Int<bM>, Int<bK>, Int<kStage>>{}));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomAB{},
                                        Shape<Int<bN>, Int<bK>, Int<kStage>>{}));

    using SmemLayoutBq = Layout< Shape<Int<bN>, Int<bK_q>, Int<kStage>>, 
                                   Stride<Int<1>,  Int<bN>,   Int<bN * bK_q>>>;
    /*
    
    using SmemLayoutA = Layout<     Shape<Int<bM>, Int<bK>,   Int<kStage>>, 
                                   Stride<Int<bK>, Int<1>,    Int<bM * bK>>  >;
    using SmemLayoutB = Layout<   Shape<Int<bN>, Int<bK>,   Int<kStage>>, 
                                   Stride<Int<bK>, Int<1>,    Int<bN * bK>>  >;

    
    */


    using SmemLayoutB_zeros = Layout<Shape<Int<bN>, Int<bK>>, Stride<Int<1>, Int<0>>>;
    using SmemLayoutB_scales = Layout<Shape<Int<bN>, Int<bK>>, Stride<Int<1>, Int<0>>>;
    using SmemLayoutC = Layout< Shape<Int<bM>, Int<bN>>, Stride<Int<bN>, Int<1>> >;

    using vec_copy_op = UniversalCopy<cute::uint128_t>;
    using vec_copy_traits = Copy_Traits<vec_copy_op>;
    using vec_copy_scalar_t_atom = Copy_Atom<vec_copy_traits, scalar_t>;
    using vec_copy_uint32_t_atom = Copy_Atom<vec_copy_traits, uint32_t>;

    using vec_async_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using vec_async_copy_traits = Copy_Traits<vec_async_copy_op>;
    using vec_async_copy_scalar_t_atom = Copy_Atom<vec_async_copy_traits, scalar_t>;
    using vec_async_copy_uint32_t_atom = Copy_Atom<vec_async_copy_traits, uint32_t>;

    using g2s_copyA_atom = vec_async_copy_scalar_t_atom;

    static constexpr int vec_scalar_t_copy = sizeof(cute::uint128_t) / sizeof(scalar_t);
    static constexpr int vec_uint32_t_copy = sizeof(cute::uint128_t) / sizeof(uint32_t);
    
    static constexpr int g2s_copyA_K_threads = bK / vec_scalar_t_copy;
    static constexpr int g2s_copyA_M_threads = threads / g2s_copyA_K_threads;
    using G2SCopyA =
      decltype(make_tiled_copy(g2s_copyA_atom{},
                               make_layout(make_shape(Int<g2s_copyA_M_threads>{}, Int<g2s_copyA_K_threads>{}),
                                           make_stride(Int<g2s_copyA_K_threads>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<vec_scalar_t_copy>{}))));

    static constexpr int g2s_copyB_N_threads = bN / vec_uint32_t_copy;
    static constexpr int g2s_copyB_K_threads = threads / g2s_copyB_N_threads;

    using g2s_copy_Bq_atom = vec_async_copy_uint32_t_atom;
    using G2SCopyBq =
        decltype(make_tiled_copy(g2s_copy_Bq_atom{},
                               make_layout(make_shape(Int<g2s_copyB_N_threads>{}, Int<g2s_copyB_K_threads>{}),
                                           make_stride(Int<1>{}, Int<g2s_copyB_N_threads>{})),
                               make_layout(make_shape(Int<vec_uint32_t_copy>{}, Int<1>{}))));

    using s2r_dq_copy_op = UniversalCopy<cute::uint32_t>;
    using s2r_dq_copy_traits = Copy_Traits<s2r_dq_copy_op>;
    using s2r_dq_copy_atom = Copy_Atom<s2r_dq_copy_traits, uint32_t>;

    using STORE_B_T = aligned_vector<scalar_t, q_div>;
    using r2s_dq_copy_op = UniversalCopy<STORE_B_T>;
    using r2s_dq_copy_traits = Copy_Traits<r2s_dq_copy_op>;
    using r2s_dq_copy_atom = Copy_Atom<r2s_dq_copy_traits, scalar_t>;

    static constexpr int s2s_dq_copy_N_threads = bN;
    static constexpr int s2s_dq_copy_K_threads = threads / s2s_dq_copy_N_threads;
    static constexpr int s2s_dq_copy_repeat_for_aK = aK / (s2s_dq_copy_K_threads * q_div);

    using S2RDQCopy =  decltype(make_tiled_copy(s2r_dq_copy_atom{},
                               make_layout(make_shape(Int<s2s_dq_copy_N_threads>{}, Int<s2s_dq_copy_K_threads>{}),
                                           make_stride(Int<1>{}, Int<s2s_dq_copy_N_threads>{})),
                               make_layout(make_shape(Int<1>{}, Int<s2s_dq_copy_repeat_for_aK>{}))));

    using R2SDQCopy = decltype(make_tiled_copy(r2s_dq_copy_atom{},
                               make_layout(make_shape(Int<s2s_dq_copy_N_threads>{}, Int<s2s_dq_copy_K_threads>{}),
                                           make_stride(Int<1>{}, Int<s2s_dq_copy_N_threads>{})),
                               make_layout(make_shape(Int<1>{}, Int<q_div * s2s_dq_copy_repeat_for_aK>{}))));

    using s2g_copy_C_atom = vec_copy_scalar_t_atom;
    static constexpr int s2g_copyC_N_threads = bN / vec_scalar_t_copy;
    static constexpr int s2g_copyC_M_threads = threads / s2g_copyC_N_threads;

    using S2GCopyC =   decltype(make_tiled_copy(s2g_copy_C_atom{},
                               make_layout(make_shape(Int<s2g_copyC_M_threads>{}, Int<s2g_copyC_N_threads>{}),
                                           make_stride(Int<s2g_copyC_N_threads>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<vec_scalar_t_copy>{}))));

    //0 2 4 6 1 3 5 7
    using Bq_dq_layout = Layout<Shape<Shape<_4, _2>, _1>, Stride<Stride<_2, _1>, _1>>; 

    using reduce_g2g_copy_atom = vec_copy_scalar_t_atom;
    using ReduceG2GCopyC = S2GCopyC;


};



template<typename scalar_t>
struct GptQ_Kernel_Params {
    const scalar_t* A_ptr;
    const uint32_t* Bq_ptr;
    const uint32_t* B_zeros_ptr;
    const scalar_t* B_scales_ptr;
    const int* B_g_idx_ptr;
    scalar_t* C_ptr;
    scalar_t* C_reduce_ptr;
    uint32_t* C_semaphore_ptr;
    int M;
    int N;
    int K;
    int count_m_blocks;
    int count_n_blocks;
    int split_k_slices;
    int bit;
    int group_size;
    int l2_tile;
    bool use_exllama;
};

struct Launch_Kernel_Params {
    dim3 grid;
    cudaStream_t stream; 
};

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

template<typename scalar_t>
void run_cutlass_gptq_gemm(GptQ_Kernel_Params<scalar_t> kernel_params, Launch_Kernel_Params launch_kernel_params);



}

#endif
