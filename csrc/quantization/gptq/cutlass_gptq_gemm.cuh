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
};


template<typename _scalar_t, typename _accscalar_t, 
    int _bM, int _bN, int _bK, int _kStage,
    int _bit, int _group_size>
struct GPTQ_GemmConfig {
    
    using scalar_t = _scalar_t;
    using accscalar_t = _accscalar_t;

    using mma_op = cute::conditional_t<cute::is_same_v<scalar_t, half>, SM80_16x8x16_F32F16F16F32_TN, SM80_16x8x16_F32BF16BF16F32_TN>;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static_assert(cute::is_same_v<accscalar_t, typename mma_traits::ValTypeD>, "accscalar_t must same as ValTypeD");

    static constexpr bool is_acc_type_match = cute::is_same_v<scalar_t, accscalar_t>;

    static constexpr int bM = _bM;
    static constexpr int bN = _bN;
    static constexpr int bK = _bK;
    static constexpr int kStage = _kStage;
    static constexpr int bit = _bit;
    static constexpr int group_size = _group_size;

    //how many data in one uint32_t
    static constexpr int q_div = 32 / bit;
    static constexpr int bit_mask = (1 << bit) - 1;

    static constexpr int bK_q = bK / q_div;
    static constexpr int bN_q = bN / q_div;
    static constexpr int group_size_div_q_div = group_size / q_div;
    
    //smem size
    static constexpr int smem_size_A = bM * bK * kStage * sizeof(scalar_t);
    static constexpr int smem_size_B = bN * bK * kStage * sizeof(scalar_t);
    static constexpr int smem_size_B_q = bN * bK_q * kStage * sizeof(uint32_t);
    static constexpr int smem_size_B_zeros = bN * sizeof(uint32_t);
    static constexpr int smem_size_B_scales = bN * sizeof(scalar_t);
    static constexpr int smem_size_C = bM * bN * sizeof(scalar_t);

    static constexpr int smem_size_AB = smem_size_A 
        + smem_size_B + smem_size_B_q
        + smem_size_B_zeros + smem_size_B_scales;
    static constexpr int smem_size = smem_size_C > smem_size_AB 
            ? smem_size_C : smem_size_AB;


    //using TiledMMA = decltype(make_tiled_mma(UniversalFMA<T,T,T>{},
    //                             Layout<Shape<_16,_8,_1>>{}, Tile<Int<bM>, Int<bN>, Int<bK>>{}));
    using TiledMMA = decltype(make_tiled_mma(mma_atom{}, Layout<Shape<_1,_4,_1>>{}, Tile<Int<bM>, Int<bN>, Int<bK>>{}));
    static constexpr int threads = size(TiledMMA{});
    //using CTATiler = Shape<Int<bM>, Int<bN>, Int<bK>>;

    using SmemLayoutA = Layout< Shape<Int<bM>, Int<bK>>, Stride<Int<bK>, Int<1>> >;
    using SmemLayoutB_T = Layout< Shape<Int<bN>, Int<bK>>, Stride<Int<1>, Int<bN>> >;
    using SmemLayoutB_q_T = Layout< Shape<Int<bN>,  Int<bK_q>>,
                                    Stride<Int<1>,    Int<bN>> >;

    using SmemLayoutB_zeros = Layout<Shape<Int<bN>, Int<bK>>, Stride<Int<1>, Int<0>>>;
    using SmemLayoutB_scales = Layout<Shape<Int<bN>, Int<bK>>, Stride<Int<1>, Int<0>>>;
    using SmemLayoutC = Layout< Shape<Int<bM>, Int<bN>>, Stride<Int<bN>, Int<1>> >;

    using g2s_copyA_op = UniversalCopy<cute::uint128_t>;
    using g2s_copyA_traits = Copy_Traits<g2s_copyA_op>;
    using g2s_copyA_atom = Copy_Atom<g2s_copyA_traits, scalar_t>;

    static constexpr int vec_scalar_t_copy = sizeof(cute::uint128_t) / sizeof(scalar_t);
    static constexpr int vec_accscalar_t_copy = sizeof(cute::uint128_t) / sizeof(accscalar_t);
    static constexpr int g2scopyA_K_threads = bK / vec_scalar_t_copy;
    static constexpr int g2scopyA_M_threads = threads / g2scopyA_K_threads;

    using G2SCopyA =
      decltype(make_tiled_copy(g2s_copyA_atom{},
                               make_layout(make_shape(Int<g2scopyA_M_threads>{}, Int<g2scopyA_K_threads>{}),
                                           make_stride(Int<g2scopyA_K_threads>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<vec_scalar_t_copy>{}))));

    using g2s_copyB_op = UniversalCopy<cute::uint128_t>;
    using g2s_copyB_traits = Copy_Traits<g2s_copyB_op>;
    using g2s_copyB_atom = Copy_Atom<g2s_copyB_traits, uint32_t>;


    static constexpr int vec_g2scopyB = sizeof(cute::uint128_t) / sizeof(uint32_t) > bK_q 
        ? bK_q : sizeof(cute::uint128_t) / sizeof(uint32_t);
    static constexpr int g2scopyB_N_threads = bN / vec_g2scopyB;
    static constexpr int g2scopyB_K_threads = threads / g2scopyB_N_threads;

    using G2SCopyB_q_T =
        decltype(make_tiled_copy(g2s_copyB_atom{},
                               make_layout(make_shape(Int<g2scopyB_N_threads>{}, Int<g2scopyB_K_threads>{}),
                                           make_stride(Int<1>{}, Int<g2scopyB_N_threads>{})),
                               make_layout(make_shape(Int<vec_g2scopyB>{}, Int<1>{}))));

    using s2r_dq_copy_op = UniversalCopy<cute::uint32_t>;
    using s2r_dq_copy_traits = Copy_Traits<s2r_dq_copy_op>;
    using s2r_dq_copy_atom = Copy_Atom<s2r_dq_copy_traits, uint32_t>;

    using r2s_dq_copy_op = UniversalCopy<scalar_t>;
    using r2s_dq_copy_traits = Copy_Traits<r2s_dq_copy_op>;
    using r2s_dq_copy_atom = Copy_Atom<r2s_dq_copy_traits, scalar_t>;

    static constexpr int s2r_dq_copy_K_threads = bK_q;
    static constexpr int s2r_dq_copy_N_threads = threads / s2r_dq_copy_K_threads;

    static constexpr int r2s_dq_copy_K_threads = bK_q;
    static constexpr int r2s_dq_copy_N_threads = threads / r2s_dq_copy_K_threads;

    using S2RDQCopy =  decltype(make_tiled_copy(s2r_dq_copy_atom{},
                               make_layout(make_shape(Int<s2r_dq_copy_N_threads>{}, Int<s2r_dq_copy_K_threads>{}),
                                           make_stride(Int<s2r_dq_copy_K_threads>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<1>{}))));

    using R2SDQCopy = decltype(make_tiled_copy(r2s_dq_copy_atom{},
                               make_layout(make_shape(Int<r2s_dq_copy_N_threads>{}, Int<r2s_dq_copy_K_threads>{}),
                                           make_stride(Int<r2s_dq_copy_K_threads>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<q_div>{}))));



    using s2g_copy_op = UniversalCopy<cute::uint128_t>;
    using s2g_copy_traits = Copy_Traits<s2g_copy_op>;
    using s2g_copy_atom = Copy_Atom<s2g_copy_traits, scalar_t>;
    using s2g_copy_atom_reduce = Copy_Atom<s2g_copy_traits, accscalar_t>;

    static constexpr int s2gcopyC_N_threads = bN / vec_scalar_t_copy;
    static constexpr int s2gcopyC_M_threads = threads / s2gcopyC_N_threads;

    static constexpr int s2gcopyC_N_threads_reduce = bN / vec_accscalar_t_copy;
    static constexpr int s2gcopyC_M_threads_reduce = threads / s2gcopyC_N_threads_reduce;

    using S2GCopyC =   decltype(make_tiled_copy(s2g_copy_atom{},
                               make_layout(make_shape(Int<s2gcopyC_M_threads>{}, Int<s2gcopyC_N_threads>{}),
                                           make_stride(Int<s2gcopyC_N_threads>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<vec_scalar_t_copy>{}))));

    using S2GCopyCReduce =   decltype(make_tiled_copy(s2g_copy_atom_reduce{},
                            make_layout(make_shape(Int<s2gcopyC_M_threads_reduce>{}, Int<s2gcopyC_N_threads_reduce>{}),
                                        make_stride(Int<s2gcopyC_N_threads_reduce>{}, Int<1>{})),
                            make_layout(make_shape(Int<1>{}, Int<vec_accscalar_t_copy>{}))));

    //0 2 4 6 1 3 5 7
    using B_q_dq_layout = Layout<Shape<Shape<_4, _2>, _1>, Stride<Stride<_2, _1>, _1>>; 

    //for reduce kernel
    static constexpr int reduce_threads = 1024;
    using reduce_g2r_copy_op = UniversalCopy<cute::uint128_t>;
    using reduce_g2r_copy_traits = Copy_Traits<reduce_g2r_copy_op>;
    using reduce_g2r_copy_atom = Copy_Atom<reduce_g2r_copy_traits, accscalar_t>;

    static constexpr int reduce_g2rcopyC_N_threads = bN / vec_accscalar_t_copy;
    static constexpr int reduce_g2rcopyC_M_threads = reduce_threads / reduce_g2rcopyC_N_threads;

    using ReduceG2RCopyCReduce = decltype(make_tiled_copy(reduce_g2r_copy_atom{},
                            make_layout(make_shape(Int<reduce_g2rcopyC_M_threads>{}, Int<reduce_g2rcopyC_N_threads>{}),
                                        make_stride(Int<reduce_g2rcopyC_N_threads>{}, Int<1>{})),
                            make_layout(make_shape(Int<1>{}, Int<vec_accscalar_t_copy>{}))));

    using STORE_C_T = aligned_vector<scalar_t, vec_accscalar_t_copy>;


    using reduce_r2g_copy_op = UniversalCopy<STORE_C_T>;
    using reduce_r2g_copy_traits = Copy_Traits<reduce_r2g_copy_op>;
    using reduce_r2g_copy_atom = Copy_Atom<reduce_r2g_copy_traits, scalar_t>;

    using ReduceR2GCopyC = decltype(make_tiled_copy(reduce_r2g_copy_atom{},
                            make_layout(make_shape(Int<reduce_g2rcopyC_M_threads>{}, Int<reduce_g2rcopyC_N_threads>{}),
                                        make_stride(Int<reduce_g2rcopyC_N_threads>{}, Int<1>{})),
                            make_layout(make_shape(Int<1>{}, Int<vec_accscalar_t_copy>{}))));

};



template<typename scalar_t, typename accscalar_t>
struct GptQ_Kernel_Params {
    const scalar_t* A_ptr;
    const uint32_t* B_q_ptr;
    const uint32_t* B_zeros_ptr;
    const scalar_t* B_scales_ptr;
    const int* B_g_idx_ptr;
    scalar_t* C_ptr;
    accscalar_t* C_reduce_ptr;
    int M;
    int N;
    int K;
    int split_k_slices;
    int bit;
    int group_size;
    bool use_exllama;
};


template<typename scalar_t, typename accscalar_t>
void run_cutlass_gptq_gemm(GptQ_Kernel_Params<scalar_t, accscalar_t> kernel_params);

}

#endif
