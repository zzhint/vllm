#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include "cutlass_extensions/torch_utils.hpp"

#include "core/registration.h"

#include "cutlass/cutlass.h"
#include <limits>

#include <cute/tensor.hpp>
#include "cute/arch/mma_sm80.hpp"
/*
template<typename scalar_t, typename GEMM_CONGIG>
__global__ void cutlass_gptq_gemm_kernel(
    const scalar_t* __restrict__ a, 
    const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const scalar_t* __restrict__ b_gptq_scales, 
    scalar_t* __restrict__ c,
    const int size_m, const int size_n, const int size_k,
    const int* __restrict__ b_q_perm) {

    using TiledMMA = typename GEMM_CONGIG::TiledMMA;
    using CTATiler = typename GEMM_CONGIG::CTATiler;

    using SmemLayoutA = GEMM_CONGIG::SmemLayoutA;
    using SmemLayoutB_T = GEMM_CONGIG::SmemLayoutB_T;
    using SmemLayoutB_q =  GEMM_CONGIG::SmemLayoutB_q;
    using SmemLayoutB_q_T = GEMM_CONGIG::SmemLayoutB_q_T;
    using SmemLayoutB_zeros = GEMM_CONGIG::SmemLayoutB_zeros;
    using SmemLayoutB_scales = GEMM_CONGIG::SmemLayoutB_scales;

    using G2SCopyA = GEMM_CONGIG::G2SCopyA;
    using G2SCopyB_q_T = GEMM_CONGIG::G2SCopyB_q_T;

    using S2RDQCopy = GEMM_CONGIG::S2RDQCopy;
    using R2SDQCopy = GEMM_CONGIG::R2SDQCopy;
    using B_q_dq_layout = GEMM_CONGIG::B_q_dq_layout;

    constexpr int bM = GEMM_CONGIG::bM;
    constexpr int bN = GEMM_CONGIG::bN;
    constexpr int bK = GEMM_CONGIG::bK;
    constexpr int bK_q = GEMM_CONGIG::bK_q;
    constexpr int bK_div_group_size = GEMM_CONGIG::bK_div_group_size;
    constexpr int bit = GEMM_CONGIG::bit;
    constexpr int q_div = GEMM_CONGIG::q_div;
    constexpr int bN_q = GEMM_CONGIG::bN_q;

    constexpr int smem_size_A = GEMM_CONGIG::smem_size_A;
    constexpr int smem_size_B = GEMM_CONGIG::smem_size_B;
    constexpr int smem_size_B_q = GEMM_CONGIG::smem_size_B_q;
    constexpr int smem_size_B_zeros = GEMM_CONGIG::smem_size_B_zeros;
    constexpr int smem_size_B_scales = GEMM_CONGIG::smem_size_B_scales;

    constexpr int smem_size = GEMM_CONGIG::smem_size;
    constexpr int group_size = GEMM_CONGIG::group_size;

    __shared__ scalar_t smem[smem_size];

    scalar_t* A_smem = smem;
    scalar_t* B_smem = A_smem + smem_size_A;
    scalar_t* B_q_smem = B_smem + smem_size_B;
    scalar_t* B_zeros = B_q_smem + smem_size_B_q;
    scalar_t* B_scales = B_zeros + smem_size_B_zeros;
    scalar_t* C_smem = smem;

    int block_idx_x = blockIdx.x;
    int block_idx_y = blockIdx.y;

    auto mA = make_tensor(make_gmem_ptr(a), make_shape(size_m, size_k), make_stride(size_k, Int<1>{}));
    auto mB_q = make_tensor(make_gmem_ptr(b_q_weight), make_shape(k / q_div, n), make_stride(Int<1>{}, k / q_div));
    auto mB_q_T = make_tensor(make_gmem_ptr(b_q_weight), make_shape(n, k / q_div), make_stride(k / q_div, Int<1>{}));
    auto mC = make_tensor(make_gmem_ptr(c), make_shape(m, n), make_stride(m, Int<1>{}));



    auto mB_zeros = make_tensor(make_gmem_ptr(b_gptq_qzeros),
            make_shape(make_shape(q_div, bN_q), make_shape(group_size, bK_div_group_size)),
            make_stride(make_stride(0, bK_div_group_size), make_stride(0, 1)));
    auto mB_scales = make_tensor(make_gmem_ptr(b_gptq_scales), 
            make_shape(n, make_shape(group_size, bK_div_group_size)),
            make_stride(bK_div_group_size, make_stride(0, 1)));

    Tensor gA = local_tile(mA, make_tile(Int<bM>{}, Int<bK>{}),
                            make_coord(block_idx_x, _));  
    Tensor gB_q_T = local_tile(mB_q_T, make_tile(Int<bN>{}, Int<bK_q>{}),
                            make_coord(block_idx_y, _));

    Tensor gB_zeros = local_tile(mB_zeros, make_tile(Int<bN>{}, Int<bK>{}),
                            make_coord(block_idx_y, _));
    
    Tensor gB_scales = local_tile(mB_scales, make_tile(Int<bN>{}, Int<bK>{}),
                            make_coord(block_idx_y, _));

    Tensor gC = local_tile(mC, make_tile(Int<bM>{}, Int<bN>{}),
                            make_coord(block_idx_x, block_idx_y));

    auto sA = make_tensor(make_smem_ptr(A_smem), SmemLayoutA{});
    auto sB_T = make_tensor(make_smem_ptr(B_smem), SmemLayoutB_T{});
    auto sB_q_T = make_tensor(make_smem_ptr(B_q_smem), SmemLayoutB_q_T{});     

    auto sB_zeros = make_tensor(make_smem_ptr(B_zeros), SmemLayoutB_zeros{}); 
    auto sB_scales = make_tensor(make_smem_ptr(B_scales), SmemLayoutB_scales{});                  

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCsA_mma = thr_mma.partition_A(sA);  // (MMA, MMA_M, MMA_K)
    auto tCsB_mma = thr_mma.partition_B(sB);  // (MMA, MMA_N, MMA_K)
    auto tCsC_mma = thr_mma.partition_C(gC);

    auto tCrA_mma = thr_mma.partition_fragment_A(sA(_, _, 0));  // (MMA, MMA_M, MMA_K)
    auto tCrB_mma = thr_mma.partition_fragment_B(sB(_, _, 0));  // (MMA, MMA_N, MMA_K)
    auto tCrC_mma = thr_mma.partition_fragment_C(gC);     // (MMA, MMA_M, MMA_N)   

    G2SCopyA tiled_g2s_copy_A;
    G2SCopyB_q_T tiled_g2s_copy_B_q_T;

    auto thr_g2s_copy_A = tiled_g2s_copy_A.get_slice(threadIdx.x);
    auto tAgA_g2s_copy = thr_g2s_copy_A.partition_S(gA);
    auto tAsA_g2s_copy = thr_g2s_copy_A.partition_D(sA);   

    auto thr_g2s_copy_B_q_T = tiled_g2s_copy_B_q_T.get_slice(threadIdx.x);
    auto tBgB_q_T_g2s_copy = thr_g2s_copy_B_q_T.partition_S(gB_q_T);
    auto tBsB_q_T_g2s_copy = thr_g2s_copy_B_q_T.partition_D(sB_q_T);

    S2RDQCopy tiled_s2r_dq_copy;
    R2SDQCopy tiled_r2s_dq_copy;

    auto thr_s2r_dq_copy = tiled_s2r_dq_copy.get_slice(threadIdx.x);
    auto sB_q_T_s2r_dq_copy = thr_s2r_dq_copy.partition_S(sB_q_T);
    auto rB_q_T_s2r_dq_copy = make_tensor_like(sB_q_T_s2r_dq_copy);

    auto sB_zeros_s2r_dq_copy = thr_s2r_dq_copy.partition_S(sB_zeros);
    auto rB_zeros_s2r_dq_copy = make_tensor_like(sB_zeros_s2r_dq_copy);

    auto sB_scales_s2r_dq_copy = thr_s2r_dq_copy.partition_S(sB_scales);
    auto rB_scales_s2r_dq_copy = make_tensor_like(sB_scales_s2r_dq_copy);

    auto thr_r2s_dq_copy = tiled_r2s_dq_copy.get_slice(threadIdx.x);
    auto sB_T_r2s_dq_copy = thr_r2s_dq_copy.partition_S(sB_T);


    int n_k_tile = size<2>(gA);


    for(int k_tile = 0; k_tile < n_k_tile; k_tile++) {
        

        cute::copy(tiled_g2s_copy_A, tAgA_g2s_copy(_,_,_, k_tile), tAsA_g2s_copy);
        cute::copy(tiled_g2s_copy_B_q_T, tBgB_q_T_g2s_copy(_,_,_, k_tile), tBsB_q_T_g2s_copy);
        if(threadIdx.x < bN) {
            sB_scales(threadIdx.x, 0) = gB_scales(threadIdx.x, 0);
            uint32_t zeros_val = gB_zeros(threadIdx.x, 0);
            int shift_offset = (threadIdx.x % q_div) * bit;
            zeros_val >>= shift_offset;
            zeros_val &= 0x0f;
            sB_zeros(threadIdx.x, 0) = int_to_float<scalar_t>(zeros_val);
        }
        __syncthreads();
        
        cute::copy(tiled_s2r_dq_copy, sB_q_T_s2r_dq_copy, rB_q_T_s2r_dq_copy);
        cute::copy(tiled_s2r_dq_copy, sB_zeros_s2r_dq_copy, rB_zeros_s2r_dq_copy);
        cute::copy(tiled_s2r_dq_copy, sB_scales_s2r_dq_copy, rB_scales_s2r_dq_copy);

        for(int in=0; in < size<1>(sB_q_T_s2r_dq_copy); in++) {
            for(int ik=0; ik < size<2>(sB_q_T_s2r_dq_copy); ik++) {
                uint32_t B_q_T_val = rB_q_T_s2r_dq_copy(0, in, ik);
                int thread_local_idx = threadIdx.x % q_div;
                int val_local_idx = B_q_dq_layout(thread_local_idx, 0);
                scalar_t val = dq_one_data<scalar_t, bit>(B_q_T_val, val_local_idx);
                scalar_t zero_val = rB_zeros_s2r_dq_copy(0, in, ik);
                scalar_t scale_val = rB_scales_s2r_dq_copy(0, in, ik);

                sB_T_r2s_dq_copy(0, in, ik) = (val - zero_val) * scale_val;
            }
        }
        __syncthreads();

        cute::copy(tCsA_mma, tCrA_mma);
        cute::copy(tCsB_mma, tCrB_mma);
        cute::gemm(tiled_mma, tCrC_mma, tCrA_mma, tCrB_mma, tCrC_mma);

    }
    cute::copy(tCrC_mma, tCgC_mma); 



}

*/

namespace cutlass_gptq {

using namespace cute;
template<typename scalar_t, int _bM, int _bN, int _bK, int _kStage, int _l2_tile, int _bit, int _group_size>
struct GPTQ_GemmConfig {
    
    
    using T = scalar_t;
    using mma_op = SM80_16x8x16_F16F16F16F16_TN; // SM80_16x8x16_F32BF16BF16F32_TN
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    static constexpr int bM = _bM;
    static constexpr int bN = _bN;
    static constexpr int bK = _bK;
    static constexpr int kStage = _kStage;
    static constexpr int l2_tile = _l2_tile;
    static constexpr int bit = _bit;
    static constexpr int group_size = _group_size;

    //how many data in one uint32_t
    static constexpr int q_div = 32 / bit;

    static constexpr int bK_q = bK / q_div;
    static constexpr int bK_div_group_size = bK / group_size;
    static constexpr int bN_q = bN / q_div;

    //smem size
    static constexpr int smem_size_A = bM * bK * kStage * sizeof(T);
    static constexpr int smem_size_B = bN * bK * kStage * sizeof(T);
    static constexpr int smem_size_B_q = bN * bK_q * kStage * sizeof(uint32_t);
    static constexpr int smem_size_B_zeros = bK_div_group_size * bN * sizeof(uint32_t);
    static constexpr int smem_size_B_scales = bK_div_group_size * bN * sizeof(T);
    static constexpr int smem_size_C = bM * bN * sizeof(T);

    static constexpr int smem_size_AB = smem_size_A 
        + smem_size_B + smem_size_B_q
        + smem_size_B_zeros + smem_size_B_scales;
    static constexpr int smem_size = smem_size_C > smem_size_AB 
            ? smem_size_C : smem_size_AB;


    using TiledMMA = decltype(make_tiled_mma(UniversalFMA<T,T,T>{},
                                 Layout<Shape<_16,_8,_1>>{}, Tile<Int<bM>, Int<bN>, Int<16>>{}));
    //using TiledMMA = decltype(make_tiled_mma(mma_atom{}, Layout<Shape<_2,_2,_1>>{}, Tile<Int<bM>, Int<bN>, Int<16>>{}));
    static constexpr int threads = size(TiledMMA{});
    using CTATiler = Shape<Int<bM>, Int<bN>, Int<bK>>;

    using SmemLayoutA = Layout< Shape<Int<bM>, Int<bK>>, Stride<Int<bK>, Int<1>> >;
    using SmemLayoutB_T = Layout< Shape<Int<bN>, Int<bK>>, Stride<Int<1>, Int<bN>> >;
    //using SmemLayoutB_q_T = Layout< Shape<     Int<bN>,    Shape<Int<q_div>, Int<bK_q>>>,
     //                               Stride<Stride<Int<1>>, Stride<Int<0>,    Int<bN>>> >;

    using SmemLayoutB_q_T = Layout< Shape<Int<bN>,  Int<bK_q>>,
                                    Stride<Int<1>,    Int<bN>> >;

    using g2s_copyA_op = UniversalCopy<cute::uint128_t>;
    using g2s_copyA_traits = Copy_Traits<g2s_copyA_op>;
    using g2s_copyA_atom = Copy_Atom<g2s_copyA_traits, T>;

    static constexpr int g2scopyA_K_threads = bK / 8;
    static constexpr int g2scopyA_M_threads = threads / g2scopyA_K_threads;


    
    using SmemLayoutB_zeros_q = Layout< Shape< Shape< Int<q_div>, Int<bN_q> >, Int<bK>>, 
                                        Stride< Stride<Int<0>,     Int<1>>,    Int<0>>>;
    using SmemLayoutB_zeros = Layout<Shape<Int<bN>, Int<bK>>, Stride<Int<1>, Int<0>>>;
    using SmemLayoutB_scales = Layout<Shape<Int<bN>, Int<bK>>, Stride<Int<1>, Int<0>>>;


    using SmemLayoutC = Layout< Shape<Int<bM>, Int<bN>>, Stride<Int<bN>, Int<1>> >;
    
    using G2SCopyA =
      decltype(make_tiled_copy(g2s_copyA_atom{},
                               make_layout(make_shape(Int<g2scopyA_M_threads>{}, Int<g2scopyA_K_threads>{}),
                                           make_stride(Int<g2scopyA_K_threads>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

    using g2s_copyB_op = UniversalCopy<cute::uint128_t>;
    using g2s_copyB_traits = Copy_Traits<g2s_copyB_op>;
    using g2s_copyB_atom = Copy_Atom<g2s_copyB_traits, uint32_t>;

    static constexpr int g2scopyB_N_threads = bN / 4;
    static constexpr int g2scopyB_K_threads = threads / g2scopyB_N_threads;

    using G2SCopyB_q_T =
        decltype(make_tiled_copy(g2s_copyB_atom{},
                               make_layout(make_shape(Int<g2scopyB_N_threads>{}, Int<g2scopyB_K_threads>{}),
                                           make_stride(Int<1>{}, Int<g2scopyB_N_threads>{})),
                               make_layout(make_shape(Int<4>{}, Int<1>{}))));

    using s2r_dq_copy_op = UniversalCopy<cute::uint32_t>;
    using s2r_dq_copy_traits = Copy_Traits<s2r_dq_copy_op>;
    using s2r_dq_copy_atom = Copy_Atom<s2r_dq_copy_traits, uint32_t>;

    using r2s_dq_copy_op = UniversalCopy<T>;
    using r2s_dq_copy_traits = Copy_Traits<r2s_dq_copy_op>;
    using r2s_dq_copy_atom = Copy_Atom<r2s_dq_copy_traits, T>;

    static constexpr int dq_copy_K_threads = bK;
    static constexpr int dq_copy_N_threads = threads / dq_copy_K_threads;

    using S2RDQCopy =  decltype(make_tiled_copy(s2r_dq_copy_atom{},
                               make_layout(make_shape(Int<dq_copy_N_threads>{}, Int<dq_copy_K_threads>{}),
                                           make_stride(Int<dq_copy_K_threads>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<1>{}))));

    using R2SDQCopy = decltype(make_tiled_copy(r2s_dq_copy_atom{},
                               make_layout(make_shape(Int<dq_copy_N_threads>{}, Int<dq_copy_K_threads>{}),
                                           make_stride(Int<dq_copy_K_threads>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<1>{}))));



    using s2g_copy_op = UniversalCopy<cute::uint128_t>;
    using s2g_copy_traits = Copy_Traits<s2g_copy_op>;
    using s2g_copy_atom = Copy_Atom<s2g_copy_traits, T>;

    static_assert(bN % 8 == 0, "bK must be divisible by 8!");
    static constexpr int s2gcopyC_N_threads = bN / 8;
    static constexpr int s2gcopyC_M_threads = threads / s2gcopyC_N_threads;

    using S2GCopyC =   decltype(make_tiled_copy(s2g_copy_atom{},
                               make_layout(make_shape(Int<s2gcopyC_M_threads>{}, Int<s2gcopyC_N_threads>{}),
                                           make_stride(Int<s2gcopyC_N_threads>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

    //0 2 4 6 1 3 5 7
    using B_q_dq_layout = Layout<Shape<Shape<_4, _2>, _1>, Stride<Stride<_2, _1>, _1>>; 

};


template<typename FloatT>
__forceinline__ __device__ FloatT int_to_float(int x);

template<>
__forceinline__ __device__ half int_to_float<half>(int x) {
    return __int2half_rn(x);
}

template<>
__forceinline__ __device__ __nv_bfloat16 int_to_float<__nv_bfloat16>(int x) {
    return __int2bfloat16_rn(x);
}

template<typename scalar_t, int bit, bool is_zero>
__forceinline__ __device__ scalar_t dq_one_data(uint32_t val, int val_local_idx) {
    val = val >> (val_local_idx * bit);
    val = val & ((1 << bit) - 1);
    if(is_zero) {
        val += 1;
    }
    return int_to_float<scalar_t>(val);
}


template<typename scalar_t, typename GEMM_CONGIG>
__global__ void cutlass_gptq_gemm_kernel(
    const scalar_t* __restrict__ a, 
    const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const scalar_t* __restrict__ b_gptq_scales, 
    scalar_t* __restrict__ c,
    const int size_m, const int size_n, const int size_k,
    const int* __restrict__ b_q_perm) {

    using namespace cute;
    using TiledMMA = typename GEMM_CONGIG::TiledMMA;
    using CTATiler = typename GEMM_CONGIG::CTATiler;

    using SmemLayoutA = typename GEMM_CONGIG::SmemLayoutA;
    using SmemLayoutB_T = typename GEMM_CONGIG::SmemLayoutB_T;
    //using SmemLayoutB_q =  typename GEMM_CONGIG::SmemLayoutB_q;
    using SmemLayoutB_q_T = typename GEMM_CONGIG::SmemLayoutB_q_T;
    using SmemLayoutB_zeros = typename GEMM_CONGIG::SmemLayoutB_zeros;
    using SmemLayoutB_scales = typename GEMM_CONGIG::SmemLayoutB_scales;

    using G2SCopyA = typename GEMM_CONGIG::G2SCopyA;
    using G2SCopyB_q_T = typename GEMM_CONGIG::G2SCopyB_q_T;

    using S2RDQCopy = typename GEMM_CONGIG::S2RDQCopy;
    using R2SDQCopy = typename GEMM_CONGIG::R2SDQCopy;
    using B_q_dq_layout = typename GEMM_CONGIG::B_q_dq_layout;

    constexpr int bM = GEMM_CONGIG::bM;
    constexpr int bN = GEMM_CONGIG::bN;
    constexpr int bK = GEMM_CONGIG::bK;
    constexpr int bK_q = GEMM_CONGIG::bK_q;
    constexpr int bK_div_group_size = GEMM_CONGIG::bK_div_group_size;
    constexpr int bit = GEMM_CONGIG::bit;
    constexpr int q_div = GEMM_CONGIG::q_div;
    constexpr int bN_q = GEMM_CONGIG::bN_q;

    constexpr int smem_size_A = GEMM_CONGIG::smem_size_A;
    constexpr int smem_size_B = GEMM_CONGIG::smem_size_B;
    constexpr int smem_size_B_q = GEMM_CONGIG::smem_size_B_q;
    constexpr int smem_size_B_zeros = GEMM_CONGIG::smem_size_B_zeros;
    constexpr int smem_size_B_scales = GEMM_CONGIG::smem_size_B_scales;

    constexpr int smem_size = GEMM_CONGIG::smem_size;
    constexpr int group_size = GEMM_CONGIG::group_size;

    extern __shared__ char smem_ptr[];

    scalar_t* A_smem = reinterpret_cast<scalar_t* >(smem_ptr);
    scalar_t* B_smem = reinterpret_cast<scalar_t* >(smem_ptr + smem_size_A);
    uint32_t* B_q_smem = reinterpret_cast<uint32_t* >(smem_ptr + smem_size_A + smem_size_B);
    uint32_t* B_zeros = reinterpret_cast<uint32_t* >(smem_ptr + smem_size_A + smem_size_B + smem_size_B_q);
    scalar_t* B_scales = reinterpret_cast<scalar_t* >(smem_ptr + smem_size_A + smem_size_B + smem_size_B_q + smem_size_B_zeros);
    scalar_t* C_smem = reinterpret_cast<scalar_t* >(smem_ptr);

    int block_idx_x = blockIdx.x;
    int block_idx_y = blockIdx.y;

    auto mA = make_tensor(make_gmem_ptr(a), make_shape(size_m, size_k), make_stride(size_k, Int<1>{}));
    auto mB_q = make_tensor(make_gmem_ptr(b_q_weight), make_shape(size_k / q_div, size_n), make_stride(size_n, Int<1>{}));
    auto mB_q_T = make_tensor(make_gmem_ptr(b_q_weight), make_shape(size_n, size_k / q_div), make_stride(Int<1>{}, size_n));
    auto mC = make_tensor(make_gmem_ptr(c), make_shape(size_m, size_n), make_stride(size_n, Int<1>{}));

    auto mB_zeros = make_tensor(make_gmem_ptr(b_gptq_qzeros), 
        make_shape(size_n / q_div, size_k / group_size), make_stride(Int<1>{}, size_n / q_div));

    auto mB_scales = make_tensor(make_gmem_ptr(b_gptq_scales),
        make_shape(size_n, size_k / group_size), make_stride(Int<1>{}, size_n));

    Tensor gA = local_tile(mA, make_tile(Int<bM>{}, Int<bK>{}),
                            make_coord(block_idx_x, _));  
    Tensor gB_q_T = local_tile(mB_q_T, make_tile(Int<bN>{}, Int<bK_q>{}),
                            make_coord(block_idx_y, _));

    Tensor gB_zeros = local_tile(mB_zeros, make_tile(Int<bN_q>{}, Int<bK_div_group_size>{}),
                            make_coord(block_idx_y, _));
    
    Tensor gB_scales = local_tile(mB_scales, make_tile(Int<bN>{}, Int<bK_div_group_size>{}),
                            make_coord(block_idx_y, _));

    Tensor gC = local_tile(mC, make_tile(Int<bM>{}, Int<bN>{}),
                            make_coord(block_idx_x, block_idx_y));




    auto sA = make_tensor(make_smem_ptr(A_smem), SmemLayoutA{});
    auto sB_T = make_tensor(make_smem_ptr(B_smem), SmemLayoutB_T{});
    auto sB_q_T = make_tensor(make_smem_ptr(B_q_smem), SmemLayoutB_q_T{});     

    auto sB_zeros = make_tensor(make_smem_ptr(B_zeros), SmemLayoutB_zeros{}); 
    auto sB_scales = make_tensor(make_smem_ptr(B_scales), SmemLayoutB_scales{});                  


    G2SCopyA tiled_g2s_copy_A;
    G2SCopyB_q_T tiled_g2s_copy_B_q_T;

    auto thr_g2s_copy_A = tiled_g2s_copy_A.get_slice(threadIdx.x);
    auto tAgA_g2s_copy = thr_g2s_copy_A.partition_S(gA);
    auto tAsA_g2s_copy = thr_g2s_copy_A.partition_D(sA);   

    auto thr_g2s_copy_B_q_T = tiled_g2s_copy_B_q_T.get_slice(threadIdx.x);
    auto tBgB_q_T_g2s_copy = thr_g2s_copy_B_q_T.partition_S(gB_q_T);
    auto tBsB_q_T_g2s_copy = thr_g2s_copy_B_q_T.partition_D(sB_q_T);
    int n_k_tile = size<2>(gA);

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCsA_mma = thr_mma.partition_A(sA);  // (MMA, MMA_M, MMA_K)
    auto tCsB_mma = thr_mma.partition_B(sB_T);  // (MMA, MMA_N, MMA_K)
    auto tCgC_mma = thr_mma.partition_C(gC);

    auto tCrA_mma = thr_mma.partition_fragment_A(sA);  // (MMA, MMA_M, MMA_K)
    auto tCrB_mma = thr_mma.partition_fragment_B(sB_T);  // (MMA, MMA_N, MMA_K)
    auto tCrC_mma = thr_mma.partition_fragment_C(gC);     // (MMA, MMA_M, MMA_N)   

    if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        print("mA\n");
        print(mA);
        print("\n");
        print("mB_q\n");
        print(mB_q);
        print("\n");
        print("mB_q_T\n");
        print(mB_q_T);
        print("\n");
        print("mC\n");
        print(mC);
        print("\n");
        print("mB_zeros\n");
        print(mB_zeros);
        print("\n");
        print("mB_scales\n");
        print(mB_scales);
        print("\n");
        print("gA\n");
        print(gA);
        print("\n");
        print("gB_q_T\n");
        print(gB_q_T);
        print("\n");
        print("gB_zeros\n");
        print(gB_zeros);
        print("\n");
        print("gB_scales\n");
        print(gB_scales);
        print("\n");
        print("gC\n");
        print(gC);
        print("\n");
        print("sA\n");
        print(sA);
        print("\n");
        print("sB_T\n");
        print(sB_T);
        print("\n");
        print("sB_q_T\n");
        print(sB_q_T);
        print("\n");
        print("sB_zeros\n");
        print(sB_zeros);
        print("\n");
        print("sB_scales\n");
        print(sB_scales);
        print("\n");

        print("tAgA_g2s_copy\n");
        print(tAgA_g2s_copy);
        print("\n");
        print("tAsA_g2s_copy\n");
        print(tAsA_g2s_copy);
        print("\n");
        print("tBgB_q_T_g2s_copy\n");
        print(tBgB_q_T_g2s_copy);
        print("\n");
        print("tBsB_q_T_g2s_copy\n");
        print(tBsB_q_T_g2s_copy);
        print("\n");
        print("tCsA_mma\n");
        print(tCsA_mma);
        print("\n");
        print("tCsB_mma\n");
        print(tCsB_mma);
        print("\n");
        print("tCgC_mma\n");
        print(tCgC_mma);
        print("\n");
        print("tCrA_mma\n");
        print(tCrA_mma);
        print("\n");
        print("tCrB_mma\n");
        print(tCrB_mma);
        print("\n");
        print("tCrC_mma\n");
        print(tCrC_mma);
        print("\n");

    }
    clear(tCrC_mma);

    B_q_dq_layout B_q_dq_map_idx;
    for(int k_tile = 0; k_tile < n_k_tile; k_tile++) {
        if(threadIdx.x < bN) {
            sB_scales(threadIdx.x, 0) = gB_scales(threadIdx.x, 0, k_tile);
            uint32_t zeros_val_q = gB_zeros(threadIdx.x / q_div, 0, k_tile);
            int val_local_idx = threadIdx.x % q_div;
            scalar_t thread_zero = dq_one_data<scalar_t, bit, true>(zeros_val_q, val_local_idx);
            sB_zeros(threadIdx.x, 0) = thread_zero;
        }
        cute::copy(tiled_g2s_copy_A, tAgA_g2s_copy(_,_,_, k_tile), tAsA_g2s_copy);
        cute::copy(tiled_g2s_copy_B_q_T, tBgB_q_T_g2s_copy(_,_,_, k_tile), tBsB_q_T_g2s_copy);
        __syncthreads();

        for(int i_n=0; i_n < size<0>(sB_T); i_n++) {
            scalar_t thread_zero = sB_zeros(i_n, threadIdx.x);
            scalar_t thread_scale = sB_scales(i_n, threadIdx.x);
            int thread_q_idx = threadIdx.x / q_div;
            uint32_t B_val_q = sB_q_T(i_n, thread_q_idx);
            int store_local_idx = B_q_dq_map_idx(threadIdx.x % q_div);
            scalar_t thread_b = dq_one_data<scalar_t, bit, false>(B_val_q, threadIdx.x % q_div);
            sB_T(i_n, thread_q_idx * q_div + store_local_idx) = (thread_b - thread_zero) * thread_scale;

        }
        __syncthreads();

      
      

        if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
            print("sB_T \n");
            print(sB_T);
            print("\n");
            for(int i = 0; i<size<0>(sB_T); i++) {
                for(int j=0; j<size<1>(sB_T); j++) {
                    scalar_t val = sB_T(i,j);
                    float val_float = static_cast<float>(val);
                    printf("%.2f\t", val_float);
                }
                printf("\n");
            }

        }

        cute::copy(tCsA_mma, tCrA_mma);
        cute::copy(tCsB_mma, tCrB_mma);
        cute::gemm(tiled_mma, tCrC_mma, tCrA_mma, tCrB_mma, tCrC_mma);

    }
    cute::copy(tCrC_mma, tCgC_mma); 


}

template<typename scalar_t>
void run_cutlass_gptq_gemm(const scalar_t* a,
                           const uint32_t* b_q_weight,
                           const uint32_t* b_gptq_qzeros,
                           const scalar_t* b_gptq_scales, const int* b_g_idx,
                           scalar_t* c, scalar_t* temp_dq, int size_m, int size_n,
                           int size_k, int groups, bool use_exllama, int bit) {

    using TEST_GEMM_CONGIG = GPTQ_GemmConfig<scalar_t, 32, 32, 128, 1, 1, 4, 128>;
    dim3 grid(size_m / TEST_GEMM_CONGIG::bM, size_n / TEST_GEMM_CONGIG::bN);
    dim3 block(TEST_GEMM_CONGIG::threads);
    int smem_size = TEST_GEMM_CONGIG::smem_size;

    cutlass_gptq_gemm_kernel<scalar_t, TEST_GEMM_CONGIG><<<grid, block, smem_size>>>(
        a, b_q_weight, b_gptq_qzeros, b_gptq_scales, c, size_m, size_n, size_k, b_g_idx
    );

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