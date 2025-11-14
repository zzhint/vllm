#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include "cutlass_extensions/torch_utils.hpp"

#include "core/registration.h"

#include "cutlass/cutlass.h"
#include <limits>
#include "cutlass_gptq_gemm.cuh"
#include <cute/tensor.hpp>
#include "cute/arch/mma_sm80.hpp"
/*
        if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
            print("sB \n");
            print(sB);
            print("\n");
            for(int i = 0; i<size<0>(sB); i++) {
                for(int j=0; j<size<1>(sB); j++) {
                    scalar_t val = sB(i,j);
                    float val_float = static_cast<float>(val);
                    printf("%.4f\t", val_float);
                }
                printf("\n");
            }

        }

        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0  && threadIdx.x == 0 && idx_group == 0 && idx_bK == 0) {
                
                print("mA\n");
                print(mA);
                print("\n");

                print("mBq\n");
                print(mBq);
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
                print("gBq\n");
                print(gBq);
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
                print("sB\n");
                print(sB);
                print("\n");
                print("sBq\n");
                print(sBq);
                print("\n");
                print("sB_zeros\n");
                print(sB_zeros);
                print("\n");
                print("sB_scales\n");
                print(sB_scales);
                print("\n");

                print("tA_this_group_gA_bK_g2s_copy\n");
                print(tA_this_group_gA_bK_g2s_copy);
                print("\n");
                print("tA_sA_g2s_copy\n");
                print(tA_sA_g2s_copy);
                print("\n");
                print("tB_this_group_gBq_bK_g2s_copy\n");
                print(tB_this_group_gBq_bK_g2s_copy);
                print("\n");
                print("tB_sBq_g2s_copy\n");
                print(tB_sBq_g2s_copy);
                print("\n");
                print("tC_sA_mma\n");
                print(tC_sA_mma);
                print("\n");
                print("tC_sB_mma\n");
                print(tC_sB_mma);
                print("\n");
                print("tCgC_mma\n");
                print(tCgC_mma);
                print("\n");
                print("tC_rA_mma\n");
                print(tC_rA_mma);
                print("\n");
                print("tC_rB_mma\n");
                print(tC_rB_mma);
                print("\n");
                print("tC_rC_mma\n");
                print(tC_rC_mma);
                print("\n");
                print("this_group_gA\n");
                print(this_group_gA);
                print("\n");
                print("this_group_gBq\n");
                print(this_group_gBq);
                print("\n");
                print("this_group_gB_zeros\n");
                print(this_group_gB_zeros);
                print("\n");
                print("this_group_gB_scales\n");
                print(this_group_gB_scales);
                print("\n");
                print("this_group_bK_div_gA\n");
                print(this_group_bK_div_gA);
                print("\n");
                print("this_group_bK_div_gBq\n");
                print(this_group_bK_div_gBq);
                print("\n");
                print("tB_sBq_s2r_copy_dq\n");
                print(tB_sBq_s2r_copy_dq);
                print("\n");
                print("tB_rBq_s2r_copy_dq\n");
                print(tB_rBq_s2r_copy_dq);
                print("\n");
                print("tB_sB_r2s_copy_dq\n");
                print(tB_sB_r2s_copy_dq);
                print("\n");
                print("tB_rB_r2s_copy_dq\n");
                print(tB_rB_r2s_copy_dq);
                print("\n");

                
                }
            if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && idx_bK == 0) {
                printf("sB \n");
                for(int i = 0; i<size<0>(sB); i++) {
                    for(int j=0; j<size<1>(sB); j++) {
                        scalar_t val = sB(i,j);
                        float val_float = static_cast<float>(val);
                        printf("%.4f\t", val_float);
                    }
                    printf("\n");
                }
                printf("\n");
                printf("sB end\n");

                printf("sA \n");
                for(int i = 0; i<size<0>(sA); i++) {
                    for(int j=0; j<size<1>(sA); j++) {
                        scalar_t val = sA(i,j);
                        float val_float = static_cast<float>(val);
                        printf("%.4f\t", val_float);
                    }
                    printf("\n");
                }
                printf("\n");
                printf("sA end\n");
            }
            __syncthreads();

            if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && idx_bK == 0) {
                printf("sA load\n");
                for(int i = 0; i<size<0>(sA); i++) {
                    for(int j=0; j<size<1>(sA); j++) {
                        scalar_t val = sA(i,j);
                        float val_float = static_cast<float>(val);
                        printf("%.4f\t", val_float);
                    }
                    printf("\n");
                }
                printf("\n");
                printf("sA end\n");

                print("tiled_g2s_copy_A\n");
                print(tiled_g2s_copy_A);
                print("\n");

                print("tiled_g2s_copy_Bq \n");
                print(tiled_g2s_copy_Bq);
                print("\n");
                

            }
            __syncthreads();

                if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        printf("gC \n");
        for(int i = 0; i<size<0>(gC); i++) {
            for(int j=0; j<size<1>(gC); j++) {
                scalar_t val = gC(i,j);
                float val_float = static_cast<float>(val);
                printf("%.4f\t", val_float);
            }
            printf("\n");
        }
        printf("\n");
        printf("gC_reduce_all \n");
        for(int i = 0; i<size<0>(gC_reduce_all); i++) {
            for(int j=0; j<size<1>(gC_reduce_all); j++) {
                scalar_t val = gC_reduce_all(i,j);
                float val_float = static_cast<float>(val);
                printf("%.4f\t", val_float);
            }
            printf("\n");
        }


    for(int idx_group = 0; idx_group < n_group; idx_group++) {
        Tensor this_group_gA = gA(_,_,idx_group);
        Tensor this_group_gBq = gBq(_,_,idx_group);
        Tensor this_group_gB_zeros = gB_zeros(_,_,idx_group);
        Tensor this_group_gB_scales = gB_scales(_, _, idx_group);
        
        for(int idx_n_thread = threadIdx.x; idx_n_thread < bN; idx_n_thread += blockDim.x) {
            sB_scales(idx_n_thread, 0) = this_group_gB_scales(idx_n_thread, 0);
            uint32_t zeros_val_q = this_group_gB_zeros(idx_n_thread / q_div, 0);
            int bit_offset = (idx_n_thread % q_div) * bit;
            zeros_val_q = (zeros_val_q >> bit_offset) & bit_mask;
            sB_zeros(idx_n_thread, 0) = zeros_val_q + 1;
        }
        __syncthreads();

        Tensor this_group_bK_div_gA = local_tile(this_group_gA, make_tile(Int<bM>{}, Int<bK>{}), make_coord(0, _));
        Tensor this_group_bK_div_gBq = local_tile(this_group_gBq, make_tile(Int<bN>{}, Int<bK_q>{}), make_coord(0, _));   
        
        for(int idx_bK = 0; idx_bK < size<2>(this_group_bK_div_gA); idx_bK++) {
            Tensor this_group_this_bK_gA = this_group_bK_div_gA(_,_,idx_bK);
            Tensor this_group_this_bK_tA_gA_g2s_copy = thr_g2s_copy_A.partition_S(this_group_this_bK_gA);
            cute::copy_if(tiled_g2s_copy_A, tA_pA, this_group_this_bK_tA_gA_g2s_copy, tA_sA_g2s_copy);

            Tensor this_group_this_bK_gBq = this_group_bK_div_gBq(_,_,idx_bK);
            Tensor this_group_this_bK_tB_gBq_g2s_copy = thr_g2s_copy_Bq.partition_S(this_group_this_bK_gBq);
            cute::copy(tiled_g2s_copy_Bq, this_group_this_bK_tB_gBq_g2s_copy, tB_sBq_g2s_copy);
            __syncthreads();

            
            for(int idx_dq_N = 0; idx_dq_N < size<1>(tB_sBq_s2r_copy_dq); idx_dq_N++) {
                
                for(int idx_dq_K = 0; idx_dq_K < size<2>(tB_sBq_s2r_copy_dq); idx_dq_K++) {
                    cute::copy(tiled_s2r_copy_dq, tB_sBq_s2r_copy_dq(_, idx_dq_N, idx_dq_K), tB_rBq_s2r_copy_dq);
                    int idx_N_thread = get<0>(tB_id_sB_r2s_copy_dq_get_n(0, idx_dq_N, idx_dq_K));
                    uint32_t zero_thread = sB_zeros(idx_N_thread,0);
                    scalar_t scale_thread = sB_scales(idx_N_thread, 0);
                    dq_thread_fn<scalar_t, Bq_dq_layout, bit, q_div, bit_mask>(
                        tB_rBq_s2r_copy_dq, tB_rB_r2s_copy_dq, zero_thread, scale_thread);
                    cute::copy(tiled_r2s_copy_dq, tB_rB_r2s_copy_dq, tB_sB_r2s_copy_dq(_, idx_dq_N, idx_dq_K));
                }
                
            }
            __syncthreads();
            cute::copy(tC_sA_mma, tC_rA_mma);
            cute::copy(tC_sB_mma, tC_rB_mma);
            cute::gemm(tiled_mma, tC_rC_mma, tC_rA_mma, tC_rB_mma, tC_rC_mma);
            __syncthreads();
        }
    }


        if(idx_bK % n_bK_in_one_group == 0) {
            int idx_group = idx_bK / n_bK_in_one_group;
            Tensor this_group_gB_zeros = gB_zeros(_,_,idx_group);
            Tensor this_group_gB_scales = gB_scales(_,_,idx_group);
            for(int idx_n_thread = threadIdx.x; idx_n_thread < bN; idx_n_thread += blockDim.x) {
                sB_scales(idx_n_thread, 0) = this_group_gB_scales(idx_n_thread, 0);
                uint32_t zeros_val_q = this_group_gB_zeros(idx_n_thread / q_div, 0);
                int bit_offset = (idx_n_thread % q_div) * bit;
                zeros_val_q = (zeros_val_q >> bit_offset) & bit_mask;
                sB_zeros(idx_n_thread, 0) = zeros_val_q + 1;
            }
        }
        __syncthreads();
*/




namespace cutlass_gptq {

using namespace cute;



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




template<typename scalar_t, typename Bq_dq_layout, int bit, int q_div, int bit_mask, 
typename TensorBQ, typename TensorB>
__forceinline__ __device__ void dq_thread_fn(TensorBQ& Bq_thread, 
    TensorB& B_thread, uint32_t zero_thread, scalar_t scale_thread) {
        Bq_dq_layout Bq_dq_map_idx;
        uint32_t Bq_val_thread = Bq_thread(0);
        for(int idx_q = 0; idx_q < q_div; idx_q++) {
            B_thread(Bq_dq_map_idx(idx_q)) = int_to_float<scalar_t>(
                (Bq_val_thread & bit_mask) - zero_thread) * scale_thread;
            Bq_val_thread >>= bit;
        }
}


template<typename GEMM_CONGIG, typename GptQ_Kernel_Params_T>
__global__ void __launch_bounds__(GEMM_CONGIG::threads) cutlass_gptq_gemm_kernel(
    GptQ_Kernel_Params_T kernel_params
    ) {

    using namespace cute;
    using scalar_t = typename GEMM_CONGIG::scalar_t;
    using accscalar_t = typename GEMM_CONGIG::accscalar_t;
    using TiledMMA = typename GEMM_CONGIG::TiledMMA;

    using SmemLayoutA = typename GEMM_CONGIG::SmemLayoutA;
    using SmemLayoutB = typename GEMM_CONGIG::SmemLayoutB;
    using SmemLayoutBq = typename GEMM_CONGIG::SmemLayoutBq;
    using SmemLayoutB_zeros = typename GEMM_CONGIG::SmemLayoutB_zeros;
    using SmemLayoutB_scales = typename GEMM_CONGIG::SmemLayoutB_scales;
    using SmemLayoutC = typename GEMM_CONGIG::SmemLayoutC;

    using G2SCopyA = typename GEMM_CONGIG::G2SCopyA;
    using G2SCopyBq = typename GEMM_CONGIG::G2SCopyBq;

    using S2RCopyAtomA = typename GEMM_CONGIG::S2RCopyAtomA;
    using S2RCopyAtomB = typename GEMM_CONGIG::S2RCopyAtomB;

    using S2RDQCopy = typename GEMM_CONGIG::S2RDQCopy;
    using R2SDQCopy = typename GEMM_CONGIG::R2SDQCopy;
    using Bq_dq_layout = typename GEMM_CONGIG::Bq_dq_layout;

    using S2GCopyC = typename GEMM_CONGIG::S2GCopyC;
    using ReduceG2GCopyC = typename GEMM_CONGIG::ReduceG2GCopyC;
    using STORE_B_T = typename GEMM_CONGIG::STORE_B_T;

    constexpr bool is_acc_type_match = GEMM_CONGIG::is_acc_type_match;

    constexpr int bM = GEMM_CONGIG::bM;
    constexpr int bN = GEMM_CONGIG::bN;
    constexpr int bK = GEMM_CONGIG::bK;
    constexpr int aK = GEMM_CONGIG::aK;
    constexpr int s2s_dq_copy_repeat_for_aK = GEMM_CONGIG::s2s_dq_copy_repeat_for_aK;
    
    
    constexpr int bit = GEMM_CONGIG::bit;
    constexpr int bit_mask = GEMM_CONGIG::bit_mask;
    constexpr int group_size = GEMM_CONGIG::group_size;
    constexpr int q_div = GEMM_CONGIG::q_div;
    
    constexpr int bK_q = GEMM_CONGIG::bK_q;
    constexpr int bN_q = GEMM_CONGIG::bN_q;
    constexpr int n_bK_in_one_group = GEMM_CONGIG::n_bK_in_one_group;

    constexpr int kStage = GEMM_CONGIG::kStage;

    constexpr int smem_size_A = GEMM_CONGIG::smem_size_A;
    constexpr int smem_size_B = GEMM_CONGIG::smem_size_B;
    constexpr int smem_size_Bq = GEMM_CONGIG::smem_size_Bq;
    constexpr int smem_size_B_zeros = GEMM_CONGIG::smem_size_B_zeros;
    constexpr int smem_size_B_scales = GEMM_CONGIG::smem_size_B_scales;

    constexpr int smem_size = GEMM_CONGIG::smem_size;
    

    extern __shared__ char smem_ptr[];

    scalar_t* A_smem = reinterpret_cast<scalar_t* >(smem_ptr);
    scalar_t* B_smem = reinterpret_cast<scalar_t* >(smem_ptr + smem_size_A);
    STORE_B_T* B_V_smem = reinterpret_cast<STORE_B_T* >(B_smem);
    uint32_t* Bq_smem = reinterpret_cast<uint32_t* >(smem_ptr + smem_size_A + smem_size_B);
    uint32_t* B_zeros_smem = reinterpret_cast<uint32_t* >(smem_ptr + smem_size_A + smem_size_B + smem_size_Bq);
    scalar_t* B_scales_smem = reinterpret_cast<scalar_t* >(smem_ptr + smem_size_A + smem_size_B + smem_size_Bq + smem_size_B_zeros);
    scalar_t* C_smem = reinterpret_cast<scalar_t* >(smem_ptr);

    int M = kernel_params.M;
    int N = kernel_params.N;
    int K = kernel_params.K;
    int count_m_blocks = kernel_params.count_m_blocks;
    int count_n_blocks = kernel_params.count_n_blocks;
    int split_k_slices = kernel_params.split_k_slices;
    int split_k_dim = K / split_k_slices;

    int l2_tile = kernel_params.l2_tile;

    Tensor A = make_tensor(make_gmem_ptr(kernel_params.A_ptr), 
                    make_shape( M,         split_k_dim,              split_k_slices),
                    make_stride(K,         Int<1>{},                 split_k_dim));
    Tensor id_A =  make_identity_tensor(shape(A));
    Tensor Bq = make_tensor(make_gmem_ptr(kernel_params.Bq_ptr),
                    make_shape( N,         split_k_dim / q_div,      split_k_slices),
                    make_stride(Int<1>{},  N,                        (N * (split_k_dim / q_div))));
    Tensor B_zeros = make_tensor(make_gmem_ptr(kernel_params.B_zeros_ptr), 
                    make_shape( N / q_div, split_k_dim / group_size, split_k_slices),
                    make_stride(Int<1>{},  N / q_div,                ((N / q_div) * (split_k_dim / group_size))));
    Tensor B_scales = make_tensor(make_gmem_ptr(kernel_params.B_scales_ptr),
                    make_shape( N,         split_k_dim / group_size, split_k_slices),
                    make_stride(Int<1>{},  N,                        (N * (split_k_dim / group_size))));
    Tensor C_reduce = make_tensor(make_gmem_ptr(kernel_params.C_reduce_ptr), 
                    make_shape( M,         N,                        split_k_slices), 
                    make_stride(N,         Int<1>{},                 M*N));
    Tensor mC = make_tensor(make_gmem_ptr(kernel_params.C_ptr), 
                    make_shape( M,         N), 
                    make_stride(N,         Int<1>{}));
    Tensor id_mC = make_identity_tensor(shape(mC));


    int split_k_idx = blockIdx.x;
    Tensor mA = A(_,_,split_k_idx);
    Tensor id_mA = id_A(_,_,split_k_idx);
    Tensor mBq = Bq(_,_,split_k_idx);
    Tensor mB_zeros = B_zeros(_,_,split_k_idx);
    Tensor mB_scales = B_scales(_,_,split_k_idx);
    Tensor mC_reduce = C_reduce(_,_,split_k_idx);

    int idx_m_block = blockIdx.y / l2_tile;
    int idx_n_block = blockIdx.z * l2_tile + blockIdx.y % l2_tile;

    Tensor gA = local_tile(mA, make_tile(Int<bM>{}, Int<bK>{}), make_coord(idx_m_block, _));  
    Tensor id_gA = local_tile(id_mA, make_tile(Int<bM>{}, Int<bK>{}), make_coord(idx_m_block, _));  
    Tensor gBq = local_tile(mBq, make_tile(Int<bN>{}, Int<bK_q>{}), make_coord(idx_n_block, _));
    Tensor gB_zeros = local_tile(mB_zeros, make_tile(Int<bN_q>{}, Int<1>{}), make_coord(idx_n_block, _));
    Tensor gB_scales = local_tile(mB_scales, make_tile(Int<bN>{}, Int<1>{}), make_coord(idx_n_block, _));
    Tensor gC = local_tile(mC, make_tile(Int<bM>{}, Int<bN>{}), make_coord(idx_m_block, idx_n_block));
    Tensor gC_reduce = local_tile(mC_reduce, make_tile(Int<bM>{}, Int<bN>{}), make_coord(idx_m_block, idx_n_block));
    Tensor gC_reduce_all = local_tile(C_reduce, make_tile(Int<bM>{}, Int<bN>{}), make_coord(idx_m_block, idx_n_block));
    Tensor id_gC = local_tile(id_mC, make_tile(Int<bM>{}, Int<bN>{}), make_coord(idx_m_block, idx_n_block));

    Tensor sA = make_tensor(make_smem_ptr(A_smem), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(B_smem), SmemLayoutB{});
    Tensor sBq = make_tensor(make_smem_ptr(Bq_smem), SmemLayoutBq{});     
    Tensor sB_zeros = make_tensor(make_smem_ptr(B_zeros_smem), SmemLayoutB_zeros{}); 
    Tensor sB_scales = make_tensor(make_smem_ptr(B_scales_smem), SmemLayoutB_scales{});  
    Tensor sC = make_tensor(make_smem_ptr(C_smem), SmemLayoutC{});

    G2SCopyA tiled_g2s_copy_A;
    ThrCopy thr_g2s_copy_A = tiled_g2s_copy_A.get_slice(threadIdx.x);
    Tensor tA_gA_g2s_copy = thr_g2s_copy_A.partition_S(gA);
    Tensor tA_sA_g2s_copy = thr_g2s_copy_A.partition_D(sA);  
    Tensor tA_id_A_g2s_copy =  thr_g2s_copy_A.partition_S(id_gA); //
    Tensor tA_pA = make_tensor<bool>(
    make_shape( shape<0>(tA_sA_g2s_copy),         size<1>(tA_sA_g2s_copy), size<2>(tA_sA_g2s_copy)),
    make_stride(make_stride(Int<0>{}, Int<0>{}) , Int<1>{},                Int<0>{}              ));
    for(int m = 0; m < size<1>(tA_pA); m++) {
        tA_pA(0,m,0) = elem_less(get<0>(tA_id_A_g2s_copy(0,m,0,0)), shape<0>(mA));
    }


    G2SCopyBq tiled_g2s_copy_Bq;
    ThrCopy thr_g2s_copy_Bq = tiled_g2s_copy_Bq.get_slice(threadIdx.x);
    Tensor tB_gBq_g2s_copy = thr_g2s_copy_Bq.partition_S(gBq);
    Tensor tB_sBq_g2s_copy = thr_g2s_copy_Bq.partition_D(sBq);
    
    TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tC_sA_mma = thr_mma.partition_A(sA);
    Tensor tC_sB_mma = thr_mma.partition_B(sB);
    Tensor tC_sC_mma = thr_mma.partition_C(sC);

    Tensor tC_rA_mma = thr_mma.partition_fragment_A(sA(_,_,0));
    Tensor tC_rB_mma = thr_mma.partition_fragment_B(sB(_,_,0));
    Tensor tC_rC_mma = thr_mma.partition_fragment_C(sC);
 
    clear(tC_rC_mma);

    TiledCopy tiled_s2r_copy_A = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    ThrCopy thr_s2r_copy_A = tiled_s2r_copy_A.get_slice(threadIdx.x);
    Tensor tC_sA_s2r_copy = thr_s2r_copy_A.partition_S(sA);
    Tensor tC_rA_s2r_copy = thr_s2r_copy_A.retile_D(tC_rA_mma);

    TiledCopy tiled_s2r_copy_B = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    ThrCopy thr_s2r_copy_B = tiled_s2r_copy_B.get_slice(threadIdx.x);
    Tensor tC_sB_s2r_copy = thr_s2r_copy_B.partition_S(sB);
    Tensor tC_rB_s2r_copy = thr_s2r_copy_B.retile_D(tC_rB_mma);

    S2RDQCopy tiled_s2r_dq_copy;
    ThrCopy thr_s2r_dq_copy = tiled_s2r_dq_copy.get_slice(threadIdx.x);
    Tensor tB_sBq_s2r_dq_copy = thr_s2r_dq_copy.partition_S(sBq);
    Tensor tB_rBq_s2r_dq_copy = make_fragment_like(tB_sBq_s2r_dq_copy(_,_,0,0));


    R2SDQCopy tiled_r2s_dq_copy;
    ThrCopy thr_r2s_dq_copy = tiled_r2s_dq_copy.get_slice(threadIdx.x);
    Tensor tB_sB_r2s_dq_copy = thr_r2s_dq_copy.partition_D(sB);
    Tensor tB_rB_r2s_dq_copy = make_fragment_like(tB_sB_r2s_dq_copy(_,_,0,0));
    
    Bq_dq_layout Bq_dq_map_idx;

    int ikstage_smem_read = 0;
    int ikstage_smem_write = 0;
    int ikstage_gmem_read = 0;
    int n_bK = size<2>(gA);

    constexpr int little_k_tile_0 = 0;
    constexpr int little_k_tile_1 = 1;
    
    for(int idx_bK = 0; idx_bK < kStage - 1; idx_bK++) {
        cute::copy(tiled_g2s_copy_Bq, tB_gBq_g2s_copy(_,_,_,ikstage_gmem_read), tB_sBq_g2s_copy(_,_,_,ikstage_smem_write));
        cp_async_fence();
        cute::copy_if(tiled_g2s_copy_A, tA_pA, tA_gA_g2s_copy(_,_,_,ikstage_gmem_read), tA_sA_g2s_copy(_,_,_,ikstage_smem_write));
        cp_async_fence();
        ikstage_gmem_read++;
        ikstage_smem_write = (ikstage_smem_write + 1) % kStage;
    }

    cp_async_wait<2*kStage - 3>();
    __syncthreads();
    
        
    uint32_t zero_thread = 1;
    scalar_t scale_thread = 1.1;
    for(int idx_dq = 0; idx_dq < 2; idx_dq++) {
        cute::copy(tiled_s2r_dq_copy, tB_sBq_s2r_dq_copy(_,_,idx_dq,ikstage_smem_read), tB_rBq_s2r_dq_copy);
        uint32_t Bq_val_thread = tB_rBq_s2r_dq_copy(0,0);
        for(int idx_q = 0; idx_q < q_div; idx_q++) {
            tB_rB_r2s_dq_copy(Bq_dq_map_idx(idx_q), 0) = int_to_float<scalar_t>(
                (Bq_val_thread & bit_mask) - zero_thread) * scale_thread;
            Bq_val_thread >>= bit;
        }
        cute::copy(tiled_r2s_dq_copy, tB_rB_r2s_dq_copy, tB_sB_r2s_dq_copy(_,_,idx_dq,ikstage_smem_read));
        
    }
    cp_async_wait<2*kStage - 4>();
    __syncthreads();

    cute::copy(tiled_s2r_copy_A, tC_sA_s2r_copy(_, _, little_k_tile_0, ikstage_smem_read), tC_rA_s2r_copy(_,_,little_k_tile_0));
    cute::copy(tiled_s2r_copy_B, tC_sB_s2r_copy(_, _, little_k_tile_0, ikstage_smem_read), tC_rB_s2r_copy(_,_,little_k_tile_0));

    for(int idx_dq = 2; idx_dq < 4; idx_dq++) {
        cute::copy(tiled_s2r_dq_copy, tB_sBq_s2r_dq_copy(_,_,idx_dq, ikstage_smem_read), tB_rBq_s2r_dq_copy);
        uint32_t Bq_val_thread = tB_rBq_s2r_dq_copy(0,0);
        for(int idx_q = 0; idx_q < q_div; idx_q++) {
            tB_rB_r2s_dq_copy(Bq_dq_map_idx(idx_q), 0) = int_to_float<scalar_t>(
                (Bq_val_thread & bit_mask) - zero_thread) * scale_thread;
            Bq_val_thread >>= bit;
        }
        cute::copy(tiled_r2s_dq_copy, tB_rB_r2s_dq_copy, tB_sB_r2s_dq_copy(_,_,idx_dq,ikstage_smem_read));
        
    }
    
    __syncthreads();

    for(int idx_bK = 0; idx_bK < n_bK; idx_bK++) {
        if(ikstage_gmem_read < n_bK) {
            cute::copy(tiled_g2s_copy_Bq, tB_gBq_g2s_copy(_,_,_,ikstage_gmem_read), tB_sBq_g2s_copy(_,_,_, ikstage_smem_write));
            cp_async_fence();
            cute::copy_if(tiled_g2s_copy_A, tA_pA, tA_gA_g2s_copy(_,_,_,ikstage_gmem_read), tA_sA_g2s_copy(_,_,_,ikstage_smem_write));
            cp_async_fence();
            ikstage_gmem_read++;
            ikstage_smem_write = (ikstage_smem_write + 1) % kStage;
        }
        

        cute::copy(tiled_s2r_copy_A, tC_sA_s2r_copy(_,_,little_k_tile_1,ikstage_smem_read), tC_rA_s2r_copy(_,_,little_k_tile_1));
        cute::copy(tiled_s2r_copy_B, tC_sB_s2r_copy(_,_,little_k_tile_1,ikstage_smem_read), tC_rB_s2r_copy(_,_,little_k_tile_1));

        cute::gemm(tiled_mma, tC_rC_mma, tC_rA_mma, tC_rB_mma, tC_rC_mma);
        //cute::gemm(tiled_mma, tC_rC_mma, tC_rA_mma(_,_,little_k_tile_0), tC_rB_mma(_,_,little_k_tile_0), tC_rC_mma);
        //cute::gemm(tiled_mma, tC_rC_mma, tC_rA_mma(_,_,little_k_tile_1), tC_rB_mma(_,_,little_k_tile_1), tC_rC_mma);
        ikstage_smem_read = (ikstage_smem_read + 1) % kStage;
        cp_async_wait<2*kStage - 3>();
        __syncthreads();

        

        for(int idx_dq = 0; idx_dq < 2; idx_dq++) {
            cute::copy(tiled_s2r_dq_copy, tB_sBq_s2r_dq_copy(_,_,idx_dq,ikstage_smem_read), tB_rBq_s2r_dq_copy);
            uint32_t Bq_val_thread = tB_rBq_s2r_dq_copy(0,0);
            for(int idx_q = 0; idx_q < q_div; idx_q++) {
                tB_rB_r2s_dq_copy(Bq_dq_map_idx(idx_q), 0) = int_to_float<scalar_t>(
                    (Bq_val_thread & bit_mask) - zero_thread) * scale_thread;
                Bq_val_thread >>= bit;
            }
            cute::copy(tiled_r2s_dq_copy, tB_rB_r2s_dq_copy, tB_sB_r2s_dq_copy(_,_,idx_dq,ikstage_smem_read));
            
        }
        cp_async_wait<2*kStage - 4>();
        __syncthreads();
        cute::copy(tiled_s2r_copy_A, tC_sA_s2r_copy(_,_,little_k_tile_0,ikstage_smem_read), tC_rA_s2r_copy(_,_,little_k_tile_0));
        cute::copy(tiled_s2r_copy_B, tC_sB_s2r_copy(_,_,little_k_tile_0,ikstage_smem_read), tC_rB_s2r_copy(_,_,little_k_tile_0));

        for(int idx_dq = 2; idx_dq < 4; idx_dq++) {
            cute::copy(tiled_s2r_dq_copy, tB_sBq_s2r_dq_copy(_,_,idx_dq,ikstage_smem_read), tB_rBq_s2r_dq_copy);
            uint32_t Bq_val_thread = tB_rBq_s2r_dq_copy(0,0);
            for(int idx_q = 0; idx_q < q_div; idx_q++) {
                tB_rB_r2s_dq_copy(Bq_dq_map_idx(idx_q), 0) = int_to_float<scalar_t>(
                    (Bq_val_thread & bit_mask) - zero_thread) * scale_thread;
                Bq_val_thread >>= bit;
            }
            cute::copy(tiled_r2s_dq_copy, tB_rB_r2s_dq_copy, tB_sB_r2s_dq_copy(_,_,idx_dq,ikstage_smem_read));
            
        }
        __syncthreads();
       
    }
    


    cute::copy(tC_rC_mma, tC_sC_mma);
    __syncthreads();

    S2GCopyC tiled_s2g_copy_C;
    ThrCopy thr_s2g_copy_C = tiled_s2g_copy_C.get_slice(threadIdx.x);
    Tensor tC_sC_s2g_copy = thr_s2g_copy_C.partition_S(sC);
    Tensor tC_gC_reduce_s2g_copy = thr_s2g_copy_C.partition_D(gC_reduce);
    Tensor tC_gC_s2g_copy = thr_s2g_copy_C.partition_D(gC);
    Tensor tC_id_gC_s2g_copy = thr_s2g_copy_C.partition_D(id_gC);
    Tensor tC_pC = make_tensor<bool>(make_shape(shape<0>(tC_gC_reduce_s2g_copy), size<1>(tC_gC_reduce_s2g_copy), size<2>(tC_gC_reduce_s2g_copy)),
                                    make_stride(make_stride(Int<0>{}, Int<0>{}), Int<1>{}, Int<0>{}));

    for(int m = 0; m < size<1>(tC_pC); m++) {
        tC_pC(0, m, 0) = elem_less(get<0>(tC_id_gC_s2g_copy(0,m,0)), shape<0>(mC));
    }
    if(split_k_slices == 1) {
        cute::copy_if(tiled_s2g_copy_C, tC_pC, tC_sC_s2g_copy, tC_gC_s2g_copy);
    } else {
        cute::copy_if(tiled_s2g_copy_C, tC_pC, tC_sC_s2g_copy, tC_gC_reduce_s2g_copy);
    }
    if(split_k_slices == 1) {
        return;
    }

    __threadfence();
    __syncthreads();

    __shared__ bool s_is_last_split_k;

    if(threadIdx.x == 0) {
        uint32_t* semaphore_ptr = kernel_params.C_semaphore_ptr 
            + idx_m_block * count_n_blocks + idx_n_block;
        uint32_t  split_finished_old = atomicAdd(semaphore_ptr, 1);
        s_is_last_split_k = (split_finished_old == split_k_slices - 1);
    }
    __syncthreads();

    if(!s_is_last_split_k) {
        return;
    }


    ReduceG2GCopyC tiled_reduce_g2g_copy_C;
    ThrCopy thr_reduce_g2g_copy_C = tiled_reduce_g2g_copy_C.get_slice(threadIdx.x);
    Tensor tC_gC_g2g_copy = thr_reduce_g2g_copy_C.partition_D(gC);
    Tensor tC_rC_g2g_copy = make_fragment_like(tC_gC_g2g_copy);
    clear(tC_rC_g2g_copy);
    
    Tensor tC_gC_reduce_all_g2g_copy = thr_reduce_g2g_copy_C.partition_S(gC_reduce_all);
    Tensor tC_rC_reduce_g2g_copy = make_fragment_like(tC_gC_g2g_copy);

    for(int idx_M_thread = 0; idx_M_thread < size<1>(tC_rC_g2g_copy); idx_M_thread++) {
        for(int idx_N_thread = 0; idx_N_thread < size<2>(tC_rC_g2g_copy); idx_N_thread++) {
            for(int idx_split_k = 0; idx_split_k < split_k_slices; idx_split_k++) {
                cute::copy_if(tiled_reduce_g2g_copy_C, tC_pC(_, idx_M_thread, idx_N_thread), 
                tC_gC_reduce_all_g2g_copy(_, idx_M_thread, idx_N_thread, idx_split_k),
                tC_rC_reduce_g2g_copy(_, idx_M_thread, idx_N_thread));

                for(int idx_vec = 0; idx_vec < size<0>(tC_rC_g2g_copy); idx_vec++) {
                    tC_rC_g2g_copy(idx_vec, idx_M_thread, idx_N_thread) 
                        += tC_rC_reduce_g2g_copy(idx_vec, idx_M_thread, idx_N_thread);
                }
            }
            cute::copy_if(tiled_reduce_g2g_copy_C, tC_pC(_, idx_M_thread, idx_N_thread), 
                tC_rC_g2g_copy(_, idx_M_thread, idx_N_thread),
                tC_gC_g2g_copy(_, idx_M_thread, idx_N_thread));

        }
    }



}
    
    


template<typename GEMM_CONGIG, typename GptQ_Kernel_Params_T>
void launch_cutlass_gptq_gemm_kernel(
    GptQ_Kernel_Params_T kernel_params,
    Launch_Kernel_Params launch_kernel_params) {

    int M = kernel_params.M;
    int N = kernel_params.N;

    int M_blocks = (M + GEMM_CONGIG::bM - 1) / GEMM_CONGIG::bM;
    int N_blocks = (N + GEMM_CONGIG::bN - 1) / GEMM_CONGIG::bN;

    dim3 block(GEMM_CONGIG::threads);
    int smem_size = GEMM_CONGIG::smem_size;
    cutlass_gptq_gemm_kernel<GEMM_CONGIG><<<launch_kernel_params.grid, block, smem_size, launch_kernel_params.stream>>>(kernel_params);
}




template<typename scalar_t>
void run_cutlass_gptq_gemm(GptQ_Kernel_Params<scalar_t> kernel_params, Launch_Kernel_Params launch_kernel_params) {
    using TEST_GEMM_CONGIG = GPTQ_GemmConfig<scalar_t, M_BLOCK, N_BLOCK, 32, 3, 4, 128>;
    
    launch_cutlass_gptq_gemm_kernel<TEST_GEMM_CONGIG>(kernel_params, launch_kernel_params);
}

template void run_cutlass_gptq_gemm<half>(GptQ_Kernel_Params<half>, Launch_Kernel_Params);
template void run_cutlass_gptq_gemm<__nv_bfloat16>(GptQ_Kernel_Params<__nv_bfloat16>, Launch_Kernel_Params);


}