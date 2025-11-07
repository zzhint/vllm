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
            print("sB_T \n");
            print(sB_T);
            print("\n");
            for(int i = 0; i<size<0>(sB_T); i++) {
                for(int j=0; j<size<1>(sB_T); j++) {
                    scalar_t val = sB_T(i,j);
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

                print("tA_this_group_gA_bK_g2s_copy\n");
                print(tA_this_group_gA_bK_g2s_copy);
                print("\n");
                print("tAsA_g2s_copy\n");
                print(tAsA_g2s_copy);
                print("\n");
                print("tB_this_group_gB_q_T_bK_g2s_copy\n");
                print(tB_this_group_gB_q_T_bK_g2s_copy);
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
                print("this_group_gA\n");
                print(this_group_gA);
                print("\n");
                print("this_group_gB_q_T\n");
                print(this_group_gB_q_T);
                print("\n");
                print("this_group_gB_zeros\n");
                print(this_group_gB_zeros);
                print("\n");
                print("this_group_gB_scales\n");
                print(this_group_gB_scales);
                print("\n");
                print("this_group_gA_bK_div\n");
                print(this_group_gA_bK_div);
                print("\n");
                print("this_group_gB_q_T_bK_div\n");
                print(this_group_gB_q_T_bK_div);
                print("\n");
                print("tBqTsBqT_s2r_copy\n");
                print(tBqTsBqT_s2r_copy);
                print("\n");
                print("tBqTrBqT_s2r_copy\n");
                print(tBqTrBqT_s2r_copy);
                print("\n");
                print("tBTsBT_r2s_copy\n");
                print(tBTsBT_r2s_copy);
                print("\n");
                print("tBTrBT_r2s_copy\n");
                print(tBTrBT_r2s_copy);
                print("\n");

                
                }
            if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && idx_bK == 0) {
                printf("sB_T \n");
                for(int i = 0; i<size<0>(sB_T); i++) {
                    for(int j=0; j<size<1>(sB_T); j++) {
                        scalar_t val = sB_T(i,j);
                        float val_float = static_cast<float>(val);
                        printf("%.4f\t", val_float);
                    }
                    printf("\n");
                }
                printf("\n");
                printf("sB_T end\n");

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

                print("tiled_g2s_copy_B_q_T \n");
                print(tiled_g2s_copy_B_q_T);
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
        printf("gC_reduce \n");
        for(int i = 0; i<size<0>(gC_reduce); i++) {
            for(int j=0; j<size<1>(gC_reduce); j++) {
                scalar_t val = gC_reduce(i,j);
                float val_float = static_cast<float>(val);
                printf("%.4f\t", val_float);
            }
            printf("\n");
        }
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




template<typename scalar_t, typename B_q_dq_layout, int bit, int q_div, int bit_mask, 
typename TensorBQ, typename TensorB>
__forceinline__ __device__ void dq_thread_fn(TensorBQ& B_q_thread, 
    TensorB& B_thread, uint32_t zero_thread, scalar_t scale_thread) {
        B_q_dq_layout B_q_dq_map_idx;
        uint32_t B_q_val_thread = B_q_thread(0);
        for(int idx_q = 0; idx_q < q_div; idx_q++) {
            B_thread(B_q_dq_map_idx(idx_q)) = int_to_float<scalar_t>(
                (B_q_val_thread & bit_mask) - zero_thread) * scale_thread;
            B_q_val_thread >>= bit;
        }
}


template<typename GEMM_CONGIG, typename GptQ_Kernel_Params_T>
__global__ void __launch_bounds__(GEMM_CONGIG::threads) cutlass_gptq_gemm_kernel(
    GptQ_Kernel_Params_T kernel_params
    ) {

    using namespace cute;
    using scalar_t = typename GEMM_CONGIG::scalar_t;
    using TiledMMA = typename GEMM_CONGIG::TiledMMA;

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

    constexpr bool is_acc_type_match = GEMM_CONGIG::is_acc_type_match;

    constexpr int bM = GEMM_CONGIG::bM;
    constexpr int bN = GEMM_CONGIG::bN;
    constexpr int bK = GEMM_CONGIG::bK;
    
    constexpr int bit = GEMM_CONGIG::bit;
    constexpr int bit_mask = GEMM_CONGIG::bit_mask;
    constexpr int group_size = GEMM_CONGIG::group_size;
    constexpr int q_div = GEMM_CONGIG::q_div;
    
    constexpr int bK_q = GEMM_CONGIG::bK_q;
    constexpr int bN_q = GEMM_CONGIG::bN_q;

    constexpr int group_size_div_q_div = GEMM_CONGIG::group_size_div_q_div;

    constexpr int smem_size_A = GEMM_CONGIG::smem_size_A;
    constexpr int smem_size_B = GEMM_CONGIG::smem_size_B;
    constexpr int smem_size_B_q = GEMM_CONGIG::smem_size_B_q;
    constexpr int smem_size_B_zeros = GEMM_CONGIG::smem_size_B_zeros;
    constexpr int smem_size_B_scales = GEMM_CONGIG::smem_size_B_scales;

    constexpr int smem_size = GEMM_CONGIG::smem_size;
    

    extern __shared__ char smem_ptr[];

    scalar_t* A_smem = reinterpret_cast<scalar_t* >(smem_ptr);
    scalar_t* B_smem = reinterpret_cast<scalar_t* >(smem_ptr + smem_size_A);
    uint32_t* B_q_smem = reinterpret_cast<uint32_t* >(smem_ptr + smem_size_A + smem_size_B);
    uint32_t* B_zeros_smem = reinterpret_cast<uint32_t* >(smem_ptr + smem_size_A + smem_size_B + smem_size_B_q);
    scalar_t* B_scales_smem = reinterpret_cast<scalar_t* >(smem_ptr + smem_size_A + smem_size_B + smem_size_B_q + smem_size_B_zeros);
    scalar_t* C_smem = reinterpret_cast<scalar_t* >(smem_ptr);

    int M = kernel_params.M;
    int N = kernel_params.N;
    int K = kernel_params.K;
    int split_k_slices = kernel_params.split_k_slices;

    Tensor mA = make_tensor(make_gmem_ptr(kernel_params.A_ptr), make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor mB_q_T = make_tensor(make_gmem_ptr(kernel_params.B_q_ptr), make_shape(N, K / q_div), make_stride(Int<1>{}, N));
    Tensor mB_zeros = make_tensor(make_gmem_ptr(kernel_params.B_zeros_ptr), 
        make_shape(N / q_div, K / group_size), make_stride(Int<1>{}, N / q_div));
    Tensor mB_scales = make_tensor(make_gmem_ptr(kernel_params.B_scales_ptr),
        make_shape(N, K / group_size), make_stride(Int<1>{}, N));
    Tensor mC = make_tensor(make_gmem_ptr(kernel_params.C_ptr), make_shape(M, N), make_stride(N, Int<1>{}));
    Tensor mC_reduce = make_tensor(make_gmem_ptr(kernel_params.C_reduce_ptr), 
        make_shape(M, N, split_k_slices), make_stride(N, Int<1>{}, M*N));

    int block_idx_x = blockIdx.x;
    int block_idx_y = blockIdx.y;

    Tensor gA = local_tile(mA, make_tile(Int<bM>{}, Int<group_size>{}), make_coord(block_idx_x, _));  
    Tensor gB_q_T = local_tile(mB_q_T, make_tile(Int<bN>{}, Int<group_size_div_q_div>{}), make_coord(block_idx_y, _));
    Tensor gB_zeros = local_tile(mB_zeros, make_tile(Int<bN_q>{}, Int<1>{}), make_coord(block_idx_y, _));
    Tensor gB_scales = local_tile(mB_scales, make_tile(Int<bN>{}, Int<1>{}), make_coord(block_idx_y, _));
    Tensor gC = local_tile(mC, make_tile(Int<bM>{}, Int<bN>{}), make_coord(block_idx_x, block_idx_y));
    Tensor gC_reduce = local_tile(mC_reduce(_,_, blockIdx.z), make_tile(Int<bM>{}, Int<bN>{}), make_coord(block_idx_x, block_idx_y));

    Tensor sA = make_tensor(make_smem_ptr(A_smem), SmemLayoutA{});
    Tensor sB_T = make_tensor(make_smem_ptr(B_smem), SmemLayoutB_T{});
    Tensor sB_q_T = make_tensor(make_smem_ptr(B_q_smem), SmemLayoutB_q_T{});     
    Tensor sB_zeros = make_tensor(make_smem_ptr(B_zeros_smem), SmemLayoutB_zeros{}); 
    Tensor sB_scales = make_tensor(make_smem_ptr(B_scales_smem), SmemLayoutB_scales{});                  

    G2SCopyA tiled_g2s_copy_A;
    G2SCopyB_q_T tiled_g2s_copy_B_q_T;

    ThrCopy thr_g2s_copy_A = tiled_g2s_copy_A.get_slice(threadIdx.x);
    Tensor tAsA_g2s_copy = thr_g2s_copy_A.partition_D(sA);   

    ThrCopy thr_g2s_copy_B_q_T = tiled_g2s_copy_B_q_T.get_slice(threadIdx.x);
    Tensor tBsB_q_T_g2s_copy = thr_g2s_copy_B_q_T.partition_D(sB_q_T);

    S2RDQCopy tiled_s2r_dq_copy;
    ThrCopy thr_s2r_dq_copy = tiled_s2r_dq_copy.get_slice(threadIdx.x);
    Tensor tBqTsBqT_s2r_copy = thr_s2r_dq_copy.partition_S(sB_q_T);
    Tensor tBqTrBqT_s2r_copy = make_fragment_like(tBqTsBqT_s2r_copy(_,0,0));

    R2SDQCopy tiled_r2s_dq_copy;
    ThrCopy thr_r2s_dq_copy = tiled_r2s_dq_copy.get_slice(threadIdx.x);
    Tensor tBTsBT_r2s_copy = thr_r2s_dq_copy.partition_D(sB_T);
    Tensor tBTrBT_r2s_copy = make_fragment_like(tBTsBT_r2s_copy(_,0,0));

    Tensor id_sB_T = make_identity_tensor(shape(sB_T));
    Tensor tBTidsBT_r2s_copy = thr_r2s_dq_copy.partition_D(id_sB_T);
    
    TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCsA_mma = thr_mma.partition_A(sA);
    Tensor tCsB_mma = thr_mma.partition_B(sB_T);
    Tensor tCgC_mma = thr_mma.partition_C(gC);
    Tensor tCgC_reduce_mma = thr_mma.partition_C(gC_reduce);

    Tensor tCrA_mma = thr_mma.partition_fragment_A(sA);
    Tensor tCrB_mma = thr_mma.partition_fragment_B(sB_T);
    Tensor tCrC_mma = thr_mma.partition_fragment_C(gC);

    clear(tCrC_mma);

    int n_group = size<2>(gA);
    for(int idx_group = 0; idx_group < n_group; idx_group++) {
        Tensor this_group_gA = gA(_,_,idx_group);
        Tensor this_group_gB_q_T = gB_q_T(_,_,idx_group);
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

        Tensor this_group_gA_bK_div = local_tile(this_group_gA, make_tile(Int<bM>{}, Int<bK>{}), make_coord(0, _));
        Tensor this_group_gB_q_T_bK_div = local_tile(this_group_gB_q_T, make_tile(Int<bN>{}, Int<bK_q>{}), make_coord(0, _));

        Tensor tA_this_group_gA_bK_g2s_copy = thr_g2s_copy_A.partition_S(this_group_gA_bK_div);
        Tensor tB_this_group_gB_q_T_bK_g2s_copy = thr_g2s_copy_B_q_T.partition_S(this_group_gB_q_T_bK_div);    

        int n_bK_in_one_group = size<2>(this_group_gA_bK_div);

        for(int idx_bK = 0; idx_bK < n_bK_in_one_group; idx_bK++) {
            cute::copy(tiled_g2s_copy_A, tA_this_group_gA_bK_g2s_copy(_,_,_, idx_bK), tAsA_g2s_copy);
            cute::copy(tiled_g2s_copy_B_q_T, tB_this_group_gB_q_T_bK_g2s_copy(_,_,_, idx_bK), tBsB_q_T_g2s_copy);
            __syncthreads();

            for(int idx_dq_N = 0; idx_dq_N < size<1>(tBqTsBqT_s2r_copy); idx_dq_N++) {
                for(int idx_dq_K = 0; idx_dq_K < size<2>(tBqTsBqT_s2r_copy); idx_dq_K++) {
                    cute::copy(tiled_s2r_dq_copy, tBqTsBqT_s2r_copy(_, idx_dq_N, idx_dq_K), tBqTrBqT_s2r_copy);
                    int idx_N_thread = get<0>(tBTidsBT_r2s_copy(0, idx_dq_N, idx_dq_K));
                    uint32_t zero_thread = sB_zeros(idx_N_thread,0);
                    scalar_t scale_thread = sB_scales(idx_N_thread, 0);
                    dq_thread_fn<scalar_t, B_q_dq_layout, bit, q_div, bit_mask>(
                        tBqTrBqT_s2r_copy, tBTrBT_r2s_copy, zero_thread, scale_thread);
                    cute::copy(tiled_r2s_dq_copy, tBTrBT_r2s_copy, tBTsBT_r2s_copy(_, idx_dq_N, idx_dq_K));
                }

            }
            __syncthreads();
            cute::copy(tCsA_mma, tCrA_mma);
            cute::copy(tCsB_mma, tCrB_mma);
            cute::gemm(tiled_mma, tCrC_mma, tCrA_mma, tCrB_mma, tCrC_mma);
        }
    }
    if(is_acc_type_match && (split_k_slices == 1)) {
        cute::copy(tCrC_mma, tCgC_mma);
    } else {
        cute::copy(tCrC_mma, tCgC_reduce_mma);
    }



}
    
    



template<typename GEMM_CONGIG, typename GptQ_Kernel_Params_T>
__global__ void __launch_bounds__(GEMM_CONGIG::reduce_threads) cutlass_gptq_reduce_kernel(
    GptQ_Kernel_Params_T kernel_params) {

    using scalar_t = typename GEMM_CONGIG::scalar_t;
    using accscalar_t = typename GEMM_CONGIG::accscalar_t;
    using ReduceG2RCopyCReduce = typename GEMM_CONGIG::ReduceG2RCopyCReduce;
    using ReduceR2GCopyC = typename GEMM_CONGIG::ReduceR2GCopyC;
    
    constexpr int bM = GEMM_CONGIG::bM;
    constexpr int bN = GEMM_CONGIG::bN;

    constexpr int vec_accscalar_t_copy = GEMM_CONGIG::vec_accscalar_t_copy;

    int M = kernel_params.M;
    int N = kernel_params.N;
    int split_k_slices = kernel_params.split_k_slices;

    Tensor mC = make_tensor(make_gmem_ptr(kernel_params.C_ptr), make_shape(M, N), make_stride(N, Int<1>{}));
    Tensor mC_reduce = make_tensor(make_gmem_ptr(kernel_params.C_reduce_ptr), 
        make_shape(M, N, split_k_slices), make_stride(N, Int<1>{}, M*N));

    int block_idx_x = blockIdx.x;
    int block_idx_y = blockIdx.y;

    Tensor gC = local_tile(mC, make_tile(Int<bM>{}, Int<bN>{}), make_coord(block_idx_x, block_idx_y));
    Tensor gC_reduce = local_tile(mC_reduce, make_tile(Int<bM>{}, Int<bN>{}), make_coord(block_idx_x, block_idx_y));
    
    ReduceR2GCopyC tiled_reduce_r2g_copy_c;
    ThrCopy thr_reduce_r2g_copy_c = tiled_reduce_r2g_copy_c.get_slice(threadIdx.x);
    Tensor tCgC_g2r_copy = thr_reduce_r2g_copy_c.partition_S(gC);
    Tensor tCrC_g2r_copy = make_fragment_like(tCgC_g2r_copy);

    ReduceG2RCopyCReduce tiled_reduce_g2r_copy_c_reduce;
    ThrCopy thr_reduce_g2r_copy_c_reduce = tiled_reduce_g2r_copy_c_reduce.get_slice(threadIdx.x);
    Tensor tCreducegCreduce_g2r_copy = thr_reduce_g2r_copy_c_reduce.partition_S(gC_reduce);
    Tensor tCreducerCreduce_g2r_copy = make_fragment_like(tCreducegCreduce_g2r_copy(_,_,_,0));
    Tensor tCreducerCreduce_g2r_copy_temp_acc = make_fragment_like(tCreducerCreduce_g2r_copy);
    clear(tCreducerCreduce_g2r_copy_temp_acc);

    for(int idx_M_thread = 0; idx_M_thread < size<1>(tCgC_g2r_copy); idx_M_thread++) {
        for(int idx_N_thread = 0; idx_N_thread < size<2>(tCgC_g2r_copy); idx_N_thread++) {
            for(int idx_split_k = 0; idx_split_k < split_k_slices; idx_split_k++) {
                cute::copy(tiled_reduce_g2r_copy_c_reduce, 
                    tCreducegCreduce_g2r_copy(_, idx_M_thread, idx_N_thread, idx_split_k), 
                    tCreducerCreduce_g2r_copy(_, idx_M_thread, idx_N_thread));
                for(int idx_vec = 0; idx_vec < vec_accscalar_t_copy; idx_vec++) {
                    tCreducerCreduce_g2r_copy_temp_acc(idx_vec, idx_M_thread, idx_N_thread) 
                        += tCreducerCreduce_g2r_copy(idx_vec, idx_M_thread, idx_N_thread);
                }
            }
            for(int idx_vec = 0; idx_vec < vec_accscalar_t_copy; idx_vec++) {
                tCrC_g2r_copy(idx_vec, idx_M_thread, idx_N_thread) 
                    = static_cast<scalar_t>(tCreducerCreduce_g2r_copy_temp_acc(idx_vec, idx_M_thread, idx_N_thread));
            }
            cute::copy(tiled_reduce_r2g_copy_c, 
            tCrC_g2r_copy(_, idx_M_thread, idx_N_thread), 
            tCgC_g2r_copy(_, idx_M_thread, idx_N_thread));
        }
    }
}

template<typename GEMM_CONGIG, typename GptQ_Kernel_Params_T>
void launch_cutlass_gptq_gemm_kernel(
    GptQ_Kernel_Params_T kernel_params) {

    int problem_size_m = kernel_params.M;
    int problem_size_n = kernel_params.N;

    int M_blocks = (problem_size_m + GEMM_CONGIG::bM - 1) / GEMM_CONGIG::bM;
    int N_blocks = (problem_size_n + GEMM_CONGIG::bN - 1) / GEMM_CONGIG::bN;

    dim3 grid(M_blocks, N_blocks, kernel_params.split_k_slices);
    dim3 block(GEMM_CONGIG::threads);
    int smem_size = GEMM_CONGIG::smem_size;
    cutlass_gptq_gemm_kernel<GEMM_CONGIG><<<grid, block, smem_size>>>(kernel_params);
}

template<typename GEMM_CONGIG, typename GptQ_Kernel_Params_T>
void launch_cutlass_gptq_reduce_kernel(
    GptQ_Kernel_Params_T kernel_params) {

    int problem_size_m = kernel_params.M;
    int problem_size_n = kernel_params.N;

    int M_blocks = (problem_size_m + GEMM_CONGIG::bM - 1) / GEMM_CONGIG::bM;
    int N_blocks = (problem_size_n + GEMM_CONGIG::bN - 1) / GEMM_CONGIG::bN;

    dim3 grid(M_blocks, N_blocks);
    dim3 block(GEMM_CONGIG::reduce_threads);
    cutlass_gptq_reduce_kernel<GEMM_CONGIG><<<grid, block>>>(kernel_params);
}



template<typename scalar_t, typename accscalar_t>
void run_cutlass_gptq_gemm(GptQ_Kernel_Params<scalar_t, accscalar_t> kernel_params) {
    using TEST_GEMM_CONGIG = GPTQ_GemmConfig<scalar_t, accscalar_t, 64, 128, 32, 1, 4, 128>;
    launch_cutlass_gptq_gemm_kernel<TEST_GEMM_CONGIG>(kernel_params);
    launch_cutlass_gptq_reduce_kernel<TEST_GEMM_CONGIG>(kernel_params);
}

template void run_cutlass_gptq_gemm<half, float>(GptQ_Kernel_Params<half, float>);
template void run_cutlass_gptq_gemm<__nv_bfloat16, float>(GptQ_Kernel_Params<__nv_bfloat16, float>);


}