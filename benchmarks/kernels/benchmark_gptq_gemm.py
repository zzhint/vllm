import torch
from triton.testing import Benchmark, perf_report, do_bench, do_bench_cudagraph  # 新增do_bench

from vllm._custom_ops import gptq_gemm, gptq_gemm_opt

input_tokens = 1024

TEST_CASES = [
    ((input_tokens, 1024), (128, 4096), (8, 512), (8, 4096), 4, True),
    ((input_tokens, 2048), (256, 1024), (16, 128), (16, 1024), 4, True),
    ((input_tokens, 1024), (128, 6144), (8, 768), (8, 6144), 4, True),
    ((input_tokens, 3072), (384, 1024), (24, 128), (24, 1024), 4, True),
    ((input_tokens, 4096), (512, 6144), (32, 768), (32, 6144), 4, True),
    ((input_tokens, 4096), (512, 4096), (32, 512), (32, 4096), 4, True),
    ((input_tokens, 4096), (512, 24576), (32, 3072), (32, 24576), 4, True),
    ((input_tokens, 12288), (1536, 4096), (96, 512), (96, 4096), 4, True),
]

# 算子映射
OPERATORS = {
    "gptq_gemm": gptq_gemm,
    "gptq_gemm_opt": gptq_gemm_opt,
}


def generate_inputs(a_shape, weight_shape, zeros_shape, scales_shape, device):
    """生成GPTQ测试输入"""
    a = torch.randn(a_shape, device=device, dtype=torch.float16)
    weight = torch.randint(-2000000, 2000000, weight_shape, device=device, dtype=torch.int32)
    zeros = torch.zeros(zeros_shape, device=device, dtype=torch.int32)
    scales = torch.randn(scales_shape, device=device, dtype=torch.float16)
    idx = torch.empty((0,), device=device, dtype=torch.int32)
    return a, weight, zeros, scales, idx


def benchmark_gptq_gemm(
    a_shape, weight_shape, zeros_shape, scales_shape, bit, use_exllama, use_cuda_graph=True  # 新增控制参数
):
    device = "cuda"
    a, weight, zeros, scales, idx = generate_inputs(
        a_shape, weight_shape, zeros_shape, scales_shape, device
    )
    M, K = a_shape 
    N, _ = weight_shape  
    total_ops = 2 * M * N * K  

    @perf_report(
        Benchmark(
            x_names=["operator"],
            x_vals=list(OPERATORS.keys()),
            line_arg="case",
            line_vals=[f"a={a_shape},w={weight_shape}"],
            line_names=["GPTQ GEMM"],
            ylabel="TFLOP/s (higher is better)",
            plot_name=f"gptq-bench-{a_shape}-{weight_shape}",
            args={},
        )
    )
    def run_benchmark(operator, case):
        op = OPERATORS[operator]
        
        def run():
            with torch.no_grad():
                op(a, weight, zeros, scales, idx, use_exllama, bit)
        
        if use_cuda_graph:
            ms, min_ms, max_ms = do_bench_cudagraph(run, quantiles=[0.5, 0.2, 0.8])
        else:
            ms, min_ms, max_ms = do_bench(run, quantiles=[0.5, 0.2, 0.8])
        
        to_tflops = lambda t_ms: total_ops * 1e-12 / (t_ms * 1e-3)
        return to_tflops(ms), to_tflops(max_ms), to_tflops(min_ms)

    print(f"\n=== Testing: a={a_shape}, weight={weight_shape} (CUDA Graph: {use_cuda_graph}) ===")
    run_benchmark.run(print_data=True, show_plots=False)


if __name__ == "__main__":
    for case in TEST_CASES:
        benchmark_gptq_gemm(*case, use_cuda_graph=False)
    print("\nBenchmark completed!")