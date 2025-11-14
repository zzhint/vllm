import torch
import pytest
from torch.testing import assert_close
'''
m=17
n=4096
k=4096
TEST_CASES = [
    ((m, k), (k//8, n), (k // 128, n // 8), (k // 128, n), 4, True)
]

'''


'''




TEST_CASES = [

    ((1024, 1024), (128, 4096), (8, 512), (8, 4096), 4, True)

]
'''

TEST_CASES = [
    ((10, 1024), (128, 4096), (8, 512), (8, 4096), 4, True),
    ((10, 2048), (256, 1024), (16, 128), (16, 1024), 4, True),
    ((10, 1024), (128, 6144), (8, 768), (8, 6144), 4, True),
    ((10, 3072), (384, 1024), (24, 128), (24, 1024), 4, True),
    ((10, 4096), (512, 6144), (32, 768), (32, 6144), 4, True),
    ((10, 4096), (512, 4096), (32, 512), (32, 4096), 4, True),
    ((10, 4096), (512, 24576), (32, 3072), (32, 24576), 4, True),
    ((10, 12288), (1536, 4096), (96, 512), (96, 4096), 4, True),
    ((1024, 1024), (128, 4096), (8, 512), (8, 4096), 4, True),
    ((1024, 2048), (256, 1024), (16, 128), (16, 1024), 4, True),
    ((1024, 1024), (128, 6144), (8, 768), (8, 6144), 4, True),
    ((1024, 3072), (384, 1024), (24, 128), (24, 1024), 4, True),
    ((1024, 4096), (512, 6144), (32, 768), (32, 6144), 4, True),
    ((1024, 4096), (512, 4096), (32, 512), (32, 4096), 4, True),
    ((1024, 4096), (512, 24576), (32, 3072), (32, 24576), 4, True),
    ((1024, 12288), (1536, 4096), (96, 512), (96, 4096), 4, True),
]


def set_fixed_seed(seed: int = 42):
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.mark.parametrize("a_shape, weight_shape, zeros_shape, scales_shape, bit, use_exllama", TEST_CASES)
def test_gptq_gemm_opt_correctness(a_shape, weight_shape, zeros_shape, scales_shape, bit, use_exllama):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_fixed_seed(43)
    #torch.set_printoptions(threshold=float('inf'), edgeitems=torch.inf)
    if device.type != "cuda":
        pytest.skip("需要CUDA设备进行测试")

    a = torch.randn(a_shape, device=device, dtype=torch.float16) / 10 # 输入特征
    weight = torch.randint(
        -2000000, 2000000,
        weight_shape,
        device=device,
        dtype=torch.int32
    )
    #zeros = torch.zeros(zeros_shape, device=device, dtype=torch.int32)  
    zeros = torch.randint(
        -2000000, 2000000,
        zeros_shape,
        device=device,
        dtype=torch.int32
    )
    scales = torch.rand(scales_shape, device=device, dtype=torch.float16) * 0.2 - 0.1
    print("最大值:", scales.max().item())
    print("最小值:", scales.min().item())
    print("平均值:", scales.mean().item())
    idx = torch.empty((0,), device=device, dtype=torch.int32) 

    with torch.no_grad():
        output_original = torch.ops._C.gptq_gemm(a, weight, zeros, scales, idx, use_exllama, bit)
        output_new = torch.ops._C.gptq_gemm_opt(a, weight, zeros, scales, idx, use_exllama, bit)
    torch.cuda.synchronize()
    assert output_original.shape == output_new.shape, "输出形状不一致"
    
    print(a)
    print(weight)
    print(zeros)
    print(scales)
    print(output_original)
    print(output_new)

    diff = torch.abs(output_original - output_new)
    mean_abs_diff = diff.mean().item()
    max_diff = diff.max().item()
    max_idx_flat = diff.argmax().item()
    max_row = max_idx_flat // diff.shape[1]
    max_col = max_idx_flat % diff.shape[1]

    # 打印基础信息
    print(f"差值绝对值的平均值: {mean_abs_diff:.6f}")
    print(f"最大绝对差值: {max_diff:.6f}")
    print(f"最大差值位置: (行={max_row}, 列={max_col})")
    print(f"  原始值: {output_original[max_row, max_col].item():.6f}")
    print(f"  新值: {output_new[max_row, max_col].item():.6f}")


    # 定义附近元素的范围（前后各10个，共21个元素）
    row_start = max(0, max_row - 2)  # 避免行索引小于0
    row_end = min(diff.shape[0], max_row + 3)  # 避免行索引超过上限（切片是左闭右开，+11才会包含max_row+10）
    col_start = max(0, max_col - 2)  # 避免列索引小于0
    col_end = min(diff.shape[1], max_col + 3)  # 避免列索引超过上限


    # 打印附近的原始值（output_original）
    print("\n===== 原始值附近区域（行范围: {}~{}, 列范围: {}~{}） =====".format(
        row_start, row_end-1, col_start, col_end-1
    ))
    print("原始值张量切片:")
    print(output_original[row_start:row_end, col_start:col_end].cpu().numpy())  # 转为numpy方便对齐打印


    # 打印附近的新值（output_new）
    print("\n===== 新值附近区域（行范围: {}~{}, 列范围: {}~{}） =====".format(
        row_start, row_end-1, col_start, col_end-1
    ))
    print("新值张量切片:")
    print(output_new[row_start:row_end, col_start:col_end].cpu().numpy())


    # 打印附近的差值（方便对比）
    print("\n===== 差值附近区域（行范围: {}~{}, 列范围: {}~{}） =====".format(
        row_start, row_end-1, col_start, col_end-1
    ))
    print("差值张量切片:")
    print(diff[row_start:row_end, col_start:col_end].cpu().numpy())

    print("差值张量排序:")
    diff_slice = diff[row_start:row_end, col_start:col_end].cpu().numpy()
    diff_flat = diff_slice.flatten()
    sorted_diff = sorted(diff_flat, reverse=True)  # 逆序排序（大→小）
    print(sorted_diff)

    rtol = 0.1
    atol = 1
    assert_close(
        output_original,
        output_new,
        rtol=rtol,
        atol=atol,
        msg=f"数值不一致（bit={bit}, use_exllama={use_exllama}）"
    )

