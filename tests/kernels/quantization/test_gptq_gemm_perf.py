import torch
import pytest
from torch.testing import assert_close

m=1024
n=4096
k=4096
TEST_CASES = [
    ((m, k), (k//8, n), (k // 128, n // 8), (k // 128, n), 4, True)
]

'''

input_tokens = 256

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
'''

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
    idx = torch.empty((0,), device=device, dtype=torch.int32) 

    with torch.no_grad():
        output_original = torch.ops._C.gptq_gemm(a, weight, zeros, scales, idx, use_exllama, bit)
        output_new = torch.ops._C.gptq_gemm_opt(a, weight, zeros, scales, idx, use_exllama, bit)
    torch.cuda.synchronize()
    assert output_original.shape == output_new.shape, "输出形状不一致"
    
    rtol = 0.1
    atol = 1
    assert_close(
        output_original,
        output_new,
        rtol=rtol,
        atol=atol,
        msg=f"数值不一致（bit={bit}, use_exllama={use_exllama}）"
    )

