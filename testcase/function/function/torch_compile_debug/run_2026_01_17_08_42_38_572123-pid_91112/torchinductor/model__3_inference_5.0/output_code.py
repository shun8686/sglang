# AOT ID: ['3_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import torch_npu
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
import torch_npu
has_initialized = False
from torch_npu._inductor import get_current_raw_stream as get_raw_stream
from torch_npu._inductor import get_current_raw_stream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/ux/cux3bd6nzfb6irgtnfu5buekn5exok35ptijlh7yqjmg2eq5htwv.py
# Topologically Sorted Source Nodes: [mask_expanded], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   mask_expanded => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_unk_fused_clone_0 = async_compile.triton('triton_unk_fused_clone_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._inductor.runtime import triton_helpers
from torch_npu._inductor import npu_triton_heuristics
from torch_npu._inductor import npu_triton_helpers
from torch_npu._inductor.runtime import NPUDeviceProperties
from torch_npu._inductor.npu_triton_helpers import libdevice, math as tl_math
import torch
import torch_npu

@npu_triton_heuristics.pointwise_npu_index(
    size_hints=[4008, 32], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'y0_numel': 'i32', 'x1_numel': 'i32'}, 'device': NPUDeviceProperties(type='npu', index=0, multi_processor_count=24, cc='Ascend910_9392', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_clone_0', 'mutated_arg_names': [], 'backend_hash': 'd6805cc2a15a5588c5eb46617914d023bdce2f41299f4011b1d756d8a333eed6', 'split_axis': [0], 'tiling_axis': [0, 1], 'axis_names': ['y0', 'x1'], 'low_dims': {0, 1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.int32, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_clone_0(in_ptr0, out_ptr0, y0_numel, x1_numel, Y0BLOCK : tl.constexpr, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (x1_numel + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None,:]
            x1_mask = x1 < x1_numel
            tmp0 = tl.load(in_ptr0 + (y0), y0_mask)
            tl.store(out_ptr0 + (x1 + 32*y0), tmp0, x1_mask & y0_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_root/6d/c6d6tptu2jepexdzxmrg4xyvt2nfxasawt7xyuepwrpl4jub55ro.py
# Topologically Sorted Source Nodes: [arange, bit_indices], Original ATen: [aten.arange, aten.repeat]
# Source node to ATen node mapping:
#   arange => iota
#   bit_indices => repeat
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: npu:0, requires_grad: False})
#   %repeat : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%iota, [%arg0_1]), kwargs = {})
# SchedulerNodes: [SchedulerNode(name='op1')]

triton_unk_fused_arange_repeat_1 = async_compile.triton('triton_unk_fused_arange_repeat_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._inductor.runtime import triton_helpers
from torch_npu._inductor import npu_triton_heuristics
from torch_npu._inductor import npu_triton_helpers
from torch_npu._inductor.runtime import NPUDeviceProperties
from torch_npu._inductor.npu_triton_helpers import libdevice, math as tl_math
import torch
import torch_npu

@npu_triton_heuristics.pointwise_npu_index(
    size_hints=[4008, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'x1_numel': 'i32', 'x2_numel': 'i32'}, 'device': NPUDeviceProperties(type='npu', index=0, multi_processor_count=24, cc='Ascend910_9392', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_arange_repeat_1', 'mutated_arg_names': [], 'backend_hash': 'd6805cc2a15a5588c5eb46617914d023bdce2f41299f4011b1d756d8a333eed6', 'split_axis': [0], 'tiling_axis': [0, 1], 'axis_names': ['x1', 'x2'], 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.int32, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_arange_repeat_1(out_ptr0, x1_numel, x2_numel, X1BLOCK : tl.constexpr, X1BLOCK_SUB : tl.constexpr, X2BLOCK_SUB : tl.constexpr):
    x1_offset = tl.program_id(0) * X1BLOCK
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (X1BLOCK + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    base_x2= tl.arange(0, X2BLOCK_SUB)
    loops_x2 = (x2_numel + X2BLOCK_SUB - 1) // X2BLOCK_SUB
    for loop_x1 in range(loops_x1):
        x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[:,None]
        x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
        for loop_x2 in range(loops_x2):
            x2 = (loop_x2 * X2BLOCK_SUB) + base_x2[None,:]
            x2_mask = x2 < x2_numel
            tmp0 = x2
            tl.store(out_ptr0 + (x2 + 32*x1), tmp0, x1_mask & x2_mask)
''', device_str='npu')


# kernel path: /tmp/torchinductor_root/u2/cu2gby5rahek6rmikhiia2swtteltwya2vftm563njhst4issym2.py
# Topologically Sorted Source Nodes: [eq, masked_fill_], Original ATen: [aten.eq, aten.masked_fill]
# Source node to ATen node mapping:
#   eq => eq_12
#   masked_fill_ => full_default, where
# Graph fragment:
#   %eq_12 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%slice_2, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_12, %full_default, %arg3_1), kwargs = {})
# SchedulerNodes: [SchedulerNode(name='op4')]

triton_unk_fused_eq_masked_fill_2 = async_compile.triton('triton_unk_fused_eq_masked_fill_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._inductor.runtime import triton_helpers
from torch_npu._inductor import npu_triton_heuristics
from torch_npu._inductor import npu_triton_helpers
from torch_npu._inductor.runtime import NPUDeviceProperties
from torch_npu._inductor.npu_triton_helpers import libdevice, math as tl_math
import torch
import torch_npu

@npu_triton_heuristics.pointwise_npu_index(
    size_hints=[128256], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'x0_numel': 'i32'}, 'device': NPUDeviceProperties(type='npu', index=0, multi_processor_count=24, cc='Ascend910_9392', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused_eq_masked_fill_2', 'mutated_arg_names': [], 'backend_hash': 'd6805cc2a15a5588c5eb46617914d023bdce2f41299f4011b1d756d8a333eed6', 'split_axis': [0], 'tiling_axis': [0], 'axis_names': ['x0'], 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused_eq_masked_fill_2(in_ptr0, in_ptr1, out_ptr0, x0_numel, X0BLOCK : tl.constexpr, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = tl.load(in_ptr0 + (x0), x0_mask)
        tmp5 = tl.load(in_ptr1 + (x0), x0_mask)
        tmp1 = tl.full([1], 1, tl.int32)
        tmp2 = tmp0 & tmp1
        tmp3 = tl.full([1], 0, tl.int32)
        tmp4 = tmp2 == tmp3
        tmp6 = float("-inf")
        tmp7 = tl.where(tmp4, tmp6, tmp5)
        tl.store(out_ptr0 + (x0), tmp7, x0_mask)
''', device_str='npu')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s64 = arg0_1
    s3 = arg2_1
    buf0 = empty_strided((1, s64, 32), (32*s64, 32, 1), device='npu', dtype=torch.int32)
    # Topologically Sorted Source Nodes: [mask_expanded], Original ATen: [aten.clone]
    stream0 = get_raw_stream(0)
    triton_unk_fused_clone_0.run(arg1_1, buf0, 4008, 32, stream=stream0)

    buf1 = empty_strided((32*s64, ), (1, ), device='npu', dtype=torch.int32)
    # Topologically Sorted Source Nodes: [arange, bit_indices], Original ATen: [aten.arange, aten.repeat]
    stream0 = get_raw_stream(0)
    triton_unk_fused_arange_repeat_1.run(buf1, 4008, 32, stream=stream0)
    # Topologically Sorted Source Nodes: [arange, bit_indices, rshift], Original ATen: [aten.arange, aten.repeat, aten.__rshift__]
    buf2 = torch.ops.aten.__rshift__.Tensor(reinterpret_tensor(buf0, (1, 32*s64), (0, 1), 0), buf1)


    buf3 = buf2
    assert_size_stride(buf3, (1, 32*s64), (32*s64, 1), 'torch.ops.aten.__rshift__.Tensor')
    assert_alignment(buf3, 16, 'torch.ops.aten.__rshift__.Tensor')

    buf4 = empty_strided((1, s3), (s3, 1), device='npu', dtype=torch.float32)
    # Topologically Sorted Source Nodes: [eq, masked_fill_], Original ATen: [aten.eq, aten.masked_fill]
    stream0 = get_raw_stream(0)
    triton_unk_fused_eq_masked_fill_2.run(buf3, arg3_1, buf4, 128256, stream=stream0)

    # Topologically Sorted Source Nodes: [eq, masked_fill_], Original ATen: [aten.eq, aten.masked_fill, aten.copy_]
    buf5 = torch.ops.aten.copy_.default(arg3_1, buf4)

    return ()


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 4008
    arg1_1 = rand_strided((1, 4008), (4008, 1), device='npu:0', dtype=torch.int32)
    arg2_1 = 128256
    arg3_1 = rand_strided((1, 128256), (128256, 1), device='npu:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
