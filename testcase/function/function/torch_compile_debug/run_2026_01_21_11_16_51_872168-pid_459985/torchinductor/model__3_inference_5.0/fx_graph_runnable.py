
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_root'
os.environ['SGLANG_ENABLE_TORCH_COMPILE'] = '0'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.assume_static_by_default = False
torch._inductor.config.allow_buffer_reuse = False
torch._inductor.config.compile_threads = 1
torch._inductor.config.comprehensive_padding = False
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.8.0+cpu
# torch cuda version: None
# torch git version: a1cb3cc05d46d198467bebbb6e8fba50a325d4e7


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        unsqueeze = torch.ops.aten.unsqueeze.default(arg1_1, 2);  arg1_1 = None
        expand = torch.ops.aten.expand.default(unsqueeze, [1, arg0_1, 32]);  unsqueeze = None
        clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        mul_7 = arg0_1 * 32
        view = torch.ops.aten.view.default(clone, [1, mul_7]);  clone = mul_7 = None
        iota = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int32, device = device(type='npu', index=0), requires_grad = False)
        repeat = torch.ops.aten.repeat.default(iota, [arg0_1]);  iota = arg0_1 = None
        rshift = torch.ops.aten.__rshift__.Tensor(view, repeat);  view = repeat = None
        bitwise_and = torch.ops.aten.bitwise_and.Scalar(rshift, 1);  rshift = None
        slice_2 = torch.ops.aten.slice.Tensor(bitwise_and, 1, 0, arg2_1);  bitwise_and = arg2_1 = None
        eq_12 = torch.ops.aten.eq.Scalar(slice_2, 0);  slice_2 = None
        full_default = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(eq_12, full_default, arg3_1);  eq_12 = full_default = None
        copy_ = torch.ops.aten.copy_.default(arg3_1, where);  arg3_1 = where = copy_ = None
        return ()
        
def load_args(reader):
    reader.symint(4008)  # arg0_1
    buf0 = reader.storage(None, 4*s64, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf0, (1, s64), dtype=torch.int32, is_leaf=True)  # arg1_1
    reader.symint(128256)  # arg2_1
    buf1 = reader.storage(None, 4*s3, device=device(type='npu', index=0))
    reader.tensor(buf1, (1, s3), is_leaf=True)  # arg3_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)