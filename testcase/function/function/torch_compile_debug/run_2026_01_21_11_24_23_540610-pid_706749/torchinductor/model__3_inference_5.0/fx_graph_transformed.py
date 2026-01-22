class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s64)", arg1_1: "i32[1, s64]", arg2_1: "Sym(s3)", arg3_1: "f32[1, s3]"):
         # File: /usr/local/python3.11.14/lib/python3.11/site-packages/llguidance/torch.py:22 in apply_token_bitmask_inplace_kernel, code: mask_expanded = torch.repeat_interleave(mask, 32, dim=1)
        unsqueeze: "i32[1, s64, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, 2);  arg1_1 = None
        expand: "i32[1, s64, 32]" = torch.ops.aten.expand.default(unsqueeze, [1, arg0_1, 32]);  unsqueeze = None
        clone: "i32[1, s64, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        mul_7: "Sym(32*s64)" = arg0_1 * 32
        view: "i32[1, 32*s64]" = torch.ops.aten.reshape.default(clone, [1, mul_7]);  clone = mul_7 = None
        
         # File: /usr/local/python3.11.14/lib/python3.11/site-packages/llguidance/torch.py:23 in apply_token_bitmask_inplace_kernel, code: bit_indices = torch.arange(32, device=logits.device,
        iota: "i32[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int32, device = device(type='npu', index=0), requires_grad = False)
        
         # File: /usr/local/python3.11.14/lib/python3.11/site-packages/llguidance/torch.py:24 in apply_token_bitmask_inplace_kernel, code: dtype=torch.int32).repeat(mask.shape[1])
        repeat: "i32[32*s64]" = torch.ops.aten.repeat.default(iota, [arg0_1]);  iota = arg0_1 = None
        
         # File: /usr/local/python3.11.14/lib/python3.11/site-packages/llguidance/torch.py:25 in apply_token_bitmask_inplace_kernel, code: bit_masks = (mask_expanded >> bit_indices) & 1  # Extract each bit
        rshift: "i32[1, 32*s64]" = torch.ops.aten.__rshift__.Tensor(view, repeat);  view = repeat = None
        bitwise_and: "i32[1, 32*s64]" = torch.ops.aten.bitwise_and.Scalar(rshift, 1);  rshift = None
        
         # File: /usr/local/python3.11.14/lib/python3.11/site-packages/llguidance/torch.py:26 in apply_token_bitmask_inplace_kernel, code: bit_masks = bit_masks[:, :logits.shape[1]]  # Trim to match vocab size
        slice_2: "i32[1, s3]" = torch.ops.aten.slice.Tensor(bitwise_and, 1, 0, arg2_1);  bitwise_and = arg2_1 = None
        
         # File: /usr/local/python3.11.14/lib/python3.11/site-packages/llguidance/torch.py:27 in apply_token_bitmask_inplace_kernel, code: logits.masked_fill_(bit_masks == 0, float("-inf"))  # Apply mask
        eq_12: "b8[1, s3]" = torch.ops.aten.eq.Scalar(slice_2, 0);  slice_2 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "f32[1, s3]" = torch.ops.aten.where.self(eq_12, full_default, arg3_1);  eq_12 = full_default = None
        copy_: "f32[1, s3]" = torch.ops.aten.copy_.default(arg3_1, where);  arg3_1 = where = copy_ = None
        return ()
        