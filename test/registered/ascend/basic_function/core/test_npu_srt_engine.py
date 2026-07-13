"""
NPU  SRT Engine consistency testing
Usage:
python3 -m unittest test_npu_srt_engine.TestNPUSRTEngine.test_1_engine_runtime_consistency
"""

import asyncio
import json
import multiprocessing as mp
import unittest

import torch

import sglang as sgl
from sglang.bench_offline_throughput import BenchArgs, throughput_test
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    QWEN2_1_5B_INSTRUCT_GTE_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestNPUSRTEngine(CustomTestCase):

    def test_1_engine_runtime_consistency(self):
        """验证 Engine 和 Runtime 对同一 prompt 生成一致文本"""
        if not torch.npu.is_available():
            self.skipTest("NPU device not available")

        prompt = "Today is a sunny day and I like"
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            random_seed=42,
            attention_backend="ascend",
        )
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        runtime = sgl.Runtime(
            model_path=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            random_seed=42,
            attention_backend="ascend",
        )
        out2 = json.loads(runtime.generate(prompt, sampling_params))["text"]
        runtime.shutdown()

        print("==== Answer 1 ====")
        print(out1)
        print("==== Answer 2 ====")
        print(out2)
        self.assertEqual(out1, out2)

    def test_2_engine_runtime_encode_consistency(self):
        """验证 Engine 和 Runtime 的嵌入输出一致"""
        if not torch.npu.is_available():
            self.skipTest("NPU device not available")

        prompt = "Today is a sunny day and I like"
        engine = sgl.Engine(
            model_path=QWEN2_1_5B_INSTRUCT_GTE_WEIGHTS_PATH,
            is_embedding=True,
            random_seed=42,
            attention_backend="ascend",
        )
        out1 = torch.tensor(engine.encode(prompt)["embedding"])
        engine.shutdown()

        runtime = sgl.Runtime(
            model_path=QWEN2_1_5B_INSTRUCT_GTE_WEIGHTS_PATH,
            is_embedding=True,
            random_seed=42,
            attention_backend="ascend",
        )
        out2 = torch.tensor(json.loads(runtime.encode(prompt))["embedding"])
        runtime.shutdown()

        self.assertTrue(torch.allclose(out1, out2, atol=1e-5, rtol=1e-3))

    def test_3_engine_token_ids_consistency(self):
        """验证文本 prompt 输入和 token_ids 输入生成结果一致"""
        if not torch.npu.is_available():
            self.skipTest("NPU device not available")

        prompt = "Today is a sunny day and I like"
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            random_seed=42,
            disable_radix_cache=True,
            attention_backend="ascend",
        )
        out1 = engine.generate(prompt, sampling_params)["text"]

        tokenizer = get_tokenizer(LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH)
        token_ids = tokenizer.encode(prompt)
        out2 = engine.generate(input_ids=token_ids, sampling_params=sampling_params)[
            "text"
        ]

        engine.shutdown()

        print("==== Answer 1 ====")
        print(out1)
        print("==== Answer 2 ====")
        print(out2)
        self.assertEqual(out1, out2)

    def test_6_engine_cpu_offload(self):
        """验证 CPU offload 模式下推理结果与正常模式一致（NPU 暂不支持，跳过）"""
        self.skipTest("NPU does not support cpu_offload_gb")

    def test_7_engine_offline_throughput(self):
        """验证离线吞吐量基准测试能正常运行"""
        if not torch.npu.is_available():
            self.skipTest("NPU device not available")

        server_args = ServerArgs(
            model_path=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            attention_backend="ascend",
        )
        bench_args = BenchArgs(num_prompts=10)
        result = throughput_test(server_args=server_args, bench_args=bench_args)
        # NPU 上吞吐量阈值调低（Qwen3-0.6B 在 NPU 上性能较低）
        self.assertGreater(result["total_throughput"], 100)

    def test_8_engine_async_encode_consistency(self):
        """验证同步和异步嵌入 API 输出一致"""
        if not torch.npu.is_available():
            self.skipTest("NPU device not available")

        prompt = "Today is a sunny day and I like"
        engine = sgl.Engine(
            model_path=QWEN2_1_5B_INSTRUCT_GTE_WEIGHTS_PATH,
            is_embedding=True,
            random_seed=42,
            disable_radix_cache=True,
            attention_backend="ascend",
        )

        out1 = torch.tensor(engine.encode(prompt)["embedding"])
        loop = asyncio.get_event_loop()
        out2 = torch.tensor(
            loop.run_until_complete(engine.async_encode(prompt))["embedding"]
        )

        engine.shutdown()

        print("\n==== Shapes ====")
        print(f"sync shape: {out1.shape}")
        print(f"async shape: {out2.shape}")
        self.assertTrue(
            torch.allclose(out1, out2, atol=1e-5, rtol=1e-3),
            "Sync and async embeddings are not equal within tolerance",
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    unittest.main(warnings="ignore")
