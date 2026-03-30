import unittest
import time

from sglang.test.ascend.test_ascend_utils import PHI_4_MULTIMODAL_INSTRUCT_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=800, suite="nightly-4-npu-a3", nightly=True)


class TestPhi4MultimodalLatencyCompare(TestVLMModels):
    """Testcase: Compare latency with / without --keep-mm-feature-on-device."""

    model = PHI_4_MULTIMODAL_INSTRUCT_WEIGHTS_PATH
    mmmu_accuracy = 0.2

    # 公共基础参数
    base_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-radix-cache",
    ]

    def _run_and_measure_time(self, extra_args):
        """运行测试 + 手动计时，返回耗时秒数"""
        self.other_args = self.base_args + extra_args

        # 手动开始计时
        start = time.time()

        # 执行测试
        metrics = self._run_vlm_mmmu_test()

        # 手动结束计时
        duration = time.time() - start

        print(f"本次耗时: {duration:.2f}s | metrics: {metrics}")
        return duration

    def test_vlm_mmmu_latency_compare(self):
        print("\n========== 测试 1：不带 --keep-mm-feature-on-device ==========")
        time_off = self._run_and_measure_time([])

        print("\n========== 测试 2：带 --keep-mm-feature-on-device ==========")
        time_on = self._run_and_measure_time(["--keep-mm-feature-on-device"])


        # 断言：开启参数后耗时必须更少
        self.assertLess(time_on, time_off, "开启 --keep-mm-feature-on-device 后延迟没有降低！")
        print("\n✅ 测试通过：开启参数成功降低延迟！")


if __name__ == "__main__":
    unittest.main()