import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=800, suite="nightly-1-npu-a3", nightly=True)  # 评估两次，耗时翻倍


class TestEnableMultimodalNonMlm(CustomTestCase):
    """Testcase：Verify --enable-multimodal parameter, the mmlu accuracy of enable is not less than disable.

        [Test Category] Parameter
        [Test Target] --enable-multimodal
        """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    # 实例变量，存储两次评估的分数
    score_with_param = None
    score_without_param = None

    def _launch_server(self, enable_multimodal: bool):
        """通用服务启动方法，根据参数决定是否添加--enable-multimodal"""
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        # 按需添加多模态参数
        if enable_multimodal:
            other_args.insert(1, "--enable-multimodal")

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        self.addCleanup(kill_process_tree, process.pid)  # 自动注册清理方法，无需手动tearDown
        return process

    def _verify_inference(self):
        """通用推理功能验证：健康检查+基础生成"""
        # 健康检查
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)
        # 基础生成请求验证
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

    def _run_mmlu_eval(self) -> float:
        """通用MMLU评估执行方法，返回评估分数"""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.2)  # 保留基础分数下限断言
        return metrics["score"]

    def test_01_enable_multimodal(self):
        """测试1：带--enable-multimodal参数，执行评估并保存分数"""
        # 启动服务
        self._launch_server(enable_multimodal=True)
        # 验证推理功能
        self._verify_inference()
        # 执行MMLU评估并保存分数
        self.score_with_param = self._run_mmlu_eval()

    def test_02_disable_multimodal(self):
        """测试2：不带--enable-multimodal参数，执行评估并保存分数"""
        # 启动服务
        self._launch_server(enable_multimodal=False)
        # 验证推理功能
        self._verify_inference()
        # 执行MMLU评估并保存分数
        self.score_without_param = self._run_mmlu_eval()

    def test_03_assert_score(self):
        """测试3：断言带参数的分数 ≥ 不带参数的分数"""
        # 确保两次分数都已正确获取
        self.assertIsNotNone(self.score_with_param, "带参数的MMLU分数未获取")
        self.assertIsNotNone(self.score_without_param, "不带参数的MMLU分数未获取")
        # 核心断言：带--enable-multimodal的分数 ≥ 不带的分数
        self.assertGreaterEqual(
            self.score_with_param,
            self.score_without_param,
            f"带--enable-multimodal的MMLU分数({self.score_with_param:.4f}) 小于 不带的分数({self.score_without_param:.4f})"
        )


if __name__ == "__main__":
    unittest.main()
