import unittest
from types import SimpleNamespace
import requests
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestEnableMultimodalNonMlm(CustomTestCase):
    """Testcase：Verify set --enable-multimodal parameter, the mmlu accuracy greaterequal not set --enable-multimodal.

        [Test Category] Parameter
        [Test Target] --enable-multimodal
        """
    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    # 修复1：定义为类变量（所有实例共享），用于跨测试方法传递分数
    score_with_param = None
    score_without_param = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 修复2：规范实例变量定义（如果有其他实例变量，统一放这里）
        # 本案例中核心是类变量，此处仅做规范示范

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
        # 修复3：通过类名赋值类变量，跨实例共享
        TestEnableMultimodalNonMlm.score_with_param = self._run_mmlu_eval()

    def test_02_disable_multimodal(self):
        """测试2：不带--enable-multimodal参数，执行评估并保存分数"""
        # 启动服务
        self._launch_server(enable_multimodal=False)
        # 验证推理功能
        self._verify_inference()
        # 修复3：通过类名赋值类变量，跨实例共享
        TestEnableMultimodalNonMlm.score_without_param = self._run_mmlu_eval()

    def test_03_assert_score(self):
        """测试3：断言带参数的分数 ≥ 不带参数的分数"""
        # 修复4：通过类名读取类变量，验证非空
        self.assertIsNotNone(TestEnableMultimodalNonMlm.score_with_param, "带参数的MMLU分数未获取")
        self.assertIsNotNone(TestEnableMultimodalNonMlm.score_without_param, "不带参数的MMLU分数未获取")
        # 核心断言：带--enable-multimodal的分数 ≥ 不带的分数
        self.assertGreaterEqual(
            TestEnableMultimodalNonMlm.score_with_param,
            TestEnableMultimodalNonMlm.score_without_param,
            f"带--enable-multimodal的MMLU分数({TestEnableMultimodalNonMlm.score_with_param:.4f}) 小于 不带的分数({TestEnableMultimodalNonMlm.score_without_param:.4f})"
        )


if __name__ == "__main__":
    # 可选：添加verbosity=2，打印更详细的测试日志
    unittest.main(verbosity=2)
