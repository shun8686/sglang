import os
import unittest
import time
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestTokenizerBatchDecodeCompare(CustomTestCase):
    """
    Test compare: with / without --disable-tokenizer-batch-decode
    When skip_special_tokens=True in request
    """
    model = QWEN3_0_6B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    test_prompt = "Hello, my name is"

    # 公共基础参数
    base_args = [
        "--trust-remote-code",
        "--tp-size", "1",
        "--mem-fraction-static", "0.8",
        "--attention-backend", "ascend",
        "--disable-cuda-graph",
    ]

    def _run_server(self, extra_args):
        """启动服务"""
        self.out_log = open("./tmp_out.txt", "w+")
        self.err_log = open("./tmp_err.txt", "w+")
        args = self.base_args + extra_args
        print(f"\n🚀 启动参数: {args}")

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=1200,
            other_args=args,
            return_stdout_stderr=(self.out_log, self.err_log),
        )
        time.sleep(15)  # 等待服务启动
        return process

    def _stop_server(self, process):
        """停止服务"""
        kill_process_tree(process.pid)
        self.out_log.close()
        self.err_log.close()
        for f in ["./tmp_out.txt", "./tmp_err.txt"]:
            if os.path.exists(f):
                os.remove(f)
        time.sleep(5)

    def _send_generate_request(self):
        """发送生成请求，开启 skip_special_tokens=True"""
        print(f"\n📤 发送请求: {self.test_prompt}")
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": self.test_prompt,
                "sampling_params": {
                    "temperature": 0.1,
                    "max_new_tokens": 64,
                    "skip_special_tokens": True  # 核心参数
                },
            },
        )
        result = response.json()
        text = result["text"]
        print(f"📥 返回结果: {repr(text)}")  # repr 能看见隐藏字符
        return text

    def test_compare_disable_tokenizer_batch_decode(self):
        """
        对比两组配置：
        1. 不带 --disable-tokenizer-batch-decode
        2. 带    --disable-tokenizer-batch-decode
        查看返回文本差异
        """
        # ==========================
        # 测试1：默认（不 disable）
        # ==========================
        print("\n" + "=" * 80)
        print("🔹 测试 1：不带 --disable-tokenizer-batch-decode")
        print("=" * 80)
        p1 = self._run_server([])
        text_with_batch = self._send_generate_request()
        self._stop_server(p1)

        # ==========================
        # 测试2：disable batch decode
        # ==========================
        print("\n" + "=" * 80)
        print("🔹 测试 2：带 --disable-tokenizer-batch-decode")
        print("=" * 80)
        p2 = self._run_server(["--disable-tokenizer-batch-decode"])
        text_without_batch = self._send_generate_request()
        self._stop_server(p2)

        # ==========================
        # 对比输出结果
        # ==========================
        print("\n" + "=" * 80)
        print("📊 最终结果对比")
        print("=" * 80)
        print(f"【带 batch decode】: {repr(text_with_batch)}")
        print(f"【无 batch decode】: {repr(text_without_batch)}")

        # 检查是否有特殊 token 残留
        has_special_batch = any(t in text_with_batch for t in ["<|endoftext|>", "<|pad|>", "<|unk|>"])
        has_special_nobatch = any(t in text_without_batch for t in ["<|endoftext|>", "<|pad|>", "<|unk|>"])

        print(f"\n🔍 是否含特殊token：")
        print(f"  带 batch decode: {has_special_batch}")
        print(f"  无 batch decode: {has_special_nobatch}")

        # 断言：开启 skip_special_tokens 后，理想情况两者都不应有特殊 token
        self.assertFalse(has_special_batch, "batch decode 不应输出特殊 token")
        self.assertFalse(has_special_nobatch, "disable batch decode 不应输出特殊 token")

        print("\n✅ 测试完成！")


if __name__ == "__main__":
    unittest.main()

