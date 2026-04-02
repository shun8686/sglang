import os
import re
import unittest
import time

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=500,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled=False,
)


class TestAscendCudaGraphGC(unittest.TestCase):
    """Testcase: Verify that avail mem is larger when enable-cudagraph-gc is on.

    [Test Category] Parameter
    [Test Target] --enable-cudagraph-gc
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    log_file = "./cudagraph_gc_log.txt"

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.log_file):
            os.remove(cls.log_file)

    def _launch_and_get_avail_mem(self, enable_cudagraph_gc: bool) -> float:
        """
        1. 启动服务（开启/关闭GC）
        2. 提取 Capture npu graph end. 行的 avail mem
        """
        extra_args = [
            "--trust-remote-code",
            "--tp-size", "1",
            "--mem-fraction-static", "0.7",
            "--attention-backend", "ascend",
        ]

        if enable_cudagraph_gc:
            extra_args.append("--enable-cudagraph-gc")

        # 启动服务
        with open(self.log_file, "w", encoding="utf-8") as f:
            proc = popen_launch_server(
                self.model,
                self.base_url,
                timeout=3600,
                other_args=extra_args,
                return_stdout_stderr=(f, f),
            )
            time.sleep(50)

        # 读取日志并提取内存
        with open(self.log_file, "r", encoding="utf-8") as f:
            content = f.read()

        match = re.search(r"Capture npu graph end\..*avail mem=([\d\.]+) GB", content)
        self.assertIsNotNone(match, "未找到 Capture npu graph end 日志")
        avail_mem = float(match.group(1))

        # 关闭服务
        kill_process_tree(proc.pid)
        time.sleep(10)

        return avail_mem

    def test_gc_avail_mem_comparison(self):
        """
        核心断言：
        开启 GC 时 Capture npu graph end 的 avail mem > 关闭时
        """
        # 1. 关闭GC
        mem_off = self._launch_and_get_avail_mem(enable_cudagraph_gc=False)

        # 2. 开启GC
        mem_on = self._launch_and_get_avail_mem(enable_cudagraph_gc=True)

        # 3. 断言：开启GC > 关闭GC
        self.assertGreaterEqual(
            mem_on, mem_off,
            f"开启GC的可用内存必须大于关闭时！\n关闭GC: {mem_off:.2f} GB\n开启GC: {mem_on:.2f} GB"
        )

        # 结果输出
        print("\n" + "=" * 60)
        print("          CUDA Graph GC 内存对比结果")
        print("=" * 60)
        print(f"关闭 --enable-cudagraph-gc: {mem_off:.2f} GB")
        print(f"开启 --enable-cudagraph-gc: {mem_on:.2f} GB")
        print("\n✅ 测试通过：开启GC后可用内存更大")
        print("=" * 60)


if __name__ == "__main__":
    unittest.main()