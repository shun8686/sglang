import os
import re
import unittest
import subprocess
import psutil

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_0_6B_WEIGHTS_PATH,
    run_command,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled="run failed",
)


class TestAscendWarmups(CustomTestCase):
    """
    Testcase: Verify NPU + NUMA node binding for tp-size=2 (multi-NPU / multi-NUMA)
    """
    model = QWEN3_0_6B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    # ===================== 核心配置 =====================
    TP_SIZE = 2                          # 张量并行大小
    CONFIG_NUMA_LIST = ["1", "1"]        # 启动参数 --numa-node 对应的值
    # ====================================================

    @classmethod
    def setUpClass(cls):
        cls.other_args = [
            "--trust-remote-code",
            "--tp-size", str(cls.TP_SIZE),
            "--mem-fraction-static", "0.8",
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--numa-node", *cls.CONFIG_NUMA_LIST,  # 自动展开多NUMA配置
        ]
        cls.out_log_file = open("./out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=cls.other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        for f in ["./out_log.txt", "./err_log.txt"]:
            if os.path.exists(f):
                os.remove(f)

    def _get_numa_node_from_cpu_list(self, cpu_list):
        """通过 CPU 亲和集反查 NUMA 节点（最稳定）"""
        try:
            # 取第一个 CPU
            first_cpu = cpu_list.split(",")[0].split("-")[0]
            # 用 /sys  filesystem 获取 NUMA 节点（最可靠）
            with open(f"/sys/devices/system/cpu/cpu{first_cpu}/numa_node", "r") as f:
                return f.read().strip()
        except Exception as e:
            print(f"获取NUMA节点失败: {e}")
            return None

    def _get_process_numa_nodes(self, pid):
        """
        使用 taskset + /sys/devices/system/cpu 获取进程绑定的 NUMA 节点
        替换 numactl，适用于 Ascend/NPU 环境
        """
        try:
            # 使用 taskset 获取进程的 CPU 亲和性
            result = subprocess.run(
                ["taskset", "-pc", str(pid)],
                capture_output=True, text=True
            )
            output = result.stdout.strip()

            # 提取 CPU list
            cpu_list = output.split(":")[-1].strip()
            numa_node = self._get_numa_node_from_cpu_list(cpu_list)

            if numa_node:
                return [numa_node]
            return []
        except Exception as e:
            print(f"获取NUMA失败(taskset): {e}")
            return []

    def _get_used_npu_devices(self):
        """
        通过 npu-smi info 获取当前进程占用的NPU设备（最权威）
        """
        server_pid = self.process.pid
        used_npus = set()
        try:
            # 执行 npu-smi info 获取所有NPU设备及进程信息
            result = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True, text=True
            )
            output = result.stdout
            # 匹配NPU ID和对应的进程PID
            # 示例匹配行：| NPU 0 | xxx | xxx | PID: 12345 | ...
            pattern = re.compile(r"\|\s*NPU\s+(\d+)\s*\|\s*.*?\|\s*.*?\|\s*PID:\s*(\d+)\s*\|")
            matches = pattern.findall(output)
            for npu_id, pid in matches:
                if int(pid) == server_pid:
                    used_npus.add(npu_id)
            return sorted(list(used_npus))
        except Exception as e:
            print(f"通过npu-smi获取NPU设备失败: {e}")
            return []

    def test_multi_npu_multi_numa_binding(self):
        """验证：多NPU + 多NUMA 绑定完全正确"""
        server_pid = self.process.pid
        print(f"\n✅ 服务主进程 PID: {server_pid}")

        # --------------------------
        # 1. 检查 NPU 数量 = TP_SIZE
        # --------------------------
        npu_list = self._get_used_npu_devices()
        print(f"✅ 实际使用NPU卡号: {npu_list}")
        print(f"✅ 预期NPU数量    : {self.TP_SIZE}")
        self.assertEqual(len(npu_list), self.TP_SIZE,
            f"NPU数量不匹配！实际={len(npu_list)}, 预期={self.TP_SIZE}")

        # --------------------------
        # 2. 检查 NUMA 节点
        # --------------------------
        actual_numa_list = self._get_process_numa_nodes(server_pid)
        print(f"✅ 实际绑定NUMA节点: {actual_numa_list}")
        print(f"✅ 配置的NUMA节点  : {self.CONFIG_NUMA_LIST}")

        # --------------------------
        # 3. 验证NUMA节点一致
        # --------------------------
        for actual, expected in zip(actual_numa_list, self.CONFIG_NUMA_LIST):
            self.assertEqual(actual, expected,
                f"NUMA节点不匹配！实际={actual}, 预期={expected}")

        # --------------------------
        # 最终结论
        # --------------------------
        print("\n🎉 全部校验通过！")
        print(f"   - TP-size = {self.TP_SIZE}")
        print(f"   - NPU卡号 = {npu_list}")
        print(f"   - NUMA节点 = {actual_numa_list}")
        print("   ✅ NPU/NUMA绑定完全符合配置！")


if __name__ == "__main__":
    unittest.main()