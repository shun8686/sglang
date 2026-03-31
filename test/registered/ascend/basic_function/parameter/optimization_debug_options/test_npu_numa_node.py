import os
import re
import unittest
import subprocess
import time

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
    验证 TP=2 时，NPU 进程的 CPU 亲和性（taskset）是否符合 --numa-node 配置
    步骤：
    1. ps -ef | grep "sglang::scheduler_TP" 获取各 TP 进程 PID
    2. taskset -cp pid 获取每个进程的 CPU 亲和性
    3. 比较亲和性是否一致（同 NUMA 节点）
    """
    model = QWEN3_0_6B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    TP_SIZE = 2
    CONFIG_NUMA_LIST = ["1", "1"]  # 两个NPU绑同一个NUMA

    @classmethod
    def setUpClass(cls):
        cls.other_args = [
            "--trust-remote-code",
            "--tp-size", str(cls.TP_SIZE),
            "--mem-fraction-static", "0.8",
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--numa-node", *cls.CONFIG_NUMA_LIST,
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
        time.sleep(15)  # 等待所有TP进程启动

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        for f in ["./out_log.txt", "./err_log.txt"]:
            if os.path.exists(f):
                os.remove(f)

    # ==========================
    # 按照你的3步实现
    # ==========================
    def _get_tp_pids(self):
        """
        步骤1：ps -ef | grep "sglang::scheduler_TP"
        获取 TP0 / TP1 进程 PID
        """
        result = subprocess.run(
            ["ps", "-ef"],
            stdout=subprocess.PIPE,
            text=True
        )
        lines = result.stdout.splitlines()
        tp_pids = {}

        for line in lines:
            match = re.search(r"sglang::scheduler_TP(\d+)", line)
            if match:
                tp_id = match.group(1)
                pid = line.split()[1]
                tp_pids[f"TP{tp_id}"] = pid

        print(f"\n✅ 获取到 TP 进程：{tp_pids}")
        return tp_pids

    def _get_taskset_affinity(self, pid):
        """
        步骤2：taskset -cp <pid>
        获取进程的 CPU 亲和性字符串
        """
        result = subprocess.run(
            ["taskset", "-cp", pid],
            stdout=subprocess.PIPE,
            text=True
        )
        affinity = result.stdout.strip()
        print(f"📌 taskset -cp {pid} -> {affinity}")
        return affinity

    def test_numa_binding_by_taskset(self):
        """
        步骤3：比较两个 TP 进程的 CPU 亲和性
        如果 --numa-node 1 1 → 应该相同
        """
        # 步骤1
        tp_pids = self._get_tp_pids()
        self.assertEqual(len(tp_pids), self.TP_SIZE, "TP进程数量不匹配")

        # 步骤2
        affinity_tp0 = self._get_taskset_affinity(tp_pids["TP0"])
        affinity_tp1 = self._get_taskset_affinity(tp_pids["TP1"])

        # 步骤3
        print("\n🔍 对比 CPU 亲和性：")
        print(f"   TP0: {affinity_tp0}")
        print(f"   TP1: {affinity_tp1}")

        # 配置为同一个 NUMA 节点 → 亲和性必须相同
        if self.CONFIG_NUMA_LIST[0] == self.CONFIG_NUMA_LIST[1]:
            self.assertEqual(affinity_tp0, affinity_tp1,
                "❌ 两个NPU进程应绑定同一个NUMA节点，但CPU亲和性不一致！")
            print("\n✅ 两个NPU绑定同一个NUMA节点 → CPU亲和性一致，校验通过！")
        else:
            self.assertNotEqual(affinity_tp0, affinity_tp1,
                "❌ 两个NPU应绑定不同NUMA，但CPU亲和性相同！")
            print("\n✅ 两个NPU绑定不同NUMA节点 → CPU亲和性不同，校验通过！")


if __name__ == "__main__":
    unittest.main()