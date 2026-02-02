import unittest
import requests
import os
import sys
import time
from datetime import datetime

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

# 日志文件路径（用于捕获服务端日志，验证prefill和decode同batch）
LOG_DUMP_FILE = f"test_mixed_chunk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
# 自定义服务器启动等待时间（超长上下文加载需稍久）
CUSTOM_SERVER_WAIT_TIME = 25
# 模型原生trunk size（Llama-3.2-1B通常为2048，此处设置为2048，输入长度超过该值）
MODEL_TRUNK_SIZE = 2048
# 构造超长输入文本（长度略大于MODEL_TRUNK_SIZE，确保触发mixed chunk功能）
def build_long_input_text():
    """构造超过模型原生trunk size的输入文本，触发enable-mixed-chunk功能"""
    # 基础重复文本（避免无意义字符，同时保证上下文长度达标）
    base_text = "This is a test sentence to extend the input context length. "
    # 计算需要重复的次数（使最终长度超过MODEL_TRUNK_SIZE）
    repeat_times = (MODEL_TRUNK_SIZE // len(base_text)) + 20
    # 拼接超长文本，末尾加上原查询句（确保最终返回Paris）
    long_text = (base_text * repeat_times) + "The capital of France is"
    return long_text

class TestEnableMixedChunk(CustomTestCase):
    """Testcase：Verify the correctness of --enable-mixed-chunk feature and related APIs (health/generate/server-info) availability.

    [Test Category] Parameter
    [Test Target] --enable-mixed-chunk
    """

    @classmethod
    def setUpClass(cls):
        # 1. 保存操作系统层面的原始stdout/stderr文件句柄（用于捕获服务端日志）
        cls.original_stdout_fd = os.dup(sys.stdout.fileno())
        cls.original_stderr_fd = os.dup(sys.stderr.fileno())

        # 2. 打开日志文件（操作系统层面句柄，用于重定向）
        cls.log_fd = os.open(
            LOG_DUMP_FILE,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644
        )
        cls.log_file = open(LOG_DUMP_FILE, "a+", encoding="utf-8")

        # 3. 操作系统层面重定向stdout/stderr到日志文件
        os.dup2(cls.log_fd, sys.stdout.fileno())
        os.dup2(cls.log_fd, sys.stderr.fileno())

        # 4. 启动服务器（保留原有参数，开启enable-mixed-chunk）
        other_args = [
            "--enable-mixed-chunk",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        # 5. 等待服务器完全启动（超长上下文加载需稍久）
        print(f"等待服务器启动（{CUSTOM_SERVER_WAIT_TIME}秒）...")
        time.sleep(CUSTOM_SERVER_WAIT_TIME)

    @classmethod
    def tearDownClass(cls):
        # 1. 终止服务器进程树
        kill_process_tree(cls.process.pid)

        # 2. 恢复操作系统层面的stdout/stderr
        os.dup2(cls.original_stdout_fd, sys.stdout.fileno())
        os.dup2(cls.original_stderr_fd, sys.stderr.fileno())

        # 3. 关闭所有文件句柄和文件对象
        os.close(cls.log_fd)
        os.close(cls.original_stdout_fd)
        os.close(cls.original_stderr_fd)
        cls.log_file.close()

        # 4. 打印完整日志（方便查看）
        cls.print_full_log()

        # 5. 删除日志文件（清理冗余）
        cls.delete_log_file()

    @classmethod
    def print_full_log(cls):
        """打印完整日志到控制台"""
        if not os.path.exists(LOG_DUMP_FILE):
            print("\n【日志提示】日志文件不存在，无内容可打印")
            return
        
        print("\n" + "="*80)
        print("完整服务端日志（验证prefill和decode同batch）：")
        print("="*80)
        with open(LOG_DUMP_FILE, "r", encoding="utf-8", errors="ignore") as f:
            full_log = f.read()
            # 日志过长时仅打印最后6000字符，避免刷屏
            print(full_log if len(full_log) <= 6000 else f"【日志过长，展示最后6000字符】\n{full_log[-6000:]}")
        print("="*80)
        print("日志打印完毕")

    @classmethod
    def delete_log_file(cls):
        """删除日志文件，清理冗余"""
        try:
            if os.path.exists(LOG_DUMP_FILE):
                os.remove(LOG_DUMP_FILE)
                print(f"\n日志文件已删除：{os.path.abspath(LOG_DUMP_FILE)}")
            else:
                print("\n【删除提示】日志文件不存在，无需删除")
        except Exception as e:
            print(f"\n【删除警告】日志文件删除失败：{e}")

    def read_log_file(self):
        """读取日志文件内容，返回完整字符串"""
        if not os.path.exists(LOG_DUMP_FILE):
            return ""
        
        with open(LOG_DUMP_FILE, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def test_enable_mixed_chunk(self):
        # 验证1：相关API（health_generate）可用性
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200, "health_generate API 请求失败")

        # 验证2：构造超长输入上下文（超过模型原生trunk size），调用/generate接口
        long_input_text = build_long_input_text()
        print(f"构造的输入上下文长度：{len(long_input_text)}（超过模型trunk size {MODEL_TRUNK_SIZE}）")
        
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": long_input_text,  # 使用超长输入文本
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200, "/generate 接口请求失败（超长上下文）")
        self.assertIn("Paris", response.text, "/generate 接口返回结果不包含预期值Paris")

        # 验证3：验证enable_mixed_chunk参数是否正确配置
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        self.assertEqual(response.status_code, 200, "get_server_info API 请求失败")
        self.assertEqual(response.json()["enable_mixed_chunk"], True, "enable_mixed_chunk 参数未正确开启")

        # 等待日志写入（确保prefill/decode相关日志完整输出）
        time.sleep(6)

        # 恢复IO（方便后续断言信息输出到控制台）
        os.dup2(self.original_stdout_fd, sys.stdout.fileno())
        os.dup2(self.original_stderr_fd, sys.stderr.fileno())

        # 验证4：核心断言 - prefill和decode在同一个batch内执行
        server_logs = self.read_log_file()
        # 定义目标关键字（根据sglang服务端日志格式调整，常见标识如下，可按需切换）
        target_keywords = [
            "prefill and decode in the same batch",
            "mixed chunk: prefill + decode batch",
            "prefill/decode executed in one batch"
        ]
        
        # 检查是否存在任意一个目标关键字（提高兼容性）
        batch_assert_passed = any(keyword in server_logs for keyword in target_keywords)
        self.assertTrue(
            batch_assert_passed,
            f"未在日志中找到prefill和decode同batch的标识！\n目标关键字：{target_keywords}\n日志预览（最后2000字符）：\n{server_logs[-2000:] if len(server_logs) > 2000 else server_logs}"
        )
        print("✅ 断言通过：prefill和decode在同一个batch内执行")

if __name__ == "__main__":
    unittest.main()
