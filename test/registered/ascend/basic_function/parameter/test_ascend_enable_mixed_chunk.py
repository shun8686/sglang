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

# 配置项
LOG_DUMP_FILE = f"test_mixed_chunk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
CUSTOM_SERVER_WAIT_TIME = 30  # 超长token输入，服务器启动时间适当延长
MODEL_TRUNK_SIZE = 2048  # Llama-3.2-1B 原生trunk size
TARGET_TOKEN_COUNT = 2500  # 目标输入token数，超过原生trunk size触发mixed chunk

def build_long_input_text_for_token():
    """
    构造足够token数的输入文本（确保#new-token超过MODEL_TRUNK_SIZE）
    每个base_sentence约10个token，重复后确保总token数达标
    """
    # 基础短句（约10个token，避免无意义字符，保证token统计准确）
    base_sentence = "This is a test sentence to generate enough tokens. "
    # 计算重复次数，确保总token数超过TARGET_TOKEN_COUNT
    repeat_times = (TARGET_TOKEN_COUNT // 10) + 20  # 每个短句约10个token，额外加20次兜底
    # 拼接超长文本，末尾保留查询句（确保最终返回Paris，兼容原有断言）
    long_input_text = (base_sentence * repeat_times) + "The capital of France is"
    return long_input_text

class TestEnableMixedChunk(CustomTestCase):
    """Testcase：Verify the correctness of --enable-mixed-chunk feature and related APIs (health/generate/server-info) availability.

    [Test Category] Parameter
    [Test Target] --enable-mixed-chunk
    """

    @classmethod
    def setUpClass(cls):
        # 1. 保存操作系统层面的原始stdout/stderr文件句柄（捕获子进程日志核心）
        cls.original_stdout_fd = os.dup(sys.stdout.fileno())
        cls.original_stderr_fd = os.dup(sys.stderr.fileno())

        # 2. 打开日志文件（操作系统层面句柄，支持重定向）
        cls.log_fd = os.open(
            LOG_DUMP_FILE,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644  # 文件权限：可读可写，避免权限不足
        )
        cls.log_file = open(LOG_DUMP_FILE, "a+", encoding="utf-8")  # 用于后续读取和关闭

        # 3. 操作系统层面重定向stdout/stderr到日志文件（子进程会继承该句柄）
        os.dup2(cls.log_fd, sys.stdout.fileno())
        os.dup2(cls.log_fd, sys.stderr.fileno())

        # 4. 启动服务器（开启enable-mixed-chunk，保留NPU相关参数）
        other_args = [
            "--enable-mixed-chunk",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--max-seq-length", "3072"  # 适配超长token输入，避免序列长度超限
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        # 5. 等待服务器完全启动（超长token处理需更多初始化时间）
        print(f"等待服务器启动（{CUSTOM_SERVER_WAIT_TIME}秒）...")
        time.sleep(CUSTOM_SERVER_WAIT_TIME)

    @classmethod
    def tearDownClass(cls):
        # 1. 终止服务器进程树，释放NPU资源
        kill_process_tree(cls.process.pid)

        # 2. 恢复操作系统层面的stdout/stderr（确保后续打印输出到控制台）
        os.dup2(cls.original_stdout_fd, sys.stdout.fileno())
        os.dup2(cls.original_stderr_fd, sys.stderr.fileno())

        # 3. 关闭所有文件句柄和文件对象（释放文件占用，避免删除失败）
        os.close(cls.log_fd)
        os.close(cls.original_stdout_fd)
        os.close(cls.original_stderr_fd)
        cls.log_file.close()

        # 4. 打印完整日志到控制台（方便排查问题）
        cls.print_full_log()

        # 5. 删除日志文件（清理冗余，避免文件堆积）
        cls.delete_log_file()

    @classmethod
    def print_full_log(cls):
        """打印完整服务端日志，方便查看mixed chunk相关内容"""
        if not os.path.exists(LOG_DUMP_FILE):
            print("\n【日志提示】日志文件不存在，无内容可打印")
            return
        
        print("\n" + "="*80)
        print("完整服务端日志（验证prefill和decode同batch）：")
        print("="*80)
        with open(LOG_DUMP_FILE, "r", encoding="utf-8", errors="ignore") as f:
            full_log = f.read()
            # 日志过长时仅打印最后8000字符，避免控制台刷屏，同时保留关键内容
            if len(full_log) <= 8000:
                print(full_log)
            else:
                print(f"【日志过长（总长度{len(full_log)}），仅展示最后8000字符】")
                print(full_log[-8000:])
        print("="*80)
        print("日志打印完毕")

    @classmethod
    def delete_log_file(cls):
        """删除已生成的日志文件，清理冗余文件"""
        try:
            if os.path.exists(LOG_DUMP_FILE):
                os.remove(LOG_DUMP_FILE)
                print(f"\n日志文件已成功删除：{os.path.abspath(LOG_DUMP_FILE)}")
            else:
                print("\n【删除提示】日志文件不存在，无需执行删除操作")
        except Exception as e:
            print(f"\n【删除警告】日志文件删除失败，可能被其他进程占用：{e}")

    def read_log_file(self):
        """读取日志文件完整内容，返回字符串格式，用于断言判断"""
        if not os.path.exists(LOG_DUMP_FILE):
            return ""
        
        with open(LOG_DUMP_FILE, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def test_enable_mixed_chunk(self):
        # 验证1：检查health_generate API 可用性（服务是否正常启动）
        health_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(
            health_response.status_code, 200,
            f"health_generate API 请求失败，响应状态码：{health_response.status_code}"
        )

        # 验证2：构造超长token输入，调用/generate接口，触发mixed chunk功能
        long_input_text = build_long_input_text_for_token()
        print(f"\n构造的输入文本字符长度：{len(long_input_text)}（目标token数：{TARGET_TOKEN_COUNT}，超过模型原生trunk size {MODEL_TRUNK_SIZE}）")
        
        generate_response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": long_input_text,
                "sampling_params": {
                    "temperature": 0,  # 温度设为0，确保输出结果稳定
                    "max_new_tokens": 32,  # 生成32个新token，确保返回Paris
                },
            },
            timeout=60  # 超长token输入处理耗时较长，延长请求超时时间
        )

        # 验证2.1：/generate 接口响应状态码是否为200
        self.assertEqual(
            generate_response.status_code, 200,
            f"/generate 接口请求失败，响应状态码：{generate_response.status_code}，响应内容：{generate_response.text[:500]}"
        )

        # 验证2.2：/generate 接口返回结果是否包含预期值Paris
        self.assertIn(
            "Paris", generate_response.text,
            f"/generate 接口返回结果不包含预期值'Paris'，响应内容预览：{generate_response.text[:1000]}"
        )

        # 验证3：检查server_info API，确认enable_mixed_chunk 参数是否正确开启（替换废弃的/get_server_info）
        server_info_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/server_info")
        self.assertEqual(
            server_info_response.status_code, 200,
            f"server_info API 请求失败，响应状态码：{server_info_response.status_code}（注意：/get_server_info 已废弃，需使用/server_info）"
        )

        server_info_json = server_info_response.json()
        self.assertEqual(
            server_info_json.get("enable_mixed_chunk"), True,
            f"enable_mixed_chunk 参数未正确开启，当前配置值：{server_info_json.get('enable_mixed_chunk')}"
        )

        # 关键：等待日志写入完成（超长token处理后，服务端输出日志有延迟，延长至10秒）
        print("\n等待服务端输出mixed chunk相关日志（10秒）...")
        time.sleep(10)

        # 恢复操作系统层面的stdout/stderr（确保后续断言信息输出到控制台）
        os.dup2(self.original_stdout_fd, sys.stdout.fileno())
        os.dup2(self.original_stderr_fd, sys.stderr.fileno())

        # 验证4：核心断言 - prefill和decode在同一个batch内执行（mixed chunk功能生效）
        server_logs = self.read_log_file()

        # 定义mixed chunk合并批次的目标关键字（适配sglang不同版本的日志格式）
        mixed_chunk_target_keywords = [
            "Prefill + Decode batch",
            "Mixed chunk batch",
            "prefill and decode in the same batch",
            "mixed chunk: prefill & decode in one batch"
        ]

        # 判断是否存在任意一个目标关键字，确认mixed chunk功能生效
        is_mixed_chunk_activated = any(keyword in server_logs for keyword in mixed_chunk_target_keywords)

        # 备用判断：若日志无明确合并标识，判断是否不存在独立的Prefill/Decode batch
        has_independent_prefill = "Prefill batch" in server_logs
        has_independent_decode = "Decode batch" in server_logs
        is_separate_batch = has_independent_prefill and has_independent_decode

        # 执行核心断言（优先使用合并关键字判断，备用非独立批次判断）
        self.assertTrue(
            is_mixed_chunk_activated,
            f"未在服务端日志中找到mixed chunk合并批次标识，prefill和decode为独立批次！\n"
            f"目标关键字列表：{mixed_chunk_target_keywords}\n"
            f"日志内容预览（最后3000字符）：\n{server_logs[-3000:] if len(server_logs) > 3000 else '日志内容为空'}"
        )

        # 可选：启用备用断言（注释掉上方断言，启用下方断言，适配无明确合并标识的日志）
        # self.assertFalse(
        #     is_separate_batch,
        #     f"prefill和decode为独立批次，未触发mixed chunk功能！\n"
        #     f"日志内容预览（最后3000字符）：\n{server_logs[-3000:] if len(server_logs) > 3000 else '日志内容为空'}"
        # )

        print("\n✅ 所有验证通过！--enable-mixed-chunk 功能生效，prefill和decode在同一个batch内执行。")

if __name__ == "__main__":
    unittest.main()
