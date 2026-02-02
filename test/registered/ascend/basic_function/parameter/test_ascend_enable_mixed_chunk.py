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

# 配置项：使用/tmp绝对路径保留日志，方便排查
LOG_DUMP_FILE = f"/tmp/test_mixed_chunk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
CUSTOM_SERVER_WAIT_TIME = 35  # 分块预填充初始化耗时更长，延长启动等待时间
MODEL_TRUNK_SIZE = 2048  # Llama-3.2-1B 原生trunk size
TARGET_TOKEN_COUNT = 2500  # 目标输入token数，超过原生trunk size
CHUNK_SIZE = 1024  # 分块预填充的每个chunk大小（<2048，与--chunked-prefill-size配置一致）

# 提前创建日志文件，记录参数配置
with open(LOG_DUMP_FILE, "w", encoding="utf-8") as f:
    f.write(f"=== 日志文件创建成功，时间：{datetime.now()} ===\n")
    f.write(f"=== 配置参数：--enable-mixed-chunk，--chunked-prefill-size {CHUNK_SIZE} ===\n")

def build_long_input_text_for_token():
    """
    构造足够token数的输入文本（确保#new-token超过MODEL_TRUNK_SIZE）
    每个base_sentence约10个token，重复后确保总token数达标
    """
    base_sentence = "This is a test sentence to generate enough tokens. "
    repeat_times = (TARGET_TOKEN_COUNT // 10) + 20
    return (base_sentence * repeat_times) + "The capital of France is"

class TestEnableMixedChunk(CustomTestCase):
    """Testcase：Verify the correctness of --enable-mixed-chunk feature (depend on --chunked-prefill-size).

    [Test Category] Parameter
    [Test Target] --enable-mixed-chunk & --chunked-prefill-size
    """

    @classmethod
    def setUpClass(cls):
        # 1. 保存原始IO句柄
        cls.original_stdout_fd = os.dup(sys.stdout.fileno())
        cls.original_stderr_fd = os.dup(sys.stderr.fileno())

        # 2. 打开日志文件句柄
        cls.log_fd = os.open(
            LOG_DUMP_FILE,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644
        )
        cls.log_file = open(LOG_DUMP_FILE, "a+", encoding="utf-8")

        # 3. 重定向IO到日志文件
        os.dup2(cls.log_fd, sys.stdout.fileno())
        os.dup2(cls.log_fd, sys.stderr.fileno())

        # 4. 启动服务器（核心：添加 --chunked-prefill-size {CHUNK_SIZE} 启用分块预填充）
        other_args = [
            "--enable-mixed-chunk",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--chunked-prefill-size", str(CHUNK_SIZE)  # 启用分块预填充，每个chunk最大1024个token
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        # 5. 等待服务器完全启动（分块预填充初始化+模型加载，耗时更长）
        print(f"等待服务器启动（{CUSTOM_SERVER_WAIT_TIME}秒）...")
        print(f"分块预填充配置：--chunked-prefill-size {CHUNK_SIZE}（< 模型trunk size {MODEL_TRUNK_SIZE}）")
        time.sleep(CUSTOM_SERVER_WAIT_TIME)

    @classmethod
    def tearDownClass(cls):
        # 1. 终止服务器进程
        kill_process_tree(cls.process.pid)

        # 2. 恢复IO
        os.dup2(cls.original_stdout_fd, sys.stdout.fileno())
        os.dup2(cls.original_stderr_fd, sys.stderr.fileno())

        # 3. 关闭文件句柄
        os.close(cls.log_fd)
        os.close(cls.original_stdout_fd)
        os.close(cls.original_stderr_fd)
        cls.log_file.close()

        # 4. 打印完整日志
        cls.print_full_log()

        # 5. 保留日志文件提示
        print(f"\n=== 日志文件已保留，路径：{os.path.abspath(LOG_DUMP_FILE)} ===")
        print(f"=== 查看分块/混合批次日志：cat {os.path.abspath(LOG_DUMP_FILE)} | grep -E 'Chunk|Prefill|Decode' ===")

    @classmethod
    def print_full_log(cls):
        """打印完整日志，重点展示分块预填充和mixed chunk相关内容"""
        if not os.path.exists(LOG_DUMP_FILE):
            print("\n【日志提示】日志文件不存在")
            return
        
        print("\n" + "="*80)
        print(f"完整日志（含分块预填充/{CHUNK_SIZE} & mixed chunk 内容）：")
        print("="*80)
        with open(LOG_DUMP_FILE, "r", encoding="utf-8", errors="ignore") as f:
            full_log = f.read()
            if len(full_log) <= 12000:
                print(full_log)
            else:
                print(f"【日志过长（{len(full_log)}字符），展示最后12000字符】")
                print(full_log[-12000:])
        print("="*80)
        print("日志打印完毕")

    def read_log_file(self):
        """读取日志文件内容"""
        if not os.path.exists(LOG_DUMP_FILE):
            return ""
        
        with open(LOG_DUMP_FILE, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def test_enable_mixed_chunk(self):
        # 验证1：health_generate API 可用性
        health_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(
            health_response.status_code, 200,
            f"health_generate API 失败，状态码：{health_response.status_code}"
        )

        # 验证2：超长token输入调用/generate接口
        long_input_text = build_long_input_text_for_token()
        print(f"\n构造输入字符长度：{len(long_input_text)}（目标token数：{TARGET_TOKEN_COUNT}，分块大小：{CHUNK_SIZE}）")
        
        generate_response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": long_input_text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
            timeout=70  # 分块处理耗时更长，延长请求超时
        )

        # 验证2.1：/generate 接口状态码
        self.assertEqual(
            generate_response.status_code, 200,
            f"/generate 接口失败，状态码：{generate_response.status_code}"
        )

        # 验证2.2：返回结果包含Paris
        self.assertIn(
            "Paris", generate_response.text,
            f"/generate 未返回Paris，预览：{generate_response.text[:1000]}"
        )

        # 验证3：server_info 确认参数配置正确
        server_info_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/server_info")
        self.assertEqual(server_info_response.status_code, 200)
        server_info_json = server_info_response.json()

        self.assertEqual(
            server_info_json.get("enable_mixed_chunk"), True,
            f"enable_mixed_chunk 未开启，当前值：{server_info_json.get('enable_mixed_chunk')}"
        )

        # 验证3.1：额外确认 chunked_prefill_size 配置（若接口返回该参数）
        if "chunked_prefill_size" in server_info_json:
            self.assertEqual(
                server_info_json.get("chunked_prefill_size"), CHUNK_SIZE,
                f"chunked_prefill_size 配置不匹配，当前值：{server_info_json.get('chunked_prefill_size')}"
            )
            print(f"\n✅ chunked_prefill_size 配置验证通过：{server_info_json.get('chunked_prefill_size')}")

        # 关键：等待分块/混合批次日志写入（延长至12秒）
        print("\n等待服务端输出分块/混合批次日志（12秒）...")
        time.sleep(12)

        # 恢复IO
        os.dup2(self.original_stdout_fd, sys.stdout.fileno())
        os.dup2(self.original_stderr_fd, sys.stderr.fileno())

        # 验证4：核心 - 分块预填充已启用，且mixed chunk功能生效
        server_logs = self.read_log_file()

        # 定义关键字
        chunked_prefill_keywords = [
            "chunked prefill",
            f"chunked-prefill-size {CHUNK_SIZE}",
            "Chunk [0-9]+/[0-9]+ prefill"
        ]
        mixed_chunk_keywords = [
            "Prefill + Decode batch",
            "Mixed chunk batch",
            "prefill and decode in the same batch"
        ]
        independent_batch_keywords = ["Prefill batch", "Decode batch"]

        # 判断状态
        is_chunked_activated = any(kw in server_logs for kw in chunked_prefill_keywords)
        is_mixed_activated = any(kw in server_logs for kw in mixed_chunk_keywords)
        has_independent_batch = all(kw in server_logs for kw in independent_batch_keywords)

        # 输出状态提示
        print("\n" + "-"*65)
        print("分块预填充 & Mixed Chunk 功能最终验证结果：")
        print("-"*65)
        print(f"1. 分块预填充启用状态：{'✅ 已启用' if is_chunked_activated else '❌ 未启用'}")
        print(f"2. Mixed Chunk 功能生效状态：{'✅ 已生效' if is_mixed_activated else '❌ 未生效'}")
        print(f"3. 独立批次存在状态：{'❌ 无独立批次' if not has_independent_batch else '✅ 存在独立批次'}")
        print("-"*65)

        # 核心断言（先分块，后混合）
        self.assertTrue(is_chunked_activated, f"断言失败：未启用分块预填充，无法触发Mixed Chunk！")
        self.assertTrue(is_mixed_activated, f"断言失败：分块预填充已启用，但Mixed Chunk未生效！")
        self.assertFalse(has_independent_batch, f"断言失败：Mixed Chunk已生效，但仍存在独立Prefill/Decode批次！")

        print("\n🎉 所有核心验证通过！--enable-mixed-chunk 功能完全生效，prefill和decode在同一个batch内执行！")
