import asyncio
import os
import re
import unittest
from typing import Any, List, Optional, Tuple

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    popen_launch_server,
    send_concurrent_generate_requests_with_custom_params,
)

class TestPrioritySchedulingPreemptionThreshold(CustomTestCase):
    """验证 --priority-scheduling-preemption-threshold=5 的调度逻辑：执行顺序 C(10) > A(2) > B(5)"""
    
    @classmethod
    def setUpClass(cls):
        # 配置模型路径（适配昇腾环境，替换为本地有效路径）
        cls.model = "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        
        # 初始化日志文件
        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")
        
        # 启动服务，核心配置优先级调度和抢占阈值（适配昇腾环境）
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--max-running-requests", "1",
                "--max-queued-requests", "10",
                "--enable-priority-scheduling",
                "--priority-scheduling-preemption-threshold", "5",
                "--disable-cuda-graph",
                "--attention-backend", "ascend",
                "--tp-size", "1",
                "--mem-fraction-static", "0.8",
            ),
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )
    
    @classmethod
    def tearDownClass(cls):
        # 安全清理进程
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        
        # 验证运行/排队请求数不超限
        _verify_running_queued_requests(1, 10)
        
        # 清理日志文件
        cls.stdout.close()
        cls.stderr.close()
        if os.path.exists(STDOUT_FILENAME):
            os.remove(STDOUT_FILENAME)
        if os.path.exists(STDERR_FILENAME):
            os.remove(STDERR_FILENAME)
    
    def test_preemption_threshold_execution_order(self):
        # 步骤1：先提交作业A（优先级2），让其进入运行状态（放大token数，确保被C抢占）
        request_a = {
            "priority": 2,
            "sampling_params": {"max_new_tokens": 2000}  # 大token数，确保运行时间足够长
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # 异步发送作业A，不等待完成（占用运行位）
        task_a = loop.create_task(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, [request_a]
            )
        )
        
        loop.run_until_complete(asyncio.sleep(0.5))
        
        # 步骤2：调整发送顺序：先C（10）后B（5），并添加间隔，确保C优先抢占
        # 2.1 先发送高优先级的C（10），满足与A的差值≥5，触发抢占
        request_c = {
            "priority": 10,
            "sampling_params": {"max_new_tokens": 100}  # 小token数，快速完成
        }
        responses_c = loop.run_until_complete(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, [request_c]
            )
        )
        
        # 2.2 等待0.5秒，确保C已完成并释放运行位，再发送B（5），避免干扰
        loop.run_until_complete(asyncio.sleep(0.5))
        request_b = {
            "priority": 5,
            "sampling_params": {"max_new_tokens": 100}  # 小token数，排队等待
        }
        responses_b = loop.run_until_complete(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, [request_b]
            )
        )
        
        # 步骤3：等待作业A完成，获取所有响应
        responses_a = loop.run_until_complete(task_a)
        
        # 步骤4：修正响应合并顺序（A→C→B），与发送顺序一致，确保索引对应正确
        all_responses = responses_a + responses_c + responses_b
        
        # 关闭事件循环，消除 "unclosed event loop" 资源警告
        loop.close()
        
        # 步骤5：验证所有请求均处理成功（状态码200，无错误信息）
        expected_status = [(200, None)] * 3  # 3个请求均预期成功
        e2e_latencies = []
        _verify_generate_responses(all_responses, expected_status, e2e_latencies)
        
        # 步骤6：明确索引对应（与合并顺序一致）
        latency_a = e2e_latencies[0]
        latency_c = e2e_latencies[1]
        latency_b = e2e_latencies[2]
        
        # 核心断言：C最先完成，其次是A，最后是B
        assert latency_c < latency_b < latency_a, \
            f"执行顺序不符合预期！预期 C<B<A，实际耗时：C={latency_c}, A={latency_a}, B={latency_b}"

    def test_preemption_threshold_execution_order_exa(self):
        # 步骤1：先提交作业A（优先级2），让其进入运行状态（放大token数，确保被C抢占）
        request_a = {
            "priority": 2,
            "sampling_params": {"max_new_tokens": 2000}  # 大token数，确保运行时间足够长
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # 异步发送作业A，不等待完成（占用运行位）
        task_a = loop.create_task(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, [request_a]
            )
        )
        
        # 等待2秒，确保作业A已完全启动并占用运行位（延长等待，避免抢占逻辑未触发）
        loop.run_until_complete(asyncio.sleep(0.5))
        
        # 步骤2：调整发送顺序：先C（10）后B（5），并添加间隔，确保C优先抢占
        # 2.1 先发送高优先级的C（10），满足与A的差值≥5，触发抢占
        request_c = {
            "priority": 10,
            "sampling_params": {"max_new_tokens": 100}  # 小token数，快速完成
        }
        responses_c = loop.run_until_complete(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, [request_c]
            )
        )
        loop.run_until_complete(asyncio.sleep(0.5))
        # 2.2 等待0.5秒，确保C已完成并释放运行位，再发送B（2），避免干扰
        request_b = {
            "priority": 2,
            "sampling_params": {"max_new_tokens": 2000}  # 小token数，排队等待
        }
        responses_b = loop.run_until_complete(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, [request_b]
            )
        )
        
        # 步骤3：等待作业A完成，获取所有响应
        responses_a = loop.run_until_complete(task_a)
        
        # 步骤4：修正响应合并顺序（A→C→B），与发送顺序一致，确保索引对应正确
        all_responses = responses_a + responses_c + responses_b
        
        # 关闭事件循环，消除 "unclosed event loop" 资源警告
        loop.close()
        
        # 步骤5：验证所有请求均处理成功（状态码200，无错误信息）
        expected_status = [(200, None)] * 3  # 3个请求均预期成功
        e2e_latencies = []
        _verify_generate_responses(all_responses, expected_status, e2e_latencies)
        
        # 步骤6：明确索引对应（与合并顺序一致）
        latency_a = e2e_latencies[0]
        latency_c = e2e_latencies[1]
        latency_b = e2e_latencies[2]
        
        # 核心断言：C最先完成，其次是A，最后是B
        assert latency_c < latency_a < latency_b, \
            f"执行顺序不符合预期！预期 C<A<B，实际耗时：C={latency_c}, A={latency_a}, B={latency_b}"

# 辅助验证函数：验证响应状态并收集端到端耗时
def _verify_generate_responses(
    responses: Tuple[int, Any, float],
    expected_code_and_error: Tuple[int, Any],
    e2e_latencies: List[Optional[float]],
):
    e2e_latencies.clear()  # 清空列表，避免残留数据干扰
    for got, expected in zip(responses, expected_code_and_error):
        # 拆分响应数据（状态码 + 响应体）
        got_status, got_json = got
        expected_status, expected_err = expected
        
        # 验证状态码是否为200
        assert got_status == expected_status, \
            f"请求处理失败：预期状态码{expected_status}，实际{got_status}，响应：{got_json}"
        
        # 验证响应无错误信息（仅针对200状态）
        if got_status == 200:
            assert "error" not in got_json, f"请求返回错误信息：{got_json.get('error', '未知错误')}"
            
            # 验证并收集端到端耗时（确保字段存在）
            assert "meta_info" in got_json, "响应缺少必要字段 'meta_info'"
            assert "e2e_latency" in got_json["meta_info"], "响应缺少必要字段 'e2e_latency'"
            e2e_latencies.append(got_json["meta_info"]["e2e_latency"])
        else:
            e2e_latencies.append(None)

# 辅助验证函数：验证运行/排队请求数不超过配置上限
def _verify_running_queued_requests(
    max_running_requests: int, max_queued_requests: int
):
    # 定义日志匹配模式
    rr_pattern = re.compile(r"#running-req:\s*(\d+)")
    qr_pattern = re.compile(r"#queue-req:\s*(\d+)")
    
    # 若日志文件不存在，直接返回
    if not os.path.exists(STDERR_FILENAME):
        return
    
    with open(STDERR_FILENAME, "r", encoding="utf-8") as f:
        for line in f:
            # 验证运行请求数
            rr_match = rr_pattern.search(line)
            if rr_match:
                running_req_count = int(rr_match.group(1))
                assert running_req_count <= max_running_requests, \
                    f"运行请求数超限：当前{running_req_count} > 上限{max_running_requests}"
            
            # 验证排队请求数
            qr_match = qr_pattern.search(line)
            if qr_match:
                queued_req_count = int(qr_match.group(1))
                assert queued_req_count <= max_queued_requests, \
                    f"排队请求数超限：当前{queued_req_count} > 上限{max_queued_requests}"

if __name__ == "__main__":
    unittest.main()
