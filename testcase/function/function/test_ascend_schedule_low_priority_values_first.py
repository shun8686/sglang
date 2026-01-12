import asyncio
import os
import re
import unittest
from typing import Any, List, Optional, Tuple

# 复用原测试的依赖和工具函数
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    popen_launch_server,
    send_concurrent_generate_requests_with_custom_params,
)


class TestLowPriorityFirstScheduling(CustomTestCase):
    """测试启用 --schedule-low-priority-values-first 参数后的调度逻辑：优先处理数值更低的优先级请求"""

    @classmethod
    def setUpClass(cls):
        cls.model = "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--max-running-requests", "1",
                "--max-queued-requests", "3", 
                "--enable-priority-scheduling",
                "--schedule-low-priority-values-first",
                "--disable-cuda-graph",
                "--attention-backend",
                "ascend",
            ),
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        # 清理进程和日志
        kill_process_tree(cls.process.pid)
        _verify_max_running_requests_and_max_queued_request_validation(1, 3)
        cls.stdout.close()
        cls.stderr.close()
        if os.path.exists(STDOUT_FILENAME):
            os.remove(STDOUT_FILENAME)
        if os.path.exists(STDERR_FILENAME):
            os.remove(STDERR_FILENAME)

    def test_low_priority_value_first_ordering(self):
        # 发送一组不同优先级的请求（数值从0到4递增）
        responses = asyncio.run(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url,
                [
                    # 优先级0（最低数值）- 预期第一个完成
                    {"priority": 0, "sampling_params": {"max_new_tokens": 100}},
                    # 优先级4（最高数值）- 预期最后一个完成
                    {"priority": 4, "sampling_params": {"max_new_tokens": 100}},
                    # 优先级2 - 预期第三个完成
                    {"priority": 2, "sampling_params": {"max_new_tokens": 100}},
                    # 优先级1 - 预期第二个完成
                    {"priority": 1, "sampling_params": {"max_new_tokens": 100}},
                ],
            )
        )

        # 验证所有请求都成功返回
        expected_status_and_error_messages = [
            (200, None), (200, None), (200, None), (200, None)
        ]
        e2e_latencies = []
        _verify_genereate_responses(
            responses, expected_status_and_error_messages, e2e_latencies
        )

        # 优先级0 < 优先级1 < 优先级2 < 优先级3 < 优先级4
        assert e2e_latencies[0] < e2e_latencies[3] < e2e_latencies[2] < e2e_latencies[1]

    def test_low_priority_first_abortion_logic(self):
        responses = asyncio.run(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url,
                [
                    # 优先级0（低数值）- 先运行，占用唯一并发位
                    {"priority": 10, "sampling_params": {"max_new_tokens": 10000}},
                    {"priority": 9, "sampling_params": {"max_new_tokens": 100}},
                    {"priority": 8, "sampling_params": {"max_new_tokens": 100}},
                    {"priority": 7, "sampling_params": {"max_new_tokens": 100}},
                    {"priority": 6, "sampling_params": {"max_new_tokens": 100}},
                    {"priority": 5, "sampling_params": {"max_new_tokens": 100}},
                    {"priority": 0, "sampling_params": {"max_new_tokens": 100}},
                ],
            )
        )

        expected_status_and_error_messages = [
            (200, None),
            (503, "The request is aborted by a higher priority request."),
            (503, "The request is aborted by a higher priority request."),
            (503, "The request is aborted by a higher priority request."),
            (200, None),
            (200, None),
            (200, None),
        ]
        e2e_latencies = []
        _verify_genereate_responses(
            responses, expected_status_and_error_messages, e2e_latencies
        )



# 复用原测试的辅助函数（确保代码独立可运行）
def _verify_genereate_responses(
    responses: Tuple[int, Any, float],
    expected_code_and_error_message: Tuple[int, Any],
    e2e_latencies: List[Optional[float]],
):
    """验证响应状态、错误信息，并收集端到端耗时"""
    for got, expected in zip(responses, expected_code_and_error_message):
        got_status, got_json = got
        expected_status, expected_err_msg = expected

        # 验证状态码
        assert got_status == expected_status, \
            f"状态码不符：预期 {expected_status}，实际 {got_status}"

        # 验证错误信息（非200状态）
        if got_status != 200:
            assert got_json.get("object") == "error", "非200状态但响应对象不是error"
            assert got_json.get("message") == expected_err_msg, \
                f"错误信息不符：预期 {expected_err_msg}，实际 {got_json.get('message')}"
        else:
            assert "object" not in got_json, "200状态不应包含object字段"
            assert "message" not in got_json, "200状态不应包含message字段"

        # 收集耗时（仅成功请求）
        if got_status == 200:
            e2e_latencies.append(got_json["meta_info"]["e2e_latency"])
        else:
            e2e_latencies.append(None)


def _verify_max_running_requests_and_max_queued_request_validation(
    max_running_requests: int, max_queued_requests: int
):
    rr_pattern = re.compile(r"#running-req:\s*(\d+)")
    qr_pattern = re.compile(r"#queue-req:\s*(\d+)")

    if not os.path.exists(STDERR_FILENAME):
        return

    with open(STDERR_FILENAME, "r") as f:
        for line in f:
            rr_match = rr_pattern.search(line)
            if rr_match:
                assert int(rr_match.group(1)) <= max_running_requests, \
                    f"运行请求数超过上限：{rr_match.group(1)} > {max_running_requests}"
            qr_match = qr_pattern.search(line)
            if qr_match:
                assert int(qr_match.group(1)) <= max_queued_requests, \
                    f"排队请求数超过上限：{qr_match.group(1)} > {max_queued_requests}"


if __name__ == "__main__":
    unittest.main()
