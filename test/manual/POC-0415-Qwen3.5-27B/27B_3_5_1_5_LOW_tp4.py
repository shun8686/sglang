"""
Performance benchmark for Qwen3.5-27B-W8A8 with NEXTN speculative decoding (MTP)
on Ascend NPU (TP=4, 4 NPUs).

Launch:   python test_qwen3_5_27B_mtp.py
CI suite: nightly-4-npu-a3
"""

import os
import re
import time
import subprocess
import sys
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    CustomTestCase,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "/home/weights/Qwen3.5-27B-W8A8/"
HOST = "127.0.0.1"
PORT = 6688
SERVER_URL = f"http://{HOST}:{PORT}"

# bench_serving parameters (mirrors the manual benchmark command)
BENCH_DATASET_NAME = "random"
BENCH_MAX_CONCURRENCY = 38
BENCH_NUM_PROMPTS = BENCH_MAX_CONCURRENCY * 4
BENCH_RANDOM_INPUT_LEN = 3500
BENCH_RANDOM_OUTPUT_LEN = 1500
BENCH_RANDOM_RANGE_RATIO = 1.0
BENCH_DATASET_PATH = "/home/hexq/ShareGPT_V3_unfiltered_cleaned_split.json"

# Performance thresholds — tune to ~80-90% of your baseline after first run
MIN_OUTPUT_THROUGHPUT_TOKENS_PER_SEC = 0  # tok/s
MIN_REQUEST_THROUGHPUT = 0              # req/s

# 最优配置保存 目标： 铆钉TPOT -> 寻找最大吞吐
# 配置层面 -> 算子层面 -> 调度层面
# 64K的时候


def _set_npu_env():
    """Apply all Ascend NPU / HCCL / SGLang env-vars required by this model."""
    overrides = {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "ASCEND_RT_VISIBLE_DEVICES":"0,1,2,3",
        "ASCEND_LANUCH_BLOCKING":"1",
        "STREAMS_PER_DEVICE": "32",
        "HCCL_BUFFSIZE": "3000",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_SOCKET_IFNAME": "lo",
        "GLOO_SOCKET_IFNAME": "lo",
        "SGLANG_NPU_PROFILING": "0",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
        "SGLANG_ENABLE_SPEC_V2":"1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM":"0",
    }
    # Proxy vars must be removed entirely
    for key in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"):
        os.environ.pop(key, None)
    for key, val in overrides.items():
        os.environ[key] = val


def _parse_metric(text: str, label: str):
    """Return first float from a stdout line containing `label`, or None."""
    for line in text.splitlines():
        if label.lower() in line.lower():
            m = re.search(r"\d+\.\d+", line)
            if m:
                return float(m.group())
    return None


class TestQwen35_27B_MTP_Perf(CustomTestCase):
    """End-to-end performance test: Qwen3.5-27B + NEXTN MTP on 4xAscend NPU."""

    @classmethod
    def setUpClass(cls):
        _set_npu_env()
        cls.base_url = SERVER_URL
        cls.process = popen_launch_server(
            MODEL_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend", "ascend",
                "--device", "npu",
                "--tp-size", "4",
                "--nnodes", "1",
                "--node-rank", "0",
                "--chunked-prefill-size", "-1",
                "--max-prefill-tokens", "186000",
                "--enable-prefill-delayer",
                "--prefill-delayer-max-delay-passes", "200",
                "--disable-radix-cache",
                "--mem-fraction-static", "0.94",
                "--max-total-tokens", "700000",
                "--max-running-requests", BENCH_MAX_CONCURRENCY,
                "--max-mamba-cache-size", "200",
                "--quantization", "modelslim",
                "--dtype", "bfloat16",
                "--mamba-ssm-dtype", "bfloat16",
                "--enable-multimodal",
                "--mm-attention-backend", "ascend_attn",
                "--cuda-graph-bs",
                "1","2","4","8","12","18","24","32","34","36","38",
                "--host", HOST,
                "--port", str(PORT),
                "--trust-remote-code",
                "--speculative-algorithm", "NEXTN",
                "--speculative-num-steps", "3",
                "--speculative-eagle-topk", "1",
                "--speculative-num-draft-tokens", "4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_server_health(self):
        """Sanity-check that the server responds to /health."""
        import requests
        resp = requests.get(self.base_url + "/health", timeout=10)
        # time.sleep(20000)
        self.assertEqual(resp.status_code, 200)
    
    # def test_a_gsm8k_accuracy(self):
    #     """GSM8K accuracy check via evalscope."""
    #     cmd = [
    #         "evalscope", "eval",
    #         "--model", MODEL_PATH,
    #         "--api-url", f"http://{HOST}:{PORT}/v1",
    #         "--api-key", "EMPTY",
    #         "--eval-type", "openai_api",
    #         "--generation-config", (
    #             '{"do_sample":true,"max_tokens":1024,"seed":3407,'
    #             '"top_p":0.8,"top_k":20,"temperature":0.7,"n":1,'
    #             '"presence_penalty":1.5,"repetition_penalty":1.0,'
    #             '"timeout":3600,"stream":true,'
    #             '"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
    #         ),
    #         "--datasets", "gsm8k",
    #         "--dataset-dir", "/home/hexq/dataset/gsm8k",
    #         "--eval-batch-size", "128",
    #         "--ignore-errors",
    #         "--limit", "200",
    #     ]

    #     proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    #     combined = proc.stdout + proc.stderr
    #     print("=" * 55)
    #     print("  GSM8K Accuracy (evalscope)")
    #     print("=" * 55)
    #     print(combined)

    #     self.assertEqual(proc.returncode, 0, f"evalscope failed:\n{combined[-1000:]}")

    #     # evalscope prints something like: "accuracy: 0.850" or "AverageAccuracy: 0.850"
    #     accuracy = _parse_metric(combined, "accuracy")
    #     self.assertIsNotNone(accuracy, "Could not parse accuracy from evalscope output")
    #     print(f"  GSM8K Accuracy: {accuracy:.3f}")

    #     MIN_GSM8K_ACCURACY = 0.80
    #     self.assertGreater(
    #         accuracy, MIN_GSM8K_ACCURACY,
    #         f"精度 {accuracy:.3f} < 阈值 {MIN_GSM8K_ACCURACY}"
    #     )

    def test_serving_throughput_random(self):
        """
        Run sglang.bench_serving and assert throughput thresholds.

        Mirrors the manual command:
          python -m sglang.bench_serving --dataset-name random --backend sglang
            --host 127.0.0.1 --port 6688 --max-concurrency 256
            --random-input-len 3500 --random-output-len 1500
            --num-prompts 256 --random-range-ratio 1
            --dataset-path /home/hexq/ShareGPT_V3_unfiltered_cleaned_split.json
        """
        cmd = [
            sys.executable, "-m", "sglang.bench_serving",
            "--backend", "sglang",
            "--host", HOST,
            "--port", str(PORT),
            "--dataset-name", BENCH_DATASET_NAME,
            "--dataset-path", BENCH_DATASET_PATH,
            "--max-concurrency", str(BENCH_MAX_CONCURRENCY),
            "--num-prompts", str(BENCH_NUM_PROMPTS),
            "--random-input-len", str(BENCH_RANDOM_INPUT_LEN),
            "--random-output-len", str(BENCH_RANDOM_OUTPUT_LEN),
            "--random-range-ratio", str(BENCH_RANDOM_RANGE_RATIO),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        combined = proc.stdout + proc.stderr
        print("=================================================")
        print(combined)  # tail for CI log visibility
        self.assertEqual(
            proc.returncode, 0,
            f"bench_serving failed (rc={proc.returncode})\n{combined[-1000:]}"
        )

        # bench_serving stdout labels (match exactly):
        #   Mean TTFT (ms):                     123.45
        #   Median TPOT (ms):                    45.67   (or "Mean TPOT")
        #   Mean E2E Latency (s):                12.34
        #   Output token throughput (tok/s):   1234.56
        ttft_ms   = _parse_metric(combined, "Mean TTFT")
        tpot_ms   = _parse_metric(combined, "Mean TPOT") or _parse_metric(combined, "Median TPOT")
        e2e_s     = _parse_metric(combined, "Mean E2E Latency")
        output_tps = _parse_metric(combined, "Output token throughput")
        req_tps   = _parse_metric(combined, "Request throughput")
        acceptance_rate = _parse_metric(combined, "acceptance rate")

        per_card_tps = (output_tps / 2) if output_tps is not None else None  # TP=4

        # ---- Performance summary ----
        print("\n" + "=" * 55)
        print("  Performance Summary — Qwen3.5-27B MTP (2 *NPU)")
        print("=" * 55)
        print(f"  TTFT 平均          : {ttft_ms:.2f} ms"   if ttft_ms    else "  TTFT 平均          : N/A")
        print(f"  TPOT 平均  (20ms)        : {tpot_ms:.2f} ms"   if tpot_ms    else "  TPOT 平均          : N/A")
        print(f"  E2E 平均           : {e2e_s:.3f} ms"      if e2e_s      else "  E2E 平均           : N/A")
        print(f"  输出吞吐           : {output_tps:.1f} tps" if output_tps else "  输出吞吐           : N/A")
        print(f"  单卡输出吞吐   (550 tps)    : {per_card_tps:.1f} tps" if per_card_tps else "  单卡输出吞吐       : N/A")
        print("=" * 55 + "\n")

        self.assertIsNotNone(output_tps, "Could not parse 'Output token throughput'")
        self.assertIsNotNone(req_tps,    "Could not parse 'Request throughput'")
        self.assertGreater(
            output_tps, MIN_OUTPUT_THROUGHPUT_TOKENS_PER_SEC,
            f"输出吞吐 {output_tps:.1f} tps < 阈值 {MIN_OUTPUT_THROUGHPUT_TOKENS_PER_SEC} tps",
        )
        self.assertGreater(
            req_tps, MIN_REQUEST_THROUGHPUT,
            f"请求吞吐 {req_tps:.3f} req/s < 阈值 {MIN_REQUEST_THROUGHPUT} req/s",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)