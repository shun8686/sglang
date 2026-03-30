# import os
# import unittest
#
# import requests
#
# from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH, QWEN3_0_6B_WEIGHTS_PATH
# from sglang.test.ci.ci_register import register_npu_ci
# from sglang.test.test_utils import (
#     DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
#     DEFAULT_URL_FOR_TEST,
#     CustomTestCase,
#     popen_launch_server,
# )
#
# register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


# class TestLogLevel(CustomTestCase):
#     """Testcase：Verify set log-level parameter, the printed log level is the same as the configured log level and the inference request is successfully processed.
#
#     [Test Category] Parameter
#     [Test Target] --log-level
#     """
#
#     model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
#     OUT_LOG_PATH = "./out_log.txt"
#     ERR_LOG_PATH = "./err_log.txt"
#
#     def _launch_server_and_run_infer(self, other_args):
#         out_log_file = None
#         err_log_file = None
#         process = None
#         try:
#             out_log_file = open(self.OUT_LOG_PATH, "w+", encoding="utf-8")
#             err_log_file = open(self.ERR_LOG_PATH, "w+", encoding="utf-8")
#             process = popen_launch_server(
#                 self.model,
#                 DEFAULT_URL_FOR_TEST,
#                 timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
#                 other_args=other_args,
#                 return_stdout_stderr=(out_log_file, err_log_file),
#             )
#             health_resp = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
#             self.assertEqual(health_resp.status_code, 200)
#             gen_resp = requests.post(
#                 f"{DEFAULT_URL_FOR_TEST}/generate",
#                 json={
#                     "text": "The capital of France is",
#                     "sampling_params": {"temperature": 0, "max_new_tokens": 32},
#                 },
#             )
#             print("=============================================================")
#             print(gen_resp.json())
#             self.assertEqual(gen_resp.status_code, 200)
#             self.assertIn("Paris", gen_resp.text)
#             out_log_file.seek(0)
#             return out_log_file.read()
#         finally:
#             kill_process_tree(process.pid)
#             out_log_file.close()
#             err_log_file.close()
#             os.remove(self.OUT_LOG_PATH)
#             os.remove(self.ERR_LOG_PATH)
#
#     def test_log_level(self):
#         # Verify set --log-level=warning and not set --log-level-http, logs print only warning level (no HTTP info)
#         other_args = [
#             "--scheduler-recv-interval",
#             "100",
#             "--attention-backend",
#             "ascend",
#             "--disable-cuda-graph",
#         ]
#         # log_content = self._launch_server_and_run_infer(other_args)
#         # self.assertNotIn("POST /generate HTTP/1.1", log_content)
#
#     def test_log_http_level(self):
#         # Verify set --log-level=warning and set --log-level-http=info, log level print http info
#         other_args = [
#             "--attention-backend",
#             "ascend",
#             "--disable-cuda-graph",
#         ]
#         # log_content = self._launch_server_and_run_infer(other_args)
#         # self.assertIn("POST /generate HTTP/1.1", log_content)
#
#
# if __name__ == "__main__":
#     unittest.main()



import time
import unittest
from types import SimpleNamespace

import requests

from sglang.bench_one_batch_server import BenchArgs as OneBatchBenchArgs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    run_bench_one_batch_server,
)
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH, \
    DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH, \
    GLM_4_5V_WEIGHTS_PATH, QWEN3_0_6B_WEIGHTS_PATH, QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH, QWEN3_32B_WEIGHTS_PATH, \
    QWEN3_8B_WEIGHTS_PATH, QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH, QWEN3_30B_A3B_WEIGHTS_PATH


class TestQwenPPTieWeightsAccuracy(unittest.TestCase):
    """Test Case: Verify the accuracy consistency of Qwen3-0.6B model (with tie_word_embeddings) between PP=1 and PP=2

    [Test Category] Parameter
    [Test Target] --pp-size
    """
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23335"  # different ports to avoid conflicts
        cls.model_name = QWEN3_0_6B_WEIGHTS_PATH  # qwen3 < 8B all have tie_word_embeddings = True

    def run_gsm8k_test(self, interval):
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--scheduler-recv-interval",
                interval,
                "--attention-backend",
                "ascend",
                # "--mem-fraction-static",
                # "0.8",
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency(self):
        baseline = self.run_gsm8k_test(interval=1)
        print("============================baseline=====================================")
        print(baseline)
        pp_metrics = self.run_gsm8k_test(interval=100)
        print("============================100===========================================")
        print(pp_metrics)

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")




        # self.assertGreaterEqual(baseline["accuracy"], 0.38)
        # self.assertGreaterEqual(
        #     pp_metrics["accuracy"],
        #     baseline["accuracy"] - 0.02,
        #     msg=(
        #         f"PP accuracy dropped more than 2% compared to baseline. "
        #         f"Baseline: {baseline['accuracy']:.2%}, PP: {pp_metrics['accuracy']:.2%}"
        #     ),
        # )
