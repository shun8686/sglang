import os
import unittest
from types import SimpleNamespace

# from sglang.test.ascend.test_ascend_utils import (
#     DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH,
#     LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
# )
DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH="/home/weights/DeepSeek-V2-Lite-W8A8"
LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH="/home/weights/Llama-3.1-8B-Instruct"
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_npu_ci(est_time=400, suite="full-16-npu-a3", nightly=True)


class TestDisaggregationAscendPrefillLargerTP(TestDisaggregationBase):
    """MLA model: Prefill TP=4 -> Decode TP=2"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp-size",
            "8",
            "--base-gpu-id",
            "4",
            # "--mem-fraction-static",
            # "0.7",
            "--disable-cuda-graph",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        env = {**os.environ, "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24668"}
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=env,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp-size",
            "4",
            "--base-gpu-id",
            #"4",
            "12",
            # "--mem-fraction-static",
            # "0.7",
            "--disable-cuda-graph",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        env = {**os.environ, "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24668"}
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.lb_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.60)


class TestDisaggregationAscendDecodeLargerTP(TestDisaggregationBase):
    """MLA model: Prefill TP=2 -> Decode TP=4"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp-size",
            "2",
            "--base-gpu-id",
            "8",
            "--mem-fraction-static",
            "0.9",
            "--disable-cuda-graph",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        env = {**os.environ, "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24668"}
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=env,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp-size",
            "4",
            "--base-gpu-id",
            #"2",
            "10",
            "--mem-fraction-static",
            "0.9",
            "--disable-cuda-graph",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        env = {**os.environ, "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24668"}
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.lb_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.60)

'''
class TestDisaggregationAscendMHAPrefillLargerTP(TestDisaggregationBase):
    """MHA model: Prefill TP=4 -> Decode TP=2"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp-size",
            "4",
            "--base-gpu-id",
            "8",
            "--mem-fraction-static",
            "0.9",
            "--disable-cuda-graph",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        env = {**os.environ, "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24668"}
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=env,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp-size",
            "2",
            "--base-gpu-id",
            #"4",
            "12",
            "--mem-fraction-static",
            "0.9",
            "--disable-cuda-graph",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        env = {**os.environ, "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24668"}
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.lb_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.60)


class TestDisaggregationAscendMHADecodeLargerTP(TestDisaggregationBase):
    """MHA model: Prefill TP=2 -> Decode TP=4"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp-size",
            "2",
            "--base-gpu-id",
            "8",
            "--mem-fraction-static",
            "0.9",
            "--disable-cuda-graph",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        env = {**os.environ, "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24668"}
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=env,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-transfer-backend",
            "ascend",
            "--tp-size",
            "4",
            "--base-gpu-id",
            #"2",
            "10",
            "--mem-fraction-static",
            "0.9",
            "--disable-cuda-graph",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        env = {**os.environ, "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24668"}
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.lb_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.60)
'''

if __name__ == "__main__":
    unittest.main()
