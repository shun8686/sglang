import os
import time
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)
from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)

base_port = int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")[0])
BASE_PORT_FOR_ASCEND_MF = 20000 + base_port * 1000 + 66
os.environ["ASCEND_MF_STORE_URL"] = f"tcp://127.0.0.1:{BASE_PORT_FOR_ASCEND_MF}"

class TestDisaggregationPrefillPPAccuracy(TestDisaggregationBase):
    """Test Case: Verify the accuracy of base model when only prefill enables PP parallelism in PD disaggregation scenario

    [Test Category] Parameter
    [Test Target] --pp-size
    """
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL")
        super().tearDownClass()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp-size",
            "2",
            "--pp-size",
            "2",
            "--disable-overlap-schedule",
            "--attention-backend",
            "ascend",
        ]
        env = os.environ.copy()
        prefill_args += cls.transfer_backend + cls.rdma_devices
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
            "--disaggregation-mode",
            "decode",
            "--tp-size",
            "2",
            "--base-gpu-id",
            "4",
            "--attention-backend",
            "ascend",
        ]
        env = os.environ.copy()
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.24)
        # Wait a little bit so that the memory check happens.
        time.sleep(5)


class TestDisaggregationPrefillPPDynamicChunkAccuracy(TestDisaggregationBase):
    """Test Case: Verify the accuracy of base model when prefill enables "dynamic chunking + PP parallelism" in PD disaggregation scenario

    [Test Category] Parameter
    [Test Target] --pp-size; --enable-dynamic-chunking
    """
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL")
        super().tearDownClass()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp-size",
            "2",
            "--pp-size",
            "2",
            "--disable-overlap-schedule",
            "--enable-dynamic-chunking",
            "--attention-backend",
            "ascend",
        ]
        env = os.environ.copy()
        prefill_args += cls.transfer_backend + cls.rdma_devices
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
            "--disaggregation-mode",
            "decode",
            "--tp-size",
            "2",
            "--base-gpu-id",
            "4",
            "--attention-backend",
            "ascend",
        ]
        env = os.environ.copy()
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.24)
        # Wait a little bit so that the memory check happens.
        time.sleep(5)


class TestDisaggregationDecodePPAccuracy(TestDisaggregationBase):
    """Test Case: Verify the accuracy of base model when both prefill and decode enable PP parallelism in PD disaggregation scenario

    [Test Category] Parameter
    [Test Target] --pp-size; --pp-max-micro-batch-size; --pp-max-micro-batch-size
    """
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ASCEND_MF_STORE_URL")
        super().tearDownClass()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp-size",
            "2",
            "--pp-size",
            "2",
            "--pp-async-batch-depth",
            "2",
            "--pp-max-micro-batch-size",
            "2",
            "--disable-overlap-schedule",
            "--attention-backend",
            "ascend",
        ]
        env = os.environ.copy()
        prefill_args += cls.transfer_backend + cls.rdma_devices
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
            "--disaggregation-mode",
            "decode",
            "--tp-size",
            "2",
            "--pp-size",
            "2",
            "--base-gpu-id",
            "4",
            "--attention-backend",
            "ascend",
        ]
        env = os.environ.copy()
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.24)
        # Wait a little bit so that the memory check happens.
        time.sleep(5)


if __name__ == "__main__":
    unittest.main()
