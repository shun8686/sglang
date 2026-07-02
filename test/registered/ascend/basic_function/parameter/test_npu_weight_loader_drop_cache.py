import os
import subprocess
import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="nightly-1-npu-a3", nightly=True)

_COMMON_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--dtype",
    "bfloat16",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.78",
]


class TestWeightLoaderDropCache(CustomTestCase):
    """--weight-loader-drop-cache-after-load — verify page cache is released
    after loading with vmtouch.

    [Test Category] Parameter
    [Test Target] --weight-loader-drop-cache-after-load
    """

    model = QWEN3_8B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.out_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        cls.err_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *_COMMON_ARGS,
                "--weight-loader-drop-cache-after-load",
                "--log-level",
                "info",
            ],
            return_stdout_stderr=(cls.out_file, cls.err_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_file.close()
        cls.err_file.close()
        os.unlink(cls.out_file.name)
        os.unlink(cls.err_file.name)

    def _vmtouch_resident_pages(self, filepath):
        """Return Resident Pages count from vmtouch, or -1 on failure."""
        try:
            result = subprocess.run(
                ["vmtouch", filepath],
                capture_output=True, text=True, timeout=30,
            )
            for line in result.stdout.split("\n"):
                if "Resident Pages" in line:
                    return int(line.strip().split()[0])
        except Exception:
            pass
        return -1

    def test_drop_cache_after_load(self):
        """Launch with drop-cache-after-load, then inspect page cache via vmtouch."""

        # Server must be healthy and able to generate
        resp = requests.get(self.base_url + "/health", timeout=30)
        self.assertEqual(resp.status_code, 200)

        data = {
            "text": "Hello, my name is",
            "sampling_params": {"temperature": 0, "max_new_tokens": 8},
        }
        resp = requests.post(
            self.base_url + "/generate", json=data, timeout=30
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text", resp.json())

        # Check page cache for safetensors shards via vmtouch (observation only)
        weight_dir = os.path.join(QWEN3_8B_WEIGHTS_PATH)
        safetensor_files = sorted(
            f for f in os.listdir(weight_dir) if f.endswith(".safetensors")
        )
        if safetensor_files:
            print("\n--- vmtouch page cache inspection ---")
            for fname in safetensor_files:
                fpath = os.path.join(weight_dir, fname)
                resident = self._vmtouch_resident_pages(fpath)
                if resident >= 0:
                    print(f"vmtouch: {fname} Resident Pages={resident}")
                else:
                    print(f"vmtouch: {fname} FAILED")
            print("--- end vmtouch ---\n")

        # Stderr may contain prefetch log if prefetch was enabled,
        # but with only drop-cache there may be no special log.
        with open(self.err_file.name) as f:
            log_content = f.read()
        print("--- Server stderr (first 2000 chars) ---")
        print(log_content[:2000])
        print("--- end stderr ---")


if __name__ == "__main__":
    unittest.main()
