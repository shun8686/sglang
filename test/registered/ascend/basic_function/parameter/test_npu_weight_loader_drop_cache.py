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
    "--weight-loader-disable-mmap",
]


def _vmtouch_file(filepath):
    """Run vmtouch and return (resident_pages, percentage_str, raw_stdout, raw_stderr).
    Format: Resident Pages: 910157/968961 36/36 93.9%
    """
    try:
        result = subprocess.run(
            ["vmtouch", filepath],
            capture_output=True, text=True, timeout=30,
        )
        stdout = result.stdout
        stderr = result.stderr
        resident = -1
        pct = ""
        for line in stdout.split("\n"):
            if "Resident Pages" in line:
                # Parse: " Resident Pages: 910157/968961 36/36 93.9%"
                parts = line.split("Resident Pages:")[1].strip().split()
                resident = int(parts[0].split("/")[0])
                pct = parts[2] if len(parts) >= 3 else ""
                break
        return resident, pct, stdout, stderr
    except Exception as e:
        return -1, "", "", str(e)


def _collect_vmtouch(safetensor_files, weight_dir):
    """Return list of (fname, resident_pages, percentage_str)."""
    results = []
    for fname in safetensor_files:
        fpath = os.path.join(weight_dir, fname)
        resident, pct, _, _ = _vmtouch_file(fpath)
        results.append((fname, resident, pct))
        if resident >= 0:
            print(f"vmtouch: {fname} Resident Pages={resident} ({pct})")
        else:
            print(f"vmtouch: {fname} FAILED")
    return results


class TestWeightLoaderDropCache(CustomTestCase):
    """--weight-loader-drop-cache-after-load with disable-mmap —
    page cache should be low after loading.

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

    def test_drop_cache_on(self):
        """At least one safetensors shard has page cache well below 100%."""
        resp = requests.get(self.base_url + "/health", timeout=30)
        self.assertEqual(resp.status_code, 200)

        weight_dir = QWEN3_8B_WEIGHTS_PATH
        safetensor_files = sorted(
            f for f in os.listdir(weight_dir) if f.endswith(".safetensors")
        )
        results = _collect_vmtouch(safetensor_files, weight_dir)

        self.assertGreater(len(results), 0, "No safetensor files found")
        not_full = [r for r in results if r[1] >= 0 and r[2] != "100%"]
        self.assertGreater(
            len(not_full),
            0,
            f"Expected at least one file below 100% page cache, got all 100%: {results}",
        )


class TestWeightLoaderDropCacheOff(CustomTestCase):
    """Default (no drop-cache) with disable-mmap — baseline: all page cache
    should remain at 100% after loading.

    [Test Category] Parameter
    [Test Target] --weight-loader-drop-cache-after-load (off, baseline)
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

    def test_drop_cache_off(self):
        """All safetensors shards should remain at 100% page cache."""
        resp = requests.get(self.base_url + "/health", timeout=30)
        self.assertEqual(resp.status_code, 200)

        weight_dir = QWEN3_8B_WEIGHTS_PATH
        safetensor_files = sorted(
            f for f in os.listdir(weight_dir) if f.endswith(".safetensors")
        )
        results = _collect_vmtouch(safetensor_files, weight_dir)

        self.assertGreater(len(results), 0, "No safetensor files found")
        for fname, resident, pct in results:
            self.assertEqual(
                pct,
                "100%",
                f"Expected {fname} at 100% page cache, got {pct} (resident={resident})",
            )


if __name__ == "__main__":
    unittest.main()
