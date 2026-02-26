import os
import shutil
import subprocess
import requests
import unittest
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import QWEN2_0_5B_INSTRUCT_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)

MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"


class TestAscendFastapiRootPath(CustomTestCase):
    """
    Testcase：Verify that the system correctly processes the root path prefix when configuring the root path prefix and
    correctly performs the route redirection behavior.

    [Test Category] Parameter
    [Test Target] --fastapi-root-path
    """

    fastapi_root_path = "/sglang/"

    @classmethod
    def setUpClass(cls):
        # Modify nginx configuration and start nginx service
        cls.nginx_manager = NginxConfigManager(
            nginx_conf_path="/usr/local/nginx/conf/nginx.conf",
            nginx_bin_path="/usr/local/nginx/sbin/nginx"
        )

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.nginx_manager.apply_config(cls.fastapi_root_path, cls.base_url)

        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)
        cls.common_args = [
            "--trust-remote-code",
            "--mem-fraction-static", 0.8,
            "--attention-backend", "ascend",
            "--fastapi-root-path", cls.fastapi_root_path,
        ]

        cls.out_log_file = open("./warmup_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./warmup_err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.common_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./warmup_out_log.txt")
        os.remove("./warmup_err_log.txt")
        cls.nginx_manager.clean_environment()

    def test_fastapi_root_path(self):
        response = self.send_request(f"{self.base_url}/generate")
        self.assertEqual(response.status_code, 200, "The request status code is not 200.")
        self.assertNotIn(
            self.fastapi_root_path,
            response.url,
            "The root path should not in response url."
        )
        self.assertIn("Paris", response.text, "The inference result does not include Paris.")

        self.out_log_file.seek(0)
        content = self.out_log_file.read()
        self.assertTrue(len(content) > 0)
        self.assertIn(f"POST {self.fastapi_root_path}/generate HTTP/1.1", content)

        response = self.send_request(f"{self.base_url}{self.fastapi_root_path}generate")
        self.assertEqual(response.status_code, 404, "The request status code is not 404.")

    def send_request(self, url):
        return requests.post(
            url,
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )

    # def assert_interfsend_request(self, url):
    #     return requests.post(
    #         url,
    #         json={
    #             "text": "The capital of France is",
    #             "sampling_params": {
    #                 "temperature": 0,
    #                 "max_new_tokens": 32,
    #             },
    #         },
    #     )


class TestAscendFastapiRootPathMultiLevel(TestAscendFastapiRootPath):
    fastapi_root_path = "/test/fastapi/root/path/"


class TestAscendFastapiRootPath1(TestAscendFastapiRootPath):
    fastapi_root_path = "/sglang"


class TestAscendFastapiRootPathErrorPath(TestAscendFastapiRootPath):
    fastapi_root_path = "sglang"

    def test_fastapi_root_path(self):
        response = self.send_request(f"{self.base_url}/generate")
        self.assertEqual(response.status_code, 200, "The request status code is not 200.")
        self.assertIn("Paris", response.text, "The inference result does not include Paris.")

        response = self.send_request(f"{self.base_url}/{self.fastapi_root_path}/generate")
        self.assertEqual(response.status_code, 404, "The request status code is not 404.")


class TestAscendFastapiRootPathNotSet(TestAscendFastapiRootPath):
    fastapi_root_path = "/sglang/"

    @classmethod
    def setUpClass(cls):
        # Modify nginx configuration and start nginx service
        cls.nginx_manager = NginxConfigManager(
            nginx_conf_path="/usr/local/nginx/conf/nginx.conf",
            nginx_bin_path="/usr/local/nginx/sbin/nginx"
        )

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.nginx_manager.apply_config(cls.fastapi_root_path, cls.base_url)

        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)
        cls.common_args = [
            "--trust-remote-code",
            "--mem-fraction-static", 0.8,
            "--attention-backend", "ascend",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.common_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.nginx_manager.clean_environment()

    def test_fastapi_root_path(self):
        response = self.send_request(f"{self.base_url}/generate")
        self.assertEqual(response.status_code, 404, "The request status code is not 404.")

        response = self.send_request(f"{self.base_url}{self.fastapi_root_path}generate")
        self.assertEqual(response.status_code, 404, "The request status code is not 404.")


class TestAscendFastapiRootPathWithoutNginx(TestAscendFastapiRootPath):
    fastapi_root_path = "/sglang/"

    @classmethod
    def setUpClass(cls):
        # Modify nginx configuration and start nginx service
        cls.nginx_manager = NginxConfigManager(
            nginx_conf_path="/usr/local/nginx/conf/nginx.conf",
            nginx_bin_path="/usr/local/nginx/sbin/nginx"
        )

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.nginx_manager.apply_config(cls.fastapi_root_path, cls.base_url)

        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(cls.base_url)
        cls.common_args = [
            "--trust-remote-code",
            "--mem-fraction-static", 0.8,
            "--attention-backend", "ascend",
            "--fastapi-root-path", cls.fastapi_root_path,
        ]

        cls.out_log_file = open("./warmup_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./warmup_err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.common_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./warmup_out_log.txt")
        os.remove("./warmup_err_log.txt")
        cls.nginx_manager.clean_environment()

    def test_fastapi_root_path(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200, "The request status code is not 200.")
        self.assertNotIn(
            self.fastapi_root_path,
            response.url,
            "The root path should not in response url."
        )
        self.assertIn("Paris", response.text, "The inference result does not include Paris.")

        self.out_log_file.seek(0)
        content = self.out_log_file.read()
        self.assertTrue(len(content) > 0)
        self.assertIn(f"POST {self.fastapi_root_path}/generate HTTP/1.1", content)

        response = requests.post(
            f"{self.base_url}{self.fastapi_root_path}generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 404, "The request status code is not 404.")


class NginxConfigManager:
    def __init__(self, nginx_conf_path, nginx_bin_path):
        self.nginx_conf_path = nginx_conf_path
        self.nginx_bin_path = nginx_bin_path
        self.backup_conf_path = f"{nginx_conf_path}.backup"

    def backup_original_config(self):
        if not os.path.exists(self.backup_conf_path):
            shutil.copy2(self.nginx_conf_path, self.backup_conf_path)

    def apply_config(self, location, proxy_pass):
        self.backup_original_config()

        try:
            with open(self.nginx_conf_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            lines[48] = "        location " + f"{location}" + " {\n"
            lines[49] = "            proxy_pass " + f"{proxy_pass}" + ";\n"
            lines[50] = "        }\n"

            with open(self.nginx_conf_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except FileNotFoundError:
            raise FileNotFoundError(f"file not found: {self.nginx_conf_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to modify nginx config: {e}")

        # reload Nginx
        try:
            subprocess.run(
                [self.nginx_bin_path],
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to modify nginx config: {e}")

    def restore_original_config(self):
        if os.path.exists(self.backup_conf_path):
            shutil.copy2(self.backup_conf_path, self.nginx_conf_path)

    def clean_environment(self):
        if os.path.exists(self.backup_conf_path):
            shutil.copy2(self.backup_conf_path, self.nginx_conf_path)
            os.remove(self.backup_conf_path)
        subprocess.run([self.nginx_bin_path, '-s', 'stop'])


if __name__ == "__main__":
    # unittest.main()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAscendFastapiRootPath))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendFastapiRootPathMultiLevel))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendFastapiRootPath1))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendFastapiRootPathErrorPath))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendFastapiRootPathNotSet))
    # suite.addTests(loader.loadTestsFromTestCase(TestAscendFastapiRootPathWithoutNginx))
    runner = unittest.TextTestRunner()
    runner.run(suite)
