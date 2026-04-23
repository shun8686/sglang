import logging
import os
import re
import subprocess
import threading
import time
from functools import wraps
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.gen_gsm8k_fixed_len import (
    generate_dataset_from_gsm8k,
    generate_fixed_len_dataset,
    generate_mm_dataset,
    save_jsonl,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    SERVICE_PORT,
    check_role,
    launch_pd_mix_node,
    launch_pd_separation_node,
    launch_router,
    wait_server_ready,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

AISBENCHMARK = "aisbench"
BENCHSERVING = "bench-serving"
BENCHMARK_TOOL_DEFAULT = AISBENCHMARK
AISBENCHMARK_DATASET_GSM8K = "gsm8k"
AISBENCHMARK_DATASET_DEFAULT = AISBENCHMARK_DATASET_GSM8K

GSM8K_DATASET_TEST_FILE = (
    "/root/.cache/modelscope/hub/datasets/grade_school_math/test.jsonl"
)
GSM8K_DATASET_TRAIN_FILE = (
    "/root/.cache/modelscope/hub/datasets/grade_school_math/train.jsonl"
)

PYTHON_FOR_TEST_TOOL = "test_env_transformers_tool/bin/python"
if not os.path.exists(PYTHON_FOR_TEST_TOOL) or not os.access(
    PYTHON_FOR_TEST_TOOL, os.X_OK
):
    PYTHON_FOR_TEST_TOOL = "python3"
logger.info(f"PYTHON_FOR_TEST_TOOL: {PYTHON_FOR_TEST_TOOL}")

DEEPSEEK_R1_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Howeee/DeepSeek-R1-0528-w8a8"
)
DEEPSEEK_R1_W4A8_PER_CHANNEL_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/DeepSeek-R1-0528-w4a8-per-channel"
)
DEEPSEEK_V32_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8"
)
QWEN3_8B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B-W8A8"
QWEN3_8B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Eagle3-Qwen3-8B-zh"
QWEN3_14B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-14B"
QWEN3_14B_LORA_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-14B-Lora/Qwen3-14B_lora"
)
QWEN3_14B_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-14B-W8A8-Dynamic2"
)
QWEN3_14B_EAGLE_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/AngelSlim/Qwen3-14B_eagle3"
)
QWEN3_5_27B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-27B"
QWEN3_5_27B_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3.5-27B-W8A8"
)
QWEN3_30B_A3B_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-Instruct-2507"
)
QWEN3_30B_A3B_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-w8a8"
)
QWEN3_30B_A3B_W8A8_VLLM_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-30B-A3B-W8A8"
)
QWEN3_A3B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-a3B_eagle3"
QWEN3_32B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"
QWEN3_32B_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/aleoyang/Qwen3-32B-w8a8-MindIE"
)
QWEN3_32B_EAGLE_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Eagle3-Qwen3-32B-zh"
)
QWEN3_235B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B"
QWEN3_235B_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-235B-A22B-W8A8"
)
QWEN3_235B_A22B_EAGLE_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B-Eagle3"
)
QWEN3_480B_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot"
)
QWEN3_NEXT_80B_A3B_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Next-80B-A3B-Instruct"
)
QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-Next-80B-A3B-Instruct-W8A8"
)
QWEN3_CODER_NEXT_W8A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Coder-Next-W8A8"
)
GLM_4_6_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/GLM-4.6-w8a8_WITH_MTP"

QWEN3_VL_8B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct"
QWEN3_VL_30B_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-VL-30B-A3B-Instruct"
)
QWEN3_VL_235B_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-VL-235B-A22B-Instruct"
)
QWEN2_5_VL_72B_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-72B-Instruct-w8a8"
)
KIMI_K2_5_W4A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/Eco-Tech/Kimi-K2.5-w4a8"
KIMI_K2_5_EAGLE3_MODEL_PATH = "/root/.cache/modelscope/hub/models/Kimi/kimi-k2.5-eagle3"
GLM_4_7_FLASH_MODEL_PATH = "/root/.cache/modelscope/hub/models/ZhipuAI/GLM-4.7-Flash"

QWEN3_5_397B_W4A8_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3.5-397B-A17B-w4a8-mtp"
)

ROUND_ROBIN = "round_robin"

DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 3600
MAX_SERVER_KEEP_ALIVE_TIME = 3600

# Timeouts and delays
SERVER_INITIALIZATION_DELAY = 120

# Test parameters
PROMPTS_MULTIPLIER = 4
PACKAGE_FILTER_KEYWORDS = [
    "sglang",
    "sgl",
    "torch",
    "deep-ep",
    "memfabric_hybrid",
]

if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
    DEFAULT_SERVER_PORT_FOR_TEST = (
        20000 + int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")[0]) * 100
    )
else:
    DEFAULT_SERVER_PORT_FOR_TEST = (
        20000 + int(os.environ.get("ASCEND_VISIBLE_DEVICES", "0")[0]) * 100
    )
DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{DEFAULT_SERVER_PORT_FOR_TEST + 66}"


def retry(max_attempts: int = None):
    """
        Test case retry decorator
    Args:
        max_attempts (int): Maximum number of execution attempts. If None, use self.max_attempts.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Store the last exception for final reporting
            last_exception = None

            # Get max_attempts from instance if not provided in decorator
            attempts = max_attempts or getattr(self, "max_attempts", 2)

            # Execute the test up to max_attempts times
            for attempt in range(1, attempts + 1):
                try:
                    logger.info(f"Executing test attempt {attempt}/{attempts}")
                    return func(
                        self, *args, **kwargs
                    )  # Return immediately if test passes
                except (AssertionError, Exception) as e:
                    last_exception = e
                    logger.info(f"Test failed on attempt {attempt}")

            # Raise the last exception if all attempts failed
            raise last_exception

        return wrapper

    return decorator


def get_cann_version():
    """Get CANN version info.

    Returns:
        str: CANN version info string.
    """
    cann_info_file = "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info"
    cann_ver_num = None

    try:
        with open(cann_info_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("version="):
                    cann_ver_num = line.strip().split("=")[-1]
                    break

        if cann_ver_num:
            cann_version_info = f"CANN: {cann_ver_num}"
            logger.info(cann_version_info)
            return cann_version_info
        else:
            logger.info("CANN version not found")
            return f"CANN: {cann_ver_num}"

    except FileNotFoundError:
        logger.error(f"CANN info file not found: {cann_info_file}")
        return f"CANN: {cann_ver_num}"
    except Exception as e:
        logger.error(f"Error reading CANN info: {e}")
        return f"CANN: {cann_ver_num}"


def write_pkg_info_to_file(result_file):
    """Write package information to result file.

    Args:
        result_file (str): Path to the result file.
    """
    import transformers

    try:
        pip_output = subprocess.run(
            ["pip", "list"], capture_output=True, text=True, check=False
        )
        packages = pip_output.stdout

        # Filter relevant packages using list comprehension
        filtered_packages = [
            line
            for line in packages.split("\n")
            if any(keyword in line for keyword in PACKAGE_FILTER_KEYWORDS)
        ]

        # Write to result file
        os.makedirs(os.path.dirname(os.path.abspath(result_file)), exist_ok=True)
        with open(result_file, "w", encoding="utf-8") as f:
            for pkg in filtered_packages:
                f.write(pkg + "\n")
                logger.info(pkg)
            f.write(get_cann_version() + "\n")
            transformers_version_info = (
                "transformers: " + transformers.__version__ + "\n"
            )
            f.write(transformers_version_info)
            logger.info(transformers_version_info)

    except Exception as e:
        logger.error(f"Error getting packages: {e}")

def run_aisbench(
    host,
    port,
    model_path,
    dataset_name,
):

    metrics_path = os.getenv("METRICS_DATA_FILE")
    result_path = "./aisbench_result" if not metrics_path else metrics_path
    logger.info(f"The metrics result file: {result_path}")

    cmd = f"/bin/bash /root/sglang/python/sglang/test/ascend/e2e/run_aisbench_accuracy.sh "
    cmd += f"--mode accuracy "
    cmd += f"--ip {host} "
    cmd += f"--port {str(port)} "
    cmd += f"--model {os.path.basename(model_path)} "
    cmd += f"--model-path {model_path} "
    cmd += f"--dataset-name {dataset_name} "
    cmd += f"--output-path {result_path}"

    logger.info(f"Command: {cmd}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=True,
    )

    output_lines = []
    try:
        for line in iter(process.stdout.readline, ""):
            if line.strip():
                print(line, end="")
            output_lines.append(line.strip())

        process.wait()

        if process.returncode != 0:
            logger.error(f"Command failed with return code: {process.returncode}")
            raise subprocess.CalledProcessError(process.returncode, cmd)

        logger.info("Command executed successfully")

        metrics = {}
        full_output = "\n".join(output_lines)

        matches = re.findall(r"accuracy\s+[a-zA-Z]+\s+([\d.]+)", full_output)

        if matches:
            final_accuracy = float(matches[-1])
            metrics["accuracy"] = final_accuracy
            logger.info(f"The Final Accuracy: {final_accuracy}")
        else:
            logger.info(f"Can Not Find The Accuracy")

        return metrics


    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, terminating process...")
        process.terminate()
        try:
            process.wait(timeout=5)
            logger.info("Process terminated")
        except subprocess.TimeoutExpired:
            logger.warning("Process did not terminate gracefully, killing it...")
            process.kill()
            logger.info("Process killed")
        raise
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        process.terminate()
        process.wait(timeout=5)
        raise


def assert_metrics(self, metrics):
    """Assert benchmark metrics against expected values.

    Args:
        metrics (dict): Benchmark metrics dictionary.
    """
    if not metrics:
        raise Exception("No metrics obtained from benchmark")

    if self.accuracy is not None:
        self.assertGreaterEqual(
            float(metrics["accuracy"]),
            self.accuracy,
            f"Accuracy check failed. Expected >= {self.accuracy}, Got: {metrics['accuracy']}"
        )

class TestAscendAccuracyTestCaseBase(CustomTestCase):
    model = None
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    backend = "sglang"
    dataset_name = "gsm8k_gen_4_shot_cot_str"  # gsm8k
    other_args = None
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    envs = None
    max_attempts = 2
    accuracy = 0.1

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        for key, value in env.items():
            logger.info(f"ENV_VAR_SYS {key}:{value}")
        if cls.envs:
            for key, value in cls.envs.items():
                logger.info(f"ENV_VAR_CASE {key}:{value}")
                env[key] = value

        other_args = list(cls.other_args)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            try:
                kill_process_tree(cls.process.pid)
            except Exception as e:
                logger.error(f"Error during tearDown: {e}")

    @retry()
    def run_throughput(self):
        parsed_url = urlparse(self.base_url)
        host = parsed_url.hostname
        port = parsed_url.port
        if self.benchmark_tool == AISBENCHMARK:
            metrics = run_aisbench(
                host=host,
                port=port,
                model_path=self.model,
                dataset_name=self.dataset_name,
            )
            assert_metrics(self, metrics)

class TestAscendAccuracyMultiNodePdMixTestCaseBase(CustomTestCase):
    model = None
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    backend = "sglang"
    dataset_name = "gsm8k_gen_4_shot_cot_str"  # gsm8k
    other_args = None
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    envs = None
    max_attempts = 2
    accuracy = 0.1

    @classmethod
    def setUpClass(cls):
        cls.local_ip = "127.0.0.1"
        cls.host = os.getenv("POD_IP")
        cls.port = SERVICE_PORT
        cls.base_url = f"http://{cls.host}:{cls.port}"
        cls.hostname = os.getenv("HOSTNAME")
        cls.role = "master" if cls.hostname.endswith("sglang-node-0") else "worker"
        logger.info(f"Init {cls.host} {cls.role=}!")

        cls.start_pd_mix_master_node()
        cls.start_pd_mix_worker_node()

    @classmethod
    def tearDownClass(cls):
        pass

    @classmethod
    @check_role(allowed_roles=["master"])
    def start_pd_mix_master_node(cls):
        sglang_thread = threading.Thread(
            target=launch_pd_mix_node, args=(cls.model_config,)
        )
        sglang_thread.start()

        wait_server_ready(f"{cls.base_url}/health")

        logger.info(
            f"Wait {SERVER_INITIALIZATION_DELAY}s, starting run benchmark ......"
        )
        time.sleep(SERVER_INITIALIZATION_DELAY)

    @classmethod
    @check_role(allowed_roles=["worker"])
    def start_pd_mix_worker_node(cls):
        sglang_thread = threading.Thread(
            target=launch_pd_mix_node, args=(cls.model_config,)
        )
        sglang_thread.start()

        logger.info(
            f"{cls.role} node started, keeping test alive for {MAX_SERVER_KEEP_ALIVE_TIME} seconds"
        )
        time.sleep(MAX_SERVER_KEEP_ALIVE_TIME)

    @retry()
    @check_role(allowed_roles=["master", "worker"])

    def run_throughput(self):
        parsed_url = urlparse(self.base_url)
        host = parsed_url.hostname
        port = parsed_url.port
        if self.benchmark_tool == AISBENCHMARK:
            metrics = run_aisbench(
                host=host,
                port=port,
                model_path=self.model,
                dataset_name=self.dataset_name,
            )
            assert_metrics(self, metrics)

class TestAscendPerfMultiNodePdSepTestCaseBase(CustomTestCase):
    model = None
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    backend = "sglang"
    dataset_name = "gsm8k_gen_4_shot_cot_str"  # gsm8k
    other_args = None
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    envs = None
    max_attempts = 2
    accuracy = 0.1

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.local_ip = "127.0.0.1"
        cls.host = os.getenv("POD_IP")
        cls.port = SERVICE_PORT
        cls.base_url = f"http://{cls.host}:{cls.port}"
        cls.hostname = os.getenv("HOSTNAME")
        cls.role = (
            "router"
            if "router" in cls.hostname
            else "prefill" if "prefill" in cls.hostname else "decode"
        )
        logger.info(f"Init {cls.host} {cls.role=}!")

        cls.start_pd_server()
        cls.start_router_server()

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            try:
                kill_process_tree(cls.process.pid)
            except Exception as e:
                logger.error(f"Error during tearDown: {e}")

    @classmethod
    @check_role(allowed_roles=["router"])
    def start_router_server(cls):
        logger.info(f"Starting router in thread...")
        sglang_thread = threading.Thread(target=launch_router, args=(cls.model_config,))
        sglang_thread.daemon = True
        sglang_thread.start()

        health_check_url = f"{cls.base_url}/health"
        logger.info(f"Waiting for router to be ready at {health_check_url}")
        wait_server_ready(health_check_url)

        logger.info(
            f"Waiting {SERVER_INITIALIZATION_DELAY} seconds for the server to fully initialize..."
        )
        time.sleep(SERVER_INITIALIZATION_DELAY)

    @classmethod
    @check_role(allowed_roles=["prefill", "decode"])
    def start_pd_server(cls):
        logger.info(f"Starting pd separation node...")
        cls.process = launch_pd_separation_node(cls.model_config)
        logger.info(f"Pd separation node started with PID: {cls.process.pid}")

        # Loop to check if the process is still running
        while True:
            if cls.process.poll() is None:
                # Process is still running
                time.sleep(30)
            else:
                # Process has exited
                exit_code = cls.process.poll()
                raise Exception(
                    f"Sglang process exited on node {cls.host} {cls.hostname} with exit code: {exit_code}"
                )

    @retry()
    @check_role(allowed_roles=["router"])
    def run_throughput(self):
        parsed_url = urlparse(self.base_url)
        host = parsed_url.hostname
        port = parsed_url.port
        if self.benchmark_tool == AISBENCHMARK:
            metrics = run_aisbench(
                host=host,
                port=port,
                model_path=self.model,
                dataset_name=self.dataset_name,
            )
            assert_metrics(self, metrics)
    





