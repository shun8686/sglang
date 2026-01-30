import os
import subprocess
import threading
import time
import psutil
import socket
import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DEEPSEEK_R1_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/Howeee/DeepSeek-R1-0528-w8a8"
DEEPSEEK_R1_W4A8_PER_CHANNEL_MODEL_PATH = "/root/.cache/modelscope/hub/models/DeepSeek-R1-0528-w4a8-per-channel"
DEEPSEEK_V32_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/DeepSeek-V3.2-Exp-W8A8"
QWEN3_30B_A3B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B-w8a8"
QWEN3_A3B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-a3B_eagle3"
QWEN3_32B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B"
QWEN3_32B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/aleoyang/Qwen3-32B-w8a8-MindIE"
QWEN3_32B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Eagle3-Qwen3-32B-zh"
QWEN3_235B_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B"
QWEN3_235B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-235B-A22B-W8A8"
QWEN3_235B_A22B_EAGLE_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-235B-A22B-Eagle3"
QWEN3_480B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot"
QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/vllm-ascend/Qwen3-Next-80B-A3B-Instruct-W8A8"
GLM_4_6_W8A8_MODEL_PATH = "/root/.cache/modelscope/hub/models/GLM-4.6-w8a8_WITH_MTP"

KUBE_CONFIG = os.environ.get('KUBECONFIG')
NAMESPACE = os.environ.get('NAMESPACE')
CONFIGMAP_NAME = os.environ.get('KUBE_CONFIG_MAP')

LOCAL_TIMEOUT = 3600
SERVICE_PORT = "6677"
LOCAL_PORT = "8000"

config.load_kube_config(KUBE_CONFIG)
v1 = client.CoreV1Api()

def get_nic_name():
    for nic, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and (addr.address.startswith("172.") or addr.address.startswith("192.")):
                print("The nic name matched is {}".format(nic))
                return nic
    return None

nic = get_nic_name()
NIC_NAME = "lo" if nic is None else nic

def run_command(cmd, shell=True):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=False
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command error: {e}")
        return None

# query configmap
def query_configmap(name, namespace):
    try:
        configmap = v1.read_namespaced_config_map(name, namespace)
        print(f"Successfully queried ConfigMap {name} in namespace {namespace}")
        return configmap
    except ApiException as e:
        print(f"Failed to query ConfigMap {name} in namespace {namespace}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error querying ConfigMap: {e}")
        return None

# get node count from k8s
def discover_worker_nodes():
    try:
        config.load_incluster_config()

        prefill_pods = v1.list_namespaced_pod(
            namespace=NAMESPACE, label_selector="volcano.sh/task-spec=sglang-prefill"
        )
        decode_pods = v1.list_namespaced_pod(
            namespace=NAMESPACE, label_selector="volcano.sh/task-spec=sglang-decode"
        )

        nodes_count = len(prefill_pods.items) + len(decode_pods.items)
        print(f"Discovered {nodes_count} worker nodes (prefill: {len(prefill_pods.items)}, decode: {len(decode_pods.items)})")
        return nodes_count

    except Exception as e:
        print(f"Unexpected error discovering worker nodes: {e}")
        return 0

def set_environment_variables(env_vars):
    if not env_vars:
        return

    for key, value in env_vars.items():
        print(f"Setting ENV_VAR {key}={value}")
        os.environ[key] = value

def check_port_availability(host, port, timeout=3):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex((host, int(port)))
            return result == 0

    except (socket.error, ValueError) as e:
        print(f"Port check error for {host}:{port}: {e}")
        return False

def wait_for_all_ports_ready(ips, port, timeout=LOCAL_TIMEOUT):
    start_time = time.time()
    while time.time() - start_time < timeout:
        ready_nodes = 0
        for ip in ips:
            if check_port_availability(ip, port):
                print(f"Node {ip}:{port} is ready")
                ready_nodes += 1
            else:
                print(f"Node {ip}:{port} is not ready yet")

        if ready_nodes == len(ips):
            print(f"All {len(ips)} nodes' ports are ready!")
            return True

        print(f"Waiting for {len(ips) - ready_nodes} more nodes to be ready...")
        time.sleep(15)

    print(f"Timeout: Not all nodes are ready after {timeout} seconds")
    return False

# launch master/worker node
def launch_pd_mix_node(model_config):
    print(f"Launch pd mix node start ......")
    node_ip = os.getenv("POD_IP")
    hostname = os.getenv("HOSTNAME")
    pod_index = int(hostname.rsplit("-", 1)[-1])
    if not node_ip or not hostname:
        raise RuntimeError(f"Missing required environment variables: POD_IP={node_ip}, HOSTNAME={hostname}")

    # monitor configmap to generate dist-init-addr and node-rank
    is_ready = False
    dist_init_addr = None
    while not is_ready:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if configmap.data is None:
            print(f"configmap is None, wait for 15s ......")
            time.sleep(15)
            continue
        print(f"monitor {configmap.data=}")

        master_node_ip = None
        for pod_name in configmap.data:
            if pod_name.endswith("sglang-node-0"):
                master_node_ip = configmap.data[pod_name]
                break
        if master_node_ip is None:
            print(f"Can not find master node in configmap: {configmap.data=}")
            continue

        dist_init_addr = f"{master_node_ip}:5000"
        print(f"launch_node {dist_init_addr=}")
        is_ready = True

    special_args = [
        "--dist-init-addr", dist_init_addr,
        "--node-rank", pod_index,
    ]
    other_args = model_config["other_args"]
    for sa in special_args:
        other_args.append(sa)

    for key, value in model_config["node_envs"].items():
        print(f"ENV_VAR {key}:{value}")
        os.environ[key] = value

    print(f"Starting node, {node_ip=} {other_args=}")
    return popen_launch_server(
        model_config["model_path"],
        f"http://{node_ip}:{SERVICE_PORT}",
        timeout=LOCAL_TIMEOUT,
        other_args=[
            *other_args,
        ],
    )

# launch p/d seperation node
def launch_pd_seperation_node(model_config):
    print(f"Launch pd seperation node start ......")
    node_ip = os.getenv("POD_IP")
    hostname = os.getenv("HOSTNAME")
    pod_index = int(hostname.rsplit("-", 1)[-1])
    if not node_ip or not hostname:
        raise RuntimeError(f"Missing required environment variables: POD_IP={node_ip}, HOSTNAME={hostname}")

    role = "prefill" if "prefill" in hostname else "decode"

    bootstrap_init_port = 8995
    master_prefill_ip = None
    master_decode_ip = None

    is_prefill_instance_multi_node = True if "--node-rank" not in model_config["prefill_args"] else False
    is_decode_instance_multi_node = True if "--node-rank" not in model_config["decode_args"] else False

    # monitor configmap ready
    is_ready = False
    start_time = time.time()
    while not is_ready and time.time() - start_time < LOCAL_TIMEOUT:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if not configmap or not configmap.data:
            print(f"ConfigMap data is not available yet, waiting for 15s...")
            time.sleep(15)
            continue

        print(f"Retrieved ConfigMap data: {configmap.data}")

        for pod_name, pod_ip in configmap.data.items():
            if pod_name.endswith("prefill-0"):
                master_prefill_ip = pod_ip
            if pod_name.endswith("decode-0"):
                master_decode_ip = pod_ip

        if master_prefill_ip and master_decode_ip:
            is_ready = True
        else:
            print(f"Missing master node information - prefill: {master_prefill_ip}, decode: {master_decode_ip}")
            print("Retrying in 15s...")
            time.sleep(15)
    if not is_ready:
        raise RuntimeError(f"Timeout: Failed to get master node information from ConfigMap")

    # generate p/d run command
    common_args = [
        "--trust-remote-code",
        "--attention-backend", "ascend",
        "--device", "npu",
        "--disaggregation-transfer-backend", "ascend",
    ]
    service_args = list(common_args)

    mf_addr = f"tcp://{master_prefill_ip}:24666"
    os.environ["ASCEND_MF_STORE_URL"] = mf_addr
    print(f"Setting ENV_VAR ASCEND_MF_STORE_URL={mf_addr}")

    if role == "prefill":
        # Current node is prefill
        dist_init_addr = f"{master_prefill_ip}:5000"
        print(f"Launching prefill node with dist_init_addr={dist_init_addr}")

        set_environment_variables(model_config.get("prefill_envs"))

        prefill_args = model_config["prefill_args"]
        if is_prefill_instance_multi_node:
            print("No node-rank specified - all prefill nodes will form a single instance.")
            prefill_args.extend(
                [
                    "--node-rank", pod_index,
                    "--dist-init-addr", dist_init_addr,
                    "--disaggregation-bootstrap-port", bootstrap_init_port,
                ]
            )
        else:
            print("Node-rank specified - each prefill node is an instance.")
            prefill_args.extend([
                    "--disaggregation-bootstrap-port", str(bootstrap_init_port + pod_index),
            ])

        service_args.extend(prefill_args)

    elif role == "decode":
        dist_init_addr = f"{master_decode_ip}:5000"
        print(f"Launching decode node with dist_init_addr={dist_init_addr}")

        set_environment_variables(model_config.get("decode_envs"))

        decode_args = model_config["decode_args"]
        if is_decode_instance_multi_node:
            print("No node-rank specified - all decode nodes will form a single instance.")
            decode_args.extend([
                "--node-rank", str(pod_index),
                "--dist-init-addr", dist_init_addr,
            ])
        else:
            print("Node-rank specified - each decode node is an instance.")

        service_args.extend(decode_args)

    print(f"Starting {role} node on {node_ip} with args: {service_args}")

    try:
        process = popen_launch_server(
            model_config["model_path"],
            f"http://{node_ip}:{LOCAL_PORT}",
            timeout=LOCAL_TIMEOUT,
            other_args=[
                *service_args,
            ],
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start {role} node on {node_ip}: {e}")
    return process

# launch router node
def launch_router(config):
    print(f"launch_router start ......")
    nodes_count = discover_worker_nodes()
    print(f"Discovered {nodes_count} worker nodes")

    # monitor  to generate p/d url
    prefill_url = []
    decode_url = []
    bootstrap_ports = []
    node_ip_list = []
    is_prefill_instance_multi_node = True if "--node-rank" not in config["prefill_args"] else False
    is_decode_instance_multi_node = True if "--node-rank" not in config["decode_args"] else False

    is_ready = False
    bootstrap_init_port = 8995
    start_time = time.time()
    while not is_ready and time.time() - start_time < 300:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if not configmap or not configmap.data:
            print(f"ConfigMap data is not available yet, waiting for 15s...")
            time.sleep(15)
            continue
        print(f"Retrieved ConfigMap data: {configmap.data}")
        for pod_name, pod_ip in configmap.data.items():
            pod_index = int(pod_name.rsplit("-", 1)[-1])
            prefill_keyword = "prefill-0" if is_prefill_instance_multi_node else "prefill"
            if prefill_keyword in pod_name:
                prefill_url.append(f"{pod_ip}:{LOCAL_PORT}")
                bootstrap_port = (bootstrap_init_port if is_prefill_instance_multi_node else bootstrap_init_port + pod_index)
                bootstrap_ports.append(str(bootstrap_port))
                node_ip_list.append(pod_ip)
            decode_keyword = "decode-0" if is_decode_instance_multi_node else "decode"
            if decode_keyword in pod_name:
                decode_url.append(f"{pod_ip}:{LOCAL_PORT}")
                node_ip_list.append(pod_ip)
        if prefill_url and decode_url:
            is_ready = True
        else:
            print("Incomplete node information in ConfigMap, waiting for 15s...")
            time.sleep(15)

    if not is_ready:
        raise RuntimeError(f"Timeout: Failed to get complete node information from ConfigMap")
    print(
        f"ConfigMap monitoring complete: prefill_url={prefill_url}, decode_url={decode_url}, "
        f"bootstrap_ports={bootstrap_ports}, node_ip_list={node_ip_list}"
    )

    # checkout all node port ready
    if not wait_for_all_ports_ready(ips=node_ip_list, port=LOCAL_PORT, timeout=LOCAL_TIMEOUT):
        raise RuntimeError("Failed to wait for all nodes to be ready")

    # set env var
    set_environment_variables(config.get("router_envs"))

    router_args = config["router_args"]
    # router server params
    router_command = [
        "python3",
        "-u",
        "-m",
        "sglang_router.launch_router",
        "--host",
        "127.0.0.1",
        "--port",
        SERVICE_PORT,
        "--pd-disaggregation",
        "--policy",
        "cache_aware",
        *[str(x) for x in router_args],
    ]

    for index, url in enumerate(prefill_url):
        router_command.append("--prefill")
        router_command.append(f"http://{url}")
        router_command.append(f"{bootstrap_ports[index]}")

    for url in decode_url:
        router_command.append("--decode")
        router_command.append("http://" + url)
    router_command_str = " ".join(router_command)
    print(f"Starting router with command: {router_command_str}")
    try:
        router_process = subprocess.Popen(router_command_str, shell=True)
        print(f"Router process started with PID: {router_process.pid}")
    except Exception as e:
        raise RuntimeError(f"Failed to start router process: {e}")

def wait_server_ready(url, timeout=LOCAL_TIMEOUT):
    print(f"Waiting for the server to start...")
    start_time = time.perf_counter()
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"Server {url} is ready!")
                return
        except Exception:
            pass

        if time.perf_counter() - start_time > timeout:
            raise RuntimeError(f"Server {url} failed to start in {timeout}s")
        time.sleep(10)

def run_bench_serving(host, port, model_path=None, backend="sglang", dataset_name=None, request_rate=None,
                      max_concurrency=None, num_prompts=None, input_len=None, output_len=None, random_range_ratio=1,
                      dataset_path=None):
    cmd_args = ["python3", "-m", "sglang.bench_serving", "--host", host, "--port", str(port),
                "--model", model_path, "--backend", backend]

    if dataset_name:
        cmd_args.extend(["--dataset-name", str(dataset_name)])
    if dataset_path:
        cmd_args.extend(["--dataset-path", str(dataset_path)])
    if request_rate:
        cmd_args.extend(["--request-rate", str(request_rate)])
    if max_concurrency:
        cmd_args.extend(["--max-concurrency", str(max_concurrency)])
    if num_prompts:
        cmd_args.extend(["--num-prompts", str(num_prompts)])
    if input_len:
        cmd_args.extend(["--random-input-len", str(input_len)])
    if output_len:
        cmd_args.extend(["--random-output-len", str(output_len)])
    if random_range_ratio:
        cmd_args.extend(["--random-range-ratio", str(random_range_ratio)])

    result_file = os.getenv("METRICS_DATA_FILE")
    result_file = "./bench_log.txt" if not result_file else result_file
    print(f"The metrics result file: {result_file}")
    run_command(f"pip list | grep -E 'sglang|sgl|torch|transformers|deep-ep|memfabric_hybrid' | tee {result_file}")
    cann_info="/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info"
    run_command(f"echo \"CANN: $(cat {cann_info} | grep '^version=')\" | tee -a {result_file}")

    command = " " .join(cmd_args)
    print(f"Command: {command}")

    metrics = run_command(f"{command} | tee -a {result_file}")
    print(f"metrics is {metrics}")

    mean_ttft = run_command(f"grep 'Mean TTFT' {result_file} | awk '{{print $4}}'")
    mean_tpot = run_command(f"grep 'Mean TPOT' {result_file} | awk '{{print $4}}'")
    total_tps = run_command(f"grep 'Output token throughput' {result_file} | awk '{{print $5}}'")

    return {
        'mean_ttft': mean_ttft,
        'mean_tpot': mean_tpot,
        'total_tps': total_tps
    }

class TestPerformanceTestCaseBase(CustomTestCase):
    model = None
    backend = "sglang"
    dataset_name = None
    dataset_path = None
    other_args = None
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 10
    envs = None
    request_rate = None
    max_concurrency = 8
    num_prompts = int(max_concurrency) * 4
    input_len = None
    output_len = None
    random_range_ratio = None
    ttft = None
    tpot = None
    output_token_throughput = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        if cls.envs:
            for key, value in cls.envs.items():
                print(f"ENV_VAR_CASE {key}:{value}")
        env = os.environ.copy()
        for key, value in cls.envs.items():
            print(f"ENV_VAR_OTHER {key}:{value}")
        env.update(cls.envs)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout,
            other_args=cls.other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_throughput(self, run_cycles=2):
        _, host, port = self.base_url.split(":")
        host = host[2:]
        bench_params = {
            'host': host,
            'port': port,
            'model_path': self.model,
            'backend': self.backend,
            'dataset_name': self.dataset_name,
            'request_rate': self.request_rate,
            'max_concurrency': self.max_concurrency,
            'num_prompts': self.num_prompts,
            'input_len': self.input_len,
            'output_len': self.output_len,
            'random_range_ratio': self.random_range_ratio,
            'dataset_path': self.dataset_path,
        }
        print(f"Starting benchmark with parameters: {bench_params}")

        metrics = None
        for i in range(run_cycles):
            print(f"Running benchmark, {i + 1}/{run_cycles}")
            metrics = run_bench_serving(**bench_params)

        if self.tpot:
            self.assertLessEqual(
                float(metrics['mean_tpot']),
                self.tpot + 1 if self.tpot < 50 else self.tpot * 1.02,
            )
        if self.output_token_throughput:
            self.assertGreaterEqual(
                float(metrics['total_tps']),
                self.output_token_throughput * 0.98,
            )
        if self.ttft:
            self.assertLessEqual(
                float(metrics['mean_ttft']),
                self.ttft * 1.02,
            )

class TestMultiNodePdMixTestCaseBase(CustomTestCase):
    model_config = None
    backend = "sglang"
    dataset_name = None
    dataset_path = None
    request_rate = None
    max_concurrency = None
    num_prompts = None
    input_len = None
    output_len = None
    random_range_ratio = None
    ttft = None
    tpot = None
    output_token_throughput = None

    @classmethod
    def setUpClass(cls):
        cls.local_ip = os.getenv("POD_IP")
        hostname = os.getenv("HOSTNAME")
        cls.role = "master" if hostname.endswith("sglang-node-0") else "worker"
        print(f"Init {cls.local_ip} {cls.role=}!")

    def run_throughput(self, run_cycles=2):
        sglang_thread = threading.Thread(
            target=launch_pd_mix_node, args=(self.model_config,)
        )
        sglang_thread.start()

        if self.role == "master":
            master_node_ip = os.getenv("POD_IP")
            wait_server_ready(f"http://{master_node_ip}:{SERVICE_PORT}" + "/health")
            print(f"Wait 120s, starting run benchmark ......")
            time.sleep(120)

            bench_params = {
                'host': master_node_ip,
                'port': SERVICE_PORT,
                'model_path': self.model_config.get("model_path"),
                'backend': self.backend,
                'dataset_name': self.dataset_name,
                'request_rate': self.request_rate,
                'max_concurrency': self.max_concurrency,
                'num_prompts': self.num_prompts,
                'input_len': self.input_len,
                'output_len': self.output_len,
                'random_range_ratio': self.random_range_ratio,
            }
            print(f"Starting benchmark with parameters: {bench_params}")

            metrics = None
            for i in range(run_cycles):
                print(f"Running benchmark, {i+1}/{run_cycles}")
                metrics = run_bench_serving(**bench_params)

            if self.tpot:
                self.assertLessEqual(
                    float(metrics['mean_tpot']),
                    self.tpot + 1 if self.tpot < 50 else self.tpot * 1.02,
                )
            if self.output_token_throughput:
                self.assertGreaterEqual(
                    float(metrics['total_tps']),
                    self.output_token_throughput * 0.98,
                )
            if self.ttft:
                self.assertLessEqual(
                    float(metrics['mean_ttft']),
                    self.ttft * 1.02,
                )
        else:
            print("Worker node is running.")
            time.sleep(LOCAL_TIMEOUT)

class TestAscendMultiNodePdSepTestCaseBase(CustomTestCase):
    model_config = None
    backend = "sglang"
    dataset_name = None
    request_rate = None
    max_concurrency = None
    num_prompts = None
    input_len = None
    output_len = None
    random_range_ratio = 1
    ttft = None
    tpot = None
    output_token_throughput = None

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.local_ip = os.getenv("POD_IP")
        hostname = os.getenv("HOSTNAME")
        cls.role = "router" if "router" in hostname else "prefill" if "prefill" in hostname else "decode"
        print(f"Init {cls.local_ip} {cls.role=}!")

    def run_throughput(self, run_cycles=2):
        if self.role == "router":
            print(f"Starting router in thread...")
            router_thread = threading.Thread(
                target=launch_router, args=(self.model_config,)
            )
            router_thread.start()

            health_check_url = f"http://127.0.0.1:{SERVICE_PORT}/health"
            print(f"Waiting for router to be ready at {health_check_url}")
            wait_server_ready(health_check_url)

            print(f"Waiting 120 seconds for the server to fully initialize...")
            time.sleep(120)

            bench_params = {
                'host': "127.0.0.1",
                'port': SERVICE_PORT,
                'model_path': self.model_config.get("model_path"),
                'backend': self.backend,
                'dataset_name': self.dataset_name,
                'request_rate': self.request_rate,
                'max_concurrency': self.max_concurrency,
                'num_prompts': self.num_prompts,
                'input_len': self.input_len,
                'output_len': self.output_len,
                'random_range_ratio': self.random_range_ratio,
            }
            print(f"Starting benchmark with parameters: {bench_params}")

            metrics = None
            for i in range(run_cycles):
                print(f"Running benchmark, {i+1}/{run_cycles}")
                metrics = run_bench_serving(**bench_params)

            if self.tpot:
                self.assertLessEqual(
                    float(metrics['mean_tpot']),
                    self.tpot + 1 if self.tpot < 50 else self.tpot * 1.02,
                )
            if self.output_token_throughput:
                self.assertGreaterEqual(
                    float(metrics['total_tps']),
                    self.output_token_throughput * 0.98,
                )
            if self.ttft:
                self.assertLessEqual(
                    float(metrics['mean_ttft']),
                    self.ttft * 1.02,
                )
        else:
            # launch p/d node
            sglang_thread = threading.Thread(
                target=launch_pd_seperation_node, args=(self.model_config,)
            )
            sglang_thread.start()
            keep_alive_time = LOCAL_TIMEOUT * 2
            print(f"{self.role} node started, keeping test alive for {keep_alive_time} seconds")
            time.sleep(keep_alive_time)
