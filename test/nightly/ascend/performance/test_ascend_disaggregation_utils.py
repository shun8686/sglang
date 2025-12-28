import os
import socket
import subprocess
import threading
import time
import requests

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from sglang.test.test_utils import CustomTestCase, popen_launch_server

from test_ascend_single_mix_utils import run_bench_serving


KUBE_CONFIG = os.environ.get('KUBECONFIG')
NAMESPACE = os.environ.get('NAMESPACE')
CONFIGMAP_NAME = os.environ.get('KUBE_CONFIG_MAP')
LOCAL_TIMEOUT = 3600
SERVICE_PORT = "6688"

config.load_kube_config(KUBE_CONFIG)
v1 = client.CoreV1Api()

def checkout_port(host, port, timeout=3):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

# query configmap
def query_configmap(name, namespace):
    try:
        configmap = v1.read_namespaced_config_map(name, namespace)
        print(f"query_configmap successfully!")
        return configmap
    except ApiException as e:
        print(f"query_configmap error {e=}")
        return None

# get node count from k8s
def discover_worker_nodes():
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    prefill_pods = v1.list_namespaced_pod(
        namespace=NAMESPACE, label_selector="volcano.sh/task-spec=sglang-prefill"
    )
    decode_pods = v1.list_namespaced_pod(
        namespace=NAMESPACE, label_selector="volcano.sh/task-spec=sglang-decode"
    )
    nodes_count = len(prefill_pods.items) + len(decode_pods.items)
    return nodes_count


# launch router
def launch_router(config):
    print(f"launch_router start ......")
    nodes_count = discover_worker_nodes()
    print(f"launch_router nodes_count {nodes_count=}")

    # monitor  to generate p/d url
    prefill_url = []
    decode_url = []
    bootstrap_ports = []
    node_ip_list = []
    is_prefill_instance_multi_node = True if "--node-rank" not in config["prefill_args"] else False
    is_decode_instance_multi_node = True if "--node-rank" not in config["decode_args"] else False

    is_ready = False
    bootstrap_init_port = 8995
    while not is_ready:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if configmap.data is None:
            print(f"configmap is None, wait for 15s ......")
            time.sleep(15)
            continue
        print(f"launch_router query_configmap {configmap.data=}")
        for pod_name in configmap.data:
            pod_ip = configmap.data[pod_name]
            pod_index = int(pod_name.rsplit("-", 1)[-1])
            prefill_keyword = "prefill-0" if is_prefill_instance_multi_node else "prefill"
            if prefill_keyword in pod_name:
                prefill_url.append(f"{pod_ip}:8000")
                bootstrap_port = bootstrap_init_port \
                    if is_prefill_instance_multi_node else bootstrap_init_port + pod_index
                bootstrap_ports.append(str(bootstrap_port))
                node_ip_list.append(pod_ip)
            decode_keyword = "decode-0" if is_decode_instance_multi_node else "decode"
            if decode_keyword in pod_name:
                decode_url.append(f"{pod_ip}:8000")
                node_ip_list.append(pod_ip)
        is_ready = True
    print(
        f"monitor configmap end, {prefill_url=} {decode_url=} {bootstrap_ports=} {node_ip_list=}"
    )

    # checkout all node port ready
    while True:
        success_nodes = 0
        port = 8000
        for ip in node_ip_list:
            if checkout_port(ip, port):
                print(f"{ip=} {port} is ready")
                success_nodes = success_nodes + 1
            else:
                print(f"{ip=} {port} is not ready")
        if success_nodes == len(node_ip_list):
            print(f"launch_router all node port are ready!")
            break
        time.sleep(15)

    # set env var
    for key, value in config["router_envs"].items():
        print(f"ENV_VAR {key}={value}")
        os.environ[key] = value

    # router server params
    router_command = [
        "python3",
        "-u",
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--policy",
        "cache_aware"
        "--host",
        "127.0.0.1",
        "--port",
        SERVICE_PORT,
    ]

    for index, url in enumerate(prefill_url):
        router_command.append("--prefill")
        router_command.append(f"http://{url}")
        router_command.append(f"{bootstrap_ports[index]}")

    for url in decode_url:
        router_command.append("--decode")
        router_command.append("http://" + url)
    router_command_str = " ".join(router_command)
    print(f"Starting router, {router_command_str=}")
    subprocess.Popen(router_command_str, shell=True)


# launch p/d node
def launch_node(config):
    print(f"launch_node start ......")
    node_ip = os.getenv("POD_IP")
    hostname = os.getenv("HOSTNAME")
    pod_index = int(hostname.rsplit("-", 1)[-1])
    role = "prefill" if "prefill" in hostname else "decode"
    bootstrap_init_port = 8995
    master_prefill_ip = None
    master_decode_ip = None
    is_prefill_instance_multi_node = True if "--node-rank" not in config["prefill_args"] else False
    is_decode_instance_multi_node = True if "--node-rank" not in config["decode_args"] else False

    # monitor configmap ready
    is_ready = False
    while not is_ready:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if configmap.data is None:
            print(f"configmap is None, wait for 15s ......")
            time.sleep(15)
            continue

        print(f"monitor {configmap.data=}")
        for pod_name in configmap.data:
            pod_ip = configmap.data[pod_name]
            if str(pod_name).endswith("prefill-0"):
                master_prefill_ip = pod_ip
            if str(pod_name).endswith("decode-0"):
                master_decode_ip = pod_ip

        if not master_prefill_ip or not master_decode_ip:
            print(f"Can not get the master node of prefill or decode...retry...")
            continue
        is_ready = True

    # generate p/d run command
    common_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--disaggregation-transfer-backend",
        "ascend",
    ]
    service_args = []
    service_args.extend(common_args)

    mf_addr = f"tcp://{master_prefill_ip}:24666"
    os.environ["ASCEND_MF_STORE_URL"] = mf_addr
    print(f"ENV_VAR ASCEND_MF_STORE_URL={mf_addr}")

    if role == "prefill":
        # Current node is prefill
        dist_init_addr = f"{master_prefill_ip}:5000"
        print(f"launch prefill node {dist_init_addr=}")

        for key, value in config["prefill_envs"].items():
            print(f"ENV_VAR {key}={value}")
            os.environ[key] = value

        prefill_args = config["prefill_args"]
        if is_prefill_instance_multi_node:
            print("No node-rank specified and all prefill node will form a single instance.")
            prefill_args.extend(
                [
                    "--node-rank",
                    pod_index,
                    "--dist-init-addr",
                    dist_init_addr,
                    "--disaggregation-bootstrap-port",
                    bootstrap_init_port,
                ]
            )
        else:
            print("Node-rank specified and each prefill node is a instance.")
            prefill_args.extend(
                [
                    "--disaggregation-bootstrap-port",
                    bootstrap_init_port + pod_index,
                ]
            )

        service_args.extend(prefill_args)

    if role == "decode":
        dist_init_addr = f"{master_decode_ip}:5000"
        print(f"launch decode node {dist_init_addr=}")

        for key, value in config["decode_envs"].items():
            print(f"ENV_VAR {key}={value}")
            os.environ[key] = value

        decode_args = config["decode_args"]
        if is_decode_instance_multi_node:
            print("No node-rank specified and all decode node will form a single instance.")
            decode_args.extend(
                [
                    "--dist-init-addr",
                    dist_init_addr,
                    "--node-rank",
                    pod_index,
                ]
            )
        else:
            print("Node-rank specified and each decode node is a instance.")

        service_args.extend(decode_args)

    print(f"Starting node, {node_ip=} {service_args=}")
    return popen_launch_server(
        config["model_path"],
        f"http://{node_ip}:{8000}",
        timeout=LOCAL_TIMEOUT,
        other_args=[
            *service_args,
        ],
    )


def wait_router_ready(url, timeout=LOCAL_TIMEOUT):
    start_time = time.perf_counter()
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"Router {url} is ready!")
                return
        except Exception:
            pass

        if time.perf_counter() - start_time > timeout:
            raise RuntimeError(f"Server {url} failed to start in {timeout}s")
        time.sleep(10)


class TestAscendDisaggregationUtils(CustomTestCase):
    model_config = None
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
    metrics_data_file = os.getenv("METRICS_DATA_FILE")

    @classmethod
    def setUpClass(cls):
        cls.process = None
        cls.local_ip = os.getenv("POD_IP")
        hostname = os.getenv("HOSTNAME")
        cls.role = "router" if "router" in hostname else None
        print(f"Init {cls.local_ip} {cls.role=}!")

    def run_throughput(self, retry=True):
        if self.role == "router":
            router_thread = threading.Thread(
                target=launch_router, args=(self.model_config,)
            )
            router_thread.start()
            wait_router_ready(f"http://127.0.0.1:{SERVICE_PORT}" + "/health")

            print(f"Wait 120, starting run benchmark ......")
            time.sleep(120)

            metrics = run_bench_serving(
                host="127.0.0.1",
                port=SERVICE_PORT,
                model_path=self.model_config.get("model_path"),
                dataset_name=self.dataset_name,
                request_rate=self.request_rate,
                max_concurrency=self.max_concurrency,
                num_prompts=self.num_prompts,
                input_len=self.input_len,
                output_len=self.output_len,
                random_range_ratio=self.random_range_ratio,
                result_file=self.metrics_data_file,
            )
            if retry:
                metrics = run_bench_serving(
                    host="127.0.0.1",
                    port=SERVICE_PORT,
                    model_path=self.model_config.get("model_path"),
                    dataset_name=self.dataset_name,
                    request_rate=self.request_rate,
                    max_concurrency=self.max_concurrency,
                    num_prompts=self.num_prompts,
                    input_len=self.input_len,
                    output_len=self.output_len,
                    random_range_ratio=self.random_range_ratio,
                    result_file=self.metrics_data_file,
                )
            if self.tpot:
                self.assertLessEqual(
                    float(metrics['mean_tpot']),
                    self.tpot * 1.02,
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
                target=launch_node, args=(self.model_config,)
            )
            sglang_thread.start()
            time.sleep(LOCAL_TIMEOUT)
