import re
import signal
import string
import subprocess
import sys
import time
import os
import argparse
import uuid
import random
from jinja2 import Template

import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

KUBE_CONFIG = os.environ.get('KUBECONFIG')

config.load_kube_config(KUBE_CONFIG)
core_api = client.CoreV1Api()
custom_api = client.CustomObjectsApi()
batch_api = client.BatchV1Api()
rbac_api = client.RbacAuthorizationV1Api()

LOCAL_TIMEOUT = 10800
KUBE_NAME_SPACE = os.environ.get('NAMESPACE')
KUBE_CONFIG_MAP = os.environ.get('KUBE_CONFIG_MAP')
KUBE_JOB_TYPE = os.environ.get('KUBE_JOB_TYPE')
KUBE_JOB_NAME = os.environ.get('KUBE_JOB_NAME')
MONITOR_POD_NAME = "{}-pod-0".format(KUBE_JOB_NAME) if KUBE_JOB_TYPE == "single" else \
    "{}-sglang-node-0".format(KUBE_JOB_NAME) if KUBE_JOB_TYPE == "multi" else \
    "{}-sglang-router-0".format(KUBE_JOB_NAME)
KUBE_YAML_TEMPLATE = "k8s_single.yaml.jinja2" if KUBE_JOB_TYPE == "single" else "k8s_multi_pd_mix.yaml.jinja2" if KUBE_JOB_TYPE == "multi" else "k8s_multi_pd_separation.yaml.jinja2"


def get_unique_random_string(length: int = 16, add_random: bool = True) -> str:
    uuid_str = str(uuid.uuid4()).replace("-", "")

    if add_random:
        if length < 8:
            raise ValueError("length can not be smaller than 8")
        random_length = length - 8
        char_pool = string.ascii_letters + string.digits
        random_chars = ''.join([random.choice(char_pool) for _ in range(random_length)])
        result = uuid_str[:8] + random_chars
    else:
        result = uuid_str[:length]

    return result

def create_pod_yaml(output_yaml, pod_context):
    with open(KUBE_YAML_TEMPLATE, 'r') as f:
        template = Template(f.read())
    kube_pod_yaml = template.render(pod_context)
    with open(output_yaml, 'w') as f:
        f.write(kube_pod_yaml)
    print(f"Pod YAML written to {output_yaml}")

def create_pod(yaml_file, namespace):
    with open(yaml_file, "r", encoding="utf-8") as f:
        yaml_docs = list(yaml.safe_load_all(f))

    for doc in yaml_docs:
        if not doc:
            continue

        kind = doc.get("kind")
        api_version = doc.get("apiVersion")

        try:
            if kind == "Pod" and api_version == "v1":
                core_api.create_namespaced_pod(namespace=namespace, body=doc)
                print(f"Pod {doc['metadata']['name']} is created")

            elif kind == "Job" and api_version == "batch/v1":
                batch_api.create_namespaced_job(namespace=namespace, body=doc)
                print(f"Job {doc['metadata']['name']} is created")

            elif kind == "Job" and api_version == "batch.volcano.sh/v1alpha1":
                response = custom_api.create_namespaced_custom_object(
                    group="batch.volcano.sh",
                    version="v1alpha1",
                    namespace=namespace,
                    plural="jobs",
                    body=doc
                )
                print(f"Volcano Job {doc['metadata']['name']} is created")
                print(f"Response info: {response['metadata']['name']}")

            elif kind == "ConfigMap" and api_version == "v1":
                core_api.create_namespaced_config_map(namespace=namespace, body=doc)
                print(f"ConfigMap {doc['metadata']['name']} is created")

            elif kind == "Role" and api_version == "rbac.authorization.k8s.io/v1":
                rbac_api.create_namespaced_role(
                    namespace=namespace,
                    body=doc
                )
                print(f"Role {doc['metadata']['name']} is created")

            elif kind == "RoleBinding" and api_version == "rbac.authorization.k8s.io/v1":
                rbac_api.create_namespaced_role_binding(
                    namespace=namespace,
                    body=doc
                )
                print(f"RoleBinding {doc['metadata']['name']} is created")

            else:
                raise f"Unrecognized kind: {kind}/{api_version}"
        except ApiException as e:
            print(f"create resource {kind} error: {e}")
            raise

def delete_pod(yaml_file, namespace):
    with open(yaml_file, "r", encoding="utf-8") as f:
        yaml_docs = list(yaml.safe_load_all(f))
    for doc in yaml_docs:
        if not doc:
            continue

        kind = doc.get("kind")
        api_version = doc.get("apiVersion")
        try:
            if kind == "Job" and api_version == "batch.volcano.sh/v1alpha1":
                job_name = doc["metadata"]["name"]
                response = custom_api.delete_namespaced_custom_object(
                    group="batch.volcano.sh",
                    version="v1alpha1",
                    namespace=namespace,
                    plural="jobs",
                    name=job_name,
                    body=client.V1DeleteOptions(
                        grace_period_seconds=0,
                        propagation_policy="Foreground"
                    )
                )
                print(f"Volcano Job {job_name} is deleted.")
                print(f"Response status: {response.get('status')}")
            elif kind == "ConfigMap" and api_version == "v1":
                config_map_name = doc["metadata"]["name"]
                response = core_api.delete_namespaced_config_map(name=config_map_name, namespace=namespace)
                print(f"ConfigMap {config_map_name} is deleted.")
                print(f"Response status: {response.get('status')}")
            else:
                raise f"Unrecognized kind: {kind}/{api_version}"
        except ApiException as e:
            raise f"delete resource {kind} error: {e}"

def check_pods_ready(timeout=300):
    print("Waiting all pods to running...")
    matching_string = "{}".format(os.environ.get('KUBE_JOB_NAME'))
    start_time = time.time()

    while time.time() - start_time < timeout:
        pods = core_api.list_namespaced_pod(namespace=KUBE_NAME_SPACE)

        if len(pods.items) == 0:
            time.sleep(5)
            continue

        all_running = True
        sglang_pods_found = False
        for pod in pods.items:
            pod_name = pod.metadata.name
            if matching_string not in pod_name:
                continue

            sglang_pods_found = True
            status = pod.status
            phase = status.phase
            print(f"Pod: {pod_name}, status: {phase}")
            if phase != "Running":
                all_running = False
                break

            containers_ready = True
            for condition in status.conditions:
                if condition.type == "Ready" and condition.status != "True":
                    containers_ready = False
                    break

            if not containers_ready:
                all_running = False
                break

        if not sglang_pods_found:
            print("No sglang pod, waiting...")
            time.sleep(5)
            continue
        if all_running:
            print("All sglang Pod is Running !")
            return True

        time.sleep(5)

    print(f"timeout in {timeout}s")
    return False

def create_or_update_configmap(cm_name: str, data: dict, namespace: str):
    cm_metadata = client.V1ObjectMeta(name=cm_name, namespace=namespace)
    configmap = client.V1ConfigMap(
        api_version="v1",
        kind="ConfigMap",
        metadata=cm_metadata,
        data=data)

    try:
        response = core_api.create_namespaced_config_map(
            namespace=namespace,
            body=configmap
        )
        print(f"ConfigMap '{cm_name}' create successfully!")
        print(f"data: {list(data.keys())}")
        return response
    except ApiException as e:
        if e.status == 409:
            print(f"ConfigMap {cm_name} already exists. Updating...")
            response = core_api.replace_namespaced_config_map(
                namespace=namespace,
                name=cm_name,
                body=configmap
            )
            print(f"ConfigMap {cm_name} updated successfully.")
            return response
        else:
            error_msg = f"ConfigMap create failed: {e.reason}"
            if e.body:
                error_msg += f" | details: {e.body}"
            print(error_msg)
            raise

def prepare_cm_data(pod_string):
    pods = core_api.list_namespaced_pod(namespace=KUBE_NAME_SPACE)
    data = {}
    for pod in pods.items:
        pod_name = pod.metadata.name
        if pod_string in pod_name:
            pod_ip = pod.status.pod_ip
            data[pod_name] = pod_ip
    return data

def monitor_pod_logs(pod_name, namespace=None, timeout=None):
    class TimeoutException(Exception):
        """Custom exception for timeout events"""

        pass

    def timeout_handler(signum, frame):
        """Signal handler for timeout events"""
        raise TimeoutException("Monitoring timeout")

    # Build kubectl command
    cmd = ["kubectl", "logs", "-f", pod_name]
    if namespace:
        cmd.extend(["-n", namespace])

    # Define multiline pattern to match
    pattern_lines = [r"^-{70,}$", r"^Ran \d+ test in [\d.]+s$", r"^$", r"^OK$"]

    # Compile regex patterns
    patterns = [re.compile(line_pattern) for line_pattern in pattern_lines]

    # Set up timeout handling
    if timeout:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

    process = None
    try:
        # Start kubectl logs process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        )

        print(f"Starting to monitor logs for Pod: {pod_name}")
        if namespace:
            print(f"Namespace: {namespace}")
        if timeout:
            print(f"Timeout set to: {timeout} seconds")
        match_state = 0
        matched = False

        # Process output
        while process.poll() is None and not matched:
            line = process.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            line = line.rstrip("\n")
            print(line)
            # Check if current line matches expected pattern
            if match_state < len(patterns) and patterns[match_state].match(line):
                match_state += 1
                if match_state == len(patterns):
                    matched = True
                    print("\nSuccessfully detected complete test completion pattern!")
            else:
                match_state = 0
                if patterns[0].match(line):
                    match_state = 1

        # Check if pattern was successfully matched
        if not matched:
            if process.poll() is not None:
                remaining_output, stderr_output = process.communicate()
                if remaining_output:
                    print(remaining_output)
                if stderr_output:
                    raise Exception(f"kubectl command error: {stderr_output}")
                else:
                    raise Exception(
                        "Pod logs ended but target pattern was not detected"
                    )
            else:
                raise Exception("Monitoring ended but target pattern was not detected")
        print("Monitoring completed successfully. Script exiting.")

    except TimeoutException:
        print(f"\nError: Target pattern not detected within {timeout} seconds")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nError: Monitoring interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        if timeout:
            signal.alarm(0)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Apply k8s yaml",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Docker image to use",
    )

    parser.add_argument(
        "--prefill-size",
        type=int,
        required=True,
        help="Number of prefill nodes",
    )

    parser.add_argument(
        "--decode-size",
        type=int,
        required=True,
        help="Number of decode nodes",
    )

    parser.add_argument(
        "--router-size",
        type=int,
        required=True,
        help="Number of router nodes",
    )

    parser.add_argument(
        "--sglang-source-path",
        type=str,
        required=True,
        help="Sglang source code path on shared-disk",
    )

    parser.add_argument(
        "--metrics-data-file",
        type=str,
        required=False,
        default=os.environ.get('METRICS_DATA_FILE'),
        help="Metrics data file",
    )

    parser.add_argument(
        "--test-case",
        type=str,
        required=True,
        help="Test case path",
    )

    parser.add_argument(
        "--sglang-is-in-ci",
        type=bool,
        required=False,
        default=os.environ.get('SGLANG_IS_IN_CI'),
        help="Is in CI",
    )

    parser.add_argument(
        "--install-sglang-from-source",
        type=bool,
        required=False,
        default=os.environ.get('INSTALL_SGLANG_FROM_SOURCE'),
        help="Install sglang from source",
    )

    args = parser.parse_args()

    pd_separation_context = {
        "image": args.image,
        "name_space": KUBE_NAME_SPACE,
        "kube_job_name": KUBE_JOB_NAME,
        "kube_config": KUBE_CONFIG,
        "kube_config_map": KUBE_CONFIG_MAP,
        "prefill_size": args.prefill_size,
        "decode_size": args.decode_size,
        "router_size": args.router_size,
        "sglang_source_path": args.sglang_source_path,
        "metrics_data_file": args.metrics_data_file,
        "test_case": args.test_case,
        "sglang_is_in_ci": args.sglang_is_in_ci,
        "install_sglang_from_source": args.install_sglang_from_source,
    }

    kube_yaml_file = os.environ.get('KUBE_YAML_FILE')
    if not kube_yaml_file:
        random_str = get_unique_random_string(16, True)
        kube_yaml_file = f"k8s_single_{random_str}.yaml" if KUBE_JOB_TYPE == "single" else \
            f"k8s_multi_pd_mix_{random_str}.yaml" if KUBE_JOB_TYPE == "multi" else \
                f"k8s_multi_pd_separation_{random_str}.yaml"

    try:
        print("Apply k8s yaml... KUBE_NAME_SPACE:{}, KUBE_CONFIG_MAP:{}, KUBE_JOB_TYPE:{}, KUBE_YAML_FILE:{}"
              .format(KUBE_NAME_SPACE, KUBE_CONFIG_MAP, KUBE_JOB_TYPE, kube_yaml_file))
        create_pod_yaml(output_yaml=kube_yaml_file, pod_context=pd_separation_context)
        create_pod(yaml_file=kube_yaml_file, namespace=KUBE_NAME_SPACE)

        if check_pods_ready(timeout=LOCAL_TIMEOUT):
            if KUBE_JOB_TYPE != "single":
                matching_pod_string = os.environ.get('KUBE_JOB_NAME')
                cm_data = prepare_cm_data(matching_pod_string)
                if not cm_data:
                    print(f"No sglang pod found while matching {matching_pod_string}")

                response = create_or_update_configmap(cm_name=KUBE_CONFIG_MAP, data=cm_data, namespace=KUBE_NAME_SPACE)
                print(response)
        else:
            print("Pod not ready, maybe not enough resource")

        monitor_pod_logs(MONITOR_POD_NAME, KUBE_NAME_SPACE, LOCAL_TIMEOUT)

    except Exception as e:
        print(e)
    finally:
        delete_pod(yaml_file=kube_yaml_file, namespace=KUBE_NAME_SPACE)
