import os
import socket
import subprocess
import threading
import time
import requests

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from sglang.test.test_utils import CustomTestCase, popen_launch_server

from test_ascend_performance_utils import run_bench_serving


KUBE_CONFIG = os.environ.get('KUBECONFIG')
NAMESPACE = os.environ.get('NAMESPACE')
CONFIGMAP_NAME = os.environ.get('KUBE_CONFIG_MAP')
LOCAL_TIMEOUT = 1800
SERVICE_PORT = "6688"
LOCAL_PORT = "8000"













