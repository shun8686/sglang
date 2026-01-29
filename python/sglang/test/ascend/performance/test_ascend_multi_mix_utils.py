import os
import time
import requests
import threading

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
)
from test_ascend_performance_utils import run_bench_serving






# query configmap






