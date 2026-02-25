import argparse
import os.path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from sglang.test.ascend.e2e.run_ascend_ci import run_ascend_e2e_test_case

NFS_ROOT_PATH = "/data/ascend-ci-share-pkking-sglang"

TEST_SUITE = [
    {
        "testcase": "test/manual/ascend/temp/_test_ascend_deepseek_r1_w4a8_1p1d_16p_function_test.py",
        "resource": {"prefill_size": 1, "decode_size": 1, "router_size": 1},
    },
    {
        "testcase": "test/manual/ascend/temp/test_ascend_fim.py",
        "resource": {"prefill_size": 1, "decode_size": 1, "router_size": 1},
    },
]


def concurrent_run_test_cases(
    test_cases_params: List[Dict[str, Any]], concurrency: int = 3
) -> List[Dict[str, Any]]:
    """
    Execute multiple run_ascend_e2e_test_case test cases concurrently
    :param test_cases_params: List of test case parameters (each element is a dict containing all function parameters)
    :param concurrency: Concurrency level (number of test cases running simultaneously)
    :return: List of execution results for all test cases
    """
    results = []
    start_time = time.time()

    # Create thread pool (use ThreadPool for IO-bound tasks, replace with ProcessPoolExecutor for CPU-bound tasks)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # 1. Submit all test tasks to thread pool, associate task parameters with Future objects
        future_to_params = {}
        for idx, params in enumerate(test_cases_params):
            # executor.submit supports keyword argument passing, **params unpacks dict to function parameters
            future = executor.submit(run_ascend_e2e_test_case, **params)
            future_to_params[future] = params

        # 2. Iterate over completed tasks and collect results (in completion order)
        completed_count = 0
        total_count = len(test_cases_params)
        for future in as_completed(future_to_params):
            completed_count += 1
            params = future_to_params[future]
            test_case = params["test_case"]
            try:
                # Get task execution result
                future.result()
                results.append(
                    {
                        "test_case": test_case,
                        "result": "Pass",
                    }
                )
                print(f"Progress: {completed_count}/{total_count} | Case {test_case}")
            except Exception as e:
                # Catch exceptions during task submission/execution (e.g., parameter errors)
                error_result = {
                    "test_case": test_case,
                    "result": "Error",
                    "resource_info": params.get("resource_info"),
                    "kube_job_type": params.get("kube_job_type"),
                    "kube_name_space": params.get("kube_name_space"),
                    "message": f"Task submission/execution exception: {str(e)}",
                }
                results.append(error_result)
                print(
                    f"Progress: {completed_count}/{total_count} | Case {test_case} exception: {str(e)}"
                )

    end_time = time.time()
    pass_count = len([item for item in results if item["result"] == "Pass"])
    fail_count = len([item for item in results if item["result"] == "Fail"])
    error_count = len([item for item in results if item["result"] == "Error"])
    print(
        f"All test cases completed! Total time: {end_time - start_time:.2f} seconds, "
        f"Total cases: {total_count}, Concurrency level: {concurrency}, "
        f"Pass: {pass_count} | Fail: {fail_count} | Error: {error_count}"
    )

    not_pass_testcase = [
        item["test_case"] for item in results if item["result"] != "Pass"
    ]
    if not_pass_testcase:
        print("Not Passed Test Cases:")
        print("\n".join(str(item) for item in not_pass_testcase))
    else:
        print("All Pass.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run test case", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--testcase",
        type=str,
        required=True,
        help="Specify test case name to run which should be configured in TEST_SUITE",
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Docker image to use",
    )

    parser.add_argument(
        "--sglang-source-relative-path",
        type=str,
        required=True,
        help="Sglang source code relative path on shared-disk(NFS_ROOT_PATH: /data/ascend-ci-share-pkking-sglang/)",
    )

    parser.add_argument(
        "--sglang-is-in-ci",
        action="store_true",
        help="Used to set env var SGLANG_IS_IN_CI in pod",
    )

    parser.add_argument(
        "--install-sglang-from-source",
        action="store_true",
        help="Used to set env var INSTALL_SGLANG_FROM_SOURCE in pod",
    )

    parser.add_argument(
        "--kube-name-space",
        type=str,
        required=True,
        help="K8s name space",
    )

    parser.add_argument(
        "--kube-job-type",
        type=str,
        choices=["single", "multi-pd-mix", "multi-pd-separation"],
        required=True,
        help="K8s job type [single, multi-pd-mix, multi-pd-separation]",
    )

    parser.add_argument(
        "--kube-job-name-prefix",
        type=str,
        required=True,
        help="K8s job name prefix",
    )

    parser.add_argument(
        "--metrics-data-file",
        type=str,
        required=False,
        default=os.environ.get("METRICS_DATA_FILE"),
        help="Metrics data file",
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["debug", "ci"],
        required=True,
        help="Environment type",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        required=False,
        default=1,
        help="Concurrency level (number of test cases running simultaneously)",
    )

    args = parser.parse_args()
    specified_test_case = args.testcase

    docker_image_url = args.image
    sglang_source_relative_path = args.sglang_source_relative_path
    sglang_is_in_ci = args.sglang_is_in_ci
    install_sglang_from_source = args.install_sglang_from_source

    kube_name_space = args.kube_name_space
    kube_job_type = args.kube_job_type
    kube_job_name_prefix = args.kube_job_name_prefix

    metrics_data_file = args.metrics_data_file

    env = args.env
    concurrency = args.concurrency

    test_cases = list()
    for test_case in TEST_SUITE:
        if specified_test_case and test_case.get("testcase") != specified_test_case:
            continue
        else:
            test_case_info = {
                "docker_image_url": docker_image_url,
                "kube_name_space": kube_name_space,
                "kube_job_type": kube_job_type,
                "kube_job_name_prefix": kube_job_name_prefix,
                "resource_info": test_case.get("resource"),
                "sglang_source_relative_path": sglang_source_relative_path,
                "test_case": test_case.get("testcase"),
                "metrics_data_file": metrics_data_file,
                "sglang_is_in_ci": sglang_is_in_ci,
                "install_sglang_from_source": install_sglang_from_source,
                "env": env,
            }
            test_case_file = f"{NFS_ROOT_PATH}/{sglang_source_relative_path}/{test_case_info.get('test_case')}"
            if not os.path.exists(test_case_file):
                raise FileNotFoundError(f"{test_case_file} does not exist")

            test_cases.append(test_case_info)

    concurrent_run_test_cases(test_cases, concurrency=concurrency)
