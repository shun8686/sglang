import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from sglang.test.ascend.e2e.run_ascend_ci import run_ascend_e2e_test_case

sglang_source_path = "/data/d00662834/dev-0210/sglang"
docker_image_url = "swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann8.5.0-a3-B025"
kube_name_space = "sglang-multi-debug"
kube_job_type = "multi-pd-separation"
kube_job_name_prefix = "sglang-multi-debug"
metrics_data_file = ""
sglang_is_in_ci = False
install_sglang_from_source = False
env = "debug"
concurrency = 1

TEST_SUITE = [
    {
        "testcase": "test/manual/ascend/temp/_test_ascend_deepseek_r1_w4a8_1p1d_16p_function_test.py",
        "resource": {
            "prefill_size": 1,
            "decode_size": 1,
            "router_size": 1
        }
    }
]

def concurrent_run_test_cases(
    test_cases_params: List[Dict[str, Any]],
    concurrency: int = 3
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
                task_result = future.result()
                results.append(task_result)
                print(f"Progress: {completed_count}/{total_count} | Case {test_case} result collected")
            except Exception as e:
                # Catch exceptions during task submission/execution (e.g., parameter errors)
                error_result = {
                    "test_case": test_case,
                    "kube_job_type": params.get("kube_job_type"),
                    "namespace": params.get("kube_name_space"),
                    "status": "error",
                    "message": f"Task submission/execution exception: {str(e)}",
                    "resource_info": params.get("resource_info")
                }
                results.append(error_result)
                print(f"Progress: {completed_count}/{total_count} | Case {test_case} exception: {str(e)}")

    end_time = time.time()
    print(
        f"\nAll test cases completed! Total time: {end_time - start_time:.2f} seconds, Total cases: {total_count}, Concurrency level: {concurrency}")
    return results

if __name__ == "__main__":
    test_cases = list()
    for test_case in TEST_SUITE:
        test_case_info = {
            "docker_image_url": docker_image_url,
            "kube_name_space": kube_name_space,
            "kube_job_type": kube_job_type,
            "kube_job_name_prefix": kube_job_name_prefix,
            "resource_info": test_case.get("resource"),
            "sglang_source_path": sglang_source_path,
            "test_case": test_case.get("testcase"),
            "metrics_data_file": metrics_data_file,
            "sglang_is_in_ci": sglang_is_in_ci,
            "install_sglang_from_source": install_sglang_from_source,
            "env": env,
        }
        test_cases.append(test_case_info)
    all_results = concurrent_run_test_cases(test_cases, concurrency=concurrency)
