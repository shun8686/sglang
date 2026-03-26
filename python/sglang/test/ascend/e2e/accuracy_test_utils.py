from kubernetes import client, config


def start_test(model, api_url):
    config.load_kube_config()
    api = client.BatchV1Api()

    job_name = "accuracy-test"
    namespace = "sgl-project"

    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=job_name),
        spec=client.V1JobSpec(
            backoff_limit=3,
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    restart_policy="OnFailure",
                    containers=[
                        client.V1Container(
                            name="job-container",
                            image="busybox:latest",
                            command=[
                                "bash",
                                "/root/sglang/python/sglang/test/ascend/e2e/run_npu_accuracy_test.sh",
                                model,
                                api_url,
                            ],
                        )
                    ],
                )
            ),
        ),
    )

    try:
        api_response = api.create_namespaced_job(namespace=namespace, body=job)
        print(f"Job create successfully {api_response.status}！")

    except Exception as e:
        print(f"Create failed: {e}")
