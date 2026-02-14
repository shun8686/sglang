# k8s namespace can not be modified due to pvc configuration reasons
KUBE_NAME_SPACE = "sgl-project"

function prepare_env() {
    pip config set global.index-url "https://pypi.tuna.tsinghua.edu.cn/simple"
    pip config set global.trusted-host "pypi.tuna.tsinghua.edu.cn"
    pip3 install kubernetes

    # copy utils to image path if not exist
    sglang_pkg_path=$(pip show sglang | grep Location | awk '{print $2}')
    ascend_test_util_path=${sglang_pkg_path}/sglang/test/ascend/e2e
    rm -rf ${ascend_test_util_path}
    cp -r /data/ascend-ci-share-pkking-sglang/d00662834/dev-0210/sglang/python/sglang/test/ascend/e2e ${ascend_test_util_path}

    # kubectl
    cp /data/ascend-ci-share-pkking-sglang/d00662834/debug/k8s/arm/kubectl /usr/local/sbin/
}

prepare_env

export KUBECONFIG=/data/ascend-ci-share-pkking-sglang/.cache/kb.yaml

python3 ascend_e2e_test_suites.py \
    --env debug \
    --image swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann8.5.0-a3-B025 \
    --kube-name-space KUBE_NAME_SPACE \
    --kube-job-type multi-pd-separation \
    --kube-job-name-prefix sglang-multi-debug \
    --sglang-source-relative-path multi-node-test/dev-0210/sglang \
    --testcase test/manual/ascend/temp/_test_ascend_deepseek_r1_w4a8_1p1d_16p_function_test.py

#    --sglang-is-in-ci \
#    --install-sglang-from-source \


