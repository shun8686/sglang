function prepare_env() {
    pip config set global.index-url "https://pypi.tuna.tsinghua.edu.cn/simple"
    pip config set global.trusted-host "pypi.tuna.tsinghua.edu.cn"
    pip3 install kubernetes

    # copy utils to image path if not exist
    sglang_pkg_path=$(pip show sglang | grep Location | awk '{print $2}')
    ascend_test_util_path=${sglang_pkg_path}/sglang/test/ascend/e2e
    rm -rf ${ascend_test_util_path}
    cp -r /data/d00662834/dev-0210/sglang/python/sglang/test/ascend/e2e ${ascend_test_util_path}

    # kubectl
    cp /data/d00662834/debug/k8s/arm/kubectl /usr/local/sbin/
}

prepare_env

export KUBECONFIG=/data/.cache/kb.yaml

python3 ascend_e2e_test_suites.py --testcase test/manual/ascend/temp/test_ascend_fim.py
