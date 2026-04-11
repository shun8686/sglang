#!/bin/bash
set -e
export PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin:$PATH

# download test scripts
# git clone https://github.com/shun8686/sglang.git -b dev-0210
sglang_source_path=/root/.cache/d00662834/dev-0210/sglang

# replace test utils file in image
sglang_pkg_path=$(pip show sglang | grep Location | awk '{print $2}')
rm -rf ${sglang_pkg_path}/sglang/test/ascend
cp -r ${sglang_source_path}/python/sglang/test/ascend ${sglang_pkg_path}/sglang/test/ascend

# copy or download required file
cp /root/.cache/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
#curl -o /tmp/test.jsonl -L https://gh-proxy.test.osinfra.cn/https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
cp /root/.cache/modelscope/hub/datasets/grade_school_math/test.jsonl /tmp

exit 1



# set environment of cann
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

python3 testcase.py
