#!/bin/bash

CONCURRENCY=1
kube_job_type=multi-pd-separation

sglang_source_path=$1
install_sglang_from_source=$2
image=$3

if [ -z "$sglang_source_path" ];then
  sglang_source_path=/home/d00662834/dev-0210/sglang
fi
if [ -z "$install_sglang_from_source" ];then
  install_sglang_from_source=false
fi
if [ -z "$image" ];then
  image=swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann8.5.0-a3-B092
fi
echo "image: $image"

test_set=$(cat "${sglang_source_path}/test/manual/ascend/performance/temp/testcase_v26_baseline_multi_pd_separation.txt")

count=0
for tc_info in $test_set
do
  prefill_size=$(echo $tc_info | cut -d'|' -f1)
  decode_size=$(echo $tc_info | cut -d'|' -f2)
  test_case=$(echo "$tc_info" | cut -d'|' -f3)
  echo "Testcase: $test_case"
   
  bash run_k8s_test_base.sh $sglang_source_path $test_case $image $install_sglang_from_source $kube_job_type $prefill_size $decode_size > log/${test_case##*/}.log 2>&1 &
  sleep 30

  count=$((count + 1))
   
  if [ "$count" -ge "$CONCURRENCY" ]; then
    wait
    count=0
  fi
done

wait

echo "所有测试执行完成"
for tc_info in $test_set
do
  test_case=$(echo "$tc_info" | cut -d'|' -f2)
  ok_num=$(cat log/${test_case##*/}.log | grep "^OK$" | wc -l)
  if [ "${ok_num}" -ne 1 ];then
    echo "RUN FAILED: ${test_case}"
  fi
done
