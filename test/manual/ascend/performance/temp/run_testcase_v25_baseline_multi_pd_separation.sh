#!/bin/bash

CONCURRENCY=1

sglang_source_path=/home/d00662834/dev-0210/sglang
install_sglang_from_source=false
kube_job_type=multi-pd-separation

image=swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann8.5.0-a3-B092

test_set=$(cat testcase_v25_baseline_multi_pd_mix.txt)

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
