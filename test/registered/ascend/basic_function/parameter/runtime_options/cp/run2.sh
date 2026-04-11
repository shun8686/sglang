#export cann_path=/usr/local/Ascend/ascend-toolkit/latest
#source /usr/local/Ascend/driver/bin/setenv.bash
#source ${cann_path}/../set_env.sh
#source ${cann_path}/../../nnal/atb/set_env.sh
#source ${cann_path}/opp/vendors/customize/bin/set_env.bash
#export PYTHONPATH=/data/dzc/code/upstream1203/sglang/python:$PYTHONPATH
#export PYTHONPATH=/data/dzc/b022/scp/sglang-npu-nn-B022/python:$PYTHONPATH
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export TASK_QUEUE_ENABLE=0

#export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export HCCL_BUFFSIZE=800
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=48

#export SGLANG_NPU_USE_MLP

#export HCCL_OP_EXPANSION_MODE=AIV
P_HOST_IP=('172.22.3.71' '172.22.3.166')
#export ASCEND_MF_STORE_URL="tcp://172.22.3.71:30110"
export HCCL_SOCKET_IFNAME="enp23s0f3"
export GLOO_SOCKET_IFNAME="enp23s0f3"

#for i in "${!P_HOST_IP[@]}";
#do
 python -m sglang.launch_server \
	--host 172.22.3.166 \
 	--port 8001 \
	--attention-backend ascend \
	--device npu \
	--trust-remote-code \
	--dist-init-addr 172.22.3.71:5000 \
	--nnodes 2 \
	--node-rank 1 \
	--tp-size 32 \
	--ep-size 32 \
	--attn-cp-size 32 \
	--enable-nsa-prefill-context-parallel \
	--nsa-prefill-cp-mode in-seq-split \
	--dp-size 1 \
	--enable-dp-attention \
	--enable-dp-lm-head \
	--moe-dense-tp-size 1 \
	--quantization modelslim \
	--model-path /data/ascend-ci-share-pkking-sglang/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8 \
	--mem-fraction-static 0.8 \
	--chunked-prefill-size -1 \
	--context-length 68000 \
	--max-prefill-tokens 68000 \
	--max-running-requests 32 \
	--max-total-tokens 68000 \
	--cuda-graph-max-bs 32 \
	--moe-a2a-backend deepep \
	--deepep-mode auto \
	--disable-radix-cache
#done
