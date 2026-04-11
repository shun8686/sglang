export cann_path=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/driver/bin/setenv.bash
source ${cann_path}/../set_env.sh
source ${cann_path}/../../nnal/atb/set_env.sh
source ${cann_path}/opp/vendors/customize/bin/set_env.bash
#export PYTHONPATH=/data/dzc/code/upstream1203/sglang/python:$PYTHONPATH
export PYTHONPATH=/data/dzc/b022/scp/sglang-npu-nn-B022/python:$PYTHONPATH
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

#export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export HCCL_BUFFSIZE=1536

#export SGLANG_NPU_USE_MLP

export HCCL_OP_EXPANSION_MODE=AIV
P_HOST_IP=('192.168.0.60' '192.168.0.234')
export ASCEND_MF_STORE_URL="tcp://192.168.0.60:30110"
export HCCL_SOCKET_IFNAME="enp23s0f3"
export GLOO_SOCKET_IFNAME="enp23s0f3"

#for i in "${!P_HOST_IP[@]}";
#do
 python -m sglang.launch_server \
	--host 192.168.0.234 \
 	--port 8001 \
	--attention-backend ascend \
	--device npu \
	--trust-remote-code \
	--dist-init-addr 192.168.0.60:5000 \
	--nnodes 2 \
	--node-rank 1 \
	--tp-size 32 \
	--dp-size 1 \
	--cuda-graph-bs 1 2 4 8 16 32 64 128 \
	--skip-server-warmup \
	--quantization w8a8_int8 \
	--model-path /data/ascend-ci-share-pkking-sglang/modelscope/hub/models/DeepSeek-V3.2-Exp-W8A8 \
        --mem-fraction-static 0.78 \
	--chunked-prefill-size 327680 \
	--context-length 68000 \
	--max-prefill-tokens 68000 \
	--max-total-tokens 68000 \
	--moe-a2a-backend deepep \
	--deepep-mode auto \
	--disable-radix-cache 
#done
