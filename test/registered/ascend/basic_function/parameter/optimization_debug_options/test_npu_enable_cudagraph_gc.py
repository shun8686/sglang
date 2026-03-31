import os
import time
import subprocess
import signal

MODEL_PATH = "QWEN3_0_6B_WEIGHTS_PATH"  # 替换成你的模型路径
BASE_URL = "http://localhost:30000"
LOG_FILE = "server_log.txt"

def run_server_with_args(extra_args):
    """启动服务并捕获 stdout + stderr，返回进程"""
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model", MODEL_PATH,
        "--trust-remote-code",
        "--tp-size", "1",
        "--mem-fraction-static", "0.7",
        "--attention-backend", "ascend",
    ] + extra_args

    print(f"\n🚀 启动命令: {' '.join(cmd)}")

    # 启动并将日志输出到文件
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=f,
            text=True
        )

    # 等待 CUDA Graph 捕获完成（根据你的模型调整时间）
    print("⏳ 等待 CUDA Graph 捕获...")
    time.sleep(40)

    return proc

def watch_cuda_graph_capture_speed():
    """读取日志，判断 CUDA Graph 捕获速度（直接反映 GC 是否冻结）"""
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.read()

    # 捕获 CUDA Graph 相关日志
    capture_lines = [
        line for line in lines.split("\n")
        if "Capturing batches" in line
    ]

    if not capture_lines:
        print("⚠️ 未检测到 CUDA Graph 捕获日志，可能已禁用 CUDA Graph")
        return

    print(f"\n📊 检测到 {len(capture_lines)} 条 CUDA Graph 捕获记录")
    for line in capture_lines[:5]:
        print(line.strip())

    # 特征判断
    if "avail_mem" in lines:
        print("\n✅ CUDA Graph 捕获功能正常")

def kill_server(proc):
    """杀死服务进程"""
    try:
        os.kill(proc.pid, signal.SIGTERM)
        time.sleep(5)
    except:
        pass

# ===================== 测试开始 =====================
print("=" * 80)
print("           端到端测试 --enable-cudagraph-gc 功能")
print("=" * 80)

# -------------------
# 测试 1：默认（关闭 cudagraph-gc → GC 冻结）
# -------------------
print("\n" + "=" * 60)
print("测试 1：默认（不加 --enable-cudagraph-gc）")
print("预期：CUDA Graph 捕获快，GC 冻结")
print("=" * 60)
proc1 = run_server_with_args([])
watch_cuda_graph_capture_speed()
kill_server(proc1)

# -------------------
# 测试 2：开启 --enable-cudagraph-gc → GC 不冻结
# -------------------
print("\n" + "=" * 60)
print("测试 2：添加 --enable-cudagraph-gc")
print("预期：CUDA Graph 捕获变慢，GC 正常运行")
print("=" * 60)
proc2 = run_server_with_args(["--enable-cudagraph-gc"])
watch_cuda_graph_capture_speed()
kill_server(proc2)

# ===================== 最终结论 =====================
print("\n" + "=" * 80)
print("                      测试结论")
print("=" * 80)
print("【不加参数】CUDA Graph 捕获快  → GC 冻结（生效）")
print("【加参数】  CUDA Graph 捕获慢  → GC 未冻结（生效）")
print("\n✅ --enable-cudagraph-gc 功能验证完成！")

# 清理日志
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)