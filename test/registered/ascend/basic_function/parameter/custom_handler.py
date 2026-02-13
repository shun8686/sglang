# custom_handlers.py
import logging
import sys
import time
import torch  # 若需清理GPU资源，导入torch
import signal

# 配置日志（可选，便于排查退出原因）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("sglang_exit.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("sglang_custom_handler")

def custom_sigquit_handler(signum, frame):
    """
    自定义SIGQUIT信号处理函数
    :param signum: 信号编号（SIGQUIT是3）
    :param frame: 信号触发时的栈帧（通常无需处理）
    """
    logger.info(f"接收到SIGQUIT信号（signum={signum}），开始执行自定义退出逻辑...")
    
    # 步骤1：清理GPU资源（SGLang基于GPU推理，核心清理项）

    
    # 步骤2：保存退出状态/上下文（可选）

    
    # 步骤3：自定义其他逻辑（如通知监控、关闭数据库连接等）
    logger.info("自定义退出逻辑执行完成，准备退出服务...")
    
    # 最后必须主动退出进程（否则服务会挂起）
    sys.exit(0)

# 可选：测试函数（本地验证逻辑是否正常）
if __name__ == "__main__":
    # 注册信号并触发测试
    signal.signal(signal.SIGQUIT, custom_sigquit_handler)
    logger.info("测试模式：发送SIGQUIT信号（Ctrl+\）触发自定义处理...")
    while True:
        time.sleep(1)
