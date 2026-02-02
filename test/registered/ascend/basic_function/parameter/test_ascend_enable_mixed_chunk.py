# -*- coding: utf-8 -*-
import unittest
import requests
import time
import threading
from datetime import datetime

# 必要模块导入
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

# 全局配置（统一配置，保证对比公平性）
CONFIG = {
    "TARGET_TOKEN_COUNT": 2500,  # 超长输入token数（>2048）
    "CHUNK_SIZE": 1024,          # 分块预填充大小
    "REQUEST_COUNT": 10,         # 增加请求数量（平抑浮动，建议10+）
    "TIMEOUT": 300,              # 延长超时，适配多请求场景
    "SERVER_WAIT_TIME": 30       # 服务器就绪等待时间
}

# 全局变量：存储两份测试的最终统计结果（用于后续对比）
FINAL_STATISTICS = {
    "mixed_enabled": None,
    "mixed_disabled": None
}

def build_long_input_text_for_token():
    """构造足够token数的超长输入（公共函数，保证两份测试输入一致）"""
    base_sentence = "This is a test sentence to generate enough tokens. "
    repeat_times = (CONFIG["TARGET_TOKEN_COUNT"] // 10) + 20
    return (base_sentence * repeat_times) + "The capital of France is"

def send_generate_request(task_id, request_results):
    """单个请求发送函数（公共函数，记录单个请求耗时）"""
    try:
        # 1. 记录单个请求开始时间
        single_start_time = time.time()
        
        # 2. 构造超长输入
        long_input_text = build_long_input_text_for_token()
        
        # 3. 发送/generate请求
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": long_input_text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
            timeout=CONFIG["TIMEOUT"]
        )
        
        # 4. 记录单个请求结束时间并计算耗时
        single_end_time = time.time()
        single_elapsed_time = round(single_end_time - single_start_time, 4)
        
        # 5. 记录请求结果
        request_results.append({
            "task_id": task_id,
            "status_code": response.status_code,
            "has_paris": "Paris" in response.text,
            "single_elapsed_time": single_elapsed_time
        })
        
        print(f"【任务 {task_id}】请求完成，状态码：{response.status_code}，单个耗时：{single_elapsed_time} 秒")
    except Exception as e:
        single_end_time = time.time()
        single_elapsed_time = round(single_end_time - time.time(), 4)
        request_results.append({
            "task_id": task_id,
            "status_code": -1,
            "error": str(e),
            "single_elapsed_time": single_elapsed_time
        })
        print(f"【任务 {task_id}】请求失败，报错：{e}，耗时：{single_elapsed_time} 秒")

def start_server(with_mixed: bool):
    """公共服务器启动函数：根据参数决定是否开启mixed chunk"""
    other_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--chunked-prefill-size", str(CONFIG["CHUNK_SIZE"])
    ]
    
    # 按需添加 --enable-mixed-chunk 参数
    if with_mixed:
        other_args.insert(0, "--enable-mixed-chunk")
    
    # 启动服务器
    process = popen_launch_server(
        LLAMA_3_2_1B_WEIGHTS_PATH,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )
    
    # 等待服务器就绪
    time.sleep(CONFIG["SERVER_WAIT_TIME"])
    return process

def calculate_statistics(request_results):
    """公共统计函数：计算耗时相关指标（平均、最大、最小、整体）"""
    success_requests = [r for r in request_results if r["status_code"] == 200]
    if not success_requests:
        return None
    
    # 提取成功请求耗时
    elapsed_times = [r["single_elapsed_time"] for r in success_requests]
    
    # 计算统计指标
    return {
        "success_count": len(success_requests),
        "total_count": len(request_results),
        "avg_elapsed": round(sum(elapsed_times) / len(elapsed_times), 4),
        "max_elapsed": round(max(elapsed_times), 4),
        "min_elapsed": round(min(elapsed_times), 4),
        "overall_elapsed": round(sum(elapsed_times), 4)  # 所有成功请求耗时总和
    }

class TestMixedChunkEnabled(CustomTestCase):
    """测试类1：开启 --enable-mixed-chunk"""
    @classmethod
    def setUpClass(cls):
        """启动服务器（开启mix）"""
        print("\n" + "="*60)
        print("=== 开始启动服务器（开启 --enable-mixed-chunk）===")
        cls.process = start_server(with_mixed=True)

    @classmethod
    def tearDownClass(cls):
        """终止服务器"""
        kill_process_tree(cls.process.pid)
        print("=== 服务器已终止（开启 --enable-mixed-chunk）===")
        print("="*60 + "\n")

    def test_mixed_chunk_with_multi_requests(self):
        """多请求测试（开启mix），统计耗时"""
        request_results = []
        overall_start_time = time.time()
        
        # 打印测试开始信息
        print(f"\n=== 【开启mix】整体任务开始，时间戳：{datetime.fromtimestamp(overall_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')} ===")
        print(f"=== 开始启动 {CONFIG['REQUEST_COUNT']} 个请求线程，制造排队场景 ===")
        
        # 创建并启动多线程
        threads = []
        for task_id in range(CONFIG["REQUEST_COUNT"]):
            t = threading.Thread(target=send_generate_request, args=(task_id, request_results))
            threads.append(t)
        
        for t in threads:
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 计算整体耗时和统计指标
        overall_end_time = time.time()
        overall_elapsed_time = round(overall_end_time - overall_start_time, 4)
        statistics = calculate_statistics(request_results)
        
        # 存储最终统计结果
        FINAL_STATISTICS["mixed_enabled"] = {
            "overall_elapsed": overall_elapsed_time,
            "detail": statistics
        }
        
        # 输出当前测试结果
        self._print_test_result("开启mix", overall_elapsed_time, request_results, statistics)

    def _print_test_result(self, test_name, overall_elapsed, request_results, statistics):
        """辅助函数：格式化输出测试结果"""
        print(f"\n=== 【{test_name}】所有请求执行完毕，整体任务结束 ===")
        print(f"=== 【{test_name}】端到端整体耗时：{overall_elapsed} 秒 ===")
        
        if not statistics:
            print(f"=== 【{test_name}】无成功请求，无法统计详细耗时 ===")
            return
        
        print(f"=== 【{test_name}】耗时统计详情 ===")
        print(f"  成功请求数：{statistics['success_count']} / {statistics['total_count']}")
        print(f"  平均单个耗时：{statistics['avg_elapsed']} 秒")
        print(f"  最大单个耗时：{statistics['max_elapsed']} 秒")
        print(f"  最小单个耗时：{statistics['min_elapsed']} 秒")
        print(f"  所有成功请求耗时总和：{statistics['overall_elapsed']} 秒")

class TestMixedChunkDisabled(CustomTestCase):
    """测试类2：不开启 --enable-mixed-chunk"""
    @classmethod
    def setUpClass(cls):
        """启动服务器（不开启mix）"""
        print("\n" + "="*60)
        print("=== 开始启动服务器（不开启 --enable-mixed-chunk）===")
        cls.process = start_server(with_mixed=False)

    @classmethod
    def tearDownClass(cls):
        """终止服务器"""
        kill_process_tree(cls.process.pid)
        print("=== 服务器已终止（不开启 --enable-mixed-chunk）===")
        print("="*60 + "\n")

    def test_mixed_chunk_with_multi_requests(self):
        """多请求测试（不开启mix），统计耗时"""
        request_results = []
        overall_start_time = time.time()
        
        # 打印测试开始信息
        print(f"\n=== 【不开启mix】整体任务开始，时间戳：{datetime.fromtimestamp(overall_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')} ===")
        print(f"=== 开始启动 {CONFIG['REQUEST_COUNT']} 个请求线程，制造排队场景 ===")
        
        # 创建并启动多线程
        threads = []
        for task_id in range(CONFIG["REQUEST_COUNT"]):
            t = threading.Thread(target=send_generate_request, args=(task_id, request_results))
            threads.append(t)
        
        for t in threads:
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 计算整体耗时和统计指标
        overall_end_time = time.time()
        overall_elapsed_time = round(overall_end_time - overall_start_time, 4)
        statistics = calculate_statistics(request_results)
        
        # 存储最终统计结果
        FINAL_STATISTICS["mixed_disabled"] = {
            "overall_elapsed": overall_elapsed_time,
            "detail": statistics
        }
        
        # 输出当前测试结果
        self._print_test_result("不开启mix", overall_elapsed_time, request_results, statistics)

    def _print_test_result(self, test_name, overall_elapsed, request_results, statistics):
        """辅助函数：格式化输出测试结果"""
        print(f"\n=== 【{test_name}】所有请求执行完毕，整体任务结束 ===")
        print(f"=== 【{test_name}】端到端整体耗时：{overall_elapsed} 秒 ===")
        
        if not statistics:
            print(f"=== 【{test_name}】无成功请求，无法统计详细耗时 ===")
            return
        
        print(f"=== 【{test_name}】耗时统计详情 ===")
        print(f"  成功请求数：{statistics['success_count']} / {statistics['total_count']}")
        print(f"  平均单个耗时：{statistics['avg_elapsed']} 秒")
        print(f"  最大单个耗时：{statistics['max_elapsed']} 秒")
        print(f"  最小单个耗时：{statistics['min_elapsed']} 秒")
        print(f"  所有成功请求耗时总和：{statistics['overall_elapsed']} 秒")

def print_final_comparison():
    """最终对比汇总：打印两份测试的核心差异"""
    print("\n" + "="*80)
    print("=== 最终对比汇总（开启mix vs 不开启mix）===")
    print("="*80)
    
    enabled = FINAL_STATISTICS["mixed_enabled"]
    disabled = FINAL_STATISTICS["mixed_disabled"]
    
    if not enabled or not disabled or not enabled["detail"] or not disabled["detail"]:
        print("=== 存在测试数据不完整，无法进行有效对比 ===")
        return
    
    # 提取核心数据
    enabled_overall = enabled["overall_elapsed"]
    disabled_overall = disabled["overall_elapsed"]
    enabled_avg = enabled["detail"]["avg_elapsed"]
    disabled_avg = disabled["detail"]["avg_elapsed"]
    enabled_max = enabled["detail"]["max_elapsed"]
    disabled_max = disabled["detail"]["max_elapsed"]
    
    # 计算优化率
    overall_optimize_rate = round(((disabled_overall - enabled_overall) / disabled_overall) * 100, 2)
    avg_optimize_rate = round(((disabled_avg - enabled_avg) / disabled_avg) * 100, 2)
    max_optimize_rate = round(((disabled_max - enabled_max) / disabled_max) * 100, 2)
    
    # 格式化输出对比
    print(f"\n1. 端到端整体耗时")
    print(f"   开启mix：{enabled_overall} 秒 | 不开启mix：{disabled_overall} 秒 | 优化率：{overall_optimize_rate}%")
    print(f"\n2. 平均单个请求耗时")
    print(f"   开启mix：{enabled_avg} 秒 | 不开启mix：{disabled_avg} 秒 | 优化率：{avg_optimize_rate}%")
    print(f"\n3. 最大单个请求耗时（长尾效应）")
    print(f"   开启mix：{enabled_max} 秒 | 不开启mix：{disabled_max} 秒 | 优化率：{max_optimize_rate}%")
    print(f"\n=== 总结：{'开启mix表现更优' if overall_optimize_rate > 0 else '未体现明显优化'} ===")

if __name__ == "__main__":
    # 执行所有测试用例
    unittest.main(verbosity=2, exit=False)
    
    # 打印最终对比汇总
    print_final_comparison()
