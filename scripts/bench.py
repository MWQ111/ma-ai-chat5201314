"""性能基准测试：模拟 100 次问答请求，统计缓存命中率与 P50/P95 延迟

说明（用数字证明缓存层的价值，任何机器都能跑出可复现的结果）：
- 不调用真实 AI：用带随机耗时的 mock 生成器模拟一次完整生成（默认 0.8~2.5s）；
- 不依赖 Redis：用进程内本地缓存模拟缓存层（与 core.cache 相同的 get/set 语义）；
- 40% 的请求会重复最近问过的问题（模拟真实场景中的追问/重复提问），
  其余为全新问题，命中率由该比例决定；
- 跑两组对照：开缓存 vs 关缓存（同一随机种子，关缓存每次都模拟完整生成），
  用「提升 = 关缓存 - 开缓存」量化缓存层拉低了多少延迟；
- 固定随机种子，结果可复现。

用法：
    python scripts/bench.py            # 默认 100 次请求
    python scripts/bench.py 10         # 快速冒烟：10 次请求

输出示例：
    性能基准（100 次请求）：

    指标          开缓存      关缓存      提升
    --------------------------------------------------
    P50 延迟      0.90s       1.66s       0.76s
    P95 延迟      2.42s       2.43s       0.01s
    命中率        44.0%       0.0%
"""

import math
import random
import sys
import time
import unicodedata

N_REQUESTS = 100                        # 默认请求数
GENERATION_DELAY_RANGE = (0.8, 2.5)     # 模拟一次真实 AI 生成的耗时区间（秒）
CACHE_HIT_DELAY_RANGE = (0.001, 0.008)  # 模拟一次缓存命中的耗时区间（秒，约等于 Redis 读取）
REPEAT_PROBABILITY = 0.4                # 重复提问概率（决定命中率，真实场景中用户常追问）
RECENT_WINDOW = 20                      # 「最近问过」窗口大小
RANDOM_SEED = 42                        # 固定随机种子，结果可复现

# 新问题的主题池（仅用于生成问题文本，不影响统计）
TOPICS = [
    "数据库优化", "缓存策略", "Agent 规划", "RAG 检索", "API 设计",
    "并发控制", "前端渲染", "部署运维", "代码规范", "测试方法",
]


class MockCache:
    """进程内缓存：与 core.cache 相同的 get/set 语义（基准运行在秒级，无需 TTL）"""

    def __init__(self):
        self._data = {}

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        self._data[key] = value


def mock_generate(question):
    """模拟一次真实 AI 生成：随机耗时后返回回答（不发起任何网络请求）"""
    time.sleep(random.uniform(*GENERATION_DELAY_RANGE))
    return f"模拟回答：{question}"


def percentile(sorted_values, p):
    """最邻近秩百分位数：P50=中位数，P95=第 95 百分位"""
    if not sorted_values:
        return 0.0
    idx = max(0, int(math.ceil(p / 100 * len(sorted_values))) - 1)
    return sorted_values[idx]


def _pad(text, width):
    """按终端显示宽度补齐到 width 列：中文字符占 2 列，直接用 str.ljust 会与表头错位"""
    display = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - display)


def run_benchmark(n_requests, use_cache=True):
    """执行基准，返回 (升序延迟列表, 缓存命中次数)

    use_cache=False 时强制不走缓存（对照组），每次请求都模拟一次完整生成。
    """
    cache = MockCache()
    recent = []
    latencies = []
    hits = 0
    unique = 0

    for _ in range(n_requests):
        if recent and random.random() < REPEAT_PROBABILITY:
            question = random.choice(recent)  # 模拟追问/重复提问
        else:
            unique += 1
            question = f"基准测试问题{unique}：{random.choice(TOPICS)}"
            recent.append(question)
            if len(recent) > RECENT_WINDOW:
                recent.pop(0)

        t0 = time.perf_counter()
        if use_cache:
            answer = cache.get(question)
        else:
            answer = None          # 关缓存：强制不走缓存
        if answer is None:
            answer = mock_generate(question)  # 未命中：模拟一次完整生成
            cache.set(question, answer)
        else:
            hits += 1
            time.sleep(random.uniform(*CACHE_HIT_DELAY_RANGE))  # 命中：模拟缓存读取

        latencies.append(time.perf_counter() - t0)

    latencies.sort()
    return latencies, hits


def main() -> int:
    n = N_REQUESTS
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(f"用法：python {sys.argv[0]} [请求数]")
            return 2

    random.seed(RANDOM_SEED)
    lat_with, hits_with = run_benchmark(n, use_cache=True)
    hit_rate_with = hits_with / n if n else 0.0

    random.seed(RANDOM_SEED)          # 重置种子，保证两组输入完全一样
    lat_without, hits_without = run_benchmark(n, use_cache=False)
    hit_rate_without = hits_without / n if n else 0.0

    p50_with = percentile(lat_with, 50)
    p95_with = percentile(lat_with, 95)
    p50_without = percentile(lat_without, 50)
    p95_without = percentile(lat_without, 95)

    print(f"性能基准（{n} 次请求）：")
    print()
    print(_pad("指标", 14) + _pad("开缓存", 12) + _pad("关缓存", 12) + _pad("提升", 12))
    print("-" * 50)
    print(_pad("P50 延迟", 14)
          + _pad(f"{p50_with:.2f}s", 12)
          + _pad(f"{p50_without:.2f}s", 12)
          + _pad(f"{p50_without - p50_with:.2f}s", 12))
    print(_pad("P95 延迟", 14)
          + _pad(f"{p95_with:.2f}s", 12)
          + _pad(f"{p95_without:.2f}s", 12)
          + _pad(f"{p95_without - p95_with:.2f}s", 12))
    print(_pad("命中率", 14)
          + _pad(f"{hit_rate_with:.1%}", 12)
          + _pad(f"{hit_rate_without:.1%}", 12))
    print()
    print(f"（方法：本地 mock 缓存 + 模拟生成耗时 "
          f"{GENERATION_DELAY_RANGE[0]}~{GENERATION_DELAY_RANGE[1]}s，")
    print(f" 未调用真实 AI / Redis；开缓存组命中 {hits_with} 次，未命中 {n - hits_with} 次；")
    print(" 对照组：同一随机种子下，关缓存强制不走缓存，每次都模拟完整生成。）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
