"""语义缓存阈值实验：对比不同余弦相似度阈值下的命中与误判

用法：
    python scripts/eval_semantic_cache.py

背景：
core/cache.py 的语义缓存（v2）在精确匹配未命中时，把问题转成向量与历史问题
做余弦相似度比较，相似度 >= SEMANTIC_CACHE_THRESHOLD 即视为命中。阈值定得越低
命中越多，但把「意思不同」的问题误判为同一问题的风险也越大，而缓存误判会直接
返回错误答案，比不命中严重得多。

本脚本用固定数据集给出可复现的取舍依据：
1. 13 组「同义改写」问题对（围绕本项目常见场景），同一意图的不同问法，理应命中；
2. 7 组「难负例」问题对，意图相反或完全不同，绝不应命中；
3. 用 core.cache.get_embedding 计算向量（默认 Ollama bge-m3，1024 维）；
4. 对每个阈值（0.95 / 0.90）统计同义命中数与难负例误判数。

输出示例（Ollama bge-m3）：
    语义缓存阈值实验（嵌入来源 Ollama bge-m3，1024 维）

    阈值 0.95：

    同义命中：6/13

    误判：0/7

    阈值 0.90：

    同义命中：10/13

    误判：1/7（示例：“如何导出对话？” vs “如何导入对话？”，相似度 0.93）

    结论：0.95 是零误判的安全阈值，0.90 虽提升命中但会出现语义误判。

说明：该结果依赖所用嵌入模型。默认的 Ollama bge-m3 为多语言模型，中文区分度好；
若回退到英文模型 all-MiniLM-L6-v2，语义无关的中文句子也会得到接近 1.00 的
相似度，实测无零误判阈值（参照 README 历史结论）。
"""

import sys
from pathlib import Path

# 允许从任意目录运行（scripts/ 位于项目根目录下）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.cache import (  # noqa: E402
    _cosine_similarity,
    get_embedding,
    get_embedding_source,
)

# ====================== 实验数据 ======================
# 同义改写：同一意图的不同问法，理应命中缓存（命中越多越好）
SYNONYM_PAIRS = [
    ("如何上传文档？", "怎么上传文件？"),
    ("怎么删除对话？", "如何删除一个会话？"),
    ("今天天气怎么样？", "今天天气如何？"),
    ("帮我计算 12 乘 8", "计算一下 12 乘以 8"),
    ("如何切换模型？", "怎么更换使用的模型？"),
    ("怎么开启缓存？", "如何启用缓存功能？"),
    ("如何配置数据库？", "怎么设置数据库连接？"),
    ("Agent 模式怎么用？", "如何使用 Agent 模式？"),
    ("支持上传哪些文档格式？", "可以上传什么类型的文档？"),
    ("如何导出对话记录？", "怎么把历史对话导出？"),
    ("网络搜索怎么配置？", "如何设置联网搜索？"),
    ("如何清空缓存？", "怎么清除所有缓存？"),
    ("怎样修改系统提示词？", "如何更改系统提示词？"),
]

# 难负例：意图相反或完全不同的问题，绝不应被判为同一问题（误判必须为 0）
HARD_NEGATIVE_PAIRS = [
    ("如何上传文档？", "如何删除文档？"),
    ("怎么开启缓存？", "如何关闭缓存？"),
    ("如何新建对话？", "如何删除对话？"),
    ("如何导出对话？", "如何导入对话？"),
    ("今天天气怎么样？", "帮我计算 12 乘 8"),
    ("如何切换模型？", "如何配置数据库？"),
    ("DeepSeek 支持哪些模型？", "Ollama 支持哪些模型？"),
]

# 待对比的阈值（高阈值更保守，低阈值更激进）
THRESHOLDS = [0.95, 0.90]


def _embed_all(questions):
    """批量计算问题向量，返回 {问题: 向量}；任一问题嵌入失败返回 None"""
    vectors = {}
    for question in questions:
        vector = get_embedding(question)
        if vector is None:
            return None
        vectors[question] = vector
    return vectors


def count_matches(pairs, vectors, threshold):
    """统计相似度 >= 阈值的问题对数，并返回其明细

    Returns:
        tuple: (命中对数, [(问题A, 问题B, 相似度), ...])
    """
    matched = []
    for question_a, question_b in pairs:
        score = _cosine_similarity(vectors[question_a], vectors[question_b])
        if score >= threshold:
            matched.append((question_a, question_b, score))
    return len(matched), matched


def main() -> int:
    questions = []
    for question_a, question_b in SYNONYM_PAIRS + HARD_NEGATIVE_PAIRS:
        for question in (question_a, question_b):
            if question not in questions:
                questions.append(question)

    vectors = _embed_all(questions)
    if vectors is None:
        print("⚠️ 嵌入函数不可用，无法运行实验（未能计算问题向量）。")
        print("   常见原因：Ollama 未启动 / 未执行 ollama pull bge-m3、未安装 chromadb，")
        print("   或回退用的本地模型 all-MiniLM-L6-v2 下载失败。")
        print("   可尝试：启动 Ollama（ollama pull bge-m3）或 pip install chromadb，")
        print("   或配置 OPENAI_API_KEY 改用在线嵌入后重跑。")
        return 1

    dim = len(next(iter(vectors.values())))
    print(f"语义缓存阈值实验（嵌入来源 {get_embedding_source()}，{dim} 维）")
    print()

    synonym_total = len(SYNONYM_PAIRS)
    negative_total = len(HARD_NEGATIVE_PAIRS)
    synonym_hits = {}
    false_positives = {}

    for threshold in THRESHOLDS:
        synonym_hits[threshold], _ = count_matches(SYNONYM_PAIRS, vectors, threshold)
        false_positives[threshold], fp_details = count_matches(
            HARD_NEGATIVE_PAIRS, vectors, threshold)

        print(f"阈值 {threshold:.2f}：")
        print()
        print(f"同义命中：{synonym_hits[threshold]}/{synonym_total}")
        print()
        if false_positives[threshold] == 0:
            print(f"误判：0/{negative_total}")
        elif false_positives[threshold] == 1:
            question_a, question_b, score = fp_details[0]
            print(f"误判：1/{negative_total}"
                  f"（示例：“{question_a}” vs “{question_b}”，相似度 {score:.2f}）")
        else:
            detail = "；".join(
                f"“{a}” vs “{b}”，相似度 {s:.2f}" for a, b, s in fp_details)
            print(f"误判：{false_positives[threshold]}/{negative_total}（{detail}）")
        print()

    # 结论由实际结果推导，不写死
    safe = [t for t in THRESHOLDS if false_positives[t] == 0]
    risky = [t for t in THRESHOLDS if false_positives[t] > 0]
    if safe and risky:
        print(f"结论：{max(safe):.2f} 是零误判的安全阈值，"
              f"{min(risky):.2f} 虽提升命中但会出现语义误判。")
    elif safe:
        print(f"结论：{max(safe):.2f} 是零误判的安全阈值。")
    else:
        print("结论：所有阈值均出现误判，语义缓存不建议启用，或需换用更强的嵌入模型。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
