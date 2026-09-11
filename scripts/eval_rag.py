"""RAG 检索评估：Recall@3 与 MRR（Mean Reciprocal Rank）

用法：
    python scripts/eval_rag.py

评估方法：
1. 内置 20 条「问题-标准答案」对（答案即期望出现在检索片段中的关键文本，
   与项目知识库文档相关；更换知识库后请按实际文档内容调整 QA_PAIRS）；
2. 对每个问题执行与线上一致的 RAG 检索（core.rag.search，top_k=3）；
3. 判断标准答案是否出现在任一检索片段中，并记录首次命中的排名；
4. 汇总输出 Recall@3 与 MRR。

示例输出：
    RAG 评估结果（20 条测试集）：

    Recall@3: 0.78（78% 的问题在 top3 中找到了相关文档）
    MRR: 0.65

注意：评估结果取决于当前向量库内容；上传/删除文档后请重新运行本脚本。
"""

import sys
from pathlib import Path

# 允许从任意目录运行（scripts/ 位于项目根目录下）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.rag import get_document_list, search  # noqa: E402

# (问题, 标准答案关键文本)：标准答案必须能在知识库文档的某个片段中原样找到，
# 检索到的 top3 片段中出现该文本即判为命中（首次命中的排名参与 MRR 计算）。
# 答案尽量选择短小、稳定的专有名词/数字短语，避免切分后无法精确匹配。
QA_PAIRS = [
    ("项目用什么框架实现 Agent 的自主决策？", "LangGraph"),
    ("RAG 检索使用的向量数据库是什么？", "ChromaDB"),
    ("网络搜索工具的主源是什么？", "Tavily"),
    ("网络搜索的备用源是什么？", "pixserp"),
    ("缓存系统使用什么中间件？", "Redis"),
    ("会话数据持久化到哪个数据库？", "MySQL"),
    ("数学计算工具如何防止代码注入？", "AST"),
    ("MySQL 不可用时应用如何降级？", "JSON"),
    ("界面用什么框架构建？", "Streamlit"),
    ("本地嵌入使用的模型是什么？", "ONNXMiniLM"),
    ("知识库支持上传哪些格式的文档？", "PDF"),
    ("一键部署用什么工具编排？", "docker-compose"),
    ("全局回答缓存的默认过期时间是多少秒？", "3600"),
    ("搜索工具内部缓存的默认有效期是多少秒？", "600"),
    ("Agent 默认最大规划步数由哪个环境变量控制？", "AGENT_MAX_STEPS"),
    ("思考过程面板针对哪类模型设计？", "DeepSeek-Reasoner"),
    ("REST API 的流式接口支持什么能力？", "取消"),
    ("嵌入方式的默认取值是什么？", "auto"),
    ("项目支持哪三种模型提供方？", "Ollama"),
    ("默认的向量集合名称是什么？", "documents"),
]


def evaluate(qa_pairs, search_fn, top_k=3):
    """执行评估，返回 (recall, mrr, 明细列表)

    明细列表元素：(问题, 首次命中排名或 None)。检索逻辑与线上一致：
    任一 top_k 片段中包含标准答案文本即判为命中。

    Args:
        qa_pairs: [(问题, 标准答案关键文本), ...]
        search_fn: 检索函数，签名与 core.rag.search 一致
        top_k: 每个问题的检索片段数，默认 3

    Returns:
        tuple: (Recall@3, MRR, [(问题, 命中排名或 None), ...])
    """
    hits = 0
    rr_sum = 0.0
    details = []
    for question, expected in qa_pairs:
        results = search_fn(question, top_k=top_k)
        rank = next(
            (i + 1 for i, r in enumerate(results)
             if expected in (r.get("content") or "")),
            None,
        )
        details.append((question, rank))
        if rank is not None:
            hits += 1
            rr_sum += 1.0 / rank
    total = len(qa_pairs)
    recall = hits / total if total else 0.0
    mrr = rr_sum / total if total else 0.0
    return recall, mrr, details


def main() -> int:
    docs = get_document_list()
    if not docs:
        print("⚠️ ChromaDB 向量库中没有数据，无法评估。")
        print("   请先在应用侧边栏「文档管理」中上传知识库文档（PDF/TXT/MD），")
        print("   再重新运行本脚本。")
        return 1

    print(f"向量库文档：{len(docs)} 个（共 {sum(d['chunks'] for d in docs)} 个片段）")
    print()

    recall, mrr, details = evaluate(QA_PAIRS, search)
    for i, (question, rank) in enumerate(details, 1):
        mark = f"✅ rank={rank}" if rank is not None else "❌ 未命中"
        print(f"[{i:>2}] {mark}  问题：{question}")

    print()
    print(f"RAG 评估结果（{len(QA_PAIRS)} 条测试集）：")
    print()
    print(f"Recall@3: {recall:.2f}（{recall:.0%} 的问题在 top3 中找到了相关文档）")
    print(f"MRR: {mrr:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
