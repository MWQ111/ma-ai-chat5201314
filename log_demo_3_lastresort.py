# -*- coding: utf-8 -*-
# 实验三：lastResort 兜底
# 特点：全程不调 basicConfig（root 没有 handler），
#       子 logger 也没有 handler，冒泡到 root 还是没人管，
#       这时 Python 拿出内置的 lastResort handler 救场：
#       级别 >= WARNING 才输出，格式是“光秃秃”的裸消息

import logging

# 故意什么都不配置，验证 root 确实没有 handler
root = logging.getLogger()
print("root 的 handler 列表:", root.handlers)

logger = logging.getLogger("orphan")   # 孤儿 logger：自己没有，root 也没有

logger.info("我是 INFO，低于 lastResort 的 WARNING 门槛，会被吞掉")
logger.warning("我是 WARNING，lastResort 兜底输出裸消息")

# 预期输出三行：
# root 的 handler 列表: []
# orphan:我是 INFO，低于 lastResort 的 WARNING 门槛，会被吞掉   ← 注意：这行【不会出现】
# 我是 WARNING，lastResort 兜底输出裸消息
#
# 对比记忆点：
#   实验一输出：[自有handler] + 时间 + 级别 + 名字 + 消息（格式最全）
#   实验二输出：[root的handler] + 时间 + 级别 + 名字 + 消息（root 的格式）
#   实验三输出：只有消息本身，连 WARNING 和名字都没有（裸奔）
