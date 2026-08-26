# -*- coding: utf-8 -*-
# 实验二：冒泡到 root 的 handler
# 特点：子 logger 身上【没有】任何 handler，
#       日志记录沿着 propagate 冒泡到 root，由 root 的 handler 统一处理
#       这正是 06.py 里 ai_chat 这个 logger 的用法

import logging

# basicConfig 本质 = 给 root logger 挂一个带格式的 StreamHandler
logging.basicConfig(
    level=logging.INFO,
    format="[root的handler] %(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("ai_chat")   # 子 logger，自己不挂 handler
# logger.handlers 是空的，可以 print 出来验证：
print("ai_chat 自己的 handler 列表:", logger.handlers)

logger.warning("我没有自己的 handler，冒泡到 root 输出")

# 预期输出两行：
# ai_chat 自己的 handler 列表: []
# [root的handler] 2026-08-22 14:20:00,123 WARNING ai_chat: 我没有自己的 handler，冒泡到 root 输出
#                                                ^^^^^^^ 注意：name 还是 ai_chat，但格式是 root 的
