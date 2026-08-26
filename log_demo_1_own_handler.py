# -*- coding: utf-8 -*-
# 实验一：自己的 handler
# 特点：logger 身上挂了自己的 StreamHandler，日志由它处理
# （有没有 basicConfig 都无所谓，自己的 handler 优先干活）

import logging

logger = logging.getLogger("own_handler")   # 自己起名的 logger

# 手工造一个 handler（输出到终端），并给它配上格式
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "[自有handler] %(asctime)s %(levelname)s %(name)s: %(message)s"
))

logger.addHandler(handler)   # 挂到自己身上
logger.setLevel(logging.INFO)  # 注意：单独 handler 时，logger 自身级别也要放行

logger.warning("我用自己的 handler 输出，格式是我自己配的")

# 预期输出（一条，带 [自有handler] 前缀，有时间戳）：
# [自有handler] 2026-08-22 14:20:00,123 WARNING own_handler: 我用自己的 handler 输出，格式是我自己配的
