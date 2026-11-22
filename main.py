"""项目入口。

课程要求压缩包里有一个能直接跑的 main.py，就是这个文件。
它本身不做任何计算，只把参数转交给 src/pipeline.py。

用法（不需要激活虚拟环境）：
    uv run main.py                 # 全流程：下载 → 清洗 → 特征 → 切分 → 模型 → 回测 → 报告
    uv run main.py --check         # 只自检：配置、目录、各模块能不能导入，不跑数据
    uv run main.py --stage clean   # 只跑某一段
"""

from __future__ import annotations

import sys

from src.pipeline import main as pipeline_main

if __name__ == "__main__":
    raise SystemExit(pipeline_main(sys.argv[1:]))
