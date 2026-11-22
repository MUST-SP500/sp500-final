# output/

课程要求压缩包里有一个 `output/`，放"最终图表与结果"。

项目内部的产物写在 `reports/` 和 `data/processed/`，这是给开发时看的；
打包提交之前，把下面这些复制到 `output/`，老师打开就能直接看到结果：

```text
output/
├── figures/            # reports/figures/ 里的全部 PNG
├── metrics.csv         # 所有数字的唯一来源
├── equity_curve.csv    # 六条曲线的逐周明细
├── trades.csv          # 每一笔换仓
└── summary.md          # 一页结论：谁赚多少、谁跌多少、分不分得出差别
```

`output/` 里的东西必须是脚本跑出来的，不要手工截图或手抄数字。
