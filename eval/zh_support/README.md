# 中文心理支持验收样例（起点）

`live_smoke.jsonl` 沿用上游 live text runtime 数据格式，覆盖学习压力、失恋、合理工作担忧和含糊的安全信号。它只检查粗粒度路由及少量禁语，不能证明共情质量、危机召回或临床有效性。实际目标模型接入后，可用 `eval/runners/run_live_text_runtime_eval.py --dataset eval/zh_support/live_smoke.jsonl` 运行并人工审阅生成内容；详细参数见上游 `eval/README.md`。

扩展到 Art.md 的目标评测集时，必须补充家暴等可信危险、拒绝练习、负面实验结果、未成年人、记忆撤回及跨会话轨迹，并由专业审阅者盲评。现在的四条样例未经目标模型验证，不作为发布门槛成绩。
