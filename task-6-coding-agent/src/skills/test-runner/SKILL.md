---
name: test-runner
description: 用于运行测试并分析失败原因。当 agent 完成代码修改后需要验证正确性时加载。
---

# 测试运行技能

1. 使用 `run_tests` 在仓库内运行 pytest
2. 如果测试通过，报告成功
3. 如果测试失败，分析失败输出中的关键错误信息
4. 使用 `git_diff` 查看当前修改
