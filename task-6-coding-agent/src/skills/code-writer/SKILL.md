---
name: code-writer
description: 用于编写或修改代码文件。当 agent 已经理解 bug 并确定了修复方案时加载。
---

# 代码编写技能

1. 根据分析结果确定修复方案
2. 使用 `write_file` 写入修改后的完整文件内容
3. 不要修改测试文件
4. 修改后立即使用 `run_tests` 验证
