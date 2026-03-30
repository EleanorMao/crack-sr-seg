# TODO List - Road Crack Restoration and Annotation System

## 课程论文相关任务 (DDL: 4.15)

### 1. R语言统计分析 [可选]
- [ ] 创建 `analysis/` 目录存放R脚本
- [ ] 编写 `analysis/statistical_analysis.R`
  - [ ] 方法间显著性检验 (paired t-test)
  - [ ] ggplot2可视化图表
- [ ] 将R分析结果整合到论文中

**说明**：时间紧张可跳过，当前实验数据已足够充分

### 2. 对比实验 [✅ 已完成]

#### SRCNN 超分辨率结果
| 模型 | Test PSNR | Test SSIM |
|-----|-----------|-----------|
| Basic SRCNN | 27.41 dB | 0.9076 |
| Improved SRCNN (5层) | 30.09 dB | 0.9482 |
| **Improved SRCNN (7层)** | **30.81 dB** | **0.9545** |

#### U-Net 分割结果
| 方法 | Test IoU | Test Dice | Test Acc |
|-----|----------|-----------|----------|
| Bilinear + U-Net | 36.96% | 49.41% | 95.70% |
| Bicubic + U-Net | 36.96% | 49.41% | 95.70% |
| Original HR + U-Net | 59.55% | 72.87% | 97.20% |
| Basic SRCNN + U-Net | 86.84% | 92.47% | 99.44% |
| **Improved SRCNN + U-Net** | **87.28%** | **92.67%** | **99.47%** |
| Improved 3x3 + U-Net | 87.21% | 92.61% | 99.47% |

#### 关键结论
- SRCNN vs Bilinear: +50.32% IoU (相对提升 +135.9%) ✅
- SRCNN vs Original HR: +27.73% IoU (相对提升 +46.5%) ✅
- 证明学习型超分 >> 传统插值 ✅

### 3. 可视化 [✅ 已完成]
- [x] `scripts/visualize.py` - 可视化脚本
- [x] `scripts/generate_comparison_table.py` - 对比表格和图表
- [x] `scripts/compare_srcnn_models.py` - SRCNN模型对比

生成的图表：
- `figures/srcnn_comparison.png` - SRCNN对比
- `figures/methods_comparison.png` - 方法对比
- `figures/methods_comparison_detailed.png` - 详细对比

### 4. 论文写作 [进行中]

| 章节 | 状态 | 文件 |
|-----|------|------|
| 摘要 | ✅ 已完成 | `abstract.md` |
| 引言 | ✅ 已完成 | `introduction.md` |
| 论文大纲 | ✅ 已完成 | `headline.md` |
| 参考文献 | ✅ 已完成 | `references.md` |
| 相关工作 | ⏳ 待写 | - |
| 方法论 | ⏳ 待写 | - |
| 实验部分 | ⏳ 待写 | - |
| 结论 | ⏳ 待写 | - |

---

## 实验记录 [✅ 已完成]

详细实验记录见 `record.md`

### 已完成的训练
- [x] Basic SRCNN (PSNR: 27.41 dB)
- [x] Improved SRCNN (PSNR: 30.09 dB)
- [x] Improved SRCNN 3x3 (PSNR: 30.81 dB)
- [x] U-Net with Original HR (IoU: 59.55%)
- [x] U-Net with Basic SRCNN (IoU: 86.84%)
- [x] U-Net with Improved SRCNN (IoU: 87.28%)
- [x] U-Net with Improved 3x3 (IoU: 87.21%)

### 已完成的测试
- [x] SRCNN测试 (train/val/test)
- [x] U-Net测试 (test)
- [x] 插值baseline (Bilinear, Bicubic)

### 不需要执行
- [x] ~~智能降质消融实验~~ (数据预处理策略，消融意义不大)
- [x] ~~正样本权重消融实验~~ (可选，时间紧可跳过)

---

## 脚本状态 [✅ 全部完成]

| 脚本 | 状态 | 说明 |
|-----|------|------|
| `scripts/run_baselines.py` | ✅ | 插值对比实验 |
| `scripts/compare_srcnn_models.py` | ✅ | SRCNN模型对比 |
| `scripts/visualize.py` | ✅ | 可视化生成 |
| `scripts/generate_comparison_table.py` | ✅ | 表格和图表 |
| `scripts/ablation_study.py` | ✅ | 消融实验脚本(备用) |

---

## 进度追踪

| 任务 | 状态 | 完成日期 |
|-----|------|---------|
| 基础Pipeline | ✅ 完成 | 2024-03-24 |
| SRCNN训练 (3个版本) | ✅ 完成 | - |
| U-Net训练 (4个版本) | ✅ 完成 | - |
| 测试评估 | ✅ 完成 | - |
| 插值对比实验 | ✅ 完成 | - |
| 可视化图表 | ✅ 完成 | - |
| 摘要 | ✅ 完成 | - |
| 引言 | ✅ 完成 | - |
| 论文大纲 | ✅ 完成 | - |
| R语言分析 | ⏳ 可选 | - |
| 论文正文 | ⏳ 进行中 | - |

---

## 论文核心数据

### 最佳结果
- **方法**: Improved SRCNN + U-Net
- **IoU**: 87.28%
- **Dice**: 92.67%
- **Accuracy**: 99.47%

### 核心贡献
1. 两阶段SRCNN+U-Net框架
2. 证明学习型超分 >> 传统插值 (+135.9%)
3. 在CRACK500数据集上达到SOTA水平
