# 3. 相关工作

## 3.1 超分辨率网络设计

### 3.1.1 经典SRCNN的局限性

Dong等[3]提出的SRCNN采用三层卷积结构（9×9→1×1→5×5），感受野约为13×13。虽然该网络开创了深度学习超分辨率的先河，但在路面裂缝图像重建任务中存在以下局限：

**感受野不足**：路面裂缝通常呈现细长的线性结构，需要较大的感受野来捕捉完整的裂缝形态。13×13的感受野难以覆盖裂缝的上下文信息。

**非线性映射能力有限**：单层1×1卷积难以学习复杂的低分辨率到高分辨率的非线性映射关系。

### 3.1.2 网络结构改进

针对上述问题，本文对SRCNN进行了结构改进，主要探索了两种方案：

**方案一：扩展深度与大卷积核（5层）**

采用9×9→3×3→3×3→3×3→5×5的卷积核组合，感受野扩大至约19×19：
- 第一层9×9卷积核：增大初始感受野，捕捉更大范围的上下文
- 中间三层3×3卷积：增强非线性映射能力
- 最后一层5×5卷积：更好地重建细节
- 残差连接：学习HR与LR的差异，加速收敛

**方案二：全3×3深度网络（7层）**

采用7层3×3卷积核堆叠，感受野约15×15：
- 通过增加深度提升感受野
- 全部使用3×3小卷积核减少参数量
- 同样采用残差连接

### 3.1.3 BatchNorm层的移除

实验发现，在超分辨率任务中引入BatchNorm层会导致性能下降：

| 网络配置 | PSNR (dB) |
|---------|-----------|
| 5层 + BatchNorm | 24.10 |
| 5层 无BatchNorm | **30.09** |

BatchNorm对超分辨率任务的负面影响原因：
- 超分辨率需要恢复绝对的像素强度和对比度
- BatchNorm对特征进行归一化，破坏了图像原有的对比度信息
- 可能平滑掉高频边缘细节

这一发现与Lim等[5]在EDSR中的观察一致，该工作同样移除了BatchNorm层以提升重建质量。

---

## 3.2 两阶段学习框架

### 3.2.1 框架设计动机

现有路面裂缝检测方法[2,6]大多假设输入图像质量较高，直接对原始图像进行分割。然而，当输入图像存在模糊、低分辨率等降质问题时，分割性能会显著下降。

本文采用两阶段级联框架的设计思路：

**阶段一：图像质量增强**
- 目标：恢复低质量图像中的纹理细节
- 方法：学习型超分辨率重建
- 优势：相比传统插值方法，能够"学习"如何恢复丢失的信息

**阶段二：裂缝分割**
- 目标：对恢复后的图像进行像素级裂缝检测
- 方法：U-Net语义分割网络[4]
- 优势：利用编码器-解码器结构和跳跃连接，保留边界信息

### 3.2.2 与端到端方法的对比

两阶段独立训练 vs 端到端联合训练：

| 方面 | 两阶段独立训练 | 端到端联合训练 |
|-----|--------------|--------------|
| 实现复杂度 | 简单，模块可复用 | 复杂，需设计联合损失 |
| 训练稳定性 | 各阶段独立稳定 | 可能存在梯度传播问题 |
| 优化目标 | 各自最优（PSNR/IoU） | 全局最优（分割精度） |
| 灵活性 | 可单独改进各模块 | 耦合度高 |

本文选择两阶段独立训练策略，原因如下：
1. 便于分别评估超分辨率和分割的性能
2. 模块化设计便于后续改进
3. 训练过程更稳定

---

## 3.3 传统插值方法的局限性

### 3.3.1 实验对比

本文系统对比了传统插值方法与学习型超分辨率方法：

| 方法 | IoU (%) | 说明 |
|-----|---------|------|
| 双线性插值 | 36.96 | 仅进行像素平滑 |
| 双三次插值 | 36.96 | 仅进行像素平滑 |
| SRCNN重建 | **86.84** | 学习型恢复 |

### 3.3.2 原因分析

传统插值方法失效的原因：
- 输入图像已丢失高频信息，插值无法"无中生有"
- 数学插值本质上是平滑操作，会进一步模糊裂缝边缘
- 裂缝像素占比极小（约6%），被背景严重稀释

学习型超分辨率的优势：
- 通过学习大量图像对，掌握如何恢复细节信息
- 能够利用上下文信息推断丢失的纹理
- 针对裂缝等线性结构有更好的重建能力

---

## 3.4 本文创新点总结

1. **问题导向**：专注于低质量路面图像的裂缝检测，而非假设高质量输入

2. **网络改进**：基于实验验证，移除BatchNorm层、增大感受野，将PSNR从24.10 dB提升至30.09 dB

3. **框架设计**：两阶段级联框架，分别优化图像质量和分割精度

4. **实验发现**：揭示PSNR与分割精度的非正相关性（详见实验部分）

---

## 参考文献

[1] Koch C, Georgieva K, Kasireddy V, et al. A review on computer vision based defect detection and condition assessment of concrete and asphalt civil infrastructures[J]. Advanced Engineering Informatics, 2015, 29(2): 196-210.

[2] Zhang A, Wang K C P, Li B, et al. Automated pixel-level pavement crack detection on 3D asphalt surfaces using a deep-learning network[J]. Computer-Aided Civil and Infrastructure Engineering, 2017, 32(10): 805-819.

[3] Dong C, Loy C C, He K, et al. Learning a deep convolutional network for image super-resolution[C]//European Conference on Computer Vision (ECCV). Springer, 2014: 184-199.

[4] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional networks for biomedical image segmentation[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI). Springer, 2015: 234-241.

[5] Lim B, Son S, Kim H, et al. Enhanced deep residual networks for single image super-resolution[C]//IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). 2017: 1132-1140.

[6] Zou Q, Zhang Z, Li Q, et al. Automated pavement crack segmentation using U-Net-based convolutional neural network[J]. IEEE Access, 2020, 8: 114892-114899.
