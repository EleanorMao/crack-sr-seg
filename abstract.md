# 摘要

路面裂缝检测是道路养护的重要环节。但在实际工程中采集的图像常因设备限制、环境干扰等因素存在模糊、低分辨率等降质问题，严重影响检测精度。针对这一问题，本文提出一种基于超分辨率重建的路面裂缝检测方法。该方法采用两阶段深度学习框架：第一阶段通过对经典的SRCNN网络进行结构优化，成功将PSNR提升至30.09 dB，对低质量图像进行超分辨率重建，恢复路面纹理细节；第二阶段采用U-Net网络对重建后的图像进行裂缝像素级分割。实验结果表明，本文提出的级联方法在分割任务中取得了高达87.28%的IoU和92.67%的Dice系数，相较于传统插值方法相对性能提升136.1%。此外，消融实验表明，PSNR指标与裂缝检测精度并非正相关。本研究为低质量路面图像的裂缝检测提供了有效方案，也为任务驱动的图像增强网络设计提供了参考。

**关键词：** 超分辨率重建；路面裂缝检测；深度学习；U-Net；语义分割


## gemini优化版

路面裂缝的自动化检测是道路养护评估的核心环节。然而，在实际工程场景中采集的图像常因设备限制与物理环境干扰，存在分辨率低、边缘模糊等严重降质问题，导致常规深度学习模型的分割精度呈现断崖式下降。针对此问题，本文提出了一种基于超分辨率重建（SR）与像素级语义分割相融合的两阶段深度学习框架。在前端重建阶段，本文对经典的SRCNN网络进行了结构重塑，通过剥离易平滑图像高频细节的批量归一化（BatchNorm）层，并设计了具有不同感受野尺寸的5层卷积架构（Improved SRCNN），成功将测试集峰值信噪比（PSNR）提升至30.09 dB。在后端分割阶段，采用U-Net网络对重建后的高清特征进行裂缝拓扑提取。在路面裂缝数据集上的实证研究表明，本文提出的级联方法在分割任务中取得了高达87.28%的交并比（IoU）和92.67%的Dice系数；相较于传统的双线性与双三次插值基线（IoU仅为36.96%），相对性能提升突破了136.1%。此外，本文的消融实验进一步揭示了面向人类视觉感知的PSNR指标与下游机器视觉理解任务之间存在的细微非正相关性。本研究不仅为低质量路面图像的裂缝检测提供了高鲁棒性的解决方案，也为面向特定任务的图像增强网络设计提供了新的实证依据。

---

# Abstract

Road crack detection is an essential component of pavement maintenance. However, images collected in real-world engineering scenarios often suffer from degradation issues such as blur and low resolution due to equipment limitations and environmental interference, severely affecting detection accuracy. To address this issue, this paper proposes a road crack detection method based on super-resolution reconstruction. The method adopts a two-stage deep learning framework: the first stage performs structural optimization on the classic SRCNN network, successfully improving PSNR to 30.09 dB, and conducts super-resolution reconstruction on low-quality images to recover pavement texture details; the second stage employs a U-Net network for pixel-level crack segmentation on the reconstructed images. Experimental results demonstrate that the proposed cascaded method achieves 87.28% IoU and 92.67% Dice coefficient in segmentation tasks, with a 136.1% relative performance improvement compared to traditional interpolation methods. Furthermore, ablation experiments reveal that PSNR is not positively correlated with crack detection accuracy. This study provides an effective solution for crack detection in low-quality pavement images and offers reference for task-driven image enhancement network design.

**Keywords:** Super-Resolution Reconstruction; Road Crack Detection; Deep Learning; U-Net; Semantic Segmentation

---

## 实验结果汇总

### SRCNN 超分辨率性能
| 模型 | PSNR (dB) | SSIM |
|-----|-----------|------|
| Basic SRCNN | 27.41 | 0.9076 |
| Improved SRCNN (5层) | 30.09 | 0.9482 |
| Improved SRCNN (7层, 全3x3) | 30.81 | 0.9545 |

### U-Net 分割性能
| 方法 | IoU | Dice | Accuracy |
|-----|-----|------|----------|
| Bilinear + U-Net | 36.96% | 49.41% | 95.70% |
| Original HR + U-Net | 59.55% | 72.87% | 97.20% |
| Basic SRCNN + U-Net | 86.84% | 92.47% | 99.44% |
| **Improved SRCNN + U-Net** | **87.28%** | **92.67%** | **99.47%** |

### 关键提升
- SRCNN vs Bilinear: +49.32% IoU (相对提升 +135.9%)
- SRCNN vs Original HR: +27.73% IoU (相对提升 +46.5%)
