# 摘要

路面裂缝检测是道路养护的重要环节，但实际采集的图像常因设备限制、环境干扰等因素存在模糊、低分辨率等质量问题，严重影响检测精度。针对这一问题，本文提出一种基于超分辨率增强的路面裂缝检测方法。该方法采用两阶段深度学习框架：第一阶段利用改进的SRCNN网络对低质量图像进行超分辨率重建，恢复路面纹理细节；第二阶段采用U-Net网络对恢复后的图像进行裂缝像素级分割。在CRACK500数据集上的实验结果表明，本文方法在裂缝分割任务上取得了较好的效果，IoU达到87.28%，相比双线性插值等传统方法提升了135.9%。本研究为低质量路面图像的裂缝检测提供了一种有效的解决方案。

**关键词：** 超分辨率；路面裂缝检测；深度学习；U-Net；图像分割

---

# Abstract

Road crack detection is crucial for pavement maintenance, but images collected in real-world scenarios often suffer from quality degradation such as blur and low resolution due to equipment limitations and environmental factors, which severely affects detection accuracy. To address this issue, this paper proposes a road crack detection method based on super-resolution enhancement. The method adopts a two-stage deep learning framework: the first stage utilizes an improved SRCNN network for super-resolution reconstruction of low-quality images to recover pavement texture details; the second stage employs a U-Net network for pixel-level crack segmentation on the restored images. Experimental results on the CRACK500 dataset demonstrate that our method achieves promising performance in crack segmentation tasks, with an IoU of 87.28%, improving by 135.9% compared to traditional methods such as bilinear interpolation. This study provides an effective solution for crack detection in low-quality pavement images.

**Keywords:** Super-Resolution; Road Crack Detection; Deep Learning; U-Net; Image Segmentation

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
