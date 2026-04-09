# 2. 相关工作

## 2.1 图像超分辨率

图像超分辨率（Super-Resolution, SR）旨在从低分辨率图像重建高分辨率图像。早期方法主要依赖双线性插值（Bilinear）和双三次插值（Bicubic），通过对相邻像素加权实现放大。这类方法计算效率高，但本质上是对相邻像素的平滑操作，因此难以恢复丢失的高频纹理，难以满足检测路面微小裂缝的需求。

随着深度学习的兴起，Dong等[3]提出了SRCNN，首次通过三层卷积网络实现了从低分辨率到高分辨率的端到端非线性映射，显著超越了传统插值算法。后续的研究中，Lim等[5]提出的EDSR移除了批量归一化层（BatchNorm），其研究表明批量归一化层会限制网络特征的动态范围灵活性，反而容易抹除图像的高频边缘细节。

然而，现有的超分辨率研究多聚焦于自然图像的重建，对特定下游任务的影响探讨较少。Lim等[5]提出的理论发现为本文在改进SRCNN结构以及进行消融实验时提供了坚实的理论依据。

## 2.2 语义分割

语义分割旨在为图像中每一个像素分配类别标签。Ronneberger等[4]提出的U-Net网络采用了经典的编码器-解码器对称架构，并通过引入跳跃连接将浅层的细节传递给深层，有效弥补了多次采样下的空间信息丢失。其在路面裂缝检测领域已被广泛采用。Zhang等[2]将卷积神经网络应用于三维沥青表面的像素级裂缝检测；Yang等[6]结合特征金字塔网络改善了多尺度裂缝的识别；Zou等[7]验证了U-Net在裂缝分割中的有效性。然而，上述研究均建立在输入图像质量较高的假设之上。当输入为低质量或者仅通过双线性插值放大的模糊图像时，由于丢失了高频边界特征，其鲁棒性的探讨有限。这促使本文在进行U-Net分割前引入超分辨率模型进行前置处理。

## 2.3 路面裂缝检测与数据集

早期的路面检测主要依靠如阈值分割、边缘检测等传统图像处理技术[1]，这类方法对光照变化和复杂的沥青纹理非常敏感，因此难以适应真实的工程环境。而近年来，深度学习方法逐渐凭借其强大的特征提取能力成为主流。在数据集方面，CRACK500是广泛使用的基准数据集之一。其原始图像尺寸较大且包含了真实路面干扰。本文的实验基于对其重新划分了自定义子集，以确保模型训练的可靠性。

## 2.4 现有方法的不足

综合上述分析，现有研究存在以下局限：

1. 超分辨率与裂缝分割通常被视为独立任务，缺乏面向具体任务的联合优化策略；

2. 超分辨率网络设计通常追求通用重建质量，未考虑如裂缝等细节的重建需求；

3. 现有的裂缝检测方法通常基于高质量图像，对实际工程中的低质量场景覆盖不足。

针对以上问题，本研究提出超分辨率增强与裂缝分割的级联框架，并通过实验探究重建指标与分割性能的关系，为低质量路面图像的检测提供有效的解决方案。

---

## 参考文献

[1] Koch C, Georgieva K, Kasireddy V, et al. A review on computer vision based defect detection and condition assessment of concrete and asphalt civil infrastructures[J]. Advanced Engineering Informatics, 2015, 29(2): 196-210.

[2] Zhang A, Wang K C P, Li B, et al. Automated pixel-level pavement crack detection on 3D asphalt surfaces using a deep-learning network[J]. Computer-Aided Civil and Infrastructure Engineering, 2017, 32(10): 805-819.

[3] Dong C, Loy C C, He K, et al. Learning a deep convolutional network for image super-resolution[C]//European Conference on Computer Vision (ECCV). Springer, 2014: 184-199.

[4] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional networks for biomedical image segmentation[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI). Springer, 2015: 234-241.

[5] Lim B, Son S, Kim H, et al. Enhanced deep residual networks for single image super-resolution[C]//IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). 2017: 1132-1140.

[6] Yang F, Zhang L, Yu S, et al. Feature pyramid and hierarchical boosting network for pavement crack detection[J]. IEEE Transactions on Intelligent Transportation Systems, 2019, 21(4): 1525-1535.

[7] Zou Q, Zhang Z, Li Q, et al. Automated pavement crack segmentation using U-Net-based convolutional neural network[J]. IEEE Access, 2020, 8: 114892-114899.
