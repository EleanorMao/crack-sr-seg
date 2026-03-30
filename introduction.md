# 1. 引言

## 1.1 研究背景与意义

道路基础设施是现代城市和国家发展的重要基石，其健康状况直接关系到交通运输安全和经济发展。路面裂缝作为道路病害的早期表现形式，若不及时发现和修复，将逐渐扩展为坑槽、车辙等严重病害，不仅影响行车舒适度，更可能引发交通安全事故[1]。因此，建立高效、准确的路面裂缝检测机制，对于保障道路安全、降低维护成本、延长道路使用寿命具有重要的现实意义。

传统的路面裂缝检测主要依赖人工巡检，这种方法存在效率低、主观性强、存在安全隐患等明显缺陷。随着计算机视觉和深度学习技术的快速发展，基于图像的自动化裂缝检测方法逐渐成为研究热点，深度学习方法在裂缝检测任务中展现出优异的性能。

## 1.2 问题描述

尽管深度学习方法在路面裂缝检测领域取得了显著进展，但现有研究大多假设输入图像具有较高的质量。然而，在实际应用场景中，采集的路面图像常因以下因素导致质量下降：

（1）**设备限制**：无人机或车载采集设备在运动过程中产生的运动模糊、散焦等问题；
（2）**环境干扰**：光照条件不佳、天气因素（雨、雾）等造成的图像对比度降低；
（3）**传输损失**：图像在无线传输过程中因带宽限制而遭受压缩失真。

这些低质量图像中，裂缝的边缘信息模糊、纹理细节丢失，直接应用于现有的裂缝检测模型时，性能将显著下降。如何提升低质量路面图像的裂缝检测精度，成为亟待解决的实际问题。

## 1.3 相关工作

### 1.3.1 图像超分辨率

图像超分辨率（Super-Resolution, SR）旨在从低分辨率图像恢复出高分辨率图像。Dong等[2]开创性地提出了SRCNN（Super-Resolution Convolutional Neural Network），首次将深度学习引入单图像超分辨率领域，通过卷积神经网络学习低分辨率到高分辨率的端到端映射，取得了超越传统插值方法的效果，奠定了深度学习超分辨率研究的基础。

### 1.3.2 语义分割

语义分割是计算机视觉的核心任务之一。Ronneberger等[3]提出的U-Net网络针对生物医学图像分割任务，设计了编码器-解码器结构配合跳跃连接的架构，在少量标注数据下取得了优异的分割效果。U-Net因其简洁高效的设计，已被广泛应用于医学影像、遥感图像、工业检测等多个领域，同样适用于路面裂缝分割任务。

## 1.4 本文贡献

针对低质量路面图像裂缝检测的挑战，本文提出一种基于超分辨率增强的两阶段深度学习方法。主要贡献如下：

（1）**两阶段检测框架**：第一阶段利用改进的SRCNN网络对低质量图像进行超分辨率重建；第二阶段采用U-Net网络对恢复后的图像进行裂缝像素级分割。

（2）**显著性能提升**：在CRACK500数据集上，本文方法在裂缝分割任务上达到87.28% IoU，相比传统双线性插值方法（36.96%）提升了135.9%。

（3）**全面实验验证**：提供了详细的对比实验，验证了学习型超分辨率方法相比传统插值方法的显著优势。

---

## 参考文献

[1] Koch C, Georgieva K, Kasireddy V, et al. A review on computer vision based defect detection and condition assessment of concrete and asphalt civil infrastructures[J]. Advanced Engineering Informatics, 2015, 29(2): 196-210. DOI: 10.1016/j.aei.2015.01.008

[2] Dong C, Loy C C, He K, et al. Learning a deep convolutional network for image super-resolution[C]//European Conference on Computer Vision (ECCV). Springer, 2014: 184-199.

[3] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional networks for biomedical image segmentation[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI). Springer, 2015: 234-241.

---

## 格式说明

- 小四字体下，引言部分约 1 页
- 引用 3 篇核心文献（符合2-5个要求）
- 包含 4 个小节：背景意义、问题描述、相关工作、本文贡献
