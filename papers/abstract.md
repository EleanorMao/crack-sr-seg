# 摘要

路面裂缝检测是道路养护的重要环节。但在实际工程中采集的图像常因设备限制、环境干扰等因素存在模糊、低分辨率等降质问题，严重影响检测精度。针对这一问题，本文提出一种基于超分辨率重建的路面裂缝检测方法。该方法采用两阶段深度学习框架：第一阶段通过对经典的SRCNN网络进行结构优化，成功将PSNR提升至30.09 dB，恢复了低质量图像的路面纹理细节；第二阶段采用U-Net网络对重建后的图像进行裂缝像素级分割。实验结果表明，本文提出的级联方法在分割任务中取得了高达87.28%的IoU和92.67%的Dice系数，相较于传统插值方法相对性能提升136.1%。此外，消融实验表明，PSNR指标与裂缝检测精度并非正相关。本研究为低质量路面图像的裂缝检测提供了有效方案，也为任务驱动的图像重建网络设计提供了参考。

**关键词：** 超分辨率重建；路面裂缝检测；深度学习；U-Net；语义分割

---

# Abstract

Road crack detection is an essential component of pavement maintenance. However, images collected in real-world engineering scenarios often suffer from degradation issues such as blur and low resolution due to equipment limitations and environmental interference, severely affecting detection accuracy. To address this issue, this paper proposes a road crack detection method based on super-resolution reconstruction. The method adopts a two-stage deep learning framework: the first stage performs structural optimization on the classic SRCNN network, successfully improving PSNR to 30.09 dB, and conducts super-resolution reconstruction on low-quality images to recover pavement texture details; the second stage employs a U-Net network for pixel-level crack segmentation on the reconstructed images. Experimental results demonstrate that the proposed cascaded method achieves 87.28% IoU and 92.67% Dice coefficient in segmentation tasks, with a 136.1% relative performance improvement compared to traditional interpolation methods. Furthermore, ablation experiments reveal that PSNR is not positively correlated with crack detection accuracy. This study provides an effective solution for crack detection in low-quality pavement images and offers reference for task-driven image enhancement network design.

**Keywords:** Super-Resolution Reconstruction; Road Crack Detection; Deep Learning; U-Net; Semantic Segmentation
