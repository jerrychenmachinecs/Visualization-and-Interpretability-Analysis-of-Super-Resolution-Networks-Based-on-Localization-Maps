# 基于局部定位图（LAM）的超分网络可视化与可解释性分析

## 项目概述

本项目包含三个独立的Python子程序，用于超分辨率神经网络的可视化和可解释性分析。每个模块都是完整的、可独立运行的。

---

## 模块说明

### 1. LAM可视化模块 (lam_visualization_module.py)
**行数**: ~600行  
**功能**: 局部定位图生成和可视化

#### 主要功能：
- 实现多种LAM生成方法：
  - 滑动窗口遮挡法 (Sliding Window Occlusion)
  - 梯度归因法 (Gradient-based Attribution)
  - 积分梯度法 (Integrated Gradients)
- 提供丰富的可视化工具
- 支持统计分析和方法对比

#### 核心类：
- `SimpleSRNetwork`: 简化的超分辨率网络
- `LAMGenerator`: LAM生成器
- `LAMVisualizer`: LAM可视化工具

#### 运行方式：
```bash
python lam_visualization_module.py
```

---

### 2. 超分网络训练模块 (sr_training_module.py)
**行数**: ~700行  
**功能**: 完整的超分辨率网络训练框架

#### 主要功能：
- 实现先进的超分网络架构（带通道注意力机制）
- 完整的训练/验证流程
- 多种优化器和学习率调度器
- 综合的性能评估（PSNR、SSIM等）
- 自动检查点保存和训练可视化

#### 核心类：
- `AdvancedSRNetwork`: 高级超分网络（带RCAB）
- `SyntheticSRDataset`: 合成数据集生成器
- `SRTrainer`: 训练器
- `MetricsCalculator`: 评估指标计算器

#### 运行方式：
```bash
python sr_training_module.py
```

---

### 3. 可解释性分析模块 (interpretability_analysis_module.py)
**行数**: ~800行  
**功能**: 多种可解释性方法的对比分析

#### 主要功能：
- 实现多种可解释性方法：
  - Grad-CAM
  - SmoothGrad
  - DeepLIFT
  - Vanilla Gradient
- 量化方法一致性分析
- 灵敏度和稀疏性分析
- 特征图可视化
- 综合对比报告生成

#### 核心类：
- `GradCAM`: Grad-CAM实现
- `SmoothGrad`: SmoothGrad实现
- `DeepLIFT`: DeepLIFT实现
- `FeatureVisualizer`: 特征可视化器
- `InterpretabilityComparator`: 可解释性对比框架

#### 运行方式：
```bash
python interpretability_analysis_module.py
```

---

## 依赖环境

### 必需库：
```bash
pip install torch torchvision numpy matplotlib opencv-python pillow scipy seaborn scikit-learn tqdm
```

### 推荐环境：
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (可选，用于GPU加速)

---

## 使用说明

### 快速开始

1. **安装依赖**：
```bash
pip install torch torchvision numpy matplotlib opencv-python scipy seaborn scikit-learn tqdm
```

2. **运行单个模块**：
```bash
# 运行LAM可视化演示
python lam_visualization_module.py

# 运行训练演示
python sr_training_module.py

# 运行可解释性分析演示
python interpretability_analysis_module.py
```

### 模块间集成

三个模块可以组合使用：

```python
# 示例：训练模型后进行可解释性分析
from sr_training_module import AdvancedSRNetwork, TrainingConfig
from interpretability_analysis_module import InterpretabilityComparator

# 1. 创建和训练模型
config = TrainingConfig(scale_factor=2, num_blocks=8)
model = AdvancedSRNetwork(config)
# ... 训练过程 ...

# 2. 进行可解释性分析
comparator = InterpretabilityComparator(model)
attributions = comparator.compare_methods(test_image)
comparator.visualize_comparison(input_image, attributions)
```

---

## 输出说明

### 模块1输出：
- `./lam_results/comprehensive_lam_analysis.png`: LAM综合分析图
- `./lam_results/lam_statistics.png`: LAM统计分析图

### 模块2输出：
- `./checkpoints/`: 模型检查点
- `./logs/training_curves.png`: 训练曲线
- `./logs/training_config.json`: 训练配置

### 模块3输出：
- `./interpretability_results/methods_comparison.png`: 方法对比图
- `./interpretability_results/agreement_analysis.png`: 一致性分析图
- `./interpretability_results/sensitivity_analysis.png`: 灵敏度分析图
- `./interpretability_results/feature_maps.png`: 特征图可视化

---

## 技术特点

### 1. LAM可视化模块
- ✅ 多种归因方法实现
- ✅ 交互式可视化
- ✅ 统计分析工具
- ✅ 方法相关性分析

### 2. 训练模块
- ✅ 先进的网络架构（RCAB）
- ✅ 灵活的训练配置
- ✅ 完整的评估指标
- ✅ 自动化训练流程

### 3. 可解释性分析模块
- ✅ 5+种解释方法
- ✅ 量化对比分析
- ✅ 特征可视化
- ✅ 雷达图综合展示


---

## 应用场景

1. **学术研究**: 用于超分辨率网络的可解释性研究
2. **模型分析**: 理解网络决策机制
3. **教学演示**: 深度学习可解释性教学
4. **工程实践**: 优化网络设计和性能

---

## 扩展建议

1. 添加更多超分网络架构（ESRGAN, SwinIR等）
2. 支持真实图像数据集（DIV2K, Set5等）
3. 实现更多可解释性方法（LIME, SHAP等）
4. 添加用户交互界面（Gradio, Streamlit）

---

## 注意事项

- 首次运行会生成示例数据，可能需要几分钟
- GPU可以显著加速运算，但不是必需的
- 建议使用虚拟环境管理Python依赖

---


---

**版本**: 1.0  
**日期**: 2025-10  
**许可**: MIT License
