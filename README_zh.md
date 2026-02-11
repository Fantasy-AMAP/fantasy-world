[English](./README.md)
# FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction



[![Home Page](https://img.shields.io/badge/🌐%20%20Project-FantasyWorld-blue.svg)](https://fantasy-amap.github.io/fantasy-world/)
[![arXiv](https://img.shields.io/badge/Arxiv-2509.21657-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.21657)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-FFD21E.svg)](https://huggingface.co/acvlab/FantasyWorld)
[![Code](https://img.shields.io/badge/Code-GitHub-181717.svg?logo=GitHub)](https://github.com/Fantasy-AMAP/fantasy-world.git)

一个具备联合深度与相机估计能力的多阶段视频生成框架。

## 🔥🔥🔥 最新动态!!
  2026/02/10：👋 我们发布了 FantasyWorld 的代码与模型权重。


## 🌟 概述

FantasyWorld 是一个面向视频生成并具备 3D 场景理解能力的两阶段训练框架。它包含：
- **阶段 1**：VGGT-style模型，用于深度、点云与相机参数估计。
- **阶段 2**：将VGGT-style模型与Wan视频生成流水线进行联合集成的模型。



## 🚀 快速开始


### 安装

1. **克隆仓库**
```bash
git clone https://github.com/Fantasy-AMAP/fantasy-world.git
cd fantasy-world
```

2. **安装依赖**

```bash
conda create -n fantasyworld python=3.10
conda activate fantasyworld
pip install -r requirements.txt
pip install third_party/utils3d/
```
### 模型下载

| 模型	        |                       下载链接	                                           |    备注                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-I2V-14B-480P  |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)    🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P)     | Base model

| FantasyWorld model      |   🤗 [Huggingface]coming soon    🤖 [ModelScope] coming soon    | Our FantasyWorld weights

使用 huggingface-cli 下载模型：

``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./models/Wan2.1-I2V-14B-480P
#Our FantasyWorld weights are coming soon
```

使用 modelscope-cli 下载模型：

``` sh
pip install modelscope
modelscope download Wan-AI/Wan2.1-I2V-14B-480P --local_dir ./models/Wan2.1-I2V-14B-480P
#Our FantasyWorld weights are coming soon
```

### 推理

```bash
python inference.py \
    --wan_ckpt_path ./models/Wan2.1-I2V-14B-480P \
    --model_ckpt ./models/fantasyworld-0_model.pth \
    --image_path examples/images/input_image.png \
    --camera_json_path examples/cameras/camera_data.json \
    --prompt "In the Open Loft Living Room, sunlight streams through large windows, highlighting the sleek fireplace and elegant wooden stairs." \
    --output_dir ./output \
    --sample_steps 50 \
    --using_scale True
```

**参数说明:**
- `--wan_ckpt_path` - **必填**: Wan模型checkpoint目录
- `--model_ckpt` - **必填**: 训练好的模型checkpoint路径
- `--image_path` - **必填**: 输入图片路径
- `--camera_type` - **必填**: 相机轨迹路径，对应 `examples/cameras/camera_data_*.json`
- `--prompt` - **必填**: 文本提示词
- `--neg_prompt` - **可选**: 负面提示词
- `--output_dir` - **可选**: 输出目录，默认为输入图片所在目录
- `--fps` - **可选**: 帧率，默认16
- `--sample_steps` - **可选**: 采样步数，默认50
- `--using_scale` - **可选**: 是否使用scale归一化，默认True
- `--height` - **可选**: 视频高度，默认336
- `--width` - **可选**: 视频宽度，默认592
- `--frames` - **可选**: 帧数，默认81



## 🧩 社区工作
我们❤️喜欢来自开源社区的贡献！如果你的工作改进了FantasyWorld，请告诉我们。


## 🔗Citation
如果你觉得这个仓库有用，请考虑点个赞⭐并引用
```
@inproceedings{
dai2025fantasyworld,
title={FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction},
author={Yixiang Dai and Fan Jiang and Chiyu Wang and Mu Xu and Yonggang Qi},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=3q9vHEqsNx}
}
```

<!-- ## 📄 License

[Add your license information here] -->

## 🙏 致谢

本项目在以下开源代码库基础上构建:
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [VGGT](https://github.com/facebookresearch/vggt.git)



