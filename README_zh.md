[English](./README.md)
# FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction



[![Home Page](https://img.shields.io/badge/🌐%20%20Project-FantasyWorld-blue.svg)](https://fantasy-amap.github.io/fantasy-world/)
[![arXiv](https://img.shields.io/badge/Arxiv-2509.21657-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.21657)
[![Code](https://img.shields.io/badge/Code-GitHub-181717.svg?logo=GitHub)](https://github.com/Fantasy-AMAP/fantasy-world.git)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Wan2.1-FFD21E)](https://huggingface.co/acvlab/FantasyWorld-Wan2.1-I2V-14B-480P)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Wan2.2-FFD21E)](https://huggingface.co/acvlab/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera)
[![ModelScope](https://img.shields.io/badge/ModelScope-Wan2.1-624AFF)](https://modelscope.cn/amap_cvlab/FantasyWorld-Wan2.1-I2V-14B-480P)
[![ModelScope](https://img.shields.io/badge/ModelScope-Wan2.2-624AFF)](https://modelscope.cn/amap_cvlab/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera)


## 🔥🔥🔥 最新动态
- 👋 2026年2月：我们正式发布了 FantasyWorld 的代码和模型权重。
- 🏛 2026年1月：FantasyWorld 被 **ICLR 2026** 接收。
- 🎉 2025年12月：FantasyWorld 在 [WorldScore](https://huggingface.co/spaces/Howieeeee/WorldScore_Leaderboard) 排行榜（由斯坦福大学李飞飞教授团队推出）中荣获 **第一名**，在与全球最先进模型（SOTA）的对比中验证了我们方法的有效性。

## 🌟 概览

![Overview](assets/overview.png)

FantasyWorld 是一个用于联合视频和 3D 场景生成的统一前馈模型。其前端采用**预调节模块 (Preconditioning Blocks, PCBs)**，通过复用冻结的 WanDiT 去噪器来提供部分去噪的隐变量，确保几何路径在有意义的特征上而非纯噪声上进行操作。骨干网络由堆叠的**重建生成一体化模块 (Integrated Reconstruction and Generation, IRG) 块**组成，在多模态条件下迭代优化视频隐变量和几何特征。每个 IRG 模块包含一个非对称的双分支结构：用于外观合成的**想象先验分支 (Imagination Prior Branch)** 和用于显式 3D 推理的**几何一致性分支 (Geometry-Consistent Branch)**，两者通过轻量级适配器和交叉注意力机制进行耦合。

### 🚀 训练策略

FantasyWorld 利用一种**两阶段训练策略**来实现稳健的视频与 3D 的联合生成：

- **阶段 1 (几何预训练)：** 利用 VGGT 风格的模型，对深度、点云和相机轨迹进行精确估计。
- **阶段 2 (联合生成)：** 一个统一的模型，无缝集成了阶段 1 的几何骨干网络与 Wan 视频生成流程。

### 📦 模型库

我们提供两个版本的模型，以满足不同的研究和应用需求：

| 模型名称 | 描述 |
| :--- | :--- |
| `FantasyWorld-Wan2.1-I2V-14B-480P` | **侧重复现性：** 严格遵守我们论文中详述的原始配置。最适合学术基准测试和复现报告结果。 |
| `FantasyWorld-Wan2.2-Fun-A14B-Control-Camera` | **侧重性能：** 提供实质性的增强，包括升级的视频基础模型、更大规模的训练数据集以及更高的输出分辨率。 |

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
pip install thirdparty/utils3d/
```
### 1. FantasyWorld-Wan2.1-I2V-14B-480P

#### 1.1 模型下载

| 模型	        |                       下载链接	                                           |    备注                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-I2V-14B-480P  |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)    🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P)     | Base Model
| FantasyWorld-Wan2.1-I2V-14B-480P      |   🤗 [Huggingface](https://huggingface.co/acvlab/FantasyWorld-Wan2.1-I2V-14B-480P)    🤖 [ModelScope](https://www.modelscope.cn/models/amap_cvlab/FantasyWorld-Wan2.1-I2V-14B-480P)    | FantasyWorld

使用 `huggingface` 下载模型：

```bash
pip install -U "huggingface_hub"
hf download "Wan-AI/Wan2.1-I2V-14B-480P" --local-dir ./models/Wan-AI/Wan2.1-I2V-14B-480P
hf download "acvlab/FantasyWorld-Wan2.1-I2V-14B-480P" --local-dir ./models/FantasyWorld-Wan2.1-I2V-14B-480P/
```

使用 `modelscope` 下载模型：

```bash
pip install -U modelscope
modelscope download "Wan-AI/Wan2.1-I2V-14B-480P" --local_dir ./models/Wan-AI/Wan2.1-I2V-14B-480P
modelscope download "amap_cvlab/FantasyWorld-Wan2.1-I2V-14B-480P" --local_dir ./models/FantasyWorld-Wan2.1-I2V-14B-480P/
```

#### 1.2 推理命令

```bash
python inference_wan21.py \
    --wan_ckpt_path ./models/Wan-AI/Wan2.1-I2V-14B-480P \
    --model_ckpt ./models/FantasyWorld-Wan2.1-I2V-14B-480P/model.pth \
    --image_path ./examples/images/input_image.png \
    --camera_json_path ./examples/cameras/camera_data.json \
    --prompt "In the Open Loft Living Room, sunlight streams through large windows, highlighting the sleek fireplace and elegant wooden stairs." \
    --output_dir ./output-wan21 \
    --sample_steps 50 \
    --using_scale True 
```

**参数说明:**
- `--wan_ckpt_path` - **必填**: Wan模型checkpoint目录
- `--model_ckpt` - **必填**: 训练好的模型checkpoint路径
- `--image_path` - **必填**: 输入图片路径
- `--camera_json_path` - **必填**: 相机轨迹路径
- `--prompt` - **必填**: 文本提示词
- `--output_dir` - **可选**: 输出目录，默认为输入图片所在目录
- `--sample_steps` - **可选**: 采样步数，默认50
- `--using_scale` - **可选**: 是否使用scale归一化，默认True

### 2. FantasyWorld-Wan2.2-Fun-A14B-Control-Camera

#### 2.1 模型下载

| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.2-Fun-A14B-Control-Camera  |      🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control-Camera)    🤖 [ModelScope](https://www.modelscope.ai/models/PAI/Wan2.2-Fun-A14B-Control-Camera)     | Base Model
| Wan2.2-Fun-Reward-LoRAs      |   🤗 [Huggingface](https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs)    🤖 [ModelScope](https://www.modelscope.ai/models/PAI/Wan2.2-Fun-Reward-LoRAs)    | LoRA Model
| FantasyWorld-Wan2.2-Fun-A14B-Control-Camera      |   🤗 [Huggingface](https://huggingface.co/acvlab/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera)   🤖 [ModelScope](https://www.modelscope.ai/models/amap_cvlab/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera)    | FantasyWorld

使用 `huggingface` 下载模型：
```bash
pip install -U "huggingface_hub"
hf download "alibaba-pai/Wan2.2-Fun-A14B-Control-Camera" --local-dir ./models/PAI/Wan2.2-Fun-A14B-Control-Camera
hf download "alibaba-pai/Wan2.2-Fun-Reward-LoRAs" --local-dir ./models/PAI/Wan2.2-Fun-Reward-LoRAs
hf download "acvlab/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera" --local-dir ./models/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera/
```

使用 `modelscope` 下载模型：
```bash
pip install -U modelscope
modelscope download "PAI/Wan2.2-Fun-A14B-Control-Camera" --local_dir ./models/PAI/Wan2.2-Fun-A14B-Control-Camera
modelscope download "PAI/Wan2.2-Fun-Reward-LoRAs" --local_dir ./models/PAI/Wan2.2-Fun-Reward-LoRAs
modelscope download "amap_cvlab/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera" --local_dir ./models/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera/
```

#### 2.2 推理命令

```bash
python inference_wan22.py \
    --image_path ./examples/images/input_image.png \
    --end_image_path ./examples/images/end_image.png \
    --wan_ckpt_path ./models/ \
    --camera_json_path ./examples/cameras/camera_data.json \
    --prompt "In the Open Loft Living Room, sunlight streams through large windows, highlighting the sleek fireplace and elegant wooden stairs." \
    --model_ckpt_high ./models/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera/high_noise_model.pth \
    --model_ckpt_low ./models/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera/low_noise_model.pth \
    --output_dir ./output-wan22 \
    --sample_steps 50 \
    --using_scale True
```

**参数说明:**
- `--image_path` - **必填**: 首帧图像路径
- `--end_image_path` - **必填**: 尾帧图像路径
- `--wan_ckpt_path` - **必填**: Wan模型checkpoint目录
- `--camera_json_path` - **必填**: 相机轨迹路径
- `--prompt` - **必填**: 文本提示词
- `--output_dir` - **可选**: 输出目录
- `--sample_steps` - **可选**: 采样步数，默认50
- `--using_scale` - **可选**: 是否使用scale归一化，默认True

## 🧩 社区贡献

我们无比欢迎来自开源社区的贡献！❤️
如果您基于 FantasyWorld 进行了改进或开发了衍生项目，请务必告知我们。您也可以直接发送邮件至 [frank.jf@alibaba-inc.com](mailto://frank.jf@alibaba-inc.com)。

我们非常乐意在本项目中收录并展示您的工作，方便社区查阅、参考。


## 🔗Citation
如果 FantasyWorld 帮助到了您，还请给我们一个 star⭐ 或引用我们的论文以示鼓励。

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

我们非常感谢 [Wan](https://github.com/Wan-Video)、[VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)、[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 和 [VGGT](https://github.com/facebookresearch/vggt.git) 的出色工作。


