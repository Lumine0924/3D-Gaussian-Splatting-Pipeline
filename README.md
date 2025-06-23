# 3D-Gaussian-Splatting-Pipeline
A comprehensive end-to-end Python pipeline for 3D object reconstruction and novel-view synthesis using 3D Gaussian Splatting, with COLMAP-based camera estimation, NeRF baselines comparison, GPU/CPU support, and quantitative PSNR/SSIM evaluation.


## Repository Structure

```
3D-Gaussian-Splatting-Pipeline/
├── data/
│   ├── train_images/    # 多视角输入图片
│   ├── test_images/     # 保留评估用测试图片
│   └── videos/          # 原始视频文件（如有），例如 my_object.mp4
├── gaussian_splatting/  # 官方 3D Gaussian Splatting 代码库
├── nerf_pytorch/        # NeRF 官方/加速版代码库
├── output/
│   ├── gaussian/        # Gaussian Splatting 渲染结果
│   ├── nerf_base/       # NeRF 基线渲染结果
│   ├── nerf_acc/        # NeRF 加速版渲染结果
│   └── gaussian.mp4     # 输出的视频文件
├── environment.yml      # Conda 环境配置
├── requirements.txt     # Python 依赖列表（pip）
├── 3d_gaussian_splatting_pipeline.py  # 主脚本
└── README.md            # 文档说明
```

## Installation

使用 Conda（推荐）或 pip 安装依赖：

```bash
conda env create -f environment.yml
conda activate gaussian_splat
```

或者：

```bash
pip install -r requirements.txt
```

确保已安装：

* COLMAP（命令行工具在 PATH 中）
* CUDA Toolkit + GPU 驱动（若使用 GPU）

## Data Preparation

### 输入多视角图片

在开始之前，请先确保文件夹已创建：

```bash
mkdir -p data/train_images data/test_images
```

将所有拍摄的 `.jpg`/`.png` 图片直接放入 `data/train_images/` 文件夹。

* 如果你只有照片，不需要 `data/videos/`。
* 脚本会自动遍历 `data/train_images/` 中的所有图片进行后续处理。

### 输入视频

如果你是通过视频拍摄：

1. 将视频文件放入 `data/videos/`（例如 `data/videos/my_object.mp4`）。
2. 使用 FFmpeg 提取视频帧到 `data/train_images/`：

   ```bash
   mkdir -p data/train_images
   ffmpeg -i data/videos/my_object.mp4 data/train_images/frame_%04d.png
   ```
3. 然后仍通过 `--images data/train_images` 参数进行管道流程。

---

## Usage

```bash
python 3d_gaussian_splatting_pipeline.py \
    --images data/train_images \
    --test_images data/test_images
```

```bash
python 3d_gaussian_splatting_pipeline.py \
    --images data/train_images \
    --test_images data/test_images
```

脚本将：

1. 使用 COLMAP 执行特征提取、匹配及稀疏重建，生成 `data/colmap/transforms.json`
2. 调用 `gaussian_splatting/train.py` 训练 3D Gaussian 模型
3. 渲染新视角视频，结果保存为 `output/gaussian.mp4`
4. 训练 NeRF 基线及加速版，并在 `output/nerf_base/`、`output/nerf_acc/` 生成渲染图像
5. 对比三种方法在 `data/test_images/` 上的 PSNR/SSIM，日志输出评估结果

## Video Output Location

渲染得到的视频文件默认保存在项目根目录下的 `output/gaussian.mp4`。你也可通过脚本中的 `OUTPUT_DIR` 常量修改保存位置。

