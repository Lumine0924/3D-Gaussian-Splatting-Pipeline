# 3D Gaussian Splatting & NeRF 一键流水线

本项目提供从数据采集到新视图合成的完整自动化工作流，主要功能包括：

1. **数据准备**：支持直接导入视频并自动提取关键帧，或导入多视角拍摄图片。
2. **相机参数标定**：调用 COLMAP 完成特征提取、匹配与稀疏重建，输出相机位姿信息。
3. **模型训练与渲染**：分别运行 3D Gaussian Splatting 与 NeRF（基础与加速版本）进行训练，并在指定轨迹下渲染环绕视频。
4. **定量评估**：根据用户选择的测试集（视频或图片），计算 PSNR/SSIM 指标，比较三种方法的合成效果。
5. **结果输出**：所有中间和最终结果均保存在 `output/` 目录，包含相机参数、渲染视频、对比图像及评估报告。

---

## 安装与依赖

* **操作系统**：Windows/macOS/Linux
* **必备环境**：Python 3.8+，COLMAP（命令行工具需加入系统 PATH）
* **可选加速**：NVIDIA 驱动与 CUDA Toolkit，用于 GPU 加速训练

脚本会自动安装以下 Python 库（使用 TUNA 镜像）：

```
numpy
Pillow
scikit-image
torch
torchvision
imageio-ffmpeg
```

如需手动安装：

```bash
pip install -r requirements.txt
```

---

## 快速使用指南

1. 克隆仓库并进入目录：

   ```bash
   ```

git clone https\://<your-repo-url>.git
cd <repo-folder>

````
2. 执行脚本：
   ```bash
python main.py
````

3. 按提示完成：

   * 选择 **训练视频**（或取消后选择**训练图片**文件夹）
   * 选择 **测试视频**（或取消后选择**测试图片**文件夹）
4. 等待脚本自动完成所有步骤。

执行结束后，查看 `output/`：

* `colmap/transforms.json`: 相机参数文件
* `gaussian/gaussian.mp4`: 3D Gaussian Splatting 渲染视频
* `nerf_base/renders/`、`nerf_acc/renders/`: NeRF 渲染结果
* 日志中显示的 PSNR/SSIM 对比结果

---

## 命令行参数

```text
--gaussian_repo   3D Gaussian Splatting 仓库路径，默认 `gaussian_splatting`
--nerf_repo       NeRF 代码仓库路径，默认 `nerf_pytorch`
--render_path     自定义渲染轨迹 JSON
--output          输出目录，默认 `output`
```

示例：

```bash
python main.py \
  --gaussian_repo ./3d-gaussian-splatting \
  --nerf_repo ./nerf_pytorch \
  --render_path ./path/render_trajectory.json \
  --output ./results
```

---

## 报告与分析

可将脚本中生成的 PSNR/SSIM 数值导出为 CSV/JSON，用于撰写实验报告及性能分析。

如需进一步定制或有任何问题，请提交 Issue 或联系维护者。
