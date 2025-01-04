
# FaceDetection

本项目包含两个主要功能：
1. 将视频中的人脸检测结果保存为文本文件。
2. 处理视频并将人脸检测框绘制到输出视频中。

## 环境要求

请确保已安装以下库：
```bash
pip install opencv-python torch torchvision ultralytics
```

## 文件说明

- **`TextProduct.py`**  
  - 功能：从输入视频中检测人脸，并将每帧的检测框坐标和置信度保存到文本文件 `detections.txt`。
  - 使用方法：修改代码中的 `video_path` 指定输入视频路径，然后运行脚本。
  
- **`VideoProduct.py`**  
  - 功能：从输入视频中检测人脸，并在输出视频中绘制检测框和置信度。
  - 使用方法：修改代码中的 `video_path` 指定输入视频路径，然后运行脚本。

- **`yolov8n-face.pt`**  
  - YOLOv8 人脸检测模型的预训练权重文件。

## 使用方法

### 1. 安装依赖
在项目根目录运行以下命令：
```bash
pip install opencv-python torch torchvision ultralytics
```

### 2. 检测人脸并保存结果到文本
运行 `TextProduct.py`：
```bash
python TextProduct.py
```
结果将保存到 `detections.txt` 文件中。

### 3. 检测人脸并生成视频
运行 `VideoProduct.py`：
```bash
python VideoProduct.py
```
输出的视频文件为 `output_video.mp4`。
