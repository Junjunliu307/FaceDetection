
# Face and Mouth Detection using YOLOv8 and Mediapipe

## 简介
此项目结合了 **YOLOv8** 和 **Mediapipe** 实现了实时人脸检测与嘴唇开合判断的功能。检测结果会存储在指定的文本文件中，同时在视频中绘制相应的标注框（红色表示张嘴，绿色表示闭嘴）。

---

## 功能
1. 使用 **YOLOv8** 检测视频中的人脸。
2. 使用 **Mediapipe** 提取嘴唇关键点，判断嘴唇是否张开。
3. 将人脸坐标和嘴唇开合状态分别保存到两个文本文件中。
4. 在视频中显示人脸框和嘴唇状态。

---

## 安装依赖
运行代码之前，请确保安装以下依赖库：

```bash
pip install opencv-python torch torchvision ultralytics mediapipe
```

---

## 文件说明
- `VideoProduct.py`：项目主文件，包含人脸检测和嘴唇开合判断的逻辑。
- `yolov8n-face.pt`：YOLOv8 模型权重文件，用于人脸检测。

---

## 使用方法
1. **准备输入视频**：
   - 将输入视频命名为 `input_video.mp4` 或修改代码中的 `video_path` 变量。

2. **运行代码**：
   - 使用以下命令运行项目：
     ```bash
     python VideoProduct.py
     ```

3. **生成输出**：
   - 处理后的视频会保存为 `output_video.mp4`。
   - 检测到的人脸框坐标会保存到 `faceDetections.txt` 文件中。
   - 检测到张嘴状态的帧信息会保存到 `speakDetections.txt` 文件中。

---

## 输出示例
- **`faceDetections.txt`**：
  ```
  Frame 1: [100, 50, 200, 150] Confidence: 0.95
  Frame 2: [120, 60, 220, 160] Confidence: 0.90
  ```

- **`speakDetections.txt`**：
  ```
  Frame 1: [100, 50, 200, 150]
  Frame 3: [120, 60, 220, 160]
  ```

---

## 注意事项
1. 请确保 `yolov8n-face.pt` 文件与主代码在同一目录下。
2. 输入视频分辨率较高时，处理速度可能会变慢。
3. 如果需要调整嘴唇开合的灵敏度，请修改 `is_mouth_open` 函数中的 `threshold` 参数。
