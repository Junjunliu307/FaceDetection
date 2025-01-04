import cv2
from ultralytics import YOLO

# 加载 YOLOv8 模型（使用通用模型或人脸专用权重文件）
model = YOLO("yolov8n-face.pt")

# 定义人脸类别索引（通常 YOLOv8 的 COCO 数据集中，人脸是类别 'person' 的一部分）
FACE_CLASS_ID = 0  # 如果权重专用于人脸，cls 可能为 0，需根据模型定义调整

# 打开视频文件或摄像头
video_path = "input_video.mp4"  # 替换为你的视频路径
cap = cv2.VideoCapture(video_path)

# 打开文本文件用于保存矩形框坐标
output_file = "detections.txt"
with open(output_file, "w") as f:
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1  # 记录帧号

        # 使用 YOLOv8 模型进行检测
        results = model(frame)  # 直接返回结果列表

        # 遍历检测结果
        detections = results[0].boxes  # 获取检测结果中的 'boxes' 对象
        for box in detections:
            # 提取检测框信息
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 检测框的坐标
            conf = box.conf[0]  # 检测置信度
            cls = int(box.cls[0])  # 检测类别

            # 检查类别和置信度
            if cls == FACE_CLASS_ID and conf > 0.5:  # 设置置信度阈值为 0.5
                # 将检测框信息写入文本文件
                f.write(f"Frame {frame_count}: [{x1}, {y1}, {x2}, {y2}] Confidence: {conf:.2f}\n")

# 释放资源
cap.release()
