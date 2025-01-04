import cv2
from ultralytics import YOLO

# 加载 YOLOv8 模型（使用通用模型或人脸专用权重文件）
model = YOLO("yolov8n-face.pt")

# 定义人脸类别索引（通常 YOLOv8 的 COCO 数据集中，人脸是类别 'person' 的一部分）
FACE_CLASS_ID = 0  # 如果权重专用于人脸，cls 可能为 0，需根据模型定义调整

# 打开视频文件或摄像头
video_path = "input_video.mp4"  # 替换为你的视频路径或使用 0 打开摄像头
cap = cv2.VideoCapture(video_path)

# 获取视频帧宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 初始化视频写入
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLOv8 模型进行检测
    results = model(frame)  # 直接返回结果列表

    # 遍历检测结果
    detections = results[0].boxes  # 获取检测结果中的 'boxes' 对象
    for box in detections:
        # 提取检测框信息
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 检测框的坐标
        conf = box.conf[0]  # 检测置信度
        cls = int(box.cls[0])  # 检测类别
        # print("cls: "+str(cls))

        # 检查类别和置信度
        if cls == FACE_CLASS_ID and conf > 0.5:  # 设置置信度阈值为 0.5
            # 绘制矩形框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 标注置信度
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示检测结果
    cv2.imshow("Face Detection", frame)

    # 保存处理后的视频帧
    out.write(frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
