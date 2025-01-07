import cv2
import mediapipe as mp
from ultralytics import YOLO

# 加载 YOLOv8 模型（使用通用模型或人脸专用权重文件）
model = YOLO("yolov8n-face.pt")

# 初始化 Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0, min_tracking_confidence=0.4)

# 定义人脸类别索引（通常 YOLOv8 的 COCO 数据集中，人脸是类别 'person' 的一部分）
FACE_CLASS_ID = 0  # 如果权重专用于人脸，cls 可能为 0，需根据模型定义调整

# 打开视频文件或摄像头
video_path = "input_video.mp4"  # 替换为你的视频路径或使用 0 打开摄像头
cap = cv2.VideoCapture(video_path)

# 获取视频帧宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义嘴唇关键点索引
UPPER_LIP = [0]  # 下嘴唇上边界点
LOWER_LIP = [17]  # 上嘴唇下边界点

# 初始化视频写入
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (frame_width, frame_height))

def is_mouth_open(landmarks, threshold=0.005):
    """
    判断嘴唇是否张开
    :param landmarks: 面部关键点
    :param threshold: 嘴唇开合的最小距离阈值（像素）
    :return: 是否张嘴（True/False）
    """
    return abs(landmarks[UPPER_LIP[0]].y - landmarks[LOWER_LIP[0]].y) > threshold

faceOutput = "faceDetections.txt"
speakOutput = "speakDetections.txt"
with open(faceOutput, "w") as f1, open(speakOutput, "w") as f2:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # 使用 YOLO 检测人脸
        results = model(frame)
        detections = results[0].boxes if len(results) > 0 else []

        for box in detections:
            # 提取人脸框的坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]

            if conf > 0.5:
                f1.write(f"Frame {frame_count}: [{x1}, {y1}, {x2}, {y2}] Confidence: {conf:.2f}\n")
                # 裁剪人脸区域
                face_region = frame[y1:y2, x1:x2]

                # 使用 Mediapipe 判断嘴部是否张开
                rgb_face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                face_results = face_mesh.process(rgb_face_region)

                mouth_open = False
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        # 判断嘴唇是否张开
                        mouth_open = is_mouth_open(face_landmarks.landmark, threshold=0.005)

                        upper_lip_y = y1 + int(face_landmarks.landmark[UPPER_LIP[0]].y * face_region.shape[0])
                        lower_lip_y = y1 + int(face_landmarks.landmark[LOWER_LIP[0]].y * face_region.shape[0])

                        if mouth_open:
                            f2.write(f"Frame {frame_count}: [{x1}, {y1}, {x2}, {y2}]\n")
                        # 绘制嘴唇区域蓝色框
                        if 0<=abs(face_landmarks.landmark[UPPER_LIP[0]].y)<=1 and 0<=abs(face_landmarks.landmark[LOWER_LIP[0]].y)<=1:
                            cv2.rectangle(frame, (x1, upper_lip_y), (x2, lower_lip_y), (255, 0, 0), 1)


                # 设置框的颜色
                color = (0, 0, 255) if mouth_open else (0, 255, 0)  # 红色：张嘴，绿色：闭嘴

                # 绘制人脸框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

                # 显示嘴唇状态
                status_text = "Speaking" if mouth_open else "Mouth Closed"
                cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 显示结果
        cv2.imshow("Face and Mouth Detection", frame)

        # 保存处理后的视频帧
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
