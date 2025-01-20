import cv2
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("/Users/li/Desktop/Projects/HandAI/best.pt")  # 替换为你的权重路径

# 打开摄像头
cap = cv2.VideoCapture(0)  # 摄像头索引，0 是默认摄像头

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 实时检测
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头帧")
        break

    # 运行模型推理
    results = model.predict(source=frame, save=False, conf=0.3, show=False)

    # 绘制检测结果
    annotated_frame = results[0].plot() 

    # 显示实时画面
    cv2.imshow("Hand Keypoints Detection", annotated_frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
