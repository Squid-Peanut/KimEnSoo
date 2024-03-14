import cv2
import numpy as np

# YOLOv3 모델과 가중치 파일 경로 설정
model_cfg = 'yolov3.cfg'
model_weights = 'yolov3.weights'

# YOLO 네트워크 불러오기
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 클래스 이름 불러오기 (여기서는 COCO 데이터셋의 클래스 사용)
classes = open('coco.names').read().strip().split('\n')

# 동영상 파일 불러오기
cap = cv2.VideoCapture('nba_free_throw.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지에서 객체 감지 실행
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # 감지된 객체에 대한 정보 처리
    for detection in detections:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # 사람과 공 감지
            if classes[class_id] in ['person', 'sports ball']:
                # 감지된 객체의 바운딩 박스 계산
                center_x, center_y, width, height = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                cv2.rectangle(frame, (x, y), (x + int(width), y + int(height)), (255, 0, 0), 2)
                cv2.putText(frame, f'{classes[class_id]}: {int(confidence * 100)}%', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 결과 이미지 표시
    cv2.imshow('NBA Free Throw Analysis', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
