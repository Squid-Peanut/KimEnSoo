import cv2
import numpy as np

# 비디오 파일을 로드합니다.
cap = cv2.VideoCapture('nba_free_throw_video.mp4')

# 공과 슈터를 인식하기 위한 분류기를 로드합니다.
# 여기서는 예시로 'ball_cascade.xml'과 'shooter_cascade.xml' 파일명을 사용합니다.
# 실제 파일명은 사용하는 분류기에 따라 다를 수 있습니다.
ball_cascade = cv2.CascadeClassifier('ball_cascade.xml')
shooter_cascade = cv2.CascadeClassifier('shooter_cascade.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일 이미지로 변환합니다.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 공을 검출합니다.
    balls = ball_cascade.detectMultiScale(gray, 1.1, 4)
    # 슈터를 검출합니다.
    shooters = shooter_cascade.detectMultiScale(gray, 1.1, 4)

    # 검출된 공과 슈터에 대한 정보를 화면에 표시합니다.
    for (x, y, w, h) in balls:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    for (x, y, w, h) in shooters:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 결과 프레임을 표시합니다.
    cv2.imshow('Frame', frame)

    # 'q'를 누르면 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처를 종료합니다.
cap.release()
cv2.destroyAllWindows()
