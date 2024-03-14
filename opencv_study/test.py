import cv2
import numpy as np


def onChange(val):
    global image, title

    add_val = val - int(image[0][0])
    print("추가 화소 값 :", add_val)
    image[:] = np.clip(image + add_val, 0, 255)  # 화소 값이 0과 255 사이로 유지되도록 함
    cv2.imshow(title, image)
    
image = np.zeros((200, 400), dtype=np.uint8)  # image를 uint8 타입으로 생성
image[:] = 200

title = 'Trackbar Event'

cv2.imshow(title, image)

# image[0][0]을 int로 캐스팅
cv2.createTrackbar("brightness", title, int(image[0][0]), 255, onChange)
cv2.waitKey(0)
cv2.destroyAllWindows()
