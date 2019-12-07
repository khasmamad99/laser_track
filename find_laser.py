import cv2
import numpy as np

cap = cv2.VideoCapture("eyup4_atis0012.avi")
cap.set(cv2.CAP_PROP_FPS, 10)

while 1:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    print(maxVal)
    print(gray)
    break
    cv2.circle(frame, maxLoc, 5, (0,255,0), 2)

    cv2.imshow("", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()