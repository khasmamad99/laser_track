import numpy as np
import cv2

img = cv2.imread("target/data/human_bw_cnt.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, 0)
cnts, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = lambda x: cv2.contourArea(x), reverse=True)

#np.save("human.npy", cnts, allow_pickle=True)
for cnt in cnts:
    cnt_img = cv2.drawContours(img.copy(), [cnt], 0, (0,255,0), 3) 
    cv2.imshow("cnt", cnt_img)
    cv2.waitKey()