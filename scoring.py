import numpy as np
import cv2
from utils import letterbox_image
from PIL import Image
import math

def score(conts, x, y):
	for i, cont in enumerate(conts):
		test = cv2.pointPolygonTest(cont, (x, y), True)
		print(test)
		if test	>= 0:
			return 10 - i
	return 0


img = cv2.imread("target/circular1.jpg")
# img = np.array(letterbox_image(Image.fromarray(img), (800, 800)))
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(img, (5,5), 0)
edges = cv2.Canny(gray, 100, 200)
edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)


conts, hier = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
conts = sorted(conts, key = lambda x: cv2.contourArea(x))[-20:]
conts = [conts[i] for i in range(1, 20, 2)]
# conts = np.load("target/circular1.npy", allow_pickle=True)


# print(score(conts, 539, 162))



np.save("circular1.npy", np.array(conts))


# print(conts[0].shape)
# print(conts[0])
# ellipse = cv2.fitEllipse(conts[0])
# center_coords, axes_length, angle = ellipse
# axes_length = [i / 2 for i in axes_length]
# p = checkpoint(*center_coords, 81, 67, *axes_length)
# print(p)
# if p > 1: 
#     print ("Outside") 

# elif p == 1: 
#     print("On the ellipse") 

# else: 
#     print("Inside") 

# print(ellipse)
# el = img.copy()
# for cont in conts:
#     ellipse = cv2.fitEllipse(cont)
#     cv2.ellipse(el, ellipse, (0,255,0), 2)



cv2.drawContours(img, conts, -1, (0,255,0), 3)
cv2.imshow("edges", edges)
cv2.imshow("conts", img)
# cv2.imshow("ellipse", el)
cv2.waitKey(0) 

	# # Convert the circle parameters a, b and r to integers. 
	# detected_circles = np.uint16(np.around(detected_circles))[0][:10]
	# detected_circles = sorted(detected_circles, key = lambda x: x[2], reverse=True)
	# for pt in detected_circles: 
	#     a, b, r = pt[0], pt[1], pt[2]    
  
	#     # Draw the circumference of the circle. 
	#     cv2.circle(img, (a, b), r, (0, 255, 0), 2) 
  
	#     # Draw a small circle (of radius 1) to show the center. 
	#     cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
	#     cv2.imshow("Detected Circle", img) 
	#     cv2.waitKey(0) 


