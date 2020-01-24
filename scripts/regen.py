import glob, os
import cv2
import numpy as np

def convert(size, box):
	dw = 1./size[0]
	dh = 1./size[1]
	x = box[0] + box[2]/2.0
	y = box[1] + box[3]/2.0
	w = box[2]
	h = box[3]
	x = x*dw
	w = w*dw
	y = y*dh
	h = h*dh
	return (x,y,w,h)



def detect_laser(image, dilate=False, erode=False, subtractor=cv2.bgsegm.createBackgroundSubtractorMOG()):
	# resize image
	h, w, c = image.shape
	scale = min(640/w, 480/h)
	im = cv2.resize(image, (int(w*scale), int(h*scale)))

	# find mask
	mask = subtractor.apply(image, None, learningRate=0)
	kernel = np.ones((5, 5), np.uint8)
	if erode:
		mask = cv2.erode(mask, kernel, iterations=1)
	if dilate:
		mask = cv2.dilate(mask, kernel, iterations=1)

	#cv2.imshow("mask", mask)

	# find avg
	avg = cv2.mean(mask)[0]
	if avg > 1 or avg < 0.0001:
		# if change in the image is too much (change in light/brightness), ignore it
		if avg > 1:
			print("Make sure that the camera is stable!")
		return None

	else:
		contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		if contours:
			cnt = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[0]
			cont = cv2.drawContours(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), [cnt], 0, (0, 255, 0), 3)
			#cv2.imshow("cont", cont)
			print("AREA:", cv2.contourArea(cnt))
			if cv2.contourArea(cnt) > 0:
				# M = cv2.moments(cnt)
				# cx = int(M['m10']/M['m00'])
				# cy = int(M['m01']/M['m00'])
				x, y, w, h = cv2.boundingRect(cnt)
				return (x, y, w, h)

			else:
				return None
		else:
			return None




# images = glob.glob("data/vid_1*gray.jpg")
# _ = detect_laser(cv2.imread("data/vid_1_0_gray.jpg"))
images = glob.glob("data/vid_5*gray.jpg")
_ = detect_laser(cv2.imread("data/vid_5_0_gray.jpg"))
for img in images:
	name = os.path.splitext(img)[0]
	print(name)
	frame = cv2.imread(img)
	frame_h, frame_w, _ = frame.shape
	box = detect_laser(frame, dilate=True)
	text = ""
	frame_box = frame.copy()
	print(box)
	if box:
		x, y, w, h = box
		frame_box = cv2.rectangle(frame.copy(),(x,y),(x+w,y+h),(0,255,0),2)
		box_norm = convert((frame_w, frame_h), box)
		text = "1 " + " ".join([str(a) for a in box_norm])
	open(name+".txt", "w").write(text)
	cv2.imshow("frame", frame_box)
	key = cv2.waitKey(1)
	if key == ord('q'):
		break